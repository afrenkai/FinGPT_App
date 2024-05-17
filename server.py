import os
import re
import time
import json
import random
import finnhub
import torch
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import date, datetime, timedelta
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
from collections import defaultdict

app = Flask(__name__)
CORS(app)

access_token = os.getenv("HF_TOKEN")
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token,
    trust_remote_code=True, 
    device_map="auto",
    load_in_8bit=True,
    offload_folder="offload/"
)
model = PeftModel.from_pretrained(
    base_model,
    'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora',
    offload_folder="offload/"
)
model = model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token=access_token
)

streamer = TextStreamer(tokenizer)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns "
    "for companies based on relevant news and basic financials from the past weeks, then provide an analysis and "
    "prediction for the companies' stock price movement for the upcoming week. "
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."
)

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def get_curday():
    return date.today().strftime("%Y-%m-%d")

def n_weeks_before(date_string, n):
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    return date.strftime("%Y-%m-%d")

def get_stock_data(stock_symbol, steps):
    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    if len(stock_data) == 0:
        raise Exception(f"Failed to download stock price data for symbol {stock_symbol} from yfinance!")
    
    dates, prices = [], []
    available_dates = stock_data.index.format()
    
    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })

def get_news(symbol, data):
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        time.sleep(1)  # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        if len(weekly_news) == 0:
            raise Exception(f"No company news found for symbol {symbol} from finnhub!")
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data

def get_company_prompt(symbol):
    profile = finnhub_client.company_profile2(symbol=symbol)
    if not profile:
        raise Exception(f"Failed to find company profile for symbol {symbol} from finnhub!")
        
    company_template = (
        "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. "
        "Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. "
        "As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding.\n\n"
        "{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."
    )

    formatted_str = company_template.format(**profile)
    return formatted_str

def get_prompt_by_row(symbol, row):
    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics

def sample_news(news, k=5):
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]

def get_current_basics(symbol, curday):
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    if not basic_financials['series']:
        raise Exception(f"Failed to find basic financials for symbol {symbol} from finnhub!")
        
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic

def get_all_prompts_online(symbol, data, curday, with_basics=True):
    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        prompt += "\n".join(sampled_news)
    
    if with_basics:
        prompt += "\n" + prev_rows[0][2]
    
    return company_prompt + prompt

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol')
    weeks = data.get('weeks', 4)
    
    curday = get_curday()
    steps = [n_weeks_before(curday, i) for i in range(weeks+1)][::-1]
    
    stock_data = get_stock_data(symbol, steps)
    news_data = get_news(symbol, stock_data)
    
    company_prompt = get_company_prompt(symbol)
    all_prompts = get_all_prompts_online(symbol, news_data, curday)
    
    input_prompt = SYSTEM_PROMPT + B_INST + B_SYS + company_prompt + E_SYS + all_prompts + E_INST
    
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, streamer=streamer)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'prediction': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
