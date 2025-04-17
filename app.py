# app.py — Complete NSE Stock Prediction App with Date Range, Chart, CSV, and UI

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import ta

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App and templates setup
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)
PRED_HISTORY_PATH = BASE_DIR / "prediction_history.json"
if not PRED_HISTORY_PATH.exists():
    with open(PRED_HISTORY_PATH, 'w') as f:
        json.dump([], f)

# Common tickers for autocomplete
COMMON_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "WIPRO"]

# LSTM model definition
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# Utilities

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df['Return'] = df['Adj Close'].pct_change()
    df['RSI'] = ta.momentum.RSIIndicator(df['Adj Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Adj Close']).macd()
    bb = ta.volatility.BollingerBands(df['Adj Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def prepare_data(data: pd.DataFrame, lookback: int = 10) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    scaler = MinMaxScaler()
    features = data[['Adj Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].values
    scaled = scaler.fit_transform(features)
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 0])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1), scaler

def train_model(model, loader, epochs: int = 10, lr: float = 0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

def save_prediction(ticker: str, price: float):
    with open(PRED_HISTORY_PATH, 'r+') as f:
        history = json.load(f)
        history.append({"ticker": ticker, "price": price, "date": str(datetime.now())})
        f.seek(0)
        json.dump(history, f, indent=2)

# Route
@app.post("/predict", response_class=HTMLResponse)
async def predict_submit(request: Request, ticker: str = Form(...), start: str = Form(...), end: str = Form(...)):
    try:
        full_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        data = download_stock_data(full_ticker, start, end)
        X, y, scaler = prepare_data(data)
        model = StockPriceLSTM(input_size=7, hidden_size=64, num_layers=2)
        loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
        train_model(model, loader)
        prediction = model(X[-1].unsqueeze(0)).item()
        predicted_price = scaler.inverse_transform([[prediction] + [0]*6])[0, 0]
        save_prediction(ticker.upper(), predicted_price)

        with open(PRED_HISTORY_PATH) as f:
            history = json.load(f)[-5:]
        chart = data['Adj Close'].tail(30).tolist()

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": round(predicted_price, 2),
            "ticker": ticker.upper(),
            "history": history,
            "tickers": COMMON_TICKERS,
            "chart": chart
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": "Error occurred",
            "ticker": ticker,
            "history": [],
            "tickers": COMMON_TICKERS,
            "chart": []
        })

# Generate index.html
html_path = TEMPLATES_DIR / "index.html"
if not html_path.exists():
    html_path.write_text("""
<!DOCTYPE html>
<html>
<head>
  <title>NSE Stock Predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: 'Segoe UI', sans-serif; padding: 30px; background-color: #f4f4f4; }
    .card { background: #fff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); max-width: 800px; margin: auto; }
    input, button { padding: 8px; margin: 8px; border-radius: 5px; border: 1px solid #ccc; }
    h1, h2 { color: #333; }
    .chart-wrapper { margin-top: 30px; }
    label { font-weight: bold; }
  </style>
</head>
<body>
  <div class="card">
    <h1>📈 NSE Stock Predictor</h1>
    <form action="/predict" method="post">
      <label for="ticker">Ticker:</label>
      <input list="tickers" name="ticker" placeholder="e.g. RELIANCE" required>
      <datalist id="tickers">{% for t in tickers %}<option value="{{ t }}">{% endfor %}</datalist><br>

      <label for="start">Start Date:</label>
      <input type="date" name="start" required>
      <label for="end">End Date:</label>
      <input type="date" name="end" required>

      <button type="submit">Predict</button>
      <button type="button" onclick="downloadCSV()">⬇ Export Predictions</button>
    </form>

    {% if prediction %}
      <h2>Predicted Price for {{ ticker }}: ₹{{ prediction }}</h2>
      <div class="chart-wrapper">
        <canvas id="adjCloseChart"></canvas>
      </div>
    {% endif %}

    {% if history %}
      <h3>📜 Recent Predictions</h3>
      <ul>{% for item in history %}<li>{{ item.date }} — {{ item.ticker }}: ₹{{ item.price }}</li>{% endfor %}</ul>
    {% endif %}
  </div>

  <script>
    {% if chart %}
    const ctx = document.getElementById('adjCloseChart').getContext('2d');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({length: {{ chart|length }}}, (_, i) => i + 1),
        datasets: [{ label: 'Adj Close', data: {{ chart|tojson }}, borderColor: 'blue', tension: 0.3 }]
      },
      options: { responsive: true, scales: { y: { beginAtZero: false } } }
    });
    {% endif %}

    function downloadCSV() {
      const csv = `Ticker,Price,Date\n` + {{ history|tojson }}.map(p => `${p.ticker},${p.price},${p.date}`).join("\n");
      const blob = new Blob([csv], { type: 'text/csv' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'prediction_history.csv';
      link.click();
    }
  </script>
</body>
</html>
""")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
