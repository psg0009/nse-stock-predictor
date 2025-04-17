# app.py — FastAPI App with HTML UI for NSE Stock Prediction

import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import yfinance as yf
import ta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI(title="NSE Stock Price Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# LSTM Model Definition
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(StockPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# Download stock data using yfinance for NSE/BSE

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

# Prepare Data

def prepare_data(data: pd.DataFrame, lookback: int = 10) -> Tuple[torch.Tensor, torch.Tensor, MinMaxScaler]:
    scaler = MinMaxScaler()
    features = data[['Adj Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].values
    scaled_features = scaler.fit_transform(features)

    X, y = [], []
    for i in range(lookback, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_features[i, 0])

    X, y = np.array(X), np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1), scaler

# Train Model

def train_model(model, train_loader, epochs: int = 50, lr: float = 0.001):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")
    torch.save(model.state_dict(), "nse_lstm_model.pth")
    logger.info("Model saved")

# Root Route for HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API Endpoint
@app.get("/predict/{ticker}")
async def predict_price(ticker: str):
    try:
        full_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
        data = download_stock_data(full_ticker, "2020-01-01", datetime.today().strftime('%Y-%m-%d'))
        X, y, scaler = prepare_data(data)
        model = StockPriceLSTM(input_size=7, hidden_size=64, num_layers=2)
        if os.path.exists("nse_lstm_model.pth"):
            model.load_state_dict(torch.load("nse_lstm_model.pth"))
        else:
            raise HTTPException(status_code=404, detail="Model not trained")

        X_input = X[-1].unsqueeze(0)
        with torch.no_grad():
            prediction = model(X_input).item()
        predicted_price = scaler.inverse_transform([[prediction] + [0]*6])[0, 0]
        return JSONResponse(content={"ticker": ticker.upper(), "predicted_price": round(predicted_price, 2)})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run training if script
if __name__ == "__main__":
    ticker = "RELIANCE.NS"
    data = download_stock_data(ticker, "2018-01-01", "2024-12-31")
    X, y, scaler = prepare_data(data)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = StockPriceLSTM(input_size=7, hidden_size=64, num_layers=2)
    train_model(model, loader)

    uvicorn.run(app, host="0.0.0.0", port=8000)
