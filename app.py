# Stock Price Prediction Model for Indian Market (NSE/BSE) with LSTM, XGBoost, Streamlit Dashboard, and Backtesting

import numpy as np
import pandas as pd
import os
import requests
import logging
from datetime import datetime
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import yfinance as yf
import ta  # Technical indicators
import streamlit as st
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Evaluation Metrics

def evaluate_model(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
    actual = y_test.numpy()
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    direction_accuracy = np.mean(np.sign(actual[1:] - actual[:-1]) == np.sign(predictions[1:] - predictions[:-1]))
    return rmse, mae, direction_accuracy

# XGBoost Comparison

def xgboost_model(data: pd.DataFrame):
    features = data[['Adj Close', 'Volume', 'Return', 'RSI', 'MACD', 'BB_upper', 'BB_lower']]
    targets = features['Adj Close'].shift(-1).dropna()
    features = features.iloc[:-1]
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(features, targets)
    predictions = model.predict(features)
    return predictions, targets

# Backtesting Strategy

def backtest(predictions, actuals):
    signals = np.where(predictions > actuals, 1, -1)
    daily_returns = np.diff(actuals) / actuals[:-1]
    strategy_returns = signals[:-1] * daily_returns
    cumulative_return = np.cumprod(1 + strategy_returns)[-1]
    return cumulative_return

# Streamlit Dashboard

def streamlit_dashboard():
    st.title("📈 Indian Stock Price Prediction Dashboard")
    ticker = st.text_input("Enter NSE Ticker (e.g. RELIANCE)", value="RELIANCE")
    if st.button("Predict and Visualize"):
        ticker += ".NS" if not ticker.endswith(".NS") else ""
        df = download_stock_data(ticker, "2020-01-01", datetime.today().strftime('%Y-%m-%d"))
        predictions, targets = xgboost_model(df)
        cumulative = backtest(predictions, targets.values)
        st.line_chart({"Actual": targets.values, "Predicted": predictions})
        st.success(f"📊 Cumulative return of strategy: {cumulative:.2f}x")

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
        return JSONResponse(content={"ticker": ticker, "predicted_price": round(predicted_price, 2)})

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

    rmse, mae, da = evaluate_model(model, X, y, scaler)
    logger.info(f"Model Evaluation — RMSE: {rmse:.2f}, MAE: {mae:.2f}, Directional Accuracy: {da:.2%}")

    # Launch FastAPI and Streamlit (as separate processes if desired)
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To run dashboard separately: streamlit run <filename>.py
