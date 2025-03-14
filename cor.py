import yfinance as yf
import ccxt
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch BTC price from Binance
def fetch_btc_data():
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=365*2)
    btc_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df.set_index('timestamp', inplace=True)
    return btc_df[['close']]

# Function to fetch Gold and SPY data from Yahoo Finance
def fetch_yf_data(ticker):
    df = yf.download(ticker, period='2y', interval='1d')
    return df[['Adj Close']].rename(columns={'Adj Close': ticker})

# Fetch data
btc_df = fetch_btc_data()
gold_df = fetch_yf_data('GC=F')  # Gold futures
spy_df = fetch_yf_data('SPY')    # S&P 500 ETF

# Merge datasets
df = btc_df.join([gold_df, spy_df], how='inner')

# Rename columns
df.columns = ['BTC', 'Gold', 'SPY']

# Calculate daily returns
returns = df.pct_change().dropna()

# Calculate rolling correlations (90-day window)
rolling_corr_btc_gold = returns['BTC'].rolling(window=90).corr(returns['Gold'])
rolling_corr_btc_spy = returns['BTC'].rolling(window=90).corr(returns['SPY'])

# Plot correlations
plt.figure(figsize=(12, 6))
plt.plot(rolling_corr_btc_gold, label='BTC/Gold Correlation', color='gold')
plt.plot(rolling_corr_btc_spy, label='BTC/SPY Correlation', color='blue')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.6)
plt.legend()
plt.title('BTC/Gold and BTC/SPY Rolling 90-Day Correlation')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.grid(True)
plt.show()
