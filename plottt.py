import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
import time
import numpy as np
# Import Historic-Crypto library
from Historic_Crypto import HistoricalData
from Historic_Crypto import Cryptocurrencies

def fetch_bitcoin_data(days=1825):  # 5 years = 365 * 5 = 1825 days
    """
    Fetch Bitcoin price data using Historic-Crypto
    
    Parameters:
    days (int): Number of days of historical data to fetch (default: 1825)
    
    Returns:
    pandas.DataFrame: DataFrame with date and price data
    """
    try:
        print("Fetching Bitcoin data using Historic-Crypto...")
        
        # Calculate start date in the required format for Historic-Crypto: YYYY-MM-DD-HH-MM
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d-00-00')
        
        # Historic-Crypto uses granularity in seconds
        # 86400 = daily data (24*60*60)
        granularity = 86400
        
        # Use Historic-Crypto to fetch the data
        # BTC-USD is the ticker for Bitcoin on Coinbase Pro
        btc_data = HistoricalData('BTC-USD', granularity, start_date).retrieve_data()
        
        if not btc_data.empty:
            print("Successfully fetched Bitcoin data from Historic-Crypto")
            
            # Rename columns to match our expected format
            # Historic-Crypto returns columns with names like 'close', 'open', etc.
            # Convert the index to a 'date' column and use 'close' as our 'price'
            df = btc_data.reset_index().rename(columns={'time': 'date', 'close': 'price'})
            
            # Keep only the date and price columns
            df = df[['date', 'price']]
            
            # Convert date to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])
            
            return df
        else:
            print("No Bitcoin data returned from Historic-Crypto")
            
            # Fall back to yfinance if Historic-Crypto fails
            print("Falling back to yfinance for Bitcoin data...")
            start_date_yf = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date_yf = datetime.now().strftime('%Y-%m-%d')
            
            # BTC-USD is the Yahoo Finance ticker for Bitcoin
            btc_data_yf = yf.download('BTC-USD', start=start_date_yf, end=end_date_yf, progress=False)
            
            if not btc_data_yf.empty:
                print("Successfully fetched Bitcoin data from yfinance")
                
                # Use Adj Close if available, otherwise use Close
                if 'Adj Close' in btc_data_yf.columns:
                    price_col = 'Adj Close'
                else:
                    price_col = 'Close'
                
                # Create a DataFrame with date and price
                df = pd.DataFrame({
                    'date': btc_data_yf.index,
                    'price': btc_data_yf[price_col].values
                })
                
                return df
            else:
                print("No Bitcoin data returned from yfinance")
                return None
    
    except Exception as e:
        print(f"Error fetching Bitcoin data from Historic-Crypto: {e}")
        
        # Fall back to yfinance if Historic-Crypto fails
        try:
            print("Falling back to yfinance for Bitcoin data...")
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # BTC-USD is the Yahoo Finance ticker for Bitcoin
            btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
            
            if not btc_data.empty:
                print("Successfully fetched Bitcoin data from yfinance")
                
                # Use Adj Close if available, otherwise use Close
                if 'Adj Close' in btc_data.columns:
                    price_col = 'Adj Close'
                else:
                    price_col = 'Close'
                
                # Create a DataFrame with date and price
                df = pd.DataFrame({
                    'date': btc_data.index,
                    'price': btc_data[price_col].values
                })
                
                return df
            else:
                print("No Bitcoin data returned from yfinance")
                return None
        except Exception as e2:
            print(f"Error fetching Bitcoin data from yfinance: {e2}")
            return None

def fetch_stock_data(tickers, days=1825):  # 5 years = 365 * 5 = 1825 days
    """
    Fetch stock price data using yfinance
    
    Parameters:
    tickers (list): List of ticker symbols
    days (int): Number of days of historical data to fetch
    
    Returns:
    dict: Dictionary of DataFrames with date and price data for each ticker
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    result = {}
    
    # Try downloading all tickers in a single request for better performance
    try:
        # Download all tickers at once
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        print(f"Successfully downloaded data for multiple tickers")
        
        # Handle the data for each ticker separately
        for ticker in tickers:
            try:
                # Handle the multi-level columns that yfinance returns
                # The columns are like ('Adj Close', 'QQQ'), ('Close', 'QQQ'), etc.
                if isinstance(data.columns, pd.MultiIndex):
                    # Select the 'Adj Close' column for this ticker
                    ticker_data = data['Adj Close'][ticker]
                    
                    # Create a DataFrame with date and price
                    df = pd.DataFrame({
                        'date': ticker_data.index,
                        'price': ticker_data.values
                    })
                    
                    result[ticker] = df
                    print(f"Successfully processed {ticker} data from multi-ticker download")
                else:
                    print(f"Unexpected column format in multi-ticker download")
                    # Fall back to individual downloads
                    raise ValueError("Column format not as expected")
                    
            except Exception as e:
                print(f"Error processing {ticker} from multi-ticker download: {e}")
                # If we failed for this ticker, we'll try downloading it individually
                result[ticker] = None
    
    except Exception as e:
        print(f"Error in multi-ticker download: {e}")
        # If multi-ticker download fails, fall back to individual downloads
        print("Falling back to individual ticker downloads")
    
    # For any tickers that failed in the multi-ticker approach, try them individually
    for ticker in tickers:
        if ticker not in result or result[ticker] is None:
            try:
                print(f"Downloading {ticker} individually")
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if ticker_data.empty:
                    print(f"No data returned for {ticker}")
                    result[ticker] = None
                    continue
                
                # Try to get the closing price
                if 'Adj Close' in ticker_data.columns:
                    price_col = 'Adj Close'
                elif 'Close' in ticker_data.columns:
                    price_col = 'Close'
                else:
                    price_col = ticker_data.columns[0]
                    print(f"Warning: Using {price_col} for {ticker} as a fallback")
                
                # Create a DataFrame with date and price
                df = pd.DataFrame({
                    'date': ticker_data.index,
                    'price': ticker_data[price_col].values
                })
                
                result[ticker] = df
                print(f"Successfully downloaded {ticker} individually")
                
            except Exception as e:
                print(f"Error downloading {ticker} individually: {e}")
                result[ticker] = None
    
    return result

def normalize_data(dfs):
    """
    Normalize price data to percentage change from first day
    
    Parameters:
    dfs (dict): Dictionary of DataFrames with date and price data
    
    Returns:
    dict: Dictionary of DataFrames with normalized price data
    """
    normalized = {}
    
    for name, df in dfs.items():
        if df is not None and not df.empty:
            df_norm = df.copy()
            first_price = df_norm['price'].iloc[0]
            if first_price > 0:  # Ensure we don't divide by zero
                df_norm['normalized'] = (df_norm['price'] / first_price) * 100
                normalized[name] = df_norm
            else:
                print(f"Warning: First price for {name} is zero or negative. Skipping normalization.")
    
    return normalized

def plot_comparison(dfs, title="Asset Price Comparison", save_path=None):
    """
    Plot multiple asset prices on the same chart
    
    Parameters:
    dfs (dict): Dictionary of DataFrames with date and price data
    title (str): Plot title
    save_path (str): Path to save the plot image (if None, display the plot)
    """
    if not dfs:
        print("No data available for comparison plot")
        return
    plt.figure(figsize=(14, 8))
    
    colors = {
        'BTC': 'orange',
        'QQQ': 'blue',
        'SPY': 'green',
        'GLD': 'gold'
    }
    
    # Plot normalized prices
    for name, df in dfs.items():
        if df is not None and not df.empty:
            plt.plot(df.date, df.normalized, color=colors.get(name, 'gray'), linewidth=2, label=name)
    
    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Normalized Price (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format x-axis date labels - adjust for 5 years of data
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months for 5 years
    plt.xticks(rotation=45)
    
    # Add annotations for latest values
    for name, df in dfs.items():
        if df is not None and not df.empty:
            latest_value = df.normalized.iloc[-1]
            plt.annotate(f'{name}: {latest_value:.1f}%', 
                        xy=(df.date.iloc[-1], latest_value),
                        xytext=(10, 10 * (list(dfs.keys()).index(name) - len(dfs) / 2)),
                        textcoords='offset points',
                        color=colors.get(name, 'gray'),
                        fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_individual_prices(dfs, title="Asset Prices", save_path=None):
    """
    Plot multiple asset actual prices with separate y-axes
    
    Parameters:
    dfs (dict): Dictionary of DataFrames with date and price data
    title (str): Plot title
    save_path (str): Path to save the plot image (if None, display the plot)
    """
    if not dfs:
        print("No data available for individual prices plot")
        return
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    colors = {
        'BTC': 'orange',
        'QQQ': 'blue',
        'SPY': 'green'
    }
    
    # Primary y-axis for Bitcoin
    if 'BTC' in dfs and dfs['BTC'] is not None and not dfs['BTC'].empty:
        btc_df = dfs['BTC']
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Bitcoin Price (USD)', color=colors['BTC'], fontsize=12)
        ax1.plot(btc_df.date, btc_df.price, color=colors['BTC'], linewidth=2, label='BTC')
        ax1.tick_params(axis='y', labelcolor=colors['BTC'])
        
        # Annotate Bitcoin price
        latest_btc = btc_df.price.iloc[-1]
        ax1.annotate(f'BTC: ${latest_btc:.2f}', 
                     xy=(btc_df.date.iloc[-1], latest_btc),
                     xytext=(10, 15),
                     textcoords='offset points',
                     color=colors['BTC'],
                     fontweight='bold')
    
    # Secondary y-axis for QQQ and SPY
    ax2 = ax1.twinx()
    ax2.set_ylabel('ETF Price (USD)', color='blue', fontsize=12)
    
    etf_plotted = False
    # Plot QQQ and SPY
    for ticker in ['QQQ', 'SPY']:
        if ticker in dfs and dfs[ticker] is not None and not dfs[ticker].empty:
            df = dfs[ticker]
            etf_plotted = True
            ax2.plot(df.date, df.price, color=colors[ticker], linewidth=2, label=ticker)
            
            # Annotate ETF price
            latest_price = df.price.iloc[-1]
            ax2.annotate(f'{ticker}: ${latest_price:.2f}', 
                         xy=(df.date.iloc[-1], latest_price),
                         xytext=(10, -15 if ticker == 'QQQ' else -35),
                         textcoords='offset points',
                         color=colors[ticker],
                         fontweight='bold')
    
    if etf_plotted:
        ax2.tick_params(axis='y', labelcolor='blue')
    
    # Format x-axis date labels - adjust for 5 years of data
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months for 5 years
    plt.xticks(rotation=45)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def calculate_rolling_correlation(dfs, window=30):
    """
    Calculate rolling correlation between Bitcoin and other assets
    
    Parameters:
    dfs (dict): Dictionary of DataFrames with date and price data
    window (int): Rolling window size in days
    
    Returns:
    dict: Dictionary of DataFrames with date and correlation data
    """
    correlations = {}
    
    # Ensure we have Bitcoin data
    if 'BTC' not in dfs or dfs['BTC'] is None or dfs['BTC'].empty:
        print("Bitcoin data not available for correlation calculation")
        return correlations
    
    btc_df = dfs['BTC']
    btc_returns = btc_df['price'].pct_change().dropna()
    
    for ticker in ['QQQ', 'SPY', 'GLD']:
        if ticker in dfs and dfs[ticker] is not None and not dfs[ticker].empty:
            # Get ticker data and calculate daily returns
            ticker_df = dfs[ticker]
            
            # Align dates between BTC and ticker
            common_dates = pd.merge(
                btc_df[['date', 'price']],
                ticker_df[['date', 'price']], 
                on='date', 
                how='inner',
                suffixes=('_btc', f'_{ticker.lower()}')
            )
            
            if common_dates.empty:
                print(f"No overlapping dates between BTC and {ticker}")
                continue
                
            # Calculate daily returns
            common_dates['return_btc'] = common_dates['price_btc'].pct_change()
            common_dates[f'return_{ticker.lower()}'] = common_dates[f'price_{ticker.lower()}'].pct_change()
            
            # Drop the first row (NaN returns)
            common_dates = common_dates.dropna()
            
            # Calculate rolling correlation
            common_dates[f'correlation'] = common_dates['return_btc'].rolling(window=window).corr(
                common_dates[f'return_{ticker.lower()}']
            )
            
            correlations[f'BTC_{ticker}'] = common_dates[['date', 'correlation']]
    
    return correlations

def plot_correlations(correlations, title="Rolling Correlation with Bitcoin", window=30, save_path=None):
    """
    Plot rolling correlations between Bitcoin and other assets
    
    Parameters:
    correlations (dict): Dictionary of DataFrames with date and correlation data
    title (str): Plot title
    window (int): Rolling window size in days used for calculation
    save_path (str): Path to save the plot image (if None, display the plot)
    """
    if not correlations:
        print("No correlation data to plot")
        return
        
    plt.figure(figsize=(14, 8))
    
    colors = {
        'BTC_QQQ': 'blue',
        'BTC_SPY': 'green',
        'BTC_GLD': 'gold'
    }
    
    # Plot correlation lines
    for name, df in correlations.items():
        if df is not None and not df.empty:
            plt.plot(df.date, df.correlation, color=colors.get(name, 'gray'), linewidth=2, label=name)
    
    # Add a horizontal line at zero correlation
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Set title and labels
    plt.title(f"{title} ({window}-Day Rolling Window)", fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set y-axis limits
    plt.ylim(-1.1, 1.1)
    
    # Format x-axis date labels - adjust for 5 years of data
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show every 6 months for 5 years
    plt.xticks(rotation=45)
    
    # Add annotations for latest correlation values
    for name, df in correlations.items():
        if df is not None and not df.empty and not df.correlation.isna().all():
            latest_value = df.correlation.dropna().iloc[-1]
            if not np.isnan(latest_value):
                plt.annotate(f'{name}: {latest_value:.2f}', 
                            xy=(df.date.iloc[-1], latest_value),
                            xytext=(10, 10 * (list(correlations.keys()).index(name) - len(correlations) / 2)),
                            textcoords='offset points',
                            color=colors.get(name, 'gray'),
                            fontweight='bold')
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    days = 1825  # 5 years = 365 * 5 = 1825 days
    
    print(f"Fetching asset price data for the last {days} days (approximately 5 years)...")
    
    # Fetch Bitcoin data using Historic-Crypto
    btc_data = fetch_bitcoin_data(days=days)
    
    # Fetch stock and ETF data
    stock_tickers = ['QQQ', 'SPY', 'GLD']
    stock_data = fetch_stock_data(stock_tickers, days=days)
    
    # Print raw stock data format
    print("\nRaw Stock Data Format Examples:")
    for ticker, df in stock_data.items():
        if df is not None and not df.empty:
            print(f"\n{ticker} Data Sample:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 5 rows:")
            print(df.head())
            print(f"\nData types:")
            print(df.dtypes)
    
    # Combine all data
    all_data = {'BTC': btc_data}
    all_data.update(stock_data)
    
    # Check if data was retrieved successfully
    data_status = {name: "Retrieved" if df is not None and not df.empty else "Failed" 
                  for name, df in all_data.items()}
    print("\nData Retrieval Status:")
    for name, status in data_status.items():
        print(f"{name}: {status}")
    
    # Filter out None values and empty DataFrames
    valid_data = {name: df for name, df in all_data.items() if df is not None and not df.empty}
    
    if valid_data:
        # Display statistics for each asset
        print("\nAsset Price Statistics:")
        for name, df in valid_data.items():
            if not df.empty:
                print(f"\n{name}:")
                try:
                    print(f"Current Price: ${df.price.iloc[-1]:.2f}")
                    print(f"5-Year High: ${df.price.max():.2f}")
                    print(f"5-Year Low: ${df.price.min():.2f}")
                    print(f"5-Year Change: {((df.price.iloc[-1] / df.price.iloc[0]) - 1) * 100:.2f}%")
                except (IndexError, ValueError) as e:
                    print(f"Error calculating statistics: {e}")
        
        # Normalize data for percentage comparison
        normalized_data = normalize_data(valid_data)
        
        # Plot the data
        print("\nGenerating plots...")
        plot_comparison(normalized_data, title=f"Asset Performance Comparison - Last 5 Years")
        plot_individual_prices(valid_data, title=f"Asset Prices - Last 5 Years")
        
        # Check if Bitcoin data is available for correlation analysis
        has_btc = 'BTC' in valid_data and valid_data['BTC'] is not None and not valid_data['BTC'].empty
        
        if has_btc:
            print("\nCalculating rolling correlations...")
            
            # Use 60-day rolling correlation for 5-year data (adjusted from 30 days)
            correlation_window = 60
            correlations = calculate_rolling_correlation(valid_data, window=correlation_window)
            
            if correlations:
                # Print correlation statistics
                print("\nCorrelation Statistics:")
                for name, df in correlations.items():
                    if not df.empty and not df.correlation.isna().all():
                        corr_data = df.correlation.dropna()
                        print(f"\n{name} Rolling {correlation_window}-Day Correlation:")
                        print(f"Current: {corr_data.iloc[-1]:.2f}")
                        print(f"Maximum: {corr_data.max():.2f}")
                        print(f"Minimum: {corr_data.min():.2f}")
                        print(f"Average: {corr_data.mean():.2f}")
                
                # Also calculate and plot a longer-term correlation for 5-year perspective
                long_window = 120
                long_correlations = calculate_rolling_correlation(valid_data, window=long_window)
                
                plot_correlations(correlations, window=correlation_window, 
                                 title=f"Bitcoin Rolling Correlation (Medium-Term)")
                
                if long_correlations:
                    plot_correlations(long_correlations, window=long_window, 
                                     title=f"Bitcoin Rolling Correlation (Long-Term)")
        else:
            print("\nSkipping correlation analysis because Bitcoin data is not available.")
            print("You can still see the performance of other assets in the generated plots.")
    else:
        print("Failed to fetch valid data for all assets.")

if __name__ == "__main__":
    main()