import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so we can import from simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.ibkr_broker import IBTWSAPI

def calc_adv(df, days=5):
    if df is None or df.empty:
        print("No data provided")
        return 0
    
    if 'volume' not in df.columns:
        print("DataFrame does not contain 'volume' column")
        print(f"Available columns: {list(df.columns)}")
        return 0
    
    # Exclude the last entry from the DataFrame (most recent incomplete period)
    if len(df) < 2:
        print("Not enough data points - need at least 2 entries")
        return 0
    
    df = df.iloc[:-1]  # Remove the last entry
    
    # Check if index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        # print("DataFrame index is not datetime - attempting to convert")
        try:
            df.index = pd.to_datetime(df.index)
        except:
            print("Could not convert index to datetime")
            return 0
    
    # Aggregate volume by day (sum all intraday volumes for each day)
    daily_volume = df.groupby(df.index.date)['volume'].sum()
    
    # Convert back to DataFrame for easier manipulation
    daily_df = pd.DataFrame({'volume': daily_volume})
    daily_df.index = pd.to_datetime(daily_df.index)
    
    # Today's data is already excluded by removing the last entry above
    
    # Take the last 'days' entries and calculate average volume
    if len(daily_df) < days:
        print(f"Not enough historical daily data available. Have {len(daily_df)} days, need {days}")
        # Use all available historical data if we don't have enough days
        recent_data = daily_df
        actual_days = len(daily_df)
    else:
        recent_data = daily_df.tail(days)
        actual_days = days
    
    avg_volume = recent_data['volume'].mean()
    
    # print(f"Average Daily Volume over {actual_days} days: {avg_volume:,.0f}")
    
    return avg_volume


if __name__ == "__main__":
    broker = IBTWSAPI()
    broker.connect()
    # get_volume now returns a DataFrame with all volume data
    df = broker.get_volume(symbol="AAPL", duration="11 D", bar_size="1 day")
    print(df)
    calc_adv(df, days=10)