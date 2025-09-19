import pandas as pd
import numpy as np
import sys
import os
import time

# Add parent directory to path so we can import from simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.ibkr_broker import IBTWSAPI

def calc_rvol(df, days=10):
    if df is None or df.empty:
        print("No data provided")
        return 0
    
    if 'volume' not in df.columns:
        print("DataFrame does not contain 'volume' column")
        print(f"Available columns: {list(df.columns)}")
        return 0
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            print("Could not convert index to datetime")
            return 0
    
    # Debug: Print timezone info
    # print(f"DataFrame timezone: {df.index.tz}")
    # print(f"DataFrame date range: {df.index.min()} to {df.index.max()}")
    # print(f"Sample index values: {df.index[:3].tolist()}")
    
    # Use the last entry time from the DataFrame as current time
    current_time = df.index.max()
    
    # Get today's date
    today = current_time.date()
    
    # Filter data for today up to current time
    today_data = df[df.index.date == today]
    if today_data.empty:
        print("No data available for today")
        return 0
    
    # Calculate today's volume so far
    today_volume_so_far = today_data['volume'].sum()
    
    # Get the time of day for current time (e.g., 10:00 AM)
    current_time_of_day = current_time.time()
    
    # Get past days data for comparison
    past_days_data = df[df.index.date < today]
    if past_days_data.empty:
        print("No historical data available")
        return 0
    
    # Group by date and time to get same time period volumes for each past day
    past_days_volumes = []
    
    # Get unique dates from past data
    unique_dates = sorted(pd.Series(past_days_data.index.date).unique())
    
    for past_date in unique_dates:
        past_day_data = past_days_data[past_days_data.index.date == past_date]
        
        # Filter to same time period (from market open to current time)
        # Handle timezone compatibility
        if past_day_data.index.tz is not None:
            # DataFrame has timezone info, create timezone-aware timestamps
            market_open = pd.Timestamp.combine(past_date, pd.Timestamp("09:30:00").time()).tz_localize(past_day_data.index.tz)
            current_time_past = pd.Timestamp.combine(past_date, current_time_of_day).tz_localize(past_day_data.index.tz)
        else:
            # DataFrame is timezone-naive, create timezone-naive timestamps
            market_open = pd.Timestamp.combine(past_date, pd.Timestamp("09:30:00").time())
            current_time_past = pd.Timestamp.combine(past_date, current_time_of_day)
        
        # Get volume from market open to current time for this past day
        period_data = past_day_data[
            (past_day_data.index >= market_open) & 
            (past_day_data.index <= current_time_past)
        ]
        
        if not period_data.empty:
            past_days_volumes.append(period_data['volume'].sum())
    
    if not past_days_volumes:
        print("No comparable historical data found")
        return 0
    
    # Take only the most recent 'days' for average calculation
    past_days_volumes = past_days_volumes[-days:]
    
    # Calculate average volume for the same time period over past days
    avg_volume_same_period = np.mean(past_days_volumes)
    
    # Calculate relative volume
    if avg_volume_same_period == 0:
        print("Average volume is zero - cannot calculate RVOL")
        return 0
    
    rvol = today_volume_so_far / avg_volume_same_period
    
    # print(f"Today's volume so far (until {current_time.time()}): {today_volume_so_far:,.0f}")
    # print(f"Average volume for same period over {len(past_days_volumes)} days: {avg_volume_same_period:,.0f}")
    print(f"Relative Volume (RVOL): {rvol:.2f} ({rvol*100:.1f}% of average)")
    
    return rvol


if __name__ == "__main__":
    broker = IBTWSAPI()
    broker.connect()
    # Get 15-minute interval data for past 10 days
    start = time.time()
    df = broker.get_volume(symbol="NVDA", duration="1 D", bar_size="15 mins")
    print("Sample data:")
    print(df)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    # Calculate RVOL using the last entry time as current time
    rvol = calc_rvol(df, days=10)
    