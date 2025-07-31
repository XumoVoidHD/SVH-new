import pandas as pd
import numpy as np

def calc_vwap(df):
    """
    Calculate Volume Weighted Average Price (VWAP) manually
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV columns ('high', 'low', 'close', 'volume')
    
    Returns:
    pd.Series: VWAP values
    """
    df = df.copy()
    
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate volume-weighted price
    volume_price = typical_price * df['volume']
    
    # Calculate cumulative volume and cumulative volume-price
    cumulative_volume = df['volume'].cumsum()
    cumulative_volume_price = volume_price.cumsum()
    
    # Calculate VWAP
    vwap = cumulative_volume_price / cumulative_volume
    
    return vwap

