import pandas as pd
import numpy as np

def calc_macd(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) manually
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column
    fast (int): Fast EMA period (default: 12)
    slow (int): Slow EMA period (default: 26)
    signal (int): Signal line period (default: 9)
    
    Returns:
    pd.Series: MACD line values
    """
    close = df['close']
    
    # Calculate fast and slow EMAs
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line
