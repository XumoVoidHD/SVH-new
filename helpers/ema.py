import pandas as pd
import numpy as np

def calc_ema(df, length):
    """
    Calculate Exponential Moving Average (EMA) manually
    
    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column
    length (int): EMA period
    
    Returns:
    pd.Series: EMA values
    """
    close = df['close']
    
    # Calculate EMA using pandas ewm
    ema = close.ewm(span=length, adjust=False).mean()
    
    return ema
