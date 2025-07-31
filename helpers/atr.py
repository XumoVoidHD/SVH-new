import pandas as pd
import numpy as np

def calc_atr(data, length=14):
    """
    Calculate Average True Range (ATR) manually
    
    Parameters:
    data (pd.DataFrame): DataFrame with OHLC columns ('high', 'low', 'close')
    length (int): Period for ATR calculation (default: 14)
    
    Returns:
    pd.Series: ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    # True Range is the maximum of the three
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using exponential moving average
    atr = true_range.ewm(span=length, adjust=False).mean()
    
    return atr
