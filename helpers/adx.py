import pandas as pd
import numpy as np

def calc_adx(df, length=14):
    """
    Calculate Average Directional Index (ADX) manually
    
    Parameters:
    df (pd.DataFrame): DataFrame with OHLC columns ('high', 'low', 'close')
    length (int): Period for ADX calculation (default: 14)
    
    Returns:
    pd.Series: ADX values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    
    # +DM: Current High - Previous High > Previous Low - Current Low
    plus_dm[high_diff > low_diff.abs()] = high_diff[high_diff > low_diff.abs()]
    plus_dm[high_diff <= 0] = 0
    
    # -DM: Previous Low - Current Low > Current High - Previous High
    minus_dm[low_diff.abs() > high_diff] = low_diff.abs()[low_diff.abs() > high_diff]
    minus_dm[low_diff >= 0] = 0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth the values using exponential moving average
    atr = true_range.ewm(span=length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=length, adjust=False).mean()
    
    return adx