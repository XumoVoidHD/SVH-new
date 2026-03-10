import pandas as pd
import numpy as np

def calc_macd_components(df, fast=12, slow=26, signal=9):
    """
    Calculate MACD line, signal line, and histogram.

    Returns:
      (macd_line, signal_line, histogram): tuple[pd.Series, pd.Series, pd.Series]
    """
    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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
    macd_line, _, _ = calc_macd_components(df, fast=fast, slow=slow, signal=signal)
    return macd_line

def calc_adjusted_macd(df, fast=12, slow=26, signal=9):
    """
    "Adjusted MACD" used by this project:
      adjusted_macd = MACD histogram normalized by price
                   = (macd_line - signal_line) / close

    This makes the value scale-free across different price levels.
    """
    _, _, histogram = calc_macd_components(df, fast=fast, slow=slow, signal=signal)
    close = df['close'].astype(float)
    # Avoid divide-by-zero
    return histogram / close.replace(0, np.nan)
