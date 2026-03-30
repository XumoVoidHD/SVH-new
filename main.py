from simulation import ibkr_broker
import pandas as pd
import sys
from stock_selector import StockSelector, initialize_stock_selector
import random
import threading
import time
from datetime import datetime, time as dtime, timedelta
import pytz
from helpers import vwap, ema, macd, adx, atr
from helpers.json_utils import dumps_safe, json_safe
from helpers.fetch_marketcap_csv import fetch_marketcap_csv
from log import setup_logger
from db.trades_db import trades_db
import json
from types import SimpleNamespace
from simulation.ib_broker import IBBroker
import traceback
import atexit
import os
from typing import Any, Dict, List, Optional

def update_bot_status(bot_on, initializing):
    """Update the bot status in bot_status.json"""
    try:
        status = {
            "bot_on": bot_on,
            "initializing": initializing,
            "pid": os.getpid() if bot_on else None
        }
        with open('bot_status.json', 'w') as f:
            json.dump(status, f)
    except Exception as e:
        print(f"Error updating bot status: {e}")

def on_exit():
    """Handler for script exit"""
    update_bot_status(False, False)

atexit.register(on_exit)

setup_logger()

def load_config(json_file='creds.json'):
    """Load configuration from JSON file and make it accessible like a module"""
    with open(json_file, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to object with dot notation access
    def dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            # Keep lists as lists, but preserve dicts as dicts (don't convert to SimpleNamespace)
            return [item for item in d]  # Keep original items as-is
        else:
            return d
    
    return dict_to_obj(config_dict)

creds = load_config('creds.json')
TESTING = creds.TESTING

# ---------------------------
# Frontend-friendly entry decision feed (no DB migration)
# ---------------------------
_ENTRY_DECISIONS_LOCK = threading.Lock()

def _write_entry_decision(symbol: str, decision: str, reasons: List[str], details: Optional[Dict[str, Any]] = None):
    """
    Persist a short, human-readable reason for why we did/didn't enter a stock.
    db_viewer.py reads this file and shows it above the table.
    """
    try:
        eastern_tz = pytz.timezone("US/Eastern")
        now = datetime.now(eastern_tz).strftime("%Y-%m-%d %H:%M:%S")
        item: Dict[str, Any] = {
            "time": now,
            "symbol": str(symbol),
            "decision": str(decision),  # "entered" | "rejected" | "order_failed"
            "reasons": [str(r) for r in (reasons or [])],
            # details often contains numpy/pandas scalars/booleans; sanitize to native JSON types
            "details": json_safe(details or {}),
        }

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_decisions.json")
        with _ENTRY_DECISIONS_LOCK:
            existing: Dict[str, Any] = {"updated_at": now, "decisions": []}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            existing = loaded
                except Exception:
                    existing = {"updated_at": now, "decisions": []}

            decisions = existing.get("decisions", [])
            if not isinstance(decisions, list):
                decisions = []

            decisions.append(item)
            decisions = decisions[-200:]  # bound growth

            out = {"updated_at": now, "decisions": decisions}
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
    except Exception as e:
        # Never let logging break trading
        print(f"[ENTRY FEED] Could not write entry_decisions.json: {e}")

class Strategy:
    
    def __init__(self, manager, stock, broker, config, sector='Unknown'):
        self.manager = manager
        self.stock = stock
        self.broker = broker
        self.config = config
        self.sector = sector or 'Unknown'
        self.score = 0
        self.additional_checks_passed = False
        
        # Load configurations from creds.json
        self.indicators_config = creds.INDICATORS
        self.alpha_score_config = creds.ALPHA_SCORE_CONFIG
        self.additional_checks_config = creds.ADDITIONAL_CHECKS_CONFIG
        self.risk_config = creds.RISK_CONFIG
        self.stop_loss_config = creds.STOP_LOSS_CONFIG
        self.profit_config = creds.PROFIT_CONFIG
        self.order_config = creds.ORDER_CONFIG
        self.trading_hours = creds.TRADING_HOURS
        self.hedge_config = creds.HEDGE_CONFIG
        
        # Extract configured timeframes for each indicator for easy access
        self._extract_configured_timeframes()
        self.weak_position_config = creds.WEAK_POSITION_CONFIG
        
        self.data = {}
        self.indicators = {}
        self.position_active = False 
        self.position_shares = 0
        self.current_price = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.used_capital = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.entry_time = None
        self.close_time = None
        self.trailing_exit_monitoring = False
        
        # Hedge logic is now centralized in StrategyManager (Thread 0)
        # No per-stock hedge tracking

        # Each stock can use risk_per_trade percentage of equity (default 0.4%)
        # No need to track total position size across stocks
        
        # Capital tracking
        self.used_capital = 0

    def _extract_configured_timeframes(self):
        """Extract and store configured timeframes for each indicator for easy access in alpha score calculations"""
        self.tf = {}  # Dictionary to store configured timeframes
        
        indicators_dict = vars(self.indicators_config)
        for indicator_name, indicator_config in indicators_dict.items():
            if hasattr(indicator_config, 'timeframes') and indicator_config.timeframes:
                # Store the first configured timeframe for this indicator
                self.tf[indicator_name] = indicator_config.timeframes[0]
        
        print(f"Configured timeframes: {self.tf}")

    def _safe_broker_call(self, broker_method, *args, max_retries=5, base_delay=2.0, max_delay=300.0, **kwargs):
        """
        Safely call broker methods with exponential backoff and random jitter
        
        Args:
            broker_method: The broker method to call
            *args: Arguments for the broker method
            max_retries (int): Maximum number of retry attempts
            base_delay (float): Base delay in seconds (starts at 2 seconds)
            max_delay (float): Maximum delay in seconds (5 minutes = 300 seconds)
            **kwargs: Keyword arguments for the broker method
        
        Returns:
            Result from broker method or None if all retries failed
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = broker_method(*args, **kwargs)
                if result is not None:
                    return result
                else:
                    print(f"Broker call returned None, retrying... (attempt {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                last_exception = e
                print(f"Broker call failed: {e} (attempt {attempt + 1}/{max_retries})")
            
            # Don't sleep after the last attempt
            if attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16, 32 seconds...
                exponential_delay = base_delay * (2 ** attempt)
                
                # Random jitter increases with retry attempts
                # Early retries: small random variation
                # Later retries: larger random variation up to 5 minutes
                if attempt == 0:
                    # First retry: 2 seconds ± 0.5 seconds
                    min_random = exponential_delay - 0.5
                    max_random = exponential_delay + 0.5
                elif attempt == 1:
                    # Second retry: 4 seconds ± 1 second
                    min_random = exponential_delay - 1.0
                    max_random = exponential_delay + 1.0
                elif attempt == 2:
                    # Third retry: 8 seconds ± 2 seconds
                    min_random = exponential_delay - 2.0
                    max_random = exponential_delay + 2.0
                else:
                    # Later retries: larger randomization, capped at 5 minutes
                    randomization_factor = min(attempt * 0.5, 2.0)  # Increase randomization up to 2x
                    min_random = exponential_delay * (1 - randomization_factor)
                    max_random = min(exponential_delay * (1 + randomization_factor), max_delay)
                
                # Ensure minimum delay of 2 seconds
                min_random = max(min_random, 2.0)
                
                # Generate random delay within the range
                delay = random.uniform(min_random, max_random)
                
                print(f"Retrying broker call in {delay:.2f} seconds... (exponential: {exponential_delay:.2f}s, range: {min_random:.2f}-{max_random:.2f}s)")
                time.sleep(delay)
        
        # If all retries failed, log and return None
        if last_exception:
            print(f"All {max_retries} attempts failed. Last error: {last_exception}")
        else:
            print(f"All {max_retries} attempts returned None")
        
        return None

    def get_current_price_with_retry(self, symbol):
        """Get current price with exponential backoff retry logic"""
        return self._safe_broker_call(self.broker.get_current_price, symbol)

    def get_historical_data_with_retry(self, stock, duration="3 D", bar_size="3 mins"):
        """Get historical data with exponential backoff retry logic"""
        return self._safe_broker_call(self.broker.get_historical_data_stock, stock=stock, duration=duration, bar_size=bar_size)
    
    def time_str_to_datetime(self, time_str):
        """Convert time string to datetime object for comparison"""
        return datetime.strptime(time_str, "%H:%M").time()
    
    def get_next_entry_window(self):
        """Get information about the next entry window"""
        eastern_tz = pytz.timezone(self.trading_hours.timezone)
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        # Get entry windows from configuration
        morning_start = self.trading_hours.morning_entry_start
        morning_end = self.trading_hours.morning_entry_end
        afternoon_start = self.trading_hours.afternoon_entry_start
        afternoon_end = self.trading_hours.afternoon_entry_end
        market_close = self.trading_hours.market_close
        
        if current_time_str < morning_start:
            return f"Next entry window: {morning_start}-{morning_end}"
        elif morning_start <= current_time_str <= morning_end:
            return f"Currently in morning entry window ({morning_start}-{morning_end})"
        elif morning_end < current_time_str < afternoon_start:
            return f"Next entry window: {afternoon_start}-{afternoon_end}" 
        elif afternoon_start <= current_time_str <= afternoon_end:
            return f"Currently in afternoon entry window ({afternoon_start}-{afternoon_end})"
        elif afternoon_end < current_time_str < market_close:
            return f"No more entry windows today. Market closes at {market_close}"
        else:
            return f"Market closed. Next entry window: {morning_start}-{morning_end} tomorrow"
    
    def fetch_data_by_timeframe(self):
        """Fetch and store data for each configured timeframe with retry logic and rate limiting"""
        # Get all unique timeframes from indicator configurations
        timeframes_needed = set()
        
        # Convert SimpleNamespace to dict-like iteration
        indicators_dict = vars(self.indicators_config)
        for indicator_name, indicator_config in indicators_dict.items():
            if hasattr(indicator_config, 'timeframes'):
                timeframes = indicator_config.timeframes
                if isinstance(timeframes, list):
                    timeframes_needed.update(timeframes)
                else:
                    print(f"Warning: timeframes is not a list for {indicator_name}: {type(timeframes)}")
            else:
                print(f"Warning: indicator_config {indicator_name} has no timeframes attribute")
        
        # Fetch data for each needed timeframe
        for tf_name in timeframes_needed:
            # Extract period number from timeframe name
            # Handle formats: '3min', '3mins', '3 min', '3 mins'
            period_str = tf_name.replace('mins', '').replace('min', '').strip()
            period = int(period_str)
    #         # Get historical price data for this stock
    #         historical_data = self.get_historical_data_with_retry(
    #             stock=self.stock, 
    #             bar_size="1 day", 
    #             duration=f"{days + 5} D"  # Get a few extra days to ensure we have enough
    #         )
            
    #         if historical_data is None or historical_data.empty:
    #             print(f"[Sharpe] {self.stock}: No historical data available")
    #             return 0.0
            
    #         if len(historical_data) < 5:
    #             print(f"[Sharpe] {self.stock}: Insufficient data ({len(historical_data)} < 5 days)")
    #             return 0.0
            
    #         # Calculate daily returns
    #         closes = historical_data['close'].values
    #         daily_returns = []
            
    #         for i in range(1, len(closes)):
    #             daily_return = (closes[i] - closes[i-1]) / closes[i-1]
    #             daily_returns.append(daily_return)
            
    #         if len(daily_returns) < 5:
    #             print(f"[Sharpe] {self.stock}: Insufficient returns ({len(daily_returns)} < 5)")
    #             return 0.0
            
    #         # Calculate mean and std of returns
    #         mean_return = sum(daily_returns) / len(daily_returns)
    #         variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
    #         std_return = variance ** 0.5
            
    #         if std_return == 0:
    #             print(f"[Sharpe] {self.stock}: Zero volatility")
    #             return 0.0
            
    #         # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
    #         sharpe = mean_return / std_return
            
    #         # Annualize the Sharpe ratio (optional: multiply by sqrt(252) for trading days)
    #         annualized_sharpe = sharpe * (252 ** 0.5)
            
    #         print(f"[Sharpe] {self.stock}: {annualized_sharpe:.2f} annualized (daily: {sharpe:.2f}, mean ret: {mean_return*100:.3f}%, std: {std_return*100:.3f}%, n={len(daily_returns)})")
            
    #         return sharpe
            
    #     except Exception as e:
    #         print(f"[Sharpe] Error for {self.stock}: {e}")
    #         traceback.print_exc()
    #         return 0.0
    
    # def calculate_10day_drawdown(self):
    #     try:
    #         # Get historical price data for this stock
    #         historical_data = self.get_historical_data_with_retry(
    #             stock=self.stock, 
    #             bar_size="1 day", 
    #             duration="15 D"  # Get 15 days to ensure we have at least 10
    #         )
            
    #         if historical_data is None or historical_data.empty:
    #             print(f"[DD-10d] {self.stock}: No historical data available")
    #             return 0.0
            
    #         if len(historical_data) < 2:
    #             print(f"[DD-10d] {self.stock}: Insufficient data ({len(historical_data)} < 2 days)")
    #             return 0.0
            
    #         # Get last 10 days of close prices
    #         closes = historical_data['close'].values[-10:] if len(historical_data) >= 10 else historical_data['close'].values
            
    #         # Calculate maximum drawdown
    #         peak = closes[0]
    #         max_drawdown = 0.0
            
    #         for price in closes:
    #             if price > peak:
    #                 peak = price
    #             drawdown = (peak - price) / peak if peak > 0 else 0.0
    #             max_drawdown = max(max_drawdown, drawdown)
            
    #         print(f"[DD-10d] {self.stock}: {max_drawdown*100:.2f}% (peak=${peak:.2f}, current=${closes[-1]:.2f}, n={len(closes)} days)")
            
    #         return max_drawdown
            
    #     except Exception as e:
    #         print(f"[DD-10d] Error for {self.stock}: {e}")
    #         traceback.print_exc()
    #         return 0.0
    
    def end_of_day_weak_exit(self):
        """Exit weak positions at configured weak exit time (configurable gain range)"""
        if not self.position_active:
            return
        
        current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
        
        # Exit if position is weak (configurable range)
        if self.weak_position_config.min_gain_pct <= current_gain_pct <= self.weak_position_config.max_gain_pct:
            weak_exit_time = self.trading_hours.weak_exit_time
            print(f"{weak_exit_time} Weak Exit: {current_gain_pct*100:.1f}% gain in weak range")
            self.close_position('end_of_day_weak', self.position_shares)
        else:
            weak_exit_time = self.trading_hours.weak_exit_time
            print(f"{weak_exit_time}: Position not in weak range ({current_gain_pct*100:.1f}% gain) - keeping position")

    def safety_exit_all(self):
        # Get exit time from configuration
        safety_exit_time = self.trading_hours.safety_exit_time
        
        print(f"Safety exit all positions at {safety_exit_time}")
        if self.position_active:
            print(f"{safety_exit_time} Safety Exit: Closing all positions")
            self.close_position('safety_exit', self.position_shares)
    
    def market_on_close_exit(self):
        """Market-on-close exit at 4:00 PM"""
        if self.position_active:
            print("4:00 PM Market-On-Close: Closing remaining positions")
            self.close_position('market_on_close', self.position_shares)       
    
    def fetch_data_by_timeframe(self):
        """Fetch and store data for each configured timeframe with retry logic and rate limiting"""
        # Get all unique timeframes from indicator configurations
        timeframes_needed = set()
        
        # Convert SimpleNamespace to dict-like iteration
        indicators_dict = vars(self.indicators_config)
        for indicator_name, indicator_config in indicators_dict.items():
            if hasattr(indicator_config, 'timeframes'):
                timeframes = indicator_config.timeframes
                if isinstance(timeframes, list):
                    timeframes_needed.update(timeframes)
                else:
                    print(f"Warning: timeframes is not a list for {indicator_name}: {type(timeframes)}")
            else:
                print(f"Warning: indicator_config {indicator_name} has no timeframes attribute")
        
        # Fetch data for each needed timeframe
        for tf_name in timeframes_needed:
            # Extract period number from timeframe name
            # Handle formats: '3min', '3mins', '3 min', '3 mins'
            period_str = tf_name.replace('mins', '').replace('min', '').strip()
            period = int(period_str)
            
            # Try to fetch data with retries
            max_retries = 10
            base_delay = 2  # Base delay in seconds
            
            # Add initial random delay to desynchronize threads (0-2 seconds)
            initial_jitter = random.uniform(0, 2)
            time.sleep(initial_jitter)
            
            for attempt in range(max_retries):
                # Use semaphore to limit concurrent requests across all threads
                # Only hold semaphore during the actual request, not during retry delays
                with self.manager.data_request_semaphore:
                    try:
                        data = self.get_historical_data_with_retry(stock=self.stock, bar_size=period)
                        
                        # Check if data is valid and not empty
                        if data is not None and hasattr(data, 'empty') and not data.empty and len(data) > 0:
                            self.data[tf_name] = data
                            print(f"Fetched {tf_name} data: {len(data)} candles")
                            break
                        else:
                            data = None  # Mark as failed for retry logic below
                            
                    except Exception as e:
                        print(f"Error fetching {tf_name} data (attempt {attempt + 1}/{max_retries}): {e}")
                        data = None  # Mark as failed for retry logic below
                
                    # Handle retry logic outside semaphore so other threads can proceed
                    if data is None or (hasattr(data, 'empty') and data.empty):
                        if attempt < max_retries - 1:
                            # Exponential backoff with randomization to prevent synchronized retries
                            exponential_delay = base_delay * (2 ** attempt)
                            jitter = random.uniform(0, exponential_delay * 0.5)  # Add up to 50% jitter
                            delay = exponential_delay + jitter
                            print(f"Retrying {tf_name} data fetch in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})...")
                            time.sleep(delay)
                        else:
                            print(f"Failed to fetch {tf_name} data after {max_retries} attempts")
    
    def calculate_indicators_by_timeframe(self):
        """Calculate indicators for each configured timeframe (data from fetch_data_by_timeframe).
        Each indicator uses its creds INDICATORS timeframes; VWAP default is 3 mins."""
        # Initialize indicators storage by timeframe
        for tf_name in list(self.data.keys()):
            self.indicators[tf_name] = {}
        
        # Calculate each indicator on its specified timeframes
        indicators_dict = vars(self.indicators_config)
        for indicator_name, indicator_config in indicators_dict.items():
            for tf_name in indicator_config.timeframes:
                if tf_name not in self.data or self.data[tf_name].empty:
                    continue
                
                data = self.data[tf_name]
                params = indicator_config.params
                
                # Calculate indicator based on type
                if indicator_name == 'vwap':
                    self.indicators[tf_name]['vwap'] = vwap.calc_vwap(data)
                
                elif indicator_name == 'macd':
                    self.indicators[tf_name]['macd'] = macd.calc_macd(
                        data, 
                        fast=getattr(params, 'fast', 12),
                        slow=getattr(params, 'slow', 26),
                        signal=getattr(params, 'signal', 9)
                    )
                    # Adjusted MACD (histogram normalized by price) for momentum rule
                    self.indicators[tf_name]['adjusted_macd'] = macd.calc_adjusted_macd(
                        data,
                        fast=getattr(params, 'fast', 12),
                        slow=getattr(params, 'slow', 26),
                        signal=getattr(params, 'signal', 9)
                    )
                
                elif indicator_name == 'adx':
                    self.indicators[tf_name]['adx'] = adx.calc_adx(
                        data, 
                        length=getattr(params, 'length', 14)
                    )
                
                elif indicator_name == 'ema1':
                    ema_length = getattr(params, 'length', 5)
                    self.indicators[tf_name]['ema1'] = ema.calc_ema(data, length=ema_length)
                
                elif indicator_name == 'ema2':
                    ema_length = getattr(params, 'length', 20)
                    self.indicators[tf_name]['ema2'] = ema.calc_ema(data, length=ema_length)
                
                elif indicator_name == 'volume_avg':
                    self.indicators[tf_name]['volume_avg'] = data['volume'].rolling(
                        window=getattr(params, 'window', 20)
                    ).mean()
                
                print(f"Calculated {indicator_name} for {tf_name}")
        
        # Print summary of calculated indicators
        for tf_name, indicators in self.indicators.items():
            if indicators:
                print(f"[{tf_name}] Indicators: {list(indicators.keys())}")
    
    def calculate_indicators(self):
        """Main method to fetch data and calculate all indicators"""
        # Fetch data for all timeframes
        self.fetch_data_by_timeframe()
        
        # Calculate indicators for each timeframe
        self.calculate_indicators_by_timeframe()
        
        # Calculate Alpha Score
        self.calculate_alpha_score()
        
        self.perform_additional_checks()
        
    def calculate_alpha_score(self):
        """Calculate Alpha Score based on configurable parameters"""
        self.score = 0
        self.indicator_values = {}
        self.criteria_passed = {}
        
        # Trend analysis (30%)
        if self._check_trend_conditions():
            self.score += self.alpha_score_config.trend.weight
            print(f"Score +{self.alpha_score_config.trend.weight} (Trend conditions met)")
        
        # Momentum analysis (20%)
        if self._check_momentum_conditions():
            self.score += self.alpha_score_config.momentum.weight
            print(f"Score +{self.alpha_score_config.momentum.weight} (Momentum conditions met)")
        
        # Volume/Volatility analysis (20%)
        if self._check_volume_volatility_conditions():
            self.score += self.alpha_score_config.volume_volatility.weight
            print(f"Score +{self.alpha_score_config.volume_volatility.weight} (Volume/Volatility conditions met)")
        
        # News analysis (15%) - placeholder
        self.score += self.alpha_score_config.news.weight
        self.criteria_passed['news_passed'] = True
        print(f"Score +{self.alpha_score_config.news.weight} (News check - placeholder)")
        
        # Market Calm analysis (15%)
        if self._check_market_calm_conditions():
            self.score += self.alpha_score_config.market_calm.weight
            print(f"Score +{self.alpha_score_config.market_calm.weight} (Market Calm conditions met)")
        
        print(f"\nFinal Alpha Score: {self.score}")
        
        # Update database with alpha score (indicators/criteria saved after additional_checks)
        trades_db.update_strategy_data(self.stock, score=self.score)
    
    def _check_trend_conditions(self):
        """Check trend conditions"""
        try:
            # Get configured timeframes from indicator config
            vwap_tf = self.tf.get('vwap')
            ema1_tf = self.tf.get('ema1')
            ema2_tf = self.tf.get('ema2')
            
            # Verify all required timeframes and indicators are available
            if not all([vwap_tf, ema1_tf, ema2_tf]):
                print(f"Missing timeframe configuration: vwap={vwap_tf}, ema1={ema1_tf}, ema2={ema2_tf}")
                return False
            
            # Check if timeframes exist in data and indicators
            if vwap_tf not in self.data or vwap_tf not in self.indicators:
                print(f"Missing data or indicators for {vwap_tf}")
                return False
            if ema1_tf not in self.indicators or ema2_tf not in self.indicators:
                print(f"Missing indicators for ema1={ema1_tf} or ema2={ema2_tf}")
                return False
            
            # Check if specific indicator keys exist and data is not empty
            if 'vwap' not in self.indicators[vwap_tf] or self.indicators[vwap_tf]['vwap'].empty:
                print(f"Missing or empty VWAP indicator for {vwap_tf}")
                return False
            if 'ema1' not in self.indicators[ema1_tf] or self.indicators[ema1_tf]['ema1'].empty:
                print(f"Missing or empty EMA1 indicator for {ema1_tf}")
                return False
            if 'ema2' not in self.indicators[ema2_tf] or self.indicators[ema2_tf]['ema2'].empty:
                print(f"Missing or empty EMA2 indicator for {ema2_tf}")
                return False
            
            # Check if data has 'close' column
            if 'close' not in self.data[vwap_tf].columns or self.data[vwap_tf].empty:
                print(f"Missing or empty close data for {vwap_tf}")
                return False
            
            # Price > VWAP
            close_price = float(self.data[vwap_tf]['close'].iloc[-1])
            vwap_val = float(self.indicators[vwap_tf]['vwap'].iloc[-1])
            ema1_val = float(self.indicators[ema1_tf]['ema1'].iloc[-1])
            ema2_val = float(self.indicators[ema2_tf]['ema2'].iloc[-1])
            price_vwap_ok = bool(close_price > vwap_val)
            ema_cross_ok = bool(ema1_val > ema2_val)
            trend_passed = price_vwap_ok and ema_cross_ok
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['close'] = close_price
            self.indicator_values['vwap'] = vwap_val
            self.indicator_values['ema1'] = ema1_val
            self.indicator_values['ema2'] = ema2_val
            self.criteria_passed['trend_passed'] = trend_passed
            self.criteria_passed['price_above_vwap'] = price_vwap_ok
            self.criteria_passed['ema_cross'] = ema_cross_ok
            return trend_passed
        except Exception as e:
            print(f"Error checking trend conditions: {e}")
            if hasattr(self, 'criteria_passed'):
                self.criteria_passed['trend_passed'] = False
            return False
    
    def _check_momentum_conditions(self):
        """Check momentum conditions"""
        # Get configured timeframe for MACD
        macd_tf = self.tf.get('macd')
        
        if not macd_tf or macd_tf not in self.indicators:
            print(f"Missing MACD timeframe or indicators: {macd_tf}")
            return False
        
        # Check if adjusted MACD indicator exists and is not empty
        if 'adjusted_macd' not in self.indicators[macd_tf]:
            print(f"Missing adjusted MACD indicator key for {macd_tf}")
            return False
        
        if self.indicators[macd_tf]['adjusted_macd'].empty:
            print(f"Empty adjusted MACD indicator data for {macd_tf}")
            return False
        
        try:
            macd_val = float(self.indicators[macd_tf]['adjusted_macd'].iloc[-1])
            momentum_passed = bool(macd_val > 0)
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['adjusted_macd'] = macd_val
            self.criteria_passed['momentum_passed'] = momentum_passed
            return momentum_passed
        except Exception as e:
            print(f"Error accessing MACD indicator: {e}")
            if hasattr(self, 'criteria_passed'):
                self.criteria_passed['momentum_passed'] = False
            return False
    
    def _check_volume_volatility_conditions(self):
        """Check volume and volatility conditions"""
        # Get configured timeframes for volume and ADX
        volume_avg_tf = self.tf.get('volume_avg')
        adx_tf = self.tf.get('adx')
        
        if not volume_avg_tf or volume_avg_tf not in self.data or volume_avg_tf not in self.indicators:
            print(f"Missing volume_avg timeframe or data/indicators: {volume_avg_tf}")
            return False
        
        if not adx_tf or adx_tf not in self.indicators:
            print(f"Missing ADX timeframe or indicators: {adx_tf}")
            return False
        
        # Check if specific indicator keys exist and data is not empty
        if 'volume_avg' not in self.indicators[volume_avg_tf] or self.indicators[volume_avg_tf]['volume_avg'].empty:
            print(f"Missing or empty volume_avg indicator for {volume_avg_tf}")
            return False
        
        if 'adx' not in self.indicators[adx_tf] or self.indicators[adx_tf]['adx'].empty:
            print(f"Missing or empty ADX indicator for {adx_tf}")
            return False
        
        # Check if data has 'volume' column
        if 'volume' not in self.data[volume_avg_tf].columns or self.data[volume_avg_tf].empty:
            print(f"Missing or empty volume data for {volume_avg_tf}")
            return False
        
        try:
            # Volume spike check
            # Wait for the current bar to close, then use penultimate bar for safety.
            self._wait_for_bar_to_close(volume_avg_tf, buffer_seconds=5)
            self._refresh_volume_avg_after_wait(volume_avg_tf)

            # Debug: print last 20 rows of timestamp, volume, and rolling volume average.
            try:
                vol_debug_df = self.data[volume_avg_tf].copy()
                vol_debug_df["volume_avg"] = self.indicators[volume_avg_tf]["volume_avg"]
                print(f"[{self.stock}] Volume debug ({volume_avg_tf}) - last 20 rows:")
                print(vol_debug_df.tail(20))
            except Exception as debug_err:
                print(f"[{self.stock}] Volume debug print failed ({volume_avg_tf}): {debug_err}")

            # Avoid the last candle when it's still forming: use penultimate bar when possible.
            recent_volume = float(self.data[volume_avg_tf]['volume'].iloc[-2])
            avg_volume = float(self.indicators[volume_avg_tf]['volume_avg'].iloc[-2])
            multiplier = self.alpha_score_config.volume_volatility.conditions.volume_spike.multiplier
            volume_ok = bool(recent_volume > multiplier * avg_volume)
            adx_val = float(self.indicators[adx_tf]['adx'].iloc[-1])
            adx_threshold = self.alpha_score_config.volume_volatility.conditions.adx_threshold.threshold
            adx_ok = bool(adx_val > adx_threshold)
            volume_volatility_passed = volume_ok and adx_ok
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['volume'] = recent_volume
            self.indicator_values['volume_avg'] = avg_volume
            self.indicator_values['adx'] = adx_val
            self.criteria_passed['volume_volatility_passed'] = volume_volatility_passed
            self.criteria_passed['volume_spike'] = volume_ok
            self.criteria_passed['adx_above_threshold'] = adx_ok
            return volume_volatility_passed
        except Exception as e:
            print(f"Error checking volume/volatility conditions: {e}")
            if hasattr(self, 'criteria_passed'):
                self.criteria_passed['volume_volatility_passed'] = False
            return False
    
    def _check_market_calm_conditions(self):
        """Check market calm conditions (VIX)"""
        try:
            vix_timeframe = self.alpha_score_config.market_calm.conditions.vix_threshold.timeframe
            vix_df = self._safe_broker_call(self.broker.get_historical_data_index, "VIX", duration="1 D", bar_size=vix_timeframe)
            if vix_df is None or vix_df.empty:
                if hasattr(self, 'criteria_passed'):
                    self.criteria_passed['market_calm_passed'] = False
                return False
            vix_close = vix_df['close']
            vix_threshold = self.alpha_score_config.market_calm.conditions.vix_threshold.threshold
            vix_current = float(vix_close.iloc[-1])
            vix_low = bool(vix_current < vix_threshold)
            # Parse bar size to minutes: "3 mins" -> 3, "1 min" -> 1
            tf_str = str(vix_timeframe).strip().lower()
            try:
                mins_per_bar = int(tf_str.split()[0]) if tf_str else 1
            except (ValueError, IndexError):
                mins_per_bar = 1
            bars_for_10min = max(1, 10 // mins_per_bar)  # how many bars span ~10 min

            vix_dropping = (
                len(vix_close) > bars_for_10min and
                vix_close.iloc[-1] < vix_close.iloc[-1 - bars_for_10min]
            )
            # vix_dropping = len(vix_close) >= 4 and bool(vix_close.iloc[-1] < vix_close.iloc[-4])
            market_calm_passed = vix_low and vix_dropping
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['vix'] = vix_current
            self.criteria_passed['market_calm_passed'] = market_calm_passed
            self.criteria_passed['vix_low'] = vix_low
            self.criteria_passed['vix_dropping'] = vix_dropping
            return market_calm_passed
        except Exception:
            if hasattr(self, 'criteria_passed'):
                self.criteria_passed['market_calm_passed'] = False
            return False
    
    def perform_additional_checks(self):
        """Perform additional checks after Alpha Score calculation"""
        print(f"\n--- Additional Checks ({self.stock}) ---")
        threshold = creds.RISK_CONFIG.alpha_score_threshold
        if self.score < threshold:
            self.additional_checks_passed = False
            print(f"  OVERALL: FAILED — Reason: Alpha Score too low (score={self.score} < threshold={threshold}). Skipping volume/VWAP/TRIN-TICK.")
        else:
            # Get configured timeframes
            volume_avg_tf = self.tf.get('volume_avg')
            volume_check = False
            vwap_slope_check = False
            market_conditions_check = False

            if not volume_avg_tf or volume_avg_tf not in self.data or volume_avg_tf not in self.indicators:
                self.additional_checks_passed = False
                print(f"  OVERALL: FAILED — Reason: Missing data/indicators for volume_avg timeframe '{volume_avg_tf}' (data keys: {list(self.data.keys())}, indicator keys: {list(self.indicators.keys())}).")
            elif 'volume_avg' not in self.indicators[volume_avg_tf] or self.indicators[volume_avg_tf]['volume_avg'].empty:
                self.additional_checks_passed = False
                print(f"  OVERALL: FAILED — Reason: Missing or empty volume_avg indicator for '{volume_avg_tf}'.")
            elif 'volume' not in self.data[volume_avg_tf].columns or self.data[volume_avg_tf].empty:
                self.additional_checks_passed = False
                print(f"  OVERALL: FAILED — Reason: Missing or empty volume data for '{volume_avg_tf}'.")
            else:
                try:
                    # Wait for the current bar to close, then use penultimate bar for safety.
                    self._wait_for_bar_to_close(volume_avg_tf, buffer_seconds=5)
                    self._refresh_volume_avg_after_wait(volume_avg_tf)

                    # Debug: print last 20 rows of timestamp, volume, and rolling volume average.
                    try:
                        vol_debug_df = self.data[volume_avg_tf].copy()
                        vol_debug_df["volume_avg"] = self.indicators[volume_avg_tf]["volume_avg"]
                        print(f"[{self.stock}] Additional-check volume debug ({volume_avg_tf}) - last 20 rows:")
                        print(vol_debug_df.tail(20))
                    except Exception as debug_err:
                        print(f"[{self.stock}] Additional-check volume debug print failed ({volume_avg_tf}): {debug_err}")

                    # Avoid the last candle when it's still forming: use penultimate bar when possible.
                    recent_volume = float(self.data[volume_avg_tf]['volume'].iloc[-2])
                    avg_volume = float(self.indicators[volume_avg_tf]['volume_avg'].iloc[-2])
                    volume_multiplier = self.additional_checks_config.volume_multiplier
                    required_volume = volume_multiplier * avg_volume
                    volume_check = bool(recent_volume > required_volume)
                    if not hasattr(self, 'indicator_values'):
                        self.indicator_values = {}
                    if not hasattr(self, 'criteria_passed'):
                        self.criteria_passed = {}
                    self.indicator_values['volume_recent'] = recent_volume
                    self.indicator_values['volume_avg_addl'] = avg_volume
                    self.criteria_passed['volume_check_passed'] = volume_check
                    print(f"  [1] Volume: {'PASSED' if volume_check else 'FAILED'} — recent={recent_volume:,.0f}, avg={avg_volume:,.0f}, required (>{volume_multiplier}x)={required_volume:,.0f} — {'recent > required' if volume_check else 'recent <= required'}")
                except Exception as e:
                    print(f"  [1] Volume: FAILED — Error: {e}")
                    volume_check = False
                    if hasattr(self, 'criteria_passed'):
                        self.criteria_passed['volume_check_passed'] = False

                print(f"  [2] VWAP slope:")
                vwap_slope_check = self._check_vwap_slope()

                bypass_alpha = self.additional_checks_config.trin_tick_bypass_alpha
                if self.score >= bypass_alpha:
                    market_conditions_check = True
                    print(f"  [3] TRIN/TICK: BYPASSED — Alpha Score ({self.score}) > bypass threshold ({bypass_alpha}).")
                else:
                    print(f"  [3] TRIN/TICK (score {self.score} <= {bypass_alpha}):")
                    market_conditions_check = self._check_trin_tick_conditions()
                    print(f"       TRIN/TICK result: {'PASSED' if market_conditions_check else 'FAILED'}")

                self.additional_checks_passed = bool(volume_check and vwap_slope_check and market_conditions_check)
                self.criteria_passed['trin_tick_passed'] = market_conditions_check
                failed = []
                if not volume_check:
                    failed.append("volume")
                if not vwap_slope_check:
                    failed.append("VWAP slope")
                if not market_conditions_check:
                    failed.append("TRIN/TICK")
                self.additional_checks_failed = failed
                if self.additional_checks_passed:
                    print(f"  OVERALL: PASSED — volume, VWAP slope, and market conditions all passed.")
                else:
                    print(f"  OVERALL: FAILED — Failed checks: {', '.join(failed)}.")
        if not hasattr(self, 'indicator_values'):
            self.indicator_values = {}
        if not hasattr(self, 'criteria_passed'):
            self.criteria_passed = {}

        # Always update database with additional checks status and indicator/criteria JSON (regardless of pass/fail)
        # Use dumps_safe so numpy/pandas scalars in dicts are converted to native Python types before json.dumps
        trades_db.update_strategy_data(
            self.stock,
            additional_checks_passed=bool(self.additional_checks_passed),
            indicator_values=dumps_safe(self.indicator_values),
            criteria_passed=dumps_safe(self.criteria_passed)
        )

    def _timeframe_to_minutes(self, tf_value):
        """Parse timeframe strings like '3 mins', '5 min', '1min' into integer minutes."""
        if tf_value is None:
            return None
        try:
            s = str(tf_value).strip().lower()
            # common formats: "3 mins", "3 min", "3mins", "3min"
            s = s.replace("minutes", "min").replace("minute", "min")
            s = s.replace("mins", "min")
            s = s.replace(" ", "")
            if s.endswith("min"):
                s = s[:-3]
            return int(s)
        except Exception:
            return None

    def _wait_for_bar_to_close(self, tf_name: str, buffer_seconds: int = 5, max_wait_seconds: int = 180):
        """
        Wait until the most recently fetched candle for `tf_name` is expected to be closed.

        Uses the last candle's timestamp (data[tf_name]['date']) plus bar duration,
        compared to current time in the strategy timezone.
        """
        try:
            minutes = self._timeframe_to_minutes(tf_name)
            if not minutes:
                return
            if tf_name not in self.data:
                return
            data = self.data[tf_name]
            if data is None or getattr(data, "empty", False):
                return
            if "date" not in data.columns or len(data["date"]) < 1:
                return

            eastern_tz = pytz.timezone(self.trading_hours.timezone)
            now = datetime.now(eastern_tz)
            last_bar_start = data["date"].iloc[-1]

            # Normalize timestamps to naive "local" time for safe subtraction
            if hasattr(last_bar_start, "tzinfo") and last_bar_start.tzinfo is not None:
                last_bar_start = last_bar_start.astimezone(eastern_tz).replace(tzinfo=None)
            else:
                # Assume the broker timestamp is already aligned with the strategy timezone
                last_bar_start = last_bar_start.replace(tzinfo=None)

            now_naive = now.replace(tzinfo=None)
            bar_end = last_bar_start + timedelta(minutes=minutes)
            remaining = (bar_end - now_naive).total_seconds()

            if remaining <= 0:
                return

            sleep_for = min(max_wait_seconds, remaining + buffer_seconds)
            if sleep_for > 0:
                print(f"Waiting {sleep_for:.1f}s for {tf_name} candle to close before volume calc...")
                time.sleep(sleep_for)
        except Exception:
            # Safety: never block trading logic on waiting
            return

    def _refresh_volume_avg_after_wait(self, volume_avg_tf: str):
        """Refetch volume_avg timeframe data and recompute the rolling volume_avg series."""
        try:
            minutes = self._timeframe_to_minutes(volume_avg_tf)
            if not minutes:
                return
            data_v = self.get_historical_data_with_retry(stock=self.stock, bar_size=minutes)
            if data_v is None or getattr(data_v, "empty", False):
                return
            if "volume" not in data_v.columns:
                return

            self.data[volume_avg_tf] = data_v

            window = int(getattr(self.indicators_config.volume_avg.params, "window", 20))
            self.indicators.setdefault(volume_avg_tf, {})
            self.indicators[volume_avg_tf]["volume_avg"] = data_v["volume"].rolling(window=window).mean()
        except Exception:
            return
    
    def _check_vwap_slope(self):
        """Check VWAP slope condition"""
        vwap_tf = self.tf.get('vwap')
        if not vwap_tf or vwap_tf not in self.indicators:
            print(f"       VWAP slope: FAILED — missing timeframe or indicators for '{vwap_tf}'.")
            return False
        if 'vwap' not in self.indicators[vwap_tf]:
            print(f"       VWAP slope: FAILED — missing VWAP indicator key for '{vwap_tf}'.")
            return False
        vwap_series = self.indicators[vwap_tf]['vwap']
        if vwap_series.empty or len(vwap_series) < 2:
            print(f"       VWAP slope: FAILED — insufficient data (need at least 2 bars, got {len(vwap_series)}).")
            return False
        try:
            current_vwap = float(vwap_series.iloc[-1])
            # vwap_slope_period is configured in minutes (see db_viewer label); support any VWAP timeframe.
            tf_minutes = self._timeframe_to_minutes(vwap_tf) or 3
            slope_period_minutes = float(getattr(self.additional_checks_config, "vwap_slope_period", 3))

            # choose a bars-back that best matches the requested minutes, at least 1 bar
            bars_back = max(1, int(round(slope_period_minutes / tf_minutes)))
            if len(vwap_series) < (bars_back + 1):
                print(
                    f"       VWAP slope: FAILED — insufficient data for period={slope_period_minutes}min on tf={tf_minutes}min "
                    f"(need {bars_back + 1} bars, got {len(vwap_series)})."
                )
                self.criteria_passed['vwap_slope_passed'] = False
                return False

            vwap_period_ago = float(vwap_series.iloc[-(bars_back + 1)])
            delta_minutes = tf_minutes * bars_back
            vwap_slope = (current_vwap - vwap_period_ago) / float(delta_minutes)
            threshold = self.additional_checks_config.vwap_slope_threshold
            slope_ok = bool(vwap_slope > threshold)
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['vwap_slope'] = vwap_slope
            self.criteria_passed['vwap_slope_passed'] = slope_ok
            print(
                f"       VWAP slope: {'PASSED' if slope_ok else 'FAILED'} — tf={tf_minutes}min, lookback={delta_minutes:.0f}min "
                f"(bars_back={bars_back}), current_vwap={current_vwap:.2f}, past_vwap={vwap_period_ago:.2f}, "
                f"slope={vwap_slope:.4f} $/min, threshold={threshold} (need slope > threshold)."
            )
            return slope_ok
        except Exception as e:
            print(f"       VWAP slope: FAILED — Error: {e}")
            if hasattr(self, 'criteria_passed'):
                self.criteria_passed['vwap_slope_passed'] = False
            return False
    
    def _check_trin_tick_conditions(self):
        """Check TRIN and TICK market breadth conditions"""
        if not self.additional_checks_config.trin_tick_check_enabled:
            print("       TRIN/TICK: BYPASSED — check disabled in configuration.")
            return True
        
        try:
            # Get configurable thresholds
            trin_threshold = self.additional_checks_config.trin_threshold
            tick_ma_window = self.additional_checks_config.tick_ma_window
            tick_threshold = self.additional_checks_config.tick_threshold

            # Check TRIN (NYSE TRIN index) - should be <= threshold
            # TRIN > 1.0 indicates bearish sentiment, TRIN < 1.0 indicates bullish
            # Use specialized TRIN/TICK helper so contract matches IBKR expectations
            trin_data = self.broker.get_trin_tick_data("TRIN-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")
            trin_check = False
            
            if trin_data is not None and not trin_data.empty and 'close' in trin_data.columns:
                current_trin = float(trin_data['close'].iloc[-1])
                trin_check = bool(current_trin <= trin_threshold)
                print(f"       TRIN: {'PASSED' if trin_check else 'FAILED'} — current={current_trin:.2f}, threshold={trin_threshold} (need TRIN <= threshold).")
            else:
                print("       TRIN: FAILED — TRIN-NYSE data unavailable.")
                if hasattr(self, 'criteria_passed'):
                    self.criteria_passed['trin_tick_passed'] = False
                return False
            # Check TICK (NYSE TICK) - MA should be >= threshold
            # TICK shows net upticks minus downticks, positive = more buying pressure
            tick_data = self.broker.get_trin_tick_data("TICK-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")
            tick_check = False
            current_tick_ma = None
            if tick_data is not None and not tick_data.empty and 'close' in tick_data.columns:
                tick_ma = tick_data['close'].rolling(window=tick_ma_window).mean()
                current_tick_ma = float(tick_ma.iloc[-1])
                tick_check = bool(current_tick_ma >= tick_threshold)
                print(f"       TICK: {'PASSED' if tick_check else 'FAILED'} — {tick_ma_window}-bar MA={current_tick_ma:.0f}, threshold={tick_threshold} (need TICK MA >= threshold).")
            else:
                print("       TICK: FAILED — TICK-NYSE data unavailable.")
                if hasattr(self, 'criteria_passed'):
                    self.criteria_passed['trin_tick_passed'] = False
                return False
            trin_tick_passed = bool(trin_check and tick_check)
            if not hasattr(self, 'indicator_values'):
                self.indicator_values = {}
            if not hasattr(self, 'criteria_passed'):
                self.criteria_passed = {}
            self.indicator_values['trin'] = current_trin
            self.indicator_values['tick_ma'] = current_tick_ma
            self.criteria_passed['trin_tick_passed'] = trin_tick_passed
            return trin_tick_passed
            
        except Exception as e:
            print(f"       TRIN/TICK: FAILED — Error: {e}")
            traceback.print_exc()
            return False 
        
    def calculate_position_size(self):
        account_equity = creds.EQUITY
        current_price = self.get_current_price_with_retry(self.stock)
        
        # Base risk per trade
        base_risk_per_trade = creds.RISK_CONFIG.risk_per_trade
        risk_per_trade = base_risk_per_trade
        
        # Per-sector capital cap: each sector can use at most equity * max_sector_weight
        max_sector_weight = getattr(creds.STOCK_SELECTION, 'max_sector_weight', 0.30)
        sector_cap = account_equity * max_sector_weight
        with self.manager.manager_lock:
            sector_used = self.manager.sector_used_capital.get(self.sector, 0)
        available_for_sector = max(0, sector_cap - sector_used)
        
        print(f"\n[Position Sizing] {self.stock} (sector: {self.sector})")
        print(f"  - Risk per trade: {risk_per_trade*100:.2f}%")
        print(f"  - Sector capital: ${sector_cap:,.0f} ({max_sector_weight*100:.0f}% equity), sector used: ${sector_used:,.0f}, available: ${available_for_sector:,.0f}")
        
        # Capital for this trade = sector_capital * risk_per_trade (amount each position in that sector gets)
        capital_for_trade = sector_cap * risk_per_trade
        print(f"  - Capital for trade: ${capital_for_trade:,.0f} (sector_cap * risk_per_trade)")
        capital_for_trade = min(capital_for_trade, available_for_sector)
        max_pos_equity = getattr(creds.RISK_CONFIG, 'max_position_equity_pct', 0.1)
        if capital_for_trade > (account_equity * max_pos_equity):
            capital_for_trade = account_equity * max_pos_equity
            print(f"  - Position capped at {max_pos_equity*100:.0f}% max equity")
        if capital_for_trade <= 0:
            print(f"  - No capital left for sector {self.sector} (cap {max_sector_weight*100:.0f}% already used)")
            return 0, 0, 0, 0
        
        stop_loss_pct = self.calculate_stop_loss(current_price)
        stop_loss_price = current_price * (1 - stop_loss_pct)
        shares = int(capital_for_trade / current_price)
        
        vwap_tf = self.tf.get('vwap') or '3 mins'
        vwap_minutes = self._timeframe_to_minutes(vwap_tf) or 3
        data_vwap_tf = self.get_historical_data_with_retry(stock=self.stock, bar_size=vwap_minutes)
        vwap_value = vwap.calc_vwap(data_vwap_tf).iloc[-1]
        
        if vwap_value < current_price:
            print(f"VWAP (${vwap_value:.2f}) is below current price (${current_price:.2f})")
        else:
            print(f"WARNING: VWAP (${vwap_value:.2f}) is NOT below current price (${current_price:.2f}) - skipping order")
            return 0, 0, 0, 0
        
        # Limit at VWAP ± offset (0.03% to 0.07%); offset applied as + or - at random
        offset_min = getattr(creds.ORDER_CONFIG, 'limit_offset_min', 0.0003)  # 0.03%
        offset_max = getattr(creds.ORDER_CONFIG, 'limit_offset_max', 0.0007)   # 0.07%
        offset_pct = random.uniform(offset_min, offset_max)
        sign = random.choice([1])
        limit_price = vwap_value * (1 + sign * offset_pct)
        
        print(f"Position Size: {shares} shares")
        print(f"Entry Price: ${current_price:.2f}")
        print(f"Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%)")
        print(f"VWAP: ${vwap_value:.2f}")
        print(f"Limit Order Price: ${limit_price:.2f} (VWAP {'+' if sign == 1 else '-'} {offset_pct*100:.2f}%)")
        
        # Update database with current price and position calculations
        trades_db.update_strategy_data(self.stock, 
            current_price=current_price
        )
        
        return shares, current_price, stop_loss_price, limit_price
    
    def calculate_stop_loss(self, current_price):
        """Calculate stop loss percentage based on volatility"""
        # Use configured VWAP timeframe minutes for ATR bars as well so timeframe changes continue to work.
        vwap_tf = self.tf.get('vwap') or '3 mins'
        vwap_minutes = self._timeframe_to_minutes(vwap_tf) or 3
        data_tf = self.get_historical_data_with_retry(stock=self.stock, bar_size=vwap_minutes)
        atr14 = atr.calc_atr(data_tf, creds.STOP_LOSS_CONFIG.atr_period)
        atr14 = atr14.iloc[-1]

        # Determine volatility using ATR as a % of price
        atr_pct = atr14 / current_price if current_price > 0 else 0
        vol_threshold = getattr(creds.STOP_LOSS_CONFIG, "atr_volatility_threshold", 0.02)
        is_volatile = bool(atr_pct >= vol_threshold)

        # Use higher base stop for volatile names, lower for calm names
        base_stop = (
            creds.STOP_LOSS_CONFIG.volatile_stop_loss
            if is_volatile
            else creds.STOP_LOSS_CONFIG.default_stop_loss
        )
        
        atr_stop = (atr14 * creds.STOP_LOSS_CONFIG.atr_multiplier) / current_price
        print(f"ATR14: {atr14:.3f}, ATR%: {atr_pct*100:.2f}% (threshold {vol_threshold*100:.2f}%) -> volatile={is_volatile}")
        print(f"ATR Stop: {atr_stop:.3f}, Base Stop (vol-adjusted): {base_stop:.3f}")
        
        # Final stop is at least the (vol-adjusted) base stop, at most max_stop_loss
        stop_loss = max(base_stop, atr_stop)
        stop_loss = min(stop_loss, creds.STOP_LOSS_CONFIG.max_stop_loss)
        
        return stop_loss

    def process_score(self):
        print(f"Alpha Score: {self.score}")
        print(f"Additional Checks Passed: {self.additional_checks_passed}")
        
        # Both conditions must be met to place an order
        # if self.score >= creds.RISK_CONFIG.alpha_score_threshold: 
            # Use this condition for now for all orders to get executed 
        if self.score >= creds.RISK_CONFIG.alpha_score_threshold and bool(self.additional_checks_passed):
            print(f"ENTERING POSITION - Both Alpha Score >= {creds.RISK_CONFIG.alpha_score_threshold} AND additional checks passed")
            shares, entry_price, stop_loss, limit_price = self.calculate_position_size()
            if shares > 0:
                print(f"Order Details:")
                print(f"  - Symbol: {self.stock}")
                print(f"  - Shares: {shares}")
                print(f"  - Estimated Limit Price (not used for market order): ${limit_price:.2f}")
                print(f"  - Stop Loss: ${stop_loss:.2f}")
                print(f"  - Exit: Market-On-Close at 4:00 PM ET")
                
                with self.manager.manager_lock:
                    if self.manager.available_capital < shares * entry_price:
                        print(f"Not enough available capital to place order for {self.stock}")
                        _write_entry_decision(
                            self.stock,
                            "rejected",
                            ["Not enough available capital to place order"],
                            details={
                                "score": float(self.score),
                                "alpha_threshold": float(creds.RISK_CONFIG.alpha_score_threshold),
                                "required_cash": float(shares * entry_price),
                                "available_cash": float(self.manager.available_capital),
                                "shares": int(shares),
                                "entry_price_est": float(entry_price),
                                "limit_price": float(limit_price),
                            },
                        )
                        return
                    else:
                        print(f"Enough available capital to place order for {self.stock}")

                order_type_cfg = str(getattr(creds.ORDER_CONFIG, "order_type", "MARKET")).upper()
                if order_type_cfg not in ("MARKET", "LIMIT"):
                    order_type_cfg = "MARKET"

                if order_type_cfg == "MARKET":
                    # StrategyBroker.place_order(..., order_type="MARKET") returns (order_id, avg_price, filled_qty)
                    trade = self.broker.place_order(symbol=self.stock, qty=shares, order_type="MARKET", side="BUY")
                    avg_fill_price = trade[1] if trade and len(trade) > 1 else -1
                    filled_qty = int(trade[2]) if trade and len(trade) > 2 else 0
                    if float(avg_fill_price) == -1 or filled_qty <= 0:
                        print(f"Unable to place order for {self.stock}")
                        _write_entry_decision(
                            self.stock,
                            "order_failed",
                            ["Market order was not filled (timed out or rejected)"],
                            details={
                                "score": float(self.score),
                                "alpha_threshold": float(creds.RISK_CONFIG.alpha_score_threshold),
                                "requested_shares": int(shares),
                                "filled_shares": int(filled_qty),
                            },
                        )
                        return
                    if filled_qty != shares:
                        print(f"Market order partially filled for {self.stock}: requested={shares}, filled={filled_qty}")
                    shares = filled_qty
                    print(f"Order placed for {self.stock}: {trade} (avg_fill_price={avg_fill_price}, filled={filled_qty})")
                    cost = shares * float(avg_fill_price)
                    fill_price_for_state = float(avg_fill_price)
                else:
                    # StrategyBroker.place_order(..., order_type="LIMIT") returns (order_id, avg_price, filled_qty)
                    trade = self.broker.place_order(symbol=self.stock, qty=shares, order_type="LIMIT", price=round(limit_price, 2), side="BUY")
                    avg_fill_price = trade[1] if trade and len(trade) > 1 else -1
                    filled_qty = int(trade[2]) if trade and len(trade) > 2 else 0
                    if float(avg_fill_price) == -1 or filled_qty <= 0:
                        print(f"Unable to place order for {self.stock}")
                        _write_entry_decision(
                            self.stock,
                            "order_failed",
                            ["Limit order was not filled (timed out or rejected)"],
                            details={
                                "score": float(self.score),
                                "alpha_threshold": float(creds.RISK_CONFIG.alpha_score_threshold),
                                "limit_price": float(limit_price),
                                "requested_shares": int(shares),
                                "filled_shares": int(filled_qty),
                            },
                        )
                        return
                    if filled_qty != shares:
                        print(f"Limit order partially filled for {self.stock}: requested={shares}, filled={filled_qty}")
                    shares = filled_qty
                    print(f"Order placed for {self.stock}: {trade} (avg_fill_price={avg_fill_price}, filled={filled_qty})")
                    cost = shares * float(avg_fill_price)
                    fill_price_for_state = float(avg_fill_price)

                with self.manager.manager_lock:
                    self.used_capital += cost
                    self.manager.available_capital -= cost
                    self.manager.used_capital += cost
                    self.manager.stock_invested_capital += cost
                    self.manager.sector_used_capital[self.sector] = self.manager.sector_used_capital.get(self.sector, 0) + cost
                    print(f"Used Capital: ${self.used_capital:.2f}")
                    print(f"Available Capital: ${self.manager.available_capital:.2f}")
                    print(f"Stock Invested Capital: ${self.manager.stock_invested_capital:.2f}")
                    print(f"Sector '{self.sector}' used capital: ${self.manager.sector_used_capital.get(self.sector, 0):,.0f}")

                self.entry_price = float(fill_price_for_state)
                self.stop_loss_price = stop_loss
                self.take_profit_price = self.entry_price * (1 + creds.PROFIT_CONFIG.profit_booking_levels[0]['gain'])  # Use 1% from profit booking levels
                self.position_shares = shares
                self.position_active = True
                self.profit_booking_levels_remaining = list(creds.PROFIT_CONFIG.profit_booking_levels)
                self.trailing_stop_levels_remaining = list(creds.STOP_LOSS_CONFIG.trailing_stop_levels)
                print(f"[{self.stock}] Profit booking and trailing stop levels reset for new position")

                eastern_tz = pytz.timezone('US/Eastern')
                self.entry_time = datetime.now(eastern_tz)
                self.current_price = float(fill_price_for_state)
                # self.unrealized_pnl = 0.0
                # self.realized_pnl = 0.0
                print(f"Position tracking initialized for {self.stock}:")
                print(f"  - Entry Price: ${self.entry_price:.2f}")
                print(f"  - Stop Loss: ${self.stop_loss_price:.2f}")
                print(f"  - Take Profit: ${self.take_profit_price:.2f}")
                print(f"  - Entry Time: {self.entry_time}")
                
                # Update database with position initialization
                trades_db.update_strategy_data(self.stock,
                    position_active=True,
                    position_shares=shares,
                    shares=shares,
                    entry_price=self.entry_price,
                    stop_loss_price=self.stop_loss_price,
                    take_profit_price=self.take_profit_price,
                    entry_time=self.entry_time,
                    current_price=self.current_price,
                    unrealized_pnl=self.unrealized_pnl,
                    realized_pnl=self.realized_pnl,
                    used_capital=self.used_capital,
                    sector=self.sector
                )
                _write_entry_decision(
                    self.stock,
                    "entered",
                    ["Entry conditions met and order filled"],
                    details={
                        "score": float(self.score),
                        "alpha_threshold": float(creds.RISK_CONFIG.alpha_score_threshold),
                        "filled_shares": int(shares),
                        "fill_price": float(fill_price_for_state),
                        "sector": str(self.sector),
                    },
                )
        else:
            reasons = []
            if self.score < creds.RISK_CONFIG.alpha_score_threshold:
                print(f"Alpha Score too low: {self.score} < {creds.RISK_CONFIG.alpha_score_threshold}")
                reasons.append(f"Alpha score {self.score:.1f} < required {creds.RISK_CONFIG.alpha_score_threshold}")

                # Add which alpha-score components failed (if available)
                try:
                    cp = getattr(self, "criteria_passed", {}) or {}
                    iv = getattr(self, "indicator_values", {}) or {}
                    failed_alpha = []

                    if cp.get("trend_passed") is False:
                        close_px = iv.get("close")
                        vwap_val = iv.get("vwap")
                        ema1_val = iv.get("ema1")
                        ema2_val = iv.get("ema2")
                        failed_alpha.append(
                            f"Trend failed: price_above_vwap={cp.get('price_above_vwap')} (close={close_px:.2f} vs vwap={vwap_val:.2f}) "
                            f"and ema_cross={cp.get('ema_cross')} (ema1={ema1_val:.2f} vs ema2={ema2_val:.2f})"
                            if all(isinstance(x, (int, float)) for x in [close_px, vwap_val, ema1_val, ema2_val])
                            else "Trend failed (price above VWAP and EMA cross not both true)"
                        )

                    if cp.get("momentum_passed") is False:
                        macd_val = iv.get("adjusted_macd")
                        failed_alpha.append(
                            f"Momentum failed: adjusted_macd={macd_val:.4f} (need > 0)"
                            if isinstance(macd_val, (int, float))
                            else "Momentum failed: MACD condition not met"
                        )

                    if cp.get("volume_volatility_passed") is False:
                        vol_recent = iv.get("volume")
                        vol_avg = iv.get("volume_avg")
                        adx_val = iv.get("adx")
                        vol_mult = getattr(self.alpha_score_config.volume_volatility.conditions.volume_spike, "multiplier", None)
                        adx_thr = getattr(self.alpha_score_config.volume_volatility.conditions.adx_threshold, "threshold", None)
                        parts = []
                        if isinstance(vol_recent, (int, float)) and isinstance(vol_avg, (int, float)) and isinstance(vol_mult, (int, float)):
                            parts.append(f"volume={vol_recent:,.0f} (need > {vol_mult:.2f}×avg {vol_avg:,.0f} = {vol_mult*vol_avg:,.0f})")
                        if isinstance(adx_val, (int, float)) and isinstance(adx_thr, (int, float)):
                            parts.append(f"ADX={adx_val:.1f} (need > {adx_thr})")
                        if parts:
                            failed_alpha.append("Volume/Volatility failed: " + "; ".join(parts))
                        else:
                            failed_alpha.append("Volume/Volatility failed (volume spike + ADX threshold not both met)")

                    if cp.get("market_calm_passed") is False:
                        vix_val = iv.get("vix")
                        vix_thr = getattr(self.alpha_score_config.market_calm.conditions.vix_threshold, "threshold", None)
                        vix_low = cp.get("vix_low")
                        vix_drop = cp.get("vix_dropping")
                        if isinstance(vix_val, (int, float)) and isinstance(vix_thr, (int, float)):
                            failed_alpha.append(
                                f"Market calm failed: vix_low={vix_low} (VIX={vix_val:.1f}, need < {vix_thr}); vix_dropping={vix_drop}"
                            )
                        else:
                            failed_alpha.append("Market calm failed (VIX below threshold and dropping not both true)")

                    if failed_alpha:
                        reasons.append("Alpha conditions failed: " + " | ".join(failed_alpha))
                except Exception:
                    # keep reasons minimal if anything unexpected
                    pass
            if not bool(self.additional_checks_passed):
                print("Additional checks failed")
                failed = getattr(self, "additional_checks_failed", None)
                if isinstance(failed, list) and failed:
                    # Add observed vs threshold for each failed additional check (when possible)
                    try:
                        iv = getattr(self, "indicator_values", {}) or {}
                        detail_parts = []
                        for chk in failed:
                            if chk == "volume":
                                recent = iv.get("volume_recent")
                                avg = iv.get("volume_avg_addl")
                                mult = getattr(self.additional_checks_config, "volume_multiplier", None)
                                if all(isinstance(x, (int, float)) for x in [recent, avg, mult]):
                                    detail_parts.append(f"Volume check failed: recent={recent:,.0f} ≤ required>{mult:.2f}×avg {avg:,.0f} = {mult*avg:,.0f}")
                                else:
                                    detail_parts.append("Volume check failed")
                            elif chk == "VWAP slope":
                                slope = iv.get("vwap_slope")
                                thr = getattr(self.additional_checks_config, "vwap_slope_threshold", None)
                                if isinstance(slope, (int, float)) and isinstance(thr, (int, float)):
                                    detail_parts.append(f"VWAP slope failed: slope={slope:.4f} $/min ≤ threshold {thr}")
                                else:
                                    detail_parts.append("VWAP slope failed")
                            elif chk == "TRIN/TICK":
                                trin = iv.get("trin")
                                tick_ma = iv.get("tick_ma")
                                trin_thr = getattr(self.additional_checks_config, "trin_threshold", None)
                                tick_thr = getattr(self.additional_checks_config, "tick_threshold", None)
                                parts = []
                                if isinstance(trin, (int, float)) and isinstance(trin_thr, (int, float)):
                                    parts.append(f"TRIN={trin:.2f} (need ≤ {trin_thr})")
                                if isinstance(tick_ma, (int, float)) and isinstance(tick_thr, (int, float)):
                                    parts.append(f"TICK MA={tick_ma:.0f} (need ≥ {tick_thr})")
                                if parts:
                                    detail_parts.append("TRIN/TICK failed: " + "; ".join(parts))
                                else:
                                    detail_parts.append("TRIN/TICK failed")
                            else:
                                detail_parts.append(f"{chk} failed")

                        if detail_parts:
                            reasons.append("Additional checks failed: " + " | ".join(detail_parts))
                        else:
                            reasons.append(f"Additional checks failed: {', '.join([str(x) for x in failed])}")
                    except Exception:
                        reasons.append(f"Additional checks failed: {', '.join([str(x) for x in failed])}")
                else:
                    reasons.append("Additional checks failed")
            if reasons:
                _write_entry_decision(
                    self.stock,
                    "rejected",
                    reasons,
                    details={
                        "score": float(self.score),
                        "alpha_threshold": float(creds.RISK_CONFIG.alpha_score_threshold),
                        "additional_checks_passed": bool(self.additional_checks_passed),
                        "additional_checks_failed": list(getattr(self, "additional_checks_failed", []) or []),
                        "criteria_passed": dict(getattr(self, "criteria_passed", {}) or {}),
                        "indicator_values": dict(getattr(self, "indicator_values", {}) or {}),
                    },
                )
        return -1
                    
    def run(self, i):
        while True:
            # Check for end-of-day exit times
            eastern_tz = pytz.timezone(self.trading_hours.timezone)
            current_time = datetime.now(eastern_tz).time()

            # Get exit times from configuration
            weak_exit_time = self.time_str_to_datetime(self.trading_hours.weak_exit_time)
            safety_exit_time = self.time_str_to_datetime(self.trading_hours.safety_exit_time)
            market_close_time = self.time_str_to_datetime(self.trading_hours.market_close)

            if current_time >= weak_exit_time and current_time < safety_exit_time:
                self.end_of_day_weak_exit()
                # break

            elif current_time >= safety_exit_time:
                self.safety_exit_all()
                break
                        
            # 4:00 PM - Market-on-close for any remaining positions
            elif current_time >= market_close_time:
                self.market_on_close_exit()
                break  # End trading for the day

            if not TESTING:            
                # Check if we're in entry time windows
                if self.manager.is_entry_time_window():
                    print(f"[{self.stock}] In entry time window - calculating indicators and processing scores")
                    self.calculate_indicators()
                    self.process_score()
                else:
                    # Outside entry windows - only monitor existing positions
                    window_status = self.get_next_entry_window()
                    print(f"[{self.stock}] {window_status} - only monitoring existing positions")
                    
                # Always monitor active positions regardless of time
                if self.position_active and self.position_shares > 0:
                    print(f"Starting position monitoring for {self.stock}...")
                    self.start_individual_monitoring()
                
                # Sleep for a bit before next iteration if no active position
                if not self.position_active:
                    time.sleep(60*3)
            else:
                self.calculate_indicators()
                self.process_score()

                if self.position_active and self.position_shares > 0:
                    print(f"Starting position monitoring for {self.stock}...")
                    self.start_individual_monitoring()
    
    def monitor_position(self):
        try:
            self.current_price = self.get_current_price_with_retry(self.stock)
            
            if self.current_price is None:
                print(f"Unable to get current price for {self.stock} in position monitoring")
                return
            
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.position_shares
            with self.manager.manager_lock:
                self.manager.unrealized_pnl += self.unrealized_pnl
            
            # Calculate current gain/loss percentage
            current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
            
            if self.current_price <= self.stop_loss_price:
                print(f"STOP LOSS TRIGGERED: {self.stock} - Current: ${self.current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
                self.close_position('stop_loss')
                return
            
            self.check_take_profit()

            if self.manager.stop_event.is_set():
                self.close_position('drawdown')

            print(f"Position Status - {self.stock}:")
            print(f"  - Current Price: ${self.current_price:.2f}")
            print(f"  - Entry Price: ${self.entry_price:.2f}")
            print(f"  - Unrealized PnL: ${self.unrealized_pnl:.2f} ({current_gain_pct*100:.2f}%)")
            print(f"  - Shares: {self.position_shares}")
            print(f"  - Stop Loss: ${self.stop_loss_price:.2f}")
            print(f"  - Take Profit: ${self.take_profit_price:.2f}")
            
            trades_db.update_strategy_data(self.stock,
                current_price=self.current_price,
                unrealized_pnl=self.unrealized_pnl,
                position_shares=self.position_shares,
                stop_loss_price=self.stop_loss_price,
                take_profit_price=self.take_profit_price
            )
            
        except Exception as e:
            print(f"Error monitoring position for {self.stock}: {e}")

    def check_take_profit(self):
        current_gain_pct = (self.current_price - self.entry_price) / self.entry_price

        self._check_profit_booking(current_gain_pct, self.current_price)
        
        self._check_trailing_stops(current_gain_pct, self.current_price)

        trailing_conditions = creds.PROFIT_CONFIG.trailing_exit_conditions
        gain_threshold = trailing_conditions.gain_threshold
        drop_threshold = trailing_conditions.drop_threshold
        monitor_period = trailing_conditions.monitor_period
        
        if current_gain_pct >= gain_threshold:
            if not self.trailing_exit_monitoring:
                self.trailing_exit_monitoring = True
                eastern_tz = pytz.timezone('US/Eastern')
                self.trailing_exit_start_time = datetime.now(eastern_tz)
                self.trailing_exit_start_price = self.current_price
                # Track peak price during the monitoring window so drop is from the high, not just the start price
                self.trailing_exit_peak_price = self.current_price
                print(f"Starting trailing exit monitoring at {gain_threshold*100:.1f}% gain")
                print(f"Monitoring for {drop_threshold*100:.1f}% price drop from peak for {monitor_period} minutes, checking every second")
                
                # Update database with trailing exit monitoring
                trades_db.update_strategy_data(self.stock,
                    trailing_exit_monitoring=True,
                    trailing_exit_start_time=self.trailing_exit_start_time,
                    trailing_exit_start_price=self.trailing_exit_start_price
                )
                
                # Start active monitoring for the specified period
                monitor_period_seconds = monitor_period * 60  # Convert minutes to seconds
                start_monitoring_time = datetime.now(eastern_tz)
                
                print(f"Starting active monitoring for {monitor_period_seconds} seconds...")
                
                while True:
                    try:
                        # Check if monitoring period has expired
                        current_time = datetime.now(eastern_tz)
                        elapsed_seconds = (current_time - start_monitoring_time).total_seconds()
                        
                        if elapsed_seconds >= monitor_period_seconds:
                            print(f"Trailing monitoring period ({monitor_period} minutes) expired - continuing with position")
                            self.trailing_exit_monitoring = False
                            
                            # Update database to stop monitoring
                            trades_db.update_strategy_data(self.stock,
                                trailing_exit_monitoring=False
                            )
                            break
                        
                        # Get current price
                        self.current_price = self.get_current_price_with_retry(self.stock)
                        if self.current_price is None:
                            print(f"Unable to get current price for {self.stock}, retrying...")
                            time.sleep(1)
                            continue
                        
                        # Update peak price seen during monitoring
                        if self.current_price > self.trailing_exit_peak_price:
                            self.trailing_exit_peak_price = self.current_price

                        # Calculate current price drop from peak
                        price_drop = (self.trailing_exit_peak_price - self.current_price) / self.trailing_exit_peak_price
                        
                        # Check if price drop threshold is met
                        if price_drop >= drop_threshold:
                            print(f"Trailing Exit: Price dropped by {price_drop*100:.2f}% within {elapsed_seconds/60:.1f} minutes")
                            print(f"Closing all remaining shares ({self.position_shares}) due to trailing exit")
                            self.close_position('trailing_exit', self.position_shares)
                            return  # Exit the function and stop monitoring
                        
                        # Print monitoring status every 30 seconds
                        if int(elapsed_seconds) % 30 == 0:
                            print(f"Trailing monitoring: {elapsed_seconds/60:.1f}/{monitor_period} minutes elapsed, price drop: {price_drop*100:.2f}%")
                        
                        # Check if position is still active (in case it was closed by other means)
                        if not self.position_active:
                            print("Position no longer active - exiting trailing monitoring")
                            return
                        
                        # Sleep for 1 second before next check
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error in trailing monitoring for {self.stock}: {e}")
                        time.sleep(1)
                        continue
    



    
    def close_position(self, reason, shares_to_sell=None):
        current_price = self.get_current_price_with_retry(self.stock)
        
        if shares_to_sell is None:
            shares_to_sell = self.position_shares

        # Validate shares_to_sell against available shares
        if shares_to_sell > self.position_shares:
            print(f"Warning: Requested shares ({shares_to_sell}) are more than available shares ({self.position_shares})")
            print(f"Closing only available shares: {self.position_shares}")
            shares_to_sell = self.position_shares
        elif shares_to_sell < self.position_shares:
            print(f"Partially closing position: {shares_to_sell} shares out of {self.position_shares}")
        if shares_to_sell <= 0:
            print(f"No shares to sell for {self.stock}")
            return
        
        # Calculate realized PnL for shares being sold
        
        trade = self.broker.place_order(symbol=self.stock, qty=shares_to_sell, order_type="MARKET", price=current_price, side="SELL")
        if trade is None:
            print(f"Unable to place order for {self.stock}")
            return
        
        with self.manager.manager_lock:
            self.manager.available_capital += trade[1] * shares_to_sell
            original_cost = self.entry_price * shares_to_sell
            self.manager.used_capital -= original_cost
            # Round to 0 if in range [-2, 2] to handle floating point precision errors
            new_value = self.manager.stock_invested_capital - original_cost
            if -2 <= new_value <= 2:
                self.manager.stock_invested_capital = 0
            else:
                self.manager.stock_invested_capital = new_value
            self.used_capital -= original_cost
            # Decrement sector used capital so the sector cap is freed for new positions
            self.manager.sector_used_capital[self.sector] = max(0, self.manager.sector_used_capital.get(self.sector, 0) - original_cost)
            print(f"Available Capital after closing position: ${self.manager.available_capital:.2f}")
            print(f"Used Capital after closing position: ${self.used_capital:.2f}")
            print(f"Used Total Capital after closing position: ${self.manager.used_capital:.2f}")
            print(f"Stock Invested Capital after closing position: ${self.manager.stock_invested_capital:.2f}")

        # Update remaining shares
        realized_pnl = (trade[1] - self.entry_price) * shares_to_sell
        self.position_shares -= shares_to_sell

        if self.position_shares <= 0:
            # Full position closure
            self.position_active = False
            
            # Reset profit booking and trailing stop levels for next position
            if hasattr(self, 'profit_booking_levels_remaining'):
                delattr(self, 'profit_booking_levels_remaining')
            if hasattr(self, 'trailing_stop_levels_remaining'):
                delattr(self, 'trailing_stop_levels_remaining')
            
            # Get current time in USA Eastern Time
            eastern_tz = pytz.timezone('US/Eastern')
            self.close_time = datetime.now(eastern_tz)
            
            with self.manager.manager_lock:
                self.manager.realized_pnl += realized_pnl
                self.manager.unrealized_pnl -= realized_pnl
                self.unrealized_pnl = 0
                self.realized_pnl += realized_pnl

            # Calculate position duration
            if self.entry_time:
                duration = self.close_time - self.entry_time
                print(f"Position fully closed for {self.stock}")
                print(f"Position Summary:")
                print(f"  - Entry Time: {self.entry_time}")
                print(f"  - Exit Time: {self.close_time}")
                print(f"  - Duration: {duration}")
                print(f"  - Entry Price: ${self.entry_price:.2f}")
                print(f"  - Exit Price: ${current_price:.2f}")
                print(f"  - Realized PnL: ${realized_pnl:.2f}")
                print(f"  - Return: {((current_price - self.entry_price) / self.entry_price * 100):.2f}%")
                print(f"  - Reason: {reason}")
                
                # Update database with position closure
                trades_db.update_strategy_data(self.stock,  
                    position_active=False,
                    position_shares=0,
                    realized_pnl=self.realized_pnl,
                    unrealized_pnl=0,
                    current_price=current_price,
                    used_capital=self.used_capital,
                    close_time=self.close_time
                )
        else:
            # Partial position closure
            print(f"Partial position closed. Remaining shares: {self.position_shares}")
            
            # Update unrealized PnL for remaining position
            self.unrealized_pnl = (trade[1] - self.entry_price) * self.position_shares
            
            with self.manager.manager_lock:
                self.manager.realized_pnl += realized_pnl
                self.manager.unrealized_pnl -= realized_pnl
                self.realized_pnl += realized_pnl
                
            # Update database with partial position closure
            trades_db.update_strategy_data(self.stock,
                position_shares=self.position_shares,
                unrealized_pnl=self.unrealized_pnl,
                realized_pnl=self.realized_pnl,
                current_price=current_price,
                used_capital=self.used_capital
            ) 
    
    def _check_profit_booking(self, current_gain_pct, current_price):
        """Check profit booking levels"""
        # Create a local copy of profit booking levels to avoid modifying the global config
        if not hasattr(self, 'profit_booking_levels_remaining'):
            self.profit_booking_levels_remaining = list(creds.PROFIT_CONFIG.profit_booking_levels)
            levels_str = ', '.join([f'{level["gain"]*100:.1f}%' for level in self.profit_booking_levels_remaining])
            print(f"[{self.stock}] Profit booking levels initialized: {levels_str}")
        
        # Check each remaining level
        for level in self.profit_booking_levels_remaining[:]:  # Use slice copy to avoid modification during iteration
            gain_threshold = level['gain']
            exit_pct = level['exit_pct']
            
            if current_gain_pct >= gain_threshold:
                # Calculate shares to sell
                shares_to_sell = int(self.position_shares * exit_pct)
                if shares_to_sell > 0:
                    # Close partial position
                    self.close_position('profit_booking', shares_to_sell)
                    print(f"[{self.stock}] Profit Booking: Sold {shares_to_sell} shares at {gain_threshold*100:.1f}% gain")
                    
                    # Remove this level from remaining levels for this strategy instance
                    self.profit_booking_levels_remaining.remove(level)
                    remaining_levels_str = ', '.join([f'{level["gain"]*100:.1f}%' for level in self.profit_booking_levels_remaining])
                    print(f"[{self.stock}] Remaining profit booking levels: {remaining_levels_str}")
                    
                    # Update database with profit booking
                    trades_db.update_strategy_data(self.stock,
                        profit_booked_flags={f'profit_booked_{gain_threshold*100:.0f}pct': True}
                    )
                    break
    
    def _check_trailing_stops(self, current_gain_pct, current_price):
        """Check trailing stop levels"""
        # Create a local copy of trailing stop levels to avoid modifying the global config
        if not hasattr(self, 'trailing_stop_levels_remaining'):
            self.trailing_stop_levels_remaining = list(creds.STOP_LOSS_CONFIG.trailing_stop_levels)
            levels_str = ', '.join([f'{level["gain"]*100:.1f}%' for level in self.trailing_stop_levels_remaining])
            print(f"[{self.stock}] Trailing stop levels initialized: {levels_str}")
        
        # Check each remaining level
        for level in self.trailing_stop_levels_remaining[:]:  # Use slice copy to avoid modification during iteration
            gain_threshold = level['gain']
            new_stop_pct = level['new_stop_pct']
            
            if current_gain_pct >= gain_threshold:
                # Move stop loss to new level
                new_stop_price = self.entry_price * (1 + new_stop_pct)
                self.stop_loss_price = new_stop_price
                print(f"[{self.stock}] TRAILING STOP: Moved SL to ${new_stop_price:.2f} at {gain_threshold*100:.1f}% gain")
                
                # Remove this level from remaining levels for this strategy instance
                self.trailing_stop_levels_remaining.remove(level)
                remaining_levels_str = ', '.join([f'{level["gain"]*100:.1f}%' for level in self.trailing_stop_levels_remaining])
                print(f"[{self.stock}] Remaining trailing stop levels: {remaining_levels_str}")
                
                # Update database with trailing stop
                trades_db.update_strategy_data(self.stock,
                    stop_loss_price=self.stop_loss_price,
                    trailing_stop_flags={f'trailing_stop_{gain_threshold*100:.0f}pct_set': True}
                )
                break
    
    def _check_trailing_exit(self, current_gain_pct, current_price):
        """Check trailing exit conditions after 5% gain"""
        exit_config = creds.PROFIT_CONFIG.trailing_exit_conditions
        
        if current_gain_pct >= exit_config.gain_threshold:
            # Check if we should exit based on price drop
            # Get current time in USA Central Time
            eastern_tz = pytz.timezone('US/Eastern')
            self.trailing_start_time = datetime.now(eastern_tz)
            self.max_price_since_trailing = current_price
            print(f"TRAILING EXIT MONITORING: {self.stock} - Started monitoring for exit conditions")
            
            # Update max price
            self.max_price_since_trailing = max(self.max_price_since_trailing, current_price)
            
            # Check time and price drop
            # Get current time in USA Eastern Time for calculation
            time_since_trailing = datetime.now(eastern_tz) - self.trailing_start_time
            if time_since_trailing.total_seconds() <= exit_config.monitor_period * 60:
                price_drop = (self.max_price_since_trailing - current_price) / self.max_price_since_trailing
                if price_drop >= exit_config.drop_threshold:
                    # Exit entire position
                    self.close_position('trailing_exit')
                    print(f"TRAILING EXIT: {self.stock} - Exited due to {price_drop*100:.1f}% drop from peak")
                    return True
        return False
    
    def start_individual_monitoring(self):
        """Start continuous monitoring for this individual stock every 5 minutes"""
        while self.position_active:
            try:
                # Monitor position
                self.monitor_position()

                eastern_tz = pytz.timezone(self.trading_hours.timezone)
                current_time = datetime.now(eastern_tz).time()

                # Get exit times from configuration
                weak_exit_time = self.time_str_to_datetime(self.trading_hours.weak_exit_time)
                safety_exit_time = self.time_str_to_datetime(self.trading_hours.safety_exit_time)

                if current_time >= weak_exit_time and current_time < safety_exit_time:
                    print("Weak Exit")
                    self.end_of_day_weak_exit()
                    break

                elif current_time >= safety_exit_time:
                    print("All exit")
                    self.safety_exit_all()
                    break

                time.sleep(10)
                
            except KeyboardInterrupt:
                print(f"\nPosition monitoring stopped for {self.stock}")
                break
            except Exception as e:
                print(f"Error monitoring position for {self.stock}: {e}")
                time.sleep(60)  # Wait 1 minute on error before retrying
    
    def get_position_status(self):
        """Get current position status and details"""
        if not self.position_active:
            return {
                'symbol': self.stock,
                'active': False,
                'message': 'No active position'
            }
        
        return {
            'symbol': self.stock,
            'active': self.position_active,
            'shares': self.position_shares,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss_price,
            'take_profit': self.take_profit_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.manager.realized_pnl,
            'pnl': self.unrealized_pnl,  # For backward compatibility
            'pnl_pct': ((self.current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0,
            'entry_time': self.entry_time,
            'duration': (datetime.now(pytz.timezone('US/Eastern')) - self.entry_time) if self.entry_time else None
        }
    
    def print_position_summary(self):
        """Print a detailed summary of the current position"""
        if not self.position_active:
            print(f"No active position for {self.stock}")
            return
        
        status = self.get_position_status()
        print(f"\n=== Position Summary for {self.stock} ===")
        print(f"Status: {'ACTIVE' if status['active'] else 'INACTIVE'}")
        print(f"Shares: {status['shares']}")
        print(f"Entry Price: ${status['entry_price']:.2f}")
        print(f"Current Price: ${status['current_price']:.2f}")
        print(f"Stop Loss: ${status['stop_loss']:.2f}")
        print(f"Take Profit: ${status['take_profit']:.2f}")
        print(f"PnL: ${status['pnl']:.2f} ({status['pnl_pct']:.2f}%)")
        print(f"Entry Time: {status['entry_time']}")
        if status['duration']:
            print(f"Duration: {status['duration']}")
        print("=" * 50)
    
    def reset_position_tracking(self):
        """Reset all position tracking variables"""
        self.position_active = False
        self.position_shares = 0
        self.current_price = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.entry_time = None
        self.close_time = None
        
        # Reset profit booking and trailing stop levels
        if hasattr(self, 'profit_booking_levels_remaining'):
            delattr(self, 'profit_booking_levels_remaining')
        if hasattr(self, 'trailing_stop_levels_remaining'):
            delattr(self, 'trailing_stop_levels_remaining')
        
        # Update database with reset
        trades_db.update_strategy_data(self.stock,
            position_active=False,
            position_shares=0,
            current_price=0,
            entry_price=0,
            stop_loss_price=0,
            take_profit_price=0,
            unrealized_pnl=0,
            realized_pnl=0,
            entry_time=None,
            close_time=None
        )
        
        print(f"Position tracking reset for {self.stock}")

class StrategyBroker:
    def __init__(self, host="127.0.0.1", port=7497, client_id=1):
        """Initialize StrategyBroker with IBBroker instance"""
        self.ib_broker = IBBroker()
        self.ib_broker.connect_to_ibkr(host, port, client_id)
        self.request_id_counter = 1
        self.counter_lock = threading.Lock()
        self.reset_order_counter_to_next_available()

    def get_current_price(self, symbol):
        """Get current price for symbol"""
        with self.counter_lock:
            req_id = self.request_id_counter
            self.request_id_counter += 1
        return self.ib_broker.get_current_price(symbol, req_id)

    def get_historical_data_stock(self, stock, duration="3 D", bar_size="3 mins"):
        """Get historical data for symbol"""
        # Valid bar sizes according to IBKR API
        valid_bar_sizes = {
            '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
            '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1W', '1M'
        }
        
        # Convert bar_size to proper IBKR format
        if isinstance(bar_size, (int, str)) and str(bar_size).isdigit():
            # Special case for 1: "1 min" not "1 mins"
            if int(bar_size) == 1:
                bar_size = "1 min"
            else:
                bar_size = f"{bar_size} mins"
        elif isinstance(bar_size, str) and bar_size.endswith(' min'):
            # Don't convert "1 min" to "1 mins" - it's already correct
            if not bar_size.startswith('1 '):
                bar_size = bar_size.replace(' min', ' mins')
        
        # Validate bar size
        if bar_size not in valid_bar_sizes:
            print(f"Warning: Invalid bar size '{bar_size}'. Using '5 mins' as fallback.")
            bar_size = '5 mins'
        
        with self.counter_lock:
            req_id = self.request_id_counter + 1000  # Offset for historical data
            self.request_id_counter += 1
        return self.ib_broker.get_historical_data_stock(stock, req_id, duration, bar_size)
    
    def get_historical_data_index(self, symbol, duration="1 D", bar_size="1 min"):
        """Get historical data for index symbols like TRIN-NYSE, TICK-NYSE"""
        # Valid bar sizes according to IBKR API
        valid_bar_sizes = {
            '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
            '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1W', '1M'
        }
        
        # Convert bar_size to proper IBKR format
        if isinstance(bar_size, (int, str)) and str(bar_size).isdigit():
            # Special case for 1: "1 min" not "1 mins"
            if int(bar_size) == 1:
                bar_size = "1 min"
            else:
                bar_size = f"{bar_size} mins"
        elif isinstance(bar_size, str) and bar_size.endswith(' min'):
            # Don't convert "1 min" to "1 mins" - it's already correct
            if not bar_size.startswith('1 '):
                bar_size = bar_size.replace(' min', ' mins')
        
        # Validate bar size
        if bar_size not in valid_bar_sizes:
            print(f"Warning: Invalid bar size '{bar_size}'. Using '1 min' as fallback.")
            bar_size = '1 min'
        
        with self.counter_lock:
            req_id = self.request_id_counter + 3000  # Offset for index data
            self.request_id_counter += 1
        return self.ib_broker.get_historical_data_index(symbol, req_id, duration, bar_size)

    def get_trin_tick_data(self, symbol, exchange="NYSE",
                           duration="1 D", bar_size="1 min", what_to_show="TRADES"):
        """Specialized helper for TRIN/TICK NYSE internals using explicit index contract.

        what_to_show maps to IBKR's historical data types, e.g. "TRADES", "BID", "ASK".
        """
        # Valid bar sizes according to IBKR API
        valid_bar_sizes = {
            '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
            '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1W', '1M'
        }

        # Convert bar_size to proper IBKR format
        if isinstance(bar_size, (int, str)) and str(bar_size).isdigit():
            if int(bar_size) == 1:
                bar_size = "1 min"
            else:
                bar_size = f"{bar_size} mins"
        elif isinstance(bar_size, str) and bar_size.endswith(' min'):
            if not bar_size.startswith('1 '):
                bar_size = bar_size.replace(' min', ' mins')

        if bar_size not in valid_bar_sizes:
            print(f"Warning: Invalid bar size '{bar_size}'. Using '1 min' as fallback.")
            bar_size = '1 min'

        with self.counter_lock:
            req_id = self.request_id_counter + 4500  # Separate offset for TRIN/TICK data
            self.request_id_counter += 1
        return self.ib_broker.get_trin_tick_historical(
            symbol, exchange, req_id, duration, bar_size, what_to_show
        )

    def is_connected(self):
        """Check if connected to IBKR"""
        return self.ib_broker.connected

    def place_market_order(self, symbol, quantity, action="BUY"):
        """Place market order and wait for fill; cancels after order_window seconds if not filled."""
        order_window = getattr(creds.ORDER_CONFIG, "order_window", 60)
        # Use StrategyBroker's counter to avoid conflicts
        with self.counter_lock:
            order_id = self.request_id_counter + 2000  # Offset for orders
            self.request_id_counter += 1
        return self.ib_broker.place_market_order_with_id(symbol, quantity, action, order_id, order_window=order_window)

    def place_limit_order(self, symbol, quantity, limit_price, action="BUY"):
        """Place limit order and wait for fill; cancels after order_window seconds if not filled."""
        order_window = getattr(creds.ORDER_CONFIG, "order_window", 60)
        # Use StrategyBroker's counter to avoid conflicts
        with self.counter_lock:
            order_id = self.request_id_counter + 2000  # Offset for orders
            self.request_id_counter += 1
        return self.ib_broker.place_limit_order_with_id(symbol, quantity, limit_price, action, order_id, order_window=order_window)

    def get_all_used_order_ids(self):
        """Get all order IDs that have been used (active and completed)"""
        return self.ib_broker.get_all_used_order_ids()

    def cancel_order(self, order_id):
        """Cancel an order by order ID"""
        self.ib_broker.cancel_order(order_id)

    def round_price_to_tick(self, price, tick_size=0.01):
        """Round price to the minimum tick size for the contract"""
        return round(price / tick_size) * tick_size

    def get_next_available_order_id(self):
        """Get the next available order ID directly from IBKR"""
        return self.ib_broker.get_next_order_id_from_ibkr()

    def reset_order_counter_to_next_available(self):
        """Reset the counter to start from the next available order ID from IBKR"""
        next_id = self.get_next_available_order_id()
        if next_id:
            with self.counter_lock:
                self.request_id_counter = next_id - 2000  # Adjust counter to match order ID range
            print(f"Reset order counter to start from IBKR ID: {next_id}")
        else:
            print("Could not get next order ID from IBKR")

    def disconnect(self):
        """Disconnect from IBKR"""
        self.ib_broker.disconnect()

    def place_order(self, symbol, qty, order_type='MARKET', price=None, side='BUY'):
        """Place order with automatic type selection"""
        if order_type == 'MARKET':
            # Market orders don't use price parameter
            return self.place_market_order(symbol, qty, side)
        elif order_type == 'LIMIT':
            if price is None:
                raise ValueError("Price required for LIMIT orders")
            return self.place_limit_order(symbol, qty, price, side)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def get_total_pnl(self):
        """Get total combined PnL from stock positions (realized + unrealized)"""
        try:
            # Use trades_db's lock instead of self.lock since this is in StrategyBroker
            conn = trades_db._get_connection()
            cursor = conn.cursor()
            
            # Query to get total combined PnL from stock positions only
            query = '''
                SELECT 
                    SUM(COALESCE(realized_pnl, 0) + COALESCE(unrealized_pnl, 0)) as total_pnl
                FROM stock_strategies
            '''
            
            cursor.execute(query)
            result = cursor.fetchone()
            
            conn.close()
            
            return result[0] if result and result[0] is not None else 0
                    
        except Exception as e:
            print(f"Error getting total PnL from database: {e}")
            return 0

class StrategyManager:
    
    def __init__(self):
        self.broker = StrategyBroker("127.0.0.1", 7497, client_id=1)
        self.config = creds
        self.selector = StockSelector(client_id=15)  # Use different client ID to avoid conflicts
        self.manager_lock = threading.Lock()
        self.available_capital = creds.EQUITY
        self.used_capital = 0  # Total capital used (stocks only, excludes hedge)
        self.stock_invested_capital = 0  # Capital invested in stocks (for hedge calculations)
        self.sector_used_capital = {}  # sector name -> total used_capital (stocks only); capped at equity * max_sector_weight per sector
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.stocks_list, self.stocks_dict = [], []
        self.stop_event = threading.Event()
        self.max_drawdown_triggered = False
        
        # Semaphore to limit concurrent data requests across all strategy threads
        # This prevents thundering herd problem when multiple threads request data simultaneously
        # Limit to 5 concurrent requests to avoid overwhelming the broker API
        self.data_request_semaphore = threading.Semaphore(7)
        hedge_symbol = self.config.HEDGE_CONFIG.hedge_symbol
        print(f"Adding hedge symbol {hedge_symbol} to database for all threads")
        
        # Verify database is properly initialized before adding data
        if not trades_db.verify_database():
            print("Database verification failed, reinitializing...")
            trades_db.init_database()
            if not trades_db.verify_database():
                raise Exception("Failed to initialize database properly")
        
        trades_db.add_stocks_from_list([hedge_symbol])
        
        # Initialize centralized hedge symbol
        trades_db.update_strategy_data(hedge_symbol,
            position_active=False,
            position_shares=0,
            entry_price=0,
            stop_loss_price=0,
            take_profit_price=0,
            current_price=0,
            unrealized_pnl=0,
            realized_pnl=0,
            used_capital=0                                
        )
        
        # List to track stocks that pass all entry conditions
        self.qualifying_stocks = []
        self.qualifying_stocks_lock = threading.Lock()  # Thread-safe access
        
        # Hedge symbol for centralized hedge management
        self.hedge_symbol = self.config.HEDGE_CONFIG.hedge_symbol
        self.hedge_config = self.config.HEDGE_CONFIG
    
    def _safe_broker_call(self, broker_method, *args, max_retries=5, base_delay=2.0, max_delay=300.0, **kwargs):
        """Safely call broker methods with exponential backoff and random jitter"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = broker_method(*args, **kwargs)
                if result is not None:
                    return result
                else:
                    print(f"Broker call returned None, retrying... (attempt {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                last_exception = e
                print(f"Broker call failed: {e} (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                exponential_delay = base_delay * (2 ** attempt)
                if attempt == 0:
                    min_random = exponential_delay - 0.5
                    max_random = exponential_delay + 0.5
                elif attempt == 1:
                    min_random = exponential_delay - 1.0
                    max_random = exponential_delay + 1.0
                elif attempt == 2:
                    min_random = exponential_delay - 2.0
                    max_random = exponential_delay + 2.0
                else:
                    randomization_factor = min(attempt * 0.5, 2.0)
                    min_random = exponential_delay * (1 - randomization_factor)
                    max_random = min(exponential_delay * (1 + randomization_factor), max_delay)
                
                min_random = max(min_random, 2.0)
                delay = random.uniform(min_random, max_random)
                print(f"Retrying broker call in {delay:.2f} seconds...")
                time.sleep(delay)
        
        if last_exception:
            print(f"All {max_retries} attempts failed. Last error: {last_exception}")
        else:
            print(f"All {max_retries} attempts returned None")
        
        return None
    
    def get_current_price_with_retry(self, symbol):
        """Get current price with exponential backoff retry logic"""
        return self._safe_broker_call(self.broker.get_current_price, symbol)
    
    def get_historical_data_with_retry(self, stock, duration="3 D", bar_size="3 mins"):
        """Get historical data with exponential backoff retry logic"""
        return self._safe_broker_call(self.broker.get_historical_data_stock, stock=stock, duration=duration, bar_size=bar_size)
    
    def _write_hedge_status(self, metrics, thresholds_dict, entered, level, message):
        """Write last hedge check result to JSON for frontend display."""
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hedge_status.json")
            payload = {
                "updated_at": datetime.now(pytz.timezone("US/Eastern")).strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "thresholds": thresholds_dict,
                "entered": entered,
                "level": level,
                "message": message,
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[HEDGE] Could not write hedge_status.json: {e}")

    def check_hedge_triggers(self):
        """Check if hedge triggers are met and return hedge level based on VIX and S&P (SPY proxy) conditions"""
        if not self.hedge_config.enabled:
            print("[HEDGE ENTRY] Hedge NOT entered: hedging is disabled in config (hedge_config.enabled=False)")
            return None, 0, 0
        
        try:
            # Get VIX data
            vix_timeframe = self.hedge_config.triggers.vix_timeframe
            vix_data = self._safe_broker_call(self.broker.get_historical_data_index, "VIX", duration="1 D", bar_size=vix_timeframe)
            if vix_data is None or vix_data.empty or 'close' not in vix_data.columns:
                print(f"[HEDGE ENTRY] Hedge NOT entered: VIX data unavailable or empty for hedge trigger check")
                return None, 0, 0
            
            current_vix = float(vix_data['close'].iloc[-1])
            
            # Get SPY data (15-minute bars for S&P drop calculation; SPY used as proxy for SPX)
            spy_data = self._safe_broker_call(self.broker.get_historical_data_stock, "SPY", duration="1 D", bar_size="15 mins")
            if spy_data is None or spy_data.empty or len(spy_data) < 2 or 'close' not in spy_data.columns:
                print(f"[HEDGE ENTRY] Hedge NOT entered: SPY data unavailable or insufficient for hedge trigger check")
                return None, 0, 0
            
            current_spy = float(spy_data['close'].iloc[-1])
            price_15min_ago = float(spy_data['close'].iloc[-2])
            drop_pct = (price_15min_ago - current_spy) / price_15min_ago
            
            # Config thresholds
            s = self.hedge_config.hedge_levels.severe
            m = self.hedge_config.hedge_levels.mild
            e = self.hedge_config.hedge_levels.early
            
            metrics = {"vix": round(current_vix, 1), "spy_now": round(current_spy, 2), "spy_15m_ago": round(price_15min_ago, 2), "sp500_drop_pct": round(drop_pct * 100, 2)}
            thresholds_dict = {
                "severe": {"vix": s.vix_threshold, "drop_pct": round(s.spx_drop_threshold * 100, 2)},
                "mild": {"vix": m.vix_threshold, "drop_pct": round(m.spx_drop_threshold * 100, 2)},
                "early": {"vix": e.vix_threshold, "drop_pct": round(e.spx_drop_threshold * 100, 2)},
            }
            
            # Always print current metrics and thresholds
            print(f"[HEDGE ENTRY] Metrics: VIX={current_vix:.1f} | SPY now={current_spy:.2f} | SPY 15m ago={price_15min_ago:.2f} | S&P drop={drop_pct*100:.2f}%")
            print(f"[HEDGE ENTRY] Thresholds: Severe VIX>={s.vix_threshold} drop>={s.spx_drop_threshold*100:.2f}% | Mild VIX>={m.vix_threshold} drop>={m.spx_drop_threshold*100:.2f}% | Early VIX>={e.vix_threshold} drop>={e.spx_drop_threshold*100:.2f}%")
            
            # Check hedge levels from severe to early (most restrictive first)
            # Severe: VIX >= 25 and S&P drop >= threshold
            if (current_vix >= s.vix_threshold and drop_pct >= s.spx_drop_threshold):
                hedge_level = 'severe'
                beta = s.beta
                equity_pct = s.equity_pct
                trigger_details = f"VIX {current_vix:.1f} >= {s.vix_threshold} and S&P drop {drop_pct*100:.2f}% >= {s.spx_drop_threshold*100:.2f}%"
                print(f"[HEDGE ENTRY] Hedge ENTERED: level=severe | {trigger_details}")
                print(f"[HEDGE ENTRY] Hedge params: beta={beta}, equity_pct={equity_pct*100:.1f}% of stock invested")
                self._write_hedge_status(metrics, thresholds_dict, True, "severe", f"Hedge ENTERED: level=severe | {trigger_details}")
                return hedge_level, beta, equity_pct
            
            # Mild: VIX >= 22 and S&P drop >= threshold
            if (current_vix >= m.vix_threshold and drop_pct >= m.spx_drop_threshold):
                hedge_level = 'mild'
                beta = m.beta
                equity_pct = m.equity_pct
                trigger_details = f"VIX {current_vix:.1f} >= {m.vix_threshold} and S&P drop {drop_pct*100:.2f}% >= {m.spx_drop_threshold*100:.2f}%"
                print(f"[HEDGE ENTRY] Hedge ENTERED: level=mild | {trigger_details}")
                print(f"[HEDGE ENTRY] Hedge params: beta={beta}, equity_pct={equity_pct*100:.1f}% of stock invested")
                self._write_hedge_status(metrics, thresholds_dict, True, "mild", f"Hedge ENTERED: level=mild | {trigger_details}")
                return hedge_level, beta, equity_pct
            
            # Early: VIX >= 20 and S&P drop >= threshold
            if (current_vix >= e.vix_threshold and drop_pct >= e.spx_drop_threshold):
                hedge_level = 'early'
                beta = e.beta
                equity_pct = e.equity_pct
                trigger_details = f"VIX {current_vix:.1f} >= {e.vix_threshold} and S&P drop {drop_pct*100:.2f}% >= {e.spx_drop_threshold*100:.2f}%"
                print(f"[HEDGE ENTRY] Hedge ENTERED: level=early | {trigger_details}")
                print(f"[HEDGE ENTRY] Hedge params: beta={beta}, equity_pct={equity_pct*100:.1f}% of stock invested")
                self._write_hedge_status(metrics, thresholds_dict, True, "early", f"Hedge ENTERED: level=early | {trigger_details}")
                return hedge_level, beta, equity_pct
            
            # No trigger conditions met - explain why
            reasons = []
            if current_vix < e.vix_threshold:
                reasons.append(f"VIX {current_vix:.1f} < {e.vix_threshold} (early)")
            elif current_vix < m.vix_threshold:
                reasons.append(f"VIX {current_vix:.1f} < {m.vix_threshold} (mild)")
            elif current_vix < s.vix_threshold:
                reasons.append(f"VIX {current_vix:.1f} < {s.vix_threshold} (severe)")
            if drop_pct < e.spx_drop_threshold:
                reasons.append(f"S&P drop {drop_pct*100:.2f}% < {e.spx_drop_threshold*100:.2f}% (early)")
            elif drop_pct < m.spx_drop_threshold:
                reasons.append(f"S&P drop {drop_pct*100:.2f}% < {m.spx_drop_threshold*100:.2f}% (mild)")
            elif drop_pct < s.spx_drop_threshold:
                reasons.append(f"S&P drop {drop_pct*100:.2f}% < {s.spx_drop_threshold*100:.2f}% (severe)")
            reason_msg = "; ".join(reasons)
            print(f"[HEDGE ENTRY] Hedge NOT entered. Reason: {reason_msg}")
            self._write_hedge_status(metrics, thresholds_dict, False, None, f"Hedge NOT entered. Reason: {reason_msg}")
            return None, 0, 0
            
        except Exception as e:
            print(f"[HEDGE ENTRY] Hedge NOT entered: Error checking hedge triggers: {e}")
            traceback.print_exc()
            return None, 0, 0
    
    def is_hedge_active(self):
        """Check if hedge is currently active from centralized database"""
        try:
            central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
            position_shares = central_data.get('position_shares', 0) or 0
            return position_shares > 0
        except Exception as e:
            print(f"Error checking hedge status: {e}")
            return False
    
    def execute_hedge(self, hedge_level, beta, equity_pct):
        """Execute hedge by buying hedge ETF - centralized hedge management"""
        if hedge_level is None:
            print("[HEDGE ENTRY] Execute skipped: hedge_level is None (no trigger met)")
            return
        if self.is_hedge_active():
            print(f"[HEDGE ENTRY] Execute skipped: hedge already active (would have entered level={hedge_level})")
            return
        
        try:
            # Calculate hedge size based on stock invested capital (not total equity)
            with self.manager_lock:
                stock_invested = self.stock_invested_capital
                # Round to 0 if in range [-2, 2] to handle floating point precision errors
                if -2 <= stock_invested <= 2:
                    stock_invested = 0
            hedge_amount = stock_invested * equity_pct
            
            # Get hedge symbol price
            hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if hedge_price is None:
                print(f"Unable to get {self.hedge_symbol} price for hedge")
                return
            
            # Calculate shares to buy
            hedge_shares = int(hedge_amount / hedge_price)

            trade = self.broker.place_order(symbol=self.hedge_symbol, qty=hedge_shares, order_type="MARKET", price=hedge_price, side="BUY")
            if trade is None:
                print(f"Unable to place hedge order for {self.hedge_symbol}")
                return
            
            print(f"[HEDGE ENTRY] Executing hedge: level={hedge_level} | beta={beta} | equity_pct={equity_pct*100:.1f}%")
            print(f"[HEDGE ENTRY]   Stock invested: ${stock_invested:,.0f} | Hedge amount: ${hedge_amount:,.0f} | {self.hedge_symbol} price: ${trade[1]:.2f} | Shares: {hedge_shares}")
            
            # Get current hedge position data
            central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
            current_position_shares = central_data.get('position_shares', 0) or 0
            current_cumulative_shares = central_data.get('shares', 0) or 0
            
            # Calculate new values (hedge does NOT affect used_capital)
            new_total_shares = current_position_shares + hedge_shares
            fill_price = trade[1]
            new_avg_entry_price = ((central_data.get('entry_price', 0) or 0) * current_position_shares + fill_price * hedge_shares) / new_total_shares if new_total_shares > 0 else fill_price
            
            # Calculate unrealized PnL for the new position
            hedge_unrealized_pnl = (fill_price - new_avg_entry_price) * new_total_shares if new_avg_entry_price > 0 else 0
            
            # Update manager capital (hedge uses available capital but doesn't count as used_capital)
            with self.manager_lock:
                self.available_capital -= (fill_price * hedge_shares)
                # Note: used_capital is NOT updated for hedge - it only tracks stock positions
            
            # Update database (hedge used_capital in DB is for hedge position tracking only, not manager used_capital)
            hedge_used_capital = (central_data.get('used_capital', 0) or 0) + (fill_price * hedge_shares)
            
            # Only set entry_time if creating a new position
            if current_position_shares == 0:
                trades_db.update_strategy_data(self.hedge_symbol,
                    position_active=True,
                    position_shares=new_total_shares,
                    shares=current_cumulative_shares + hedge_shares,
                    used_capital=hedge_used_capital,  # This is hedge-specific, not manager used_capital
                    entry_price=new_avg_entry_price,
                    current_price=fill_price,
                    unrealized_pnl=hedge_unrealized_pnl,
                    hedge_level=hedge_level,
                    hedge_beta=beta,
                    entry_time=datetime.now(pytz.timezone('US/Eastern'))
                )
            else:
                trades_db.update_strategy_data(self.hedge_symbol,
                    position_active=True,
                    position_shares=new_total_shares,
                    shares=current_cumulative_shares + hedge_shares,
                    used_capital=hedge_used_capital,  # This is hedge-specific, not manager used_capital
                    entry_price=new_avg_entry_price,
                    current_price=fill_price,
                    unrealized_pnl=hedge_unrealized_pnl,
                    hedge_level=hedge_level,
                    hedge_beta=beta
                )
                     
            print(f"[HEDGE ENTRY] Hedge position OPENED: bought {hedge_shares} shares of {self.hedge_symbol} at ${fill_price:.2f} (level={hedge_level})")
                
        except Exception as e:
            print(f"[HEDGE ENTRY] Hedge entry FAILED: {e}")
            traceback.print_exc()
    
    def check_hedge_exit_conditions(self):
        """Check if hedge exit conditions are met for scaling down - ALL conditions must be true"""
        if not self.is_hedge_active():
            return None
        
        signal_details = []
        all_conditions_met = True
        
        try:
            # Check VIX < 20 and falling (10-min slope negative)
            vix_timeframe = self.hedge_config.exit_conditions.vix_timeframe
            vix_data = self._safe_broker_call(self.broker.get_historical_data_index, "VIX", duration="1 D", bar_size=vix_timeframe)
            vix_condition_met = False
            if vix_data is not None and not vix_data.empty and len(vix_data) >= 4 and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                vix_exit_threshold = self.hedge_config.exit_conditions.vix_exit_threshold
                vix_10min_ago = vix_data['close'].iloc[-4]  # ~10 minutes ago
                
                if current_vix < vix_exit_threshold and current_vix < vix_10min_ago:
                    vix_condition_met = True
                    signal_details.append(f"VIX {current_vix:.1f} < {vix_exit_threshold} and falling")
                else:
                    signal_details.append(f"VIX condition NOT MET (current: {current_vix:.1f}, threshold: {vix_exit_threshold}, falling: {current_vix < vix_10min_ago})")
            else:
                print(f"VIX data unavailable for exit condition check")
                all_conditions_met = False
            
            if not vix_condition_met:
                all_conditions_met = False
            
            # Check S&P 500 up > +0.6% in 15 min after hedge added (use SPY as proxy)
            spy_data = self._safe_broker_call(self.broker.get_historical_data_stock, "SPY", duration="1 D", bar_size="15 mins")
            spy_condition_met = False
            if spy_data is not None and not spy_data.empty and len(spy_data) >= 2 and 'close' in spy_data.columns:
                current_price = spy_data['close'].iloc[-1]
                price_15min_ago = spy_data['close'].iloc[-2]
                gain_pct = (current_price - price_15min_ago) / price_15min_ago
                sp500_recovery_threshold = self.hedge_config.exit_conditions.sp500_recovery_threshold
                
                if gain_pct > sp500_recovery_threshold:
                    spy_condition_met = True
                    signal_details.append(f"S&P up {gain_pct*100:.1f}% > {sp500_recovery_threshold*100:.1f}%")
                else:
                    signal_details.append(f"S&P condition NOT MET (gain: {gain_pct*100:.1f}%, threshold: {sp500_recovery_threshold*100:.1f}%)")
            else:
                print(f"SPY data unavailable for exit condition check")
                all_conditions_met = False
            
            if not spy_condition_met:
                all_conditions_met = False
            
            # Check Nasdaq (QQQ) trades above 5-min VWAP for 2+ consecutive bars
            qqq_data = self.get_historical_data_with_retry(stock="QQQ", bar_size=5)
            qqq_condition_met = False
            if qqq_data is not None and not qqq_data.empty and len(qqq_data) >= 2 and 'close' in qqq_data.columns:
                # Calculate 5-min VWAP
                qqq_vwap = vwap.calc_vwap(qqq_data)
                qqq_consecutive_bars = self.hedge_config.exit_conditions.sqqq_vwap_consecutive_bars
                if len(qqq_vwap) >= qqq_consecutive_bars:
                    current_qqq = qqq_data['close'].iloc[-1]
                    qqq_vwap_current = qqq_vwap.iloc[-1]
                    qqq_vwap_prev = qqq_vwap.iloc[-2]
                    
                    # Check if QQQ is above VWAP for 2+ consecutive bars
                    if current_qqq > qqq_vwap_current and qqq_data['close'].iloc[-2] > qqq_vwap_prev:
                        qqq_condition_met = True
                        signal_details.append(f"QQQ above 5-min VWAP for {qqq_consecutive_bars}+ bars")
                    else:
                        signal_details.append(f"QQQ VWAP condition NOT MET")
                else:
                    signal_details.append(f"QQQ VWAP data insufficient (need {qqq_consecutive_bars} bars)")
            else:
                print(f"QQQ data unavailable for exit condition check")
                all_conditions_met = False
            
            if not qqq_condition_met:
                all_conditions_met = False
            
        except Exception as e:
            print(f"Error checking hedge exit conditions: {e}")
            traceback.print_exc()
            return None
        
        if all_conditions_met:
            print(f"ALL hedge exit conditions met - {', '.join(signal_details)}")
            return True
        else:
            print(f"Hedge exit conditions NOT met - {', '.join(signal_details)}")
            return False
    
    def scale_down_hedge(self):
        """Scale down hedge based on current level and recovery signals"""
        if not self.is_hedge_active():
            return
        
        recovery_signals = self.check_hedge_exit_conditions()
        if recovery_signals is None or recovery_signals == False:
            return 
        
        try:
            # Get current hedge position data
            central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
            current_hedge_level = central_data.get('hedge_level', 'severe')
            current_hedge_shares = central_data.get('position_shares', 0) or 0
            current_used_capital = central_data.get('used_capital', 0) or 0
            current_realized_pnl = central_data.get('realized_pnl', 0) or 0
            entry_price = central_data.get('entry_price', 0) or 0
            
            # Scale down logic
            if current_hedge_level == 'severe':
                new_level = 'mild'
                new_beta = self.hedge_config.hedge_levels.mild.beta
                new_equity_pct = self.hedge_config.hedge_levels.mild.equity_pct
                print(f"Scaling hedge: Severe → Mild (-{new_beta}β, {new_equity_pct*100:.1f}% of stock invested)")
                
            elif current_hedge_level == 'mild':
                new_level = 'early'
                new_beta = self.hedge_config.hedge_levels.early.beta
                new_equity_pct = self.hedge_config.hedge_levels.early.equity_pct
                print(f"Scaling hedge: Mild → Early (-{new_beta}β, {new_equity_pct*100:.1f}% of stock invested)")
                
            elif current_hedge_level == 'early':
                print(f"Scaling hedge: Early → Exit fully")
                self.close_hedge()
                return
            
            else:
                print(f"Unknown hedge level: {current_hedge_level}")
                return
            
            # Calculate new hedge size based on current stock invested capital
            with self.manager_lock:
                stock_invested = self.stock_invested_capital
                # Round to 0 if in range [-2, 2] to handle floating point precision errors
                if -2 <= stock_invested <= 2:
                    stock_invested = 0
            new_hedge_amount = stock_invested * new_equity_pct
            
            # Get current hedge price
            hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if hedge_price is None:
                print(f"Unable to get {self.hedge_symbol} price for hedge scaling")
                return
            
            # Calculate shares to adjust
            new_hedge_shares = int(new_hedge_amount / hedge_price)
            shares_to_close = current_hedge_shares - new_hedge_shares
            
            if shares_to_close > 0:
                print(f"Closing {shares_to_close} shares of hedge (reducing from {current_hedge_shares} to {new_hedge_shares})")
                
                # Close partial hedge position
                trade = self.broker.place_order(symbol=self.hedge_symbol, qty=shares_to_close, order_type="MARKET", price=hedge_price, side="SELL")
                if trade is None:
                    print(f"Unable to place hedge scaling order for {self.hedge_symbol}")
                    return
                
                # Calculate new values (using data already read)
                # entry_price was already retrieved earlier in the function
                fill_price = trade[1]
                partial_hedge_pnl = (fill_price - entry_price) * shares_to_close
                remaining_shares = max(0, current_hedge_shares - shares_to_close)
                # Calculate new average entry price for remaining hedge
                hedge_used_capital = max(0, current_used_capital - (entry_price * shares_to_close))
                new_avg_entry_price = (hedge_used_capital / remaining_shares) if remaining_shares > 0 else 0
                
                # Calculate unrealized PnL for remaining shares
                if remaining_shares > 0 and new_avg_entry_price > 0:
                    hedge_unrealized_pnl = (fill_price - new_avg_entry_price) * remaining_shares
                else:
                    hedge_unrealized_pnl = 0
                
                # Update manager capital (hedge does NOT affect used_capital)
                with self.manager_lock:
                    self.available_capital += (fill_price * shares_to_close)
                    # Note: used_capital is NOT updated for hedge - it only tracks stock positions
                
                # If remaining shares is 0, set hedge inactive
                position_active = remaining_shares > 0
                
                # Update database (hedge used_capital in DB is for hedge position tracking only)
                # Set close_time if all shares are sold
                if remaining_shares == 0:
                    trades_db.update_strategy_data(self.hedge_symbol,
                        position_active=position_active,
                        position_shares=remaining_shares,
                        used_capital=hedge_used_capital,  # This is hedge-specific, not manager used_capital
                        entry_price=new_avg_entry_price,
                        realized_pnl=current_realized_pnl + partial_hedge_pnl,
                        current_price=fill_price,
                        unrealized_pnl=hedge_unrealized_pnl,
                        hedge_level=new_level,
                        hedge_beta=new_beta,
                        close_time=datetime.now(pytz.timezone('US/Eastern'))
                    )
                    print(f"[HEDGE] All hedge shares sold - setting hedge inactive")
                else:
                    trades_db.update_strategy_data(self.hedge_symbol,
                        position_active=position_active,
                        position_shares=remaining_shares,
                        used_capital=hedge_used_capital,  # This is hedge-specific, not manager used_capital
                        entry_price=new_avg_entry_price,
                        realized_pnl=current_realized_pnl + partial_hedge_pnl,
                        current_price=fill_price,
                        unrealized_pnl=hedge_unrealized_pnl,
                        hedge_level=new_level,
                        hedge_beta=new_beta
                    )
                
                print(f"Hedge scaled down successfully: {current_hedge_level} → {new_level}")
                print(f"Remaining hedge: {new_hedge_shares} shares ({new_equity_pct*100:.1f}% of equity)")
            
        except Exception as e:
            print(f"Error scaling down hedge: {e}")
            traceback.print_exc()
    
    def close_hedge(self):
        """Close hedge position by selling hedge shares - centralized hedge management"""
        if not self.is_hedge_active():
            return
        
        try:
            # Get current hedge position from database
            central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
            hedge_shares = central_data.get('position_shares', 0) or 0
            
            if hedge_shares == 0:
                return
            
            print(f"[HEDGE] Initiating close for {self.hedge_symbol}: selling {hedge_shares} shares")
            
            # Get entry_price for P&L calculation
            entry_price = central_data.get('entry_price', 0) or 0
            if entry_price == 0:
                print(f"Warning: No entry_price found for hedge, cannot calculate P&L")
            
            print(f"[HEDGE] Placing SELL order: symbol={self.hedge_symbol}, qty={hedge_shares}")
            trade = self.broker.place_order(symbol=self.hedge_symbol, qty=hedge_shares, order_type="MARKET", price=0, side="SELL")
            if trade is None:
                print(f"Unable to place hedge order for {self.hedge_symbol}")
                return
            else:
                print(f"[HEDGE] SELL filled: price={trade[1]}")
            
            # Calculate PnL using actual fill price and entry_price
            hedge_pnl = (trade[1] - entry_price) * hedge_shares if entry_price > 0 else 0

            # Get current realized PnL (already have central_data from earlier)
            current_realized_pnl = central_data.get('realized_pnl', 0) or 0
            
            # Update manager capital (hedge does NOT affect used_capital)
            with self.manager_lock:
                self.available_capital += (trade[1] * hedge_shares)
                # Note: used_capital is NOT updated for hedge - it only tracks stock positions

            # Update database
            trades_db.update_strategy_data(self.hedge_symbol,
                position_active=False,
                position_shares=0,
                used_capital=0,
                realized_pnl=current_realized_pnl + hedge_pnl,
                unrealized_pnl=0,
                current_price=trade[1],  # Use actual fill price, not estimated price
                close_time=datetime.now(pytz.timezone('US/Eastern')),
                hedge_pnl=hedge_pnl
            )
            
            print(f"[HEDGE] Close complete. Hedge P&L: ${hedge_pnl:.2f}")
            print(f"[HEDGE] Centralized position for {self.hedge_symbol} updated.")
                
        except Exception as e:
            print(f"Error closing hedge: {e}")
            traceback.print_exc()
    
    def update_hedge_position_unrealized_pnl(self):
        """Update unrealized PnL for the centralized hedge position"""
        try:
            if not self.is_hedge_active():
                return
            
            # Get current hedge price
            current_hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if current_hedge_price is None:
                return
            
            # Get current hedge position data
            central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
            hedge_position_shares = central_data.get('position_shares', 0) or 0
            entry_price = central_data.get('entry_price', 0) or 0
            
            if hedge_position_shares > 0 and entry_price > 0:
                # Calculate unrealized PnL
                hedge_unrealized_pnl = (current_hedge_price - entry_price) * hedge_position_shares
                
                # Update database
                trades_db.update_strategy_data(self.hedge_symbol,
                    current_price=current_hedge_price,
                    unrealized_pnl=hedge_unrealized_pnl
                )
                
        except Exception as e:
            print(f"Error updating hedge unrealized PnL: {e}")
    
    def calculate_portfolio_atr14(self):
        try:
            atr_values = []
            active_count = 0
            
            for strategy in self.strategies:
                if strategy.position_active and strategy.position_shares > 0:
                    active_count += 1
                    try:
                        data_3min = self.broker.get_historical_data_stock(stock=strategy.stock, bar_size="3 mins")
                        if data_3min is not None and not data_3min.empty:
                            atr14 = atr.calc_atr(data_3min, 14)
                            if atr14 is not None and not atr14.empty:
                                atr_value = atr14.iloc[-1]
                                current_price = strategy.current_price if strategy.current_price > 0 else data_3min['close'].iloc[-1]
                                atr_pct = atr_value / current_price
                                atr_values.append(atr_pct)
                                print(f"[Portfolio ATR] {strategy.stock}: ATR14=${atr_value:.2f}, ATR%={atr_pct*100:.2f}%")
                    except Exception as e:
                        print(f"[Portfolio ATR] Error calculating ATR for {strategy.stock}: {e}")
                        continue
            
            if atr_values:
                portfolio_atr_pct = sum(atr_values) / len(atr_values)
                print(f"[Portfolio ATR] Average ATR% across {len(atr_values)} positions: {portfolio_atr_pct*100:.2f}%")
                return portfolio_atr_pct
            else:
                print(f"[Portfolio ATR] No active positions to calculate ATR (active_count={active_count})")
                return 0.02
                
        except Exception as e:
            print(f"[Portfolio ATR] Error in calculate_portfolio_atr14: {e}")
            return 0.02 
    
    def get_fixed_daily_limit(self):
        """Hard daily drawdown limit (e.g. 2% of equity) from config."""
        equity = creds.EQUITY
        pct = getattr(self.config.RISK_CONFIG, 'daily_drawdown_limit', 0.02)
        return equity * pct

    def calculate_dynamic_daily_limit(self):
        """Dynamic daily limit = min(lower_limit, atr_multiplier * portfolio ATR14, upper_limit)."""
        equity = creds.EQUITY
        
        limit_2pct = equity * self.config.RISK_CONFIG.lower_limit
        
        portfolio_atr_pct = self.calculate_portfolio_atr14()
        atr_multiplier = getattr(self.config.RISK_CONFIG, 'atr_multiplier', 1.5)
        limit_atr = equity * (atr_multiplier * portfolio_atr_pct)
        
        limit_3pct = equity * self.config.RISK_CONFIG.upper_limit
        
        daily_limit_dollars = min(limit_2pct, limit_atr, limit_3pct)
        daily_limit_pct = daily_limit_dollars / equity
        
        print(f"[Dynamic Daily Limit]")
        print(f"  - lower_limit: ${limit_2pct:,.0f}")
        print(f"  - {atr_multiplier} x Portfolio ATR14: ${limit_atr:,.0f} (ATR={portfolio_atr_pct*100:.2f}%)")
        print(f"  - upper_limit cap: ${limit_3pct:,.0f}")
        print(f"  - Selected limit: ${daily_limit_dollars:,.0f} ({daily_limit_pct*100:.2f}%)")
        
        return daily_limit_dollars

    def monitor_drawdown_loop(self):
        """Monitor daily PnL; if fixed or dynamic drawdown limit is hit, set stop_event to exit all strategies."""
        print("Starting global drawdown monitoring thread.")
        self.max_drawdown_triggered = False
        eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
        while not self.stop_event.is_set():
            try:
                total_pnl = self.broker.get_total_pnl()
                fixed_limit = self.get_fixed_daily_limit()
                dynamic_limit = self.calculate_dynamic_daily_limit()
                threshold = min(fixed_limit, dynamic_limit)

                now = datetime.now(eastern_tz)
                current_time_str = now.strftime("%H:%M")
                print(f"[Drawdown] PnL: ${total_pnl:.2f}, Fixed: ${fixed_limit:,.0f}, Dynamic: ${dynamic_limit:,.0f}, Threshold: ${threshold:,.0f}")

                if total_pnl <= -threshold and not self.max_drawdown_triggered:
                    which = "fixed (hard)" if threshold == fixed_limit else "dynamic"
                    print(f"Daily drawdown limit hit ({which}): loss ${-total_pnl:,.0f} >= ${threshold:,.0f}. Stopping all strategies and closing hedge.")
                    # If a centralized hedge is active, close it before stopping strategies
                    try:
                        if hasattr(self, "is_hedge_active") and self.is_hedge_active():
                            print("[Drawdown] Active hedge detected - closing hedge due to drawdown stop.")
                            self.close_hedge()
                    except Exception as hedge_err:
                        print(f"[Drawdown] Error while closing hedge on drawdown stop: {hedge_err}")
                    self.max_drawdown_triggered = True
                    self.stop_event.set()

                if current_time_str >= self.config.TRADING_HOURS.market_close:
                    print(f"Exit time reached at {current_time_str} - Closing drawdown monitor thread")
                    self.max_drawdown_triggered = True
                    self.stop_event.set()
                    break

                time.sleep(60)
            except Exception as e:
                print(f"Error in drawdown monitoring: {e}")
                time.sleep(5)
        print("Drawdown monitor thread has been closed.")

    def is_entry_time_window(self):
        """Check if current time is within entry time windows"""
        
        eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        
        morning_start = self.config.TRADING_HOURS.morning_entry_start
        morning_end = self.config.TRADING_HOURS.morning_entry_end
        afternoon_start = self.config.TRADING_HOURS.afternoon_entry_start
        afternoon_end = self.config.TRADING_HOURS.afternoon_entry_end
        
        
        def time_str_to_datetime(time_str):
            return datetime.strptime(time_str, "%H:%M").time()
        
        current_time_obj = time_str_to_datetime(current_time_str)
        morning_start_obj = time_str_to_datetime(morning_start)
        morning_end_obj = time_str_to_datetime(morning_end)
        afternoon_start_obj = time_str_to_datetime(afternoon_start)
        afternoon_end_obj = time_str_to_datetime(afternoon_end)
        
        
        in_morning_window = morning_start_obj <= current_time_obj <= morning_end_obj
        in_afternoon_window = afternoon_start_obj <= current_time_obj <= afternoon_end_obj
        
        return in_morning_window or in_afternoon_window

    def hedge_monitoring_loop(self):
        """Hedge monitoring thread (Thread 0) - handles all centralized hedge logic"""
        print("[Hedge-0] Hedge monitoring thread started.", flush=True)
        
        # Cache time conversion function and market hours
        def time_str_to_datetime(time_str):
            return datetime.strptime(time_str, "%H:%M").time()
        
        eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
        market_open_time = time_str_to_datetime(self.config.TRADING_HOURS.market_open)
        market_close_time = time_str_to_datetime(self.config.TRADING_HOURS.market_close)
        hedge_force_exit_time = self.config.TRADING_HOURS.hedge_force_exit_time
        hedge_exit_time_obj = time_str_to_datetime(hedge_force_exit_time)
        
        print(f"[Hedge-0] Monitoring active. Market hours: {self.config.TRADING_HOURS.market_open}-{self.config.TRADING_HOURS.market_close}, hedge exit at {hedge_force_exit_time}. Symbol: {self.hedge_symbol}", flush=True)
        
        while True:
            try:
                # Check if hedge exit time is met (only during market hours)
                current_time = datetime.now(eastern_tz)
                current_time_str = current_time.strftime("%H:%M")
                current_time_obj = time_str_to_datetime(current_time_str)
                
                # Only check exit time if we're within market hours
                in_market_hours = market_open_time <= current_time_obj <= market_close_time
                
                # If hedge exit time is met during market hours, close hedge and return
                if in_market_hours and current_time_obj >= hedge_exit_time_obj:
                    if self.is_hedge_active():
                        print(f"[Hedge Thread 0] Hedge exit time ({hedge_force_exit_time}) reached - closing active hedge...")
                        self.close_hedge()
                        print("[Hedge Thread 0] Hedge closed successfully")
                    else:
                        print(f"[Hedge Thread 0] Hedge exit time ({hedge_force_exit_time}) reached - no active hedge positions")
                    print("[Hedge Thread 0] Hedge monitoring loop ending.")
                    return
                
                # Only check hedge triggers if hedge is NOT active
                # Round to 0 if in range [-2, 2] to handle floating point precision errors
                stock_invested_check = self.stock_invested_capital
                print(f"[Hedge-0] Stock invested: ${stock_invested_check:,.0f}")
                if -2 <= stock_invested_check <= 2:
                    stock_invested_check = 0
                # Check if we're in entry time window (skip time check if testing is true)
                in_entry_window = self.is_entry_time_window() if not self.config.TESTING else True
                if not self.is_hedge_active() and stock_invested_check > 0 and in_entry_window:
                    hedge_level, hedge_beta, hedge_equity_pct = self.check_hedge_triggers()
                    if hedge_level:
                        print(f"[Hedge Thread 0] Hedge triggers met, executing hedge (level={hedge_level})...")
                        self.execute_hedge(hedge_level, hedge_beta, hedge_equity_pct)
                    else:
                        print(f"[Hedge-0] Hedge triggers not met")
                    # If no hedge_level, check_hedge_triggers already printed metrics and reason not entered
                
                # If hedge is active, check for scaling down/up and exit conditions
                if self.is_hedge_active():
                    with self.manager_lock:
                        # Get current stock invested capital (with lock)
                        # Round to 0 if in range [-2, 2] to handle floating point precision errors
                        stock_invested = self.stock_invested_capital
                        if -2 <= stock_invested <= 2:
                            stock_invested = 0
                    
                    # If no money is invested in stocks, wait a bit and double-check before closing hedge
                    # This prevents premature closure during brief gaps between positions
                    if stock_invested == 0:
                        print(f"[HEDGE] No stock investments detected - waiting 60 seconds before closing hedge...")
                        time.sleep(60)
                        # Double-check after delay
                        with self.manager_lock:
                            stock_invested = self.stock_invested_capital
                            # Round to 0 if in range [-2, 2] to handle floating point precision errors
                            if -2 <= stock_invested <= 2:
                                stock_invested = 0
                        if stock_invested == 0:
                            print(f"[HEDGE] Confirmed no stock investments - closing all hedges")
                            self.close_hedge()
                            # Set position as not active
                            trades_db.update_strategy_data(self.hedge_symbol,
                                position_active=False
                            )
                            print(f"[HEDGE] Hedge set to inactive")
                            # Skip rest of the logic and continue loop
                            time.sleep(30)
                            continue
                        else:
                            print(f"[HEDGE] Stock investments detected after delay - keeping hedge active")
                    
                    # Get current hedge position
                    central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
                    current_hedge_level = central_data.get('hedge_level', 'severe')
                    current_hedge_shares = central_data.get('position_shares', 0) or 0
                    
                    # Check for scaling down based on recovery signals FIRST (before adjustments)
                    # This prevents adjusting hedge up only to immediately scale it down
                    self.scale_down_hedge()
                    
                    # Re-check if hedge is still active after potential scale-down
                    if not self.is_hedge_active():
                        # Update PnL and continue
                        self.update_hedge_position_unrealized_pnl()
                        time.sleep(30)
                        continue
                    
                    # Re-fetch data after potential scale-down
                    central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
                    current_hedge_level = central_data.get('hedge_level', 'severe')
                    current_hedge_shares = central_data.get('position_shares', 0) or 0
                    
                    # Now adjust hedge based on stock invested changes
                    # Get hedge config for current level
                    if current_hedge_level == 'severe':
                        equity_pct = self.hedge_config.hedge_levels.severe.equity_pct
                    elif current_hedge_level == 'mild':
                        equity_pct = self.hedge_config.hedge_levels.mild.equity_pct
                    elif current_hedge_level == 'early':
                        equity_pct = self.hedge_config.hedge_levels.early.equity_pct
                    else:
                        equity_pct = 0
                    
                    # Calculate target hedge amount based on current stock invested
                    target_hedge_amount = stock_invested * equity_pct
                    
                    # Get current hedge price
                    hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
                    if hedge_price is not None:
                        # Calculate target shares
                        target_hedge_shares = int(target_hedge_amount / hedge_price)
                        current_hedge_value = current_hedge_shares * hedge_price
                        shares_diff = target_hedge_shares - current_hedge_shares
                        value_diff = abs(target_hedge_amount - current_hedge_value)
                        
                        # Adjust hedge if difference is significant (more than 5% or $100)
                        if abs(shares_diff) > 0 and (value_diff > 100 or (current_hedge_value > 0 and value_diff / current_hedge_value > 0.05)):
                            print(f"[HEDGE] Adjusting hedge for stock changes:")
                            print(f"  - Stock invested: ${stock_invested:,.0f}")
                            print(f"  - Current hedge: {current_hedge_shares} shares (${current_hedge_value:,.0f})")
                            print(f"  - Target hedge: {target_hedge_shares} shares (${target_hedge_amount:,.0f})")
                            print(f"  - Adjustment: {shares_diff:+d} shares")
                            
                            if shares_diff > 0:
                                # Need to buy more hedge
                                trade = self.broker.place_order(symbol=self.hedge_symbol, qty=shares_diff, order_type="MARKET", price=hedge_price, side="BUY")
                                if trade is not None:
                                    # Update hedge position
                                    new_total_shares = current_hedge_shares + shares_diff
                                    hedge_used_capital = (central_data.get('used_capital', 0) or 0) + (trade[1] * shares_diff)
                                    new_avg_entry_price = (hedge_used_capital / new_total_shares) if new_total_shares > 0 else trade[1]
                                    
                                    # Calculate unrealized PnL for the new total position
                                    fill_price = trade[1]
                                    hedge_unrealized_pnl = (fill_price - new_avg_entry_price) * new_total_shares if new_avg_entry_price > 0 else 0
                                    
                                    with self.manager_lock:
                                        self.available_capital -= (fill_price * shares_diff)
                                    
                                    # Only set entry_time if creating a new position
                                    if current_hedge_shares == 0:
                                        trades_db.update_strategy_data(self.hedge_symbol,
                                            position_shares=new_total_shares,
                                            used_capital=hedge_used_capital,
                                            entry_price=new_avg_entry_price,
                                            current_price=fill_price,
                                            unrealized_pnl=hedge_unrealized_pnl,
                                            entry_time=datetime.now(pytz.timezone('US/Eastern'))
                                        )
                                    else:
                                        trades_db.update_strategy_data(self.hedge_symbol,
                                            position_shares=new_total_shares,
                                            used_capital=hedge_used_capital,
                                            entry_price=new_avg_entry_price,
                                            current_price=fill_price,
                                            unrealized_pnl=hedge_unrealized_pnl
                                        )
                                    print(f"[HEDGE] Hedge increased by {shares_diff} shares")
                                
                            elif shares_diff < 0:
                                # Need to sell some hedge
                                shares_to_sell = abs(shares_diff)
                                trade = self.broker.place_order(symbol=self.hedge_symbol, qty=shares_to_sell, order_type="MARKET", price=hedge_price, side="SELL")
                                if trade is not None:
                                    # Update hedge position
                                    entry_price = central_data.get('entry_price', 0) or 0
                                    remaining_shares = current_hedge_shares - shares_to_sell
                                    current_realized_pnl = central_data.get('realized_pnl', 0) or 0
                                    fill_price = trade[1]
                                    partial_hedge_pnl = (fill_price - entry_price) * shares_to_sell if entry_price > 0 else 0
                                    hedge_used_capital = max(0, (central_data.get('used_capital', 0) or 0) - (entry_price * shares_to_sell))
                                    new_avg_entry_price = (hedge_used_capital / remaining_shares) if remaining_shares > 0 else 0
                                    
                                    # Calculate unrealized PnL for remaining shares
                                    if remaining_shares > 0 and new_avg_entry_price > 0:
                                        hedge_unrealized_pnl = (fill_price - new_avg_entry_price) * remaining_shares
                                    else:
                                        hedge_unrealized_pnl = 0
                                    
                                    with self.manager_lock:
                                        self.available_capital += (fill_price * shares_to_sell)
                                    
                                    # If remaining shares is 0, set hedge inactive
                                    position_active = remaining_shares > 0
                                    
                                    # Set close_time if all shares are sold
                                    if remaining_shares == 0:
                                        trades_db.update_strategy_data(self.hedge_symbol,
                                            position_active=position_active,
                                            position_shares=remaining_shares,
                                            used_capital=hedge_used_capital,
                                            entry_price=new_avg_entry_price,
                                            realized_pnl=current_realized_pnl + partial_hedge_pnl,
                                            current_price=fill_price,
                                            unrealized_pnl=hedge_unrealized_pnl,
                                            close_time=datetime.now(pytz.timezone('US/Eastern'))
                                        )
                                        print(f"[HEDGE] All hedge shares sold - setting hedge inactive")
                                    else:
                                        trades_db.update_strategy_data(self.hedge_symbol,
                                            position_active=position_active,
                                            position_shares=remaining_shares,
                                            used_capital=hedge_used_capital,
                                            entry_price=new_avg_entry_price,
                                            realized_pnl=current_realized_pnl + partial_hedge_pnl,
                                            current_price=fill_price,
                                            unrealized_pnl=hedge_unrealized_pnl
                                        )
                                    print(f"[HEDGE] Hedge decreased by {shares_to_sell} shares")
                                    print(f"[HEDGE] Realized PnL: ${partial_hedge_pnl:.2f}, Remaining unrealized PnL: ${hedge_unrealized_pnl:.2f}")
                    
                    # Update hedge position unrealized PnL
                    self.update_hedge_position_unrealized_pnl()
                
                # Heartbeat when idle so you can see the hedge thread is running
                if not self.is_hedge_active():
                    with self.manager_lock:
                        si = self.stock_invested_capital
                    if -2 <= si <= 2:
                        si = 0
                    if si == 0:
                        print(f"[Hedge-0] Monitoring (no hedge, stock_invested=0) - next check in 30s", flush=True)
                
                # Sleep to prevent tight loop
                time.sleep(30)  # Check every 30s
                
            except Exception as e:
                print(f"[Hedge Thread 0] Error in hedge monitoring: {e}")
                traceback.print_exc()
                time.sleep(60)

    def run(self):
        self.threads = []
        self.strategies = []
        
        
        print("Waiting for entry time window to begin stock selection...")
        while not self.is_entry_time_window() and not self.config.TESTING:
            eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
            current_time = datetime.now(eastern_tz)
            current_time_str = current_time.strftime("%H:%M")
            print(f"Current time: {current_time_str} - Not in entry window, waiting...")
            time.sleep(60)  # Check every minute
        
        print("Entry time window detected! Starting stock selection...")
        
        try:
            self.stocks_list, self.stocks_dict = self.selector.run()
            # self.stocks_list = ["AAPL"]
            print(f"Stock selector returned {len(self.stocks_list)} stocks")
        except Exception as e:
            print(f"Error in stock selector: {e}")
            self.stocks_list = ["AAPL"]
            self.stocks_dict = []

        # Initialize sector_used_capital from DB (existing active positions)
        self.sector_used_capital = trades_db.get_sector_used_capital()
        # Set sector in DB for each stock from selector (stocks_dict has 'sector' per symbol)
        for rec in self.stocks_dict:
            sym = rec.get('symbol')
            sector = rec.get('sector') or 'Unknown'
            if sym:
                trades_db.update_strategy_data(sym, sector=sector)

        print("Start")
        for i, stock in enumerate(self.stocks_list):
            print(f"{i}:{stock} ")
            
        self.print_qualifying_stocks()
        
        # Start Thread 0 for hedge monitoring
        hedge_thread = threading.Thread(target=self.hedge_monitoring_loop, name="Hedge-0", daemon=False)
        hedge_thread.start()
        self.threads.append(hedge_thread)
        print("[Hedge-0] Hedge thread launched (loop will print when ready).")

        # Start drawdown monitoring thread
        drawdown_thread = threading.Thread(target=self.monitor_drawdown_loop, name="DrawdownMonitor", daemon=True)
        drawdown_thread.start()
        self.threads.append(drawdown_thread)
        print("[Drawdown] Started drawdown monitoring thread")
        
        # Start strategy threads (starting from thread 1)
        for i, stock in enumerate(self.stocks_list):
            sector = 'Unknown'
            for rec in self.stocks_dict:
                if rec.get('symbol') == stock:
                    sector = rec.get('sector') or 'Unknown'
                    break
            strategy = Strategy(self, stock, self.broker, self.config, sector=sector)
            self.strategies.append(strategy)
            
            thread_name = f"Strategy-{i+1}-{stock}"  # Thread numbers start from 1
            t = threading.Thread(target=strategy.run, args=(i+1,), name=thread_name)  # Pass i+1 as thread number
            
            t.start()
            self.threads.append(t)
        
        print(f"\nWaiting for {len(self.threads)} threads to complete (hedge + drawdown + {len(self.threads)-2} strategy threads)...")
        for i, thread in enumerate(self.threads):
            thread.join()
            if i == 0:
                print(f"Hedge Thread 0 completed")
            else:
                print(f"Strategy Thread {i} completed")
        
        
    def test_simple(self, stock_symbol="AAPL"):
        """Simple test function to run strategy for one stock"""
        print("=" * 80)
        print(f"Testing Strategy for {stock_symbol}")
        print("=" * 80)
        
        # Step 1: Initialize database for the stock
        print(f"\n[Step 1] Initializing database for {stock_symbol}...")
        trades_db.add_stocks_from_list([stock_symbol])
        print(f"✓ Database initialized for {stock_symbol}")
        
        # Step 2: Create strategy instance
        print(f"\n[Step 2] Creating Strategy instance for {stock_symbol}...")
        strategy = Strategy(self, stock_symbol, self.broker, self.config)
        print(f"✓ Strategy created for {stock_symbol}")
        
        # Step 3: Run the strategy
        print(f"\n[Step 3] Running strategy for {stock_symbol}...")
        print("=" * 80)
        
        try:
            # Run the strategy (this will loop and monitor)
            # Note: In TESTING mode, this will calculate indicators once and process
            strategy.run(i=0)
            print(f"\n✓ Strategy execution completed for {stock_symbol}")
        except KeyboardInterrupt:
            print(f"\n⚠ Strategy interrupted by user")
        except Exception as e:
            print(f"\n✗ Error running strategy: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 80)
        print(f"Test completed for {stock_symbol}")
        print("=" * 80)
    
    def simulate_stock_investment(self, amount):
        """Helper method to simulate stock investment for testing hedge logic"""
        with self.manager_lock:
            self.stock_invested_capital += amount
            self.used_capital += amount
            self.available_capital -= amount
        print(f"[SIMULATION] Simulated stock investment: ${amount:,.0f}")
        print(f"  - Stock Invested Capital: ${self.stock_invested_capital:,.0f}")
        print(f"  - Available Capital: ${self.available_capital:,.0f}")
        print(f"  - Used Capital: ${self.used_capital:,.0f}")
    
    def simulate_stock_sale(self, amount):
        """Helper method to simulate stock sale for testing hedge logic"""
        with self.manager_lock:
            # Round to 0 if in range [-2, 2] to handle floating point precision errors
            new_value = self.stock_invested_capital - amount
            if -2 <= new_value <= 2:
                self.stock_invested_capital = 0
            else:
                self.stock_invested_capital = new_value
            self.used_capital = max(0, self.used_capital - amount)
            self.available_capital += amount
        print(f"[SIMULATION] Simulated stock sale: ${amount:,.0f}")
        print(f"  - Stock Invested Capital: ${self.stock_invested_capital:,.0f}")
        print(f"  - Available Capital: ${self.available_capital:,.0f}")
        print(f"  - Used Capital: ${self.used_capital:,.0f}")
    
    def test_hedge_monitoring(self, test_stock_invested=50000):
        """Test function to test hedge monitoring loop with specified stock invested capital"""
        print("=" * 80)
        print("Testing Hedge Monitoring Loop")
        print("=" * 80)
        print(f"Total Equity: ${creds.EQUITY:,.0f}")
        print(f"Stock Invested Capital: ${test_stock_invested:,.0f}")
        print(f"Hedge Symbol: {self.hedge_symbol}")
        print("=" * 80)
        
        # Set stock invested capital for testing (keep total equity as is)
        self.stock_invested_capital = test_stock_invested
        self.used_capital = test_stock_invested
        # Adjust available capital based on stock investments
        self.available_capital = creds.EQUITY - test_stock_invested
        
        print(f"\n[Setup] Capital configured:")
        print(f"  - Total Equity: ${creds.EQUITY:,.0f}")
        print(f"  - Stock Invested Capital: ${self.stock_invested_capital:,.0f}")
        print(f"  - Available Capital: ${self.available_capital:,.0f}")
        print(f"  - Used Capital: ${self.used_capital:,.0f}")
        print(f"\n[Note] Use manager.simulate_stock_investment(amount) to simulate additional stock investments")
        print(f"       Use manager.simulate_stock_sale(amount) to simulate stock sales")
        
        # Initialize hedge symbol in database
        print(f"\n[Setup] Initializing hedge symbol in database...")
        trades_db.add_stocks_from_list([self.hedge_symbol])
        trades_db.update_strategy_data(self.hedge_symbol,
            position_active=False,
            position_shares=0,
            entry_price=0,
            current_price=0,
            unrealized_pnl=0,
            realized_pnl=0,
            used_capital=0
        )
        print(f"✓ Hedge symbol {self.hedge_symbol} initialized")
        
        print(f"\n[Step 1] Starting hedge monitoring loop...")
        print(f"  - Checking triggers every 60 seconds")
        print(f"  - Monitoring hedge position adjustments")
        print(f"  - Press Ctrl+C to stop")
        print("\n" + "=" * 80)
        
        try:
            # Call hedge monitoring loop directly (not in a thread)
            self.hedge_monitoring_loop()
        except KeyboardInterrupt:
            print("\n\n[Stopping] Interrupted by user...")
            self.stop_event.set()
            print("✓ Stop event set")
        except Exception as e:
            print(f"\n[Error] Exception in hedge monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset capital tracking
            self.stock_invested_capital = 0
            self.used_capital = 0
            self.available_capital = creds.EQUITY
            print(f"\n[Cleanup] Reset capital tracking")
            print("=" * 80)
            print("Hedge monitoring test completed")
            print("=" * 80)
    
    def test(self):
        """Test function to get VIX index data and SPY (S&P proxy) data"""
        print("=" * 80)
        print("Testing VIX and SPY Data Retrieval")
        print("=" * 80)
        
        # Get VIX index data
        print("\n[Step 1] Fetching VIX index data...")
        try:
            vix_df = self.broker.get_historical_data_index("VIX", "1 D", "15 mins")
            if vix_df is not None and not vix_df.empty:
                print(f"✓ VIX data retrieved: {len(vix_df)} bars")
                print(f"  Date range: {vix_df['date'].min()} to {vix_df['date'].max()}")
                print(f"  Latest VIX: {vix_df['close'].iloc[-1]:.2f}")
                print(f"\nVIX Data (last 5 bars):")
                print(vix_df.tail())
            else:
                print("✗ Failed to retrieve VIX data")
        except Exception as e:
            print(f"✗ Error fetching VIX data: {e}")
            import traceback
            traceback.print_exc()
        
        # Get SPY data (S&P 500 proxy)
        print("\n[Step 2] Fetching SPY data...")
        try:
            spy_df = self.broker.get_historical_data_stock("SPY", "1 D", "15 mins")
            if spy_df is not None and not spy_df.empty:
                print(f"✓ SPY data retrieved: {len(spy_df)} bars")
                print(f"  Date range: {spy_df['date'].min()} to {spy_df['date'].max()}")
                print(f"  Latest SPY: {spy_df['close'].iloc[-1]:.2f}")
                print(f"\nSPY Data (last 5 bars):")
                print(spy_df.tail())
            else:
                print("✗ Failed to retrieve SPY data")
        except Exception as e:
            print(f"✗ Error fetching SPY data: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("Test completed")
        print("=" * 80)

    def print_qualifying_stocks(self):
        """Print the current qualifying stocks with their details from stocks_dict"""
        if not self.qualifying_stocks:
            print("No qualifying stocks found")
            return

        print(f"\n=== QUALIFYING STOCKS ({len(self.qualifying_stocks)}) ===")
        for i, stock_data in enumerate(self.qualifying_stocks, 1):
            symbol = stock_data['symbol']
            
            stock_info = next((item for item in self.stocks_dict if item.get('symbol') == symbol), {})
            print(f"{i}. {symbol}")
            print(f"   Alpha Score: {stock_data.get('alpha_score', 0):.1f}")
            print(f"   Entry Price: ${stock_data.get('entry_price', 0):.2f}")
            print(f"   Stop Loss: ${stock_data.get('stop_loss', 0):.2f}")
            print(f"   Shares: {stock_data.get('shares', 0)}")
            print(f"   Limit Price: ${stock_data.get('limit_price', 0):.2f}")
            if stock_info:
                for k, v in stock_info.items():
                    if k not in stock_data and k != 'symbol':
                        print(f"   {k}: {v}")
            print()
            

if __name__ == "__main__":
    lite_restart = "--lite" in sys.argv
    
    # Set status to initializing
    update_bot_status(True, True)
    
    if not lite_restart:
        print("Caching ADV and RVOL data...")
        initialize_stock_selector()
        print("ADV and RVOL data cached")
    
    # Set status to running (initialization complete)
    update_bot_status(True, False)

    print("Checking for existing database to backup...")
    trades_db.backup_database()
    
    manager = StrategyManager()
    manager.run()
    print("STOPPING MANAGER")
    update_bot_status(False, False)