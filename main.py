from simulation import ibkr_broker
import pandas as pd
from stock_selector import StockSelector, initialize_stock_selector
import random
import threading
import time
from datetime import datetime, time as dtime
import pytz
from helpers import vwap, ema, macd, adx, atr
from helpers.fetch_marketcap_csv import fetch_marketcap_csv
from log import setup_logger
from db.trades_db import trades_db
import json
from types import SimpleNamespace
from simulation.ib_broker import IBBroker
import traceback
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

class Strategy:
    
    def __init__(self, manager, stock, broker, config):
        self.manager = manager
        self.stock = stock
        self.broker = broker
        self.config = config
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
        # self.leverage_config = creds.LEVERAGE_CONFIG
        
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
        
        # Hedge and leverage tracking
        self.hedge_active = False
        self.hedge_shares = 0
        self.hedge_symbol = self.hedge_config.hedge_symbol
        # self.current_leverage = 1.0
        # self.margin_used_leverage = 0

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
    
    def is_entry_time_window(self):
        """Check if current time is within entry time windows"""
        # Get current time in USA Eastern Time
        eastern_tz = pytz.timezone(self.trading_hours.timezone)
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        # Get entry windows from configuration
        morning_start = self.trading_hours.morning_entry_start
        morning_end = self.trading_hours.morning_entry_end
        afternoon_start = self.trading_hours.afternoon_entry_start
        afternoon_end = self.trading_hours.afternoon_entry_end
        
        # Convert time strings to datetime objects for proper comparison
        def time_str_to_datetime(time_str):
            return datetime.strptime(time_str, "%H:%M").time()
        
        current_time_obj = time_str_to_datetime(current_time_str)
        morning_start_obj = time_str_to_datetime(morning_start)
        morning_end_obj = time_str_to_datetime(morning_end)
        afternoon_start_obj = time_str_to_datetime(afternoon_start)
        afternoon_end_obj = time_str_to_datetime(afternoon_end)
        
        # Check if current time is in either window
        in_morning_window = morning_start_obj <= current_time_obj <= morning_end_obj
        in_afternoon_window = afternoon_start_obj <= current_time_obj <= afternoon_end_obj
        
        return in_morning_window or in_afternoon_window
    
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
    
    def check_hedge_triggers(self):
        """Check if hedge triggers are met and return hedge level"""
        if not self.hedge_config.enabled:
            return None, 0, 0
        
        triggers_met = 0
        trigger_details = []
        
        try:
            # Check VIX trigger
            vix_timeframe = self.hedge_config.triggers.vix_timeframe
            vix_data = self.get_historical_data_with_retry(stock="VIXY", bar_size=vix_timeframe)
            if not vix_data.empty and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                if current_vix > self.hedge_config.triggers.vix_threshold:
                    triggers_met += 1
                    trigger_details.append(f"VIX {current_vix:.1f} > {self.hedge_config.triggers.vix_threshold}")
            else:
                print(f"VIX data unavailable or empty for hedge trigger check")
            
            # Check S&P 500 drop trigger (using SPY as proxy)
            spy_data = self.get_historical_data_with_retry(stock="SPY", bar_size=15)  # 15-min data
            if not spy_data.empty and len(spy_data) >= 2 and 'close' in spy_data.columns:
                current_price = spy_data['close'].iloc[-1]
                price_15min_ago = spy_data['close'].iloc[-2]
                drop_pct = (price_15min_ago - current_price) / price_15min_ago
                
                if drop_pct > self.hedge_config.triggers.sp500_drop_threshold:
                    triggers_met += 1
                    trigger_details.append(f"S&P drop {drop_pct*100:.1f}% > {self.hedge_config.triggers.sp500_drop_threshold*100:.1f}%")
            else:
                print(f"SPY data unavailable or insufficient for hedge trigger check")
            
        except Exception as e:
            print(f"Error checking hedge triggers: {e}")
            traceback.print_exc()
            return None, 0, 0
        
        # Determine hedge level
        if triggers_met == 0:
            return None, 0, 0
        elif triggers_met == 1:
            hedge_level = 'early'
            beta = self.hedge_config.hedge_levels.early.beta
            equity_pct = self.hedge_config.hedge_levels.early.equity_pct
        elif triggers_met == 2:
            hedge_level = 'mild'
            beta = self.hedge_config.hedge_levels.mild.beta
            equity_pct = self.hedge_config.hedge_levels.mild.equity_pct
        else:
            hedge_level = 'severe' 
            beta = self.hedge_config.hedge_levels.severe.beta
            equity_pct = self.hedge_config.hedge_levels.severe.equity_pct
        
        print(f"Hedge triggers: {triggers_met} met - {', '.join(trigger_details)}")
        print(f"Hedge level: {hedge_level} (-{beta}β, {equity_pct*100:.1f}% of equity)")
        
        return hedge_level, beta, equity_pct
    
    # def calculate_sharpe_ratio(self, days=30):
    #     try:
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
    
    # def check_leverage_conditions(self):
    #     if not self.leverage_config.enabled:
    #         return 1.0
    #     
    #     conditions_met = 0
    #     total_conditions = 4  # Alpha, VIX, Sharpe, Drawdown
    #     condition_details = []
    #     
    #     try:
    #         # 1. Check Alpha ≥ 85
    #         if self.score >= 85:
    #             conditions_met += 1
    #             condition_details.append(f"Alpha {self.score} ≥ 85")
    #         else:
    #             condition_details.append(f"Alpha {self.score} < 85")
    #         
    #         # 2. Check VIX < 18 and falling
    #         vix_timeframe = self.leverage_config.conditions.vix_timeframe
    #         vix_data = self.get_historical_data_with_retry(stock="VIXY", bar_size=vix_timeframe, duration="30 D")
    #         if not vix_data.empty and 'close' in vix_data.columns:
    #             current_vix = vix_data['close'].iloc[-1]
    #             
    #             if current_vix < 18:
    #                 # Check if falling (vs 10 days ago)
    #                 if len(vix_data) >= 10:
    #                     vix_10d_ago = vix_data['close'].iloc[-10]
    #                     if current_vix < vix_10d_ago:
    #                         conditions_met += 1
    #                         condition_details.append(f"VIX {current_vix:.1f} < 18 & falling")
    #                     else:
    #                         condition_details.append(f"VIX {current_vix:.1f} < 18 but rising")
    #                 else:
    #                     condition_details.append(f"VIX history insufficient")
    #             else:
    #                 condition_details.append(f"VIX {current_vix:.1f} ≥ 18")
    #         else:
    #             condition_details.append(f"VIX data unavailable")
    #         
    #         # 3. Check Sharpe ratio > 2.5 (for this stock, last 30 days)
    #         sharpe = self.calculate_sharpe_ratio(days=30)
    #         if sharpe > 2.5:
    #             conditions_met += 1
    #             condition_details.append(f"Sharpe {sharpe:.2f} > 2.5")
    #         else:
    #             condition_details.append(f"Sharpe {sharpe:.2f} ≤ 2.5")
    #         
    #         # 4. Check 10-day drawdown < 2% (for this stock)
    #         drawdown_10d = self.calculate_10day_drawdown()
    #         if drawdown_10d < 0.02:
    #             conditions_met += 1
    #             condition_details.append(f"10d DD {drawdown_10d*100:.1f}% < 2%")
    #         else:
    #             condition_details.append(f"10d DD {drawdown_10d*100:.1f}% ≥ 2%")
    #          
    #     except Exception as e:
    #         print(f"Error checking leverage conditions: {e}")
    #         traceback.print_exc()
    #         return 1.0
    #     
    #     # Determine leverage level
    #     print(f"\n[Leverage Check] {self.stock}: {conditions_met}/{total_conditions} conditions met")
    #     for detail in condition_details:
    #         print(f"  {detail}")
    #     
    #     if conditions_met == total_conditions:  # ALL conditions met
    #         leverage = 2.0
    #         print(f"All conditions met → 2.0x leverage")
    #     elif conditions_met >= 3:  # Signals weakening
    #         leverage = 1.2
    #         print(f"Partial conditions → 1.2x leverage")
    #     else:
    #         leverage = 1.0
    #         print(f"Few conditions → 1.0x (no leverage)")
    #     
    #     return leverage
    
    def execute_hedge(self, hedge_level, beta, equity_pct):
        """Execute hedge by buying SQQQ ETF"""
        if hedge_level is None or self.hedge_active:
            return
        
        try:
            # Calculate hedge size using equity_pct
            account_equity = creds.EQUITY
            hedge_amount = account_equity * equity_pct
            
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
            
            print(f"Executing {hedge_level} hedge:")
            print(f"  - Beta offset: -{beta}β")
            print(f"  - Hedge amount: ${hedge_amount:,.0f} ({equity_pct*100:.1f}% of equity)")
            print(f"  - {self.hedge_symbol} price: ${trade[1]}")
            print(f"  - Shares to buy: {hedge_shares}")
            
            # Update hedge status and track hedge position
            self.hedge_active = True
            self.hedge_shares = hedge_shares
            self.hedge_level = hedge_level
            self.hedge_entry_price = trade[1]
            
            # Update database with hedge information for the individual stock
            trades_db.update_strategy_data(self.stock,
                hedge_active=True,
                hedge_shares=hedge_shares,
                hedge_symbol=self.hedge_symbol,
                hedge_level=hedge_level,
                hedge_beta=beta,
                hedge_entry_price=trade[1],
                hedge_entry_time=datetime.now(pytz.timezone('US/Eastern'))
            )
            
            # Update the centralized hedge symbol position and manager capital
            # Read current centralized hedge data under manager lock, compute new values and adjust manager capital, then release before DB update
            with self.manager.manager_lock:
                central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
                current_position_shares = central_data.get('position_shares', 0) or 0
                current_used_capital = central_data.get('used_capital', 0) or 0
                current_cumulative_shares = central_data.get('shares', 0) or 0
                new_used_capital = current_used_capital + (trade[1] * hedge_shares)
                new_total_shares = current_position_shares + hedge_shares
                new_avg_entry_price = (new_used_capital / new_total_shares) if new_total_shares > 0 else trade[1]
                # Manager capital: spend cash and increase used capital by cost
                self.manager.available_capital -= (trade[1] * hedge_shares)
                self.manager.used_capital += (trade[1] * hedge_shares)
                trades_db.update_strategy_data(self.hedge_symbol,
                    position_active=True,
                    position_shares=new_total_shares,
                    shares=current_cumulative_shares + hedge_shares,
                    used_capital=new_used_capital,
                    entry_price=new_avg_entry_price,
                    current_price=trade[1]
                )
                     
            print(f"Hedge executed: Buy {hedge_shares} shares of {self.hedge_symbol}")
            print(f"Hedge position tracked in database for {self.stock}")
                
        except Exception as e:
            print(f"Error executing hedge: {e}")
            traceback.print_exc()
    
    def check_hedge_exit_conditions(self):
        """Check if hedge exit conditions are met for scaling down"""
        if not self.hedge_active:
            return None
        
        recovery_signals = 0
        signal_details = []
        
        try:
            # Check VIX < 20 and falling (configurable timeframe slope negative)
            vix_timeframe = self.hedge_config.exit_conditions.vix_timeframe
            vix_data = self.get_historical_data_with_retry(stock="VIXY", bar_size=vix_timeframe)
            if not vix_data.empty and len(vix_data) >= 4 and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                vix_slope_period = self.hedge_config.exit_conditions.vix_slope_period
                vix_exit_threshold = self.hedge_config.exit_conditions.vix_exit_threshold
                vix_10min_ago = vix_data['close'].iloc[-4]  # ~10 minutes ago (configurable bars)
                
                if current_vix < vix_exit_threshold and current_vix < vix_10min_ago:
                    recovery_signals += 1
                    signal_details.append(f"VIX {current_vix:.1f} < {vix_exit_threshold} and falling")
            
            # Check S&P 500 up > +0.6% in 15 min after hedge added
            spy_data = self.get_historical_data_with_retry(stock="SPY", bar_size=15)
            if not spy_data.empty and len(spy_data) >= 2 and 'close' in spy_data.columns:
                current_price = spy_data['close'].iloc[-1]
                price_15min_ago = spy_data['close'].iloc[-2]
                gain_pct = (current_price - price_15min_ago) / price_15min_ago
                sp500_recovery_threshold = self.hedge_config.exit_conditions.sp500_recovery_threshold
                
                if gain_pct > sp500_recovery_threshold:
                    recovery_signals += 1
                    signal_details.append(f"S&P up {gain_pct*100:.1f}% > {sp500_recovery_threshold*100:.1f}%")
            
            # Check Nasdaq (SQQQ) trades above 5-min VWAP for 2+ consecutive bars
            sqqq_data = self.get_historical_data_with_retry(stock="QQQ", bar_size=5)
            if not sqqq_data.empty and len(sqqq_data) >= 2 and 'close' in sqqq_data.columns:
                # Calculate 5-min VWAP
                sqqq_vwap = vwap.calc_vwap(sqqq_data)
                sqqq_consecutive_bars = self.hedge_config.exit_conditions.sqqq_vwap_consecutive_bars
                if len(sqqq_vwap) >= sqqq_consecutive_bars:
                    current_sqqq = sqqq_data['close'].iloc[-1]
                    sqqq_vwap_current = sqqq_vwap.iloc[-1]
                    sqqq_vwap_prev = sqqq_vwap.iloc[-2]
                    
                    # Check if SQQQ is above VWAP for 2+ consecutive bars
                    if current_sqqq > sqqq_vwap_current and sqqq_data['close'].iloc[-2] > sqqq_vwap_prev:
                        recovery_signals += 1
                        signal_details.append(f"QQQ above 5-min VWAP for {sqqq_consecutive_bars}+ bars")
            
        except Exception as e:
            print(f"Error checking hedge exit conditions: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        print(f"Hedge exit signals: {recovery_signals} met - {', '.join(signal_details)}")
        return recovery_signals
    
    def scale_down_hedge(self):
        """Scale down hedge based on current level and recovery signals"""
        if not self.hedge_active:
            return
        
        recovery_signals = self.check_hedge_exit_conditions()
        if recovery_signals is None or recovery_signals == 0:
            return 
        
        try:
            # Determine current hedge level from database or config
            current_hedge_level = getattr(self, 'hedge_level', 'severe')
            
            # Scale down logic
            if current_hedge_level == 'severe':
                # Severe (-0.3β) → cut to Mild (-0.15β)
                new_level = 'mild'
                new_beta = self.hedge_config.hedge_levels.mild.beta
                new_equity_pct = self.hedge_config.hedge_levels.mild.equity_pct
                print(f"Scaling hedge: Severe → Mild (-{new_beta}β, {new_equity_pct*100:.1f}% of equity)")
                
            elif current_hedge_level == 'mild':
                # Mild (-0.15β) → cut to Early (-0.1β)
                new_level = 'early'
                new_beta = self.hedge_config.hedge_levels.early.beta
                new_equity_pct = self.hedge_config.hedge_levels.early.equity_pct
                print(f"Scaling hedge: Mild → Early (-{new_beta}β, {new_equity_pct*100:.1f}% of equity)")
                
            elif current_hedge_level == 'early':
                # Early (-0.1β) → exit fully
                print(f"Scaling hedge: Early → Exit fully")
                self.close_hedge()
                return
            
            else:
                print(f"Unknown hedge level: {current_hedge_level}")
                return
            
            # Calculate new hedge size using equity_pct
            account_equity = creds.EQUITY
            new_hedge_amount = account_equity * new_equity_pct
            
            # Get current hedge price
            hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if hedge_price is None:
                print(f"Unable to get {self.hedge_symbol} price for hedge scaling")
                return
            
            # Calculate shares to adjust
            new_hedge_shares = int(new_hedge_amount / hedge_price)
            shares_to_close = self.hedge_shares - new_hedge_shares
            
            if shares_to_close > 0:
                print(f"Closing {shares_to_close} shares of hedge (reducing from {self.hedge_shares} to {new_hedge_shares})")
                
                # Close partial hedge position
                trade = self.broker.place_order(symbol=self.hedge_symbol, qty=shares_to_close, order_type="MARKET", price=hedge_price, side="SELL")
                if trade is None:
                    print(f"Unable to place hedge scaling order for {self.hedge_symbol}")
                    return
                
                # Update hedge status
                self.hedge_shares = new_hedge_shares
                self.hedge_level = new_level
                
                # Update database with hedge scaling
                trades_db.update_strategy_data(self.stock,
                    hedge_shares=self.hedge_shares,
                    hedge_level=new_level,
                    hedge_beta=new_beta
                )
                
                # Update the centralized hedge symbol position
                # Only decrease position_shares (current holding), keep shares (cumulative total) unchanged
                # Read under manager lock and compute; release before DB update
                with self.manager.manager_lock:
                    central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
                    current_position_shares = central_data.get('position_shares', 0) or 0
                    current_used_capital = central_data.get('used_capital', 0) or 0
                    current_realized_pnl = central_data.get('realized_pnl', 0) or 0
                    # PnL for the shares being closed from this thread
                    hedge_entry_price = getattr(self, 'hedge_entry_price', hedge_price)
                    partial_hedge_pnl = (trade[1] - hedge_entry_price) * shares_to_close
                    # Reduce used capital by cost basis of these shares
                    new_used_capital = max(0, current_used_capital - (hedge_entry_price * shares_to_close))
                    remaining_shares = max(0, current_position_shares - shares_to_close)
                    new_avg_entry_price = (new_used_capital / remaining_shares) if remaining_shares > 0 else 0
                    # Manager capital: receive cash from sale, release used capital at cost
                    self.manager.available_capital += (trade[1] * shares_to_close)
                    self.manager.used_capital -= (hedge_entry_price * shares_to_close)
                trades_db.update_strategy_data(self.hedge_symbol,
                    position_shares=remaining_shares,
                    used_capital=new_used_capital,
                    entry_price=new_avg_entry_price,
                    realized_pnl=current_realized_pnl + partial_hedge_pnl,
                    current_price=trade[1]
                    # shares remains unchanged (cumulative total)
                )
                
                print(f"Hedge scaled down successfully: {current_hedge_level} → {new_level}")
                print(f"Remaining hedge: {self.hedge_shares} shares ({new_equity_pct*100:.1f}% of equity)")
            
        except Exception as e:
            print(f"Error scaling down hedge: {e}")
            import traceback
            traceback.print_exc()
    
    def close_hedge(self):
        """Close hedge position by selling hedge shares"""
        if not self.hedge_active or self.hedge_shares == 0:
            return
        
        try:
            print(f"[HEDGE] Initiating close for {self.hedge_symbol}: selling {self.hedge_shares} shares")
            
            # Get current hedge symbol price for P&L calculation
            current_hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if current_hedge_price is None:
                print(f"Unable to get current {self.hedge_symbol} price for P&L calculation")
                current_hedge_price = 0
            
            # Calculate hedge P&L (profit from long position)
            hedge_pnl = 0
            if hasattr(self, 'hedge_entry_price') and self.hedge_entry_price:
                hedge_pnl = (current_hedge_price - self.hedge_entry_price) * self.hedge_shares
            
            print(f"[HEDGE] Placing SELL order: symbol={self.hedge_symbol}, qty={self.hedge_shares}, mkt_price={current_hedge_price}")
            trade = self.broker.place_order(symbol=self.hedge_symbol, qty=self.hedge_shares, order_type="MARKET", price=current_hedge_price, side="SELL")
            if trade is None:
                print(f"Unable to place hedge order for {self.hedge_symbol}")
                return
            else:
                print(f"[HEDGE] SELL filled: price={trade[1]} (order_id={trade[0] if len(trade) > 0 else 'N/A'})")

            # Store hedge_shares before resetting
            shares_to_sell = self.hedge_shares
            
            # Update hedge status
            self.hedge_active = False
            self.hedge_shares = 0
            print(f"[HEDGE] Local state updated: hedge_active={self.hedge_active}, hedge_shares={self.hedge_shares}")
            
            # Update database with hedge closure for individual stock
            trades_db.update_strategy_data(self.stock,
                hedge_active=False,
                hedge_shares=0,
                hedge_exit_price=current_hedge_price,
                hedge_exit_time=datetime.now(pytz.timezone('US/Eastern')),
                hedge_pnl=hedge_pnl
            )
            
            # Update the centralized hedge symbol position
            # Only decrease position_shares (current holding), keep shares (cumulative total) unchanged
            # Read under manager lock and compute; release before DB update
            with self.manager.manager_lock:
                central_data = trades_db.get_latest_strategy_data(self.hedge_symbol) or {}
                current_position_shares = central_data.get('position_shares', 0) or 0
                current_realized_pnl = central_data.get('realized_pnl', 0) or 0
                current_used_capital = central_data.get('used_capital', 0) or 0
                print(f"[HEDGE][DB SNAPSHOT] position_shares={current_position_shares}, used_capital={current_used_capital}, realized_pnl={current_realized_pnl}")
                remaining_hedge_shares = max(0, current_position_shares - shares_to_sell)
                hedge_entry_price = getattr(self, 'hedge_entry_price', current_hedge_price)
                new_used_capital = max(0, current_used_capital - (hedge_entry_price * shares_to_sell))
                pnl_increment = hedge_pnl
                print(f"[HEDGE][CALC] shares_to_sell={shares_to_sell}, hedge_entry_price={hedge_entry_price}, fill_price={trade[1]}")
                print(f"[HEDGE][CALC] remaining_shares={remaining_hedge_shares}, new_used_capital={new_used_capital}, pnl_increment={pnl_increment}")
                # Manager capital adjustments
                self.manager.available_capital += (trade[1] * shares_to_sell)
                self.manager.used_capital -= (hedge_entry_price * shares_to_sell)
                print(f"[HEDGE][MANAGER] available_capital={self.manager.available_capital:.2f}, used_capital={self.manager.used_capital:.2f}")

                if remaining_hedge_shares > 0:
                    # Partial closure - update position_shares only (shares remains cumulative total)
                    print(f"[HEDGE][DB UPDATE PARTIAL] symbol={self.hedge_symbol}, position_shares={remaining_hedge_shares}, used_capital={new_used_capital}, entry_price={(new_used_capital / remaining_hedge_shares) if remaining_hedge_shares > 0 else 0}, realized_pnl={current_realized_pnl + hedge_pnl}, current_price={current_hedge_price}")
                    trades_db.update_strategy_data(self.hedge_symbol,
                        position_shares=remaining_hedge_shares,
                        used_capital=new_used_capital,
                        entry_price=(new_used_capital / remaining_hedge_shares) if remaining_hedge_shares > 0 else 0,
                        realized_pnl=current_realized_pnl + hedge_pnl,
                        current_price=current_hedge_price
                    )
                else:
                    # Full closure - close the position (shares still remains as cumulative total)
                    print(f"[HEDGE][DB UPDATE FULL] symbol={self.hedge_symbol}, position_shares=0, used_capital=0, entry_price=0, realized_pnl={current_realized_pnl + hedge_pnl}, current_price={current_hedge_price}, close_time=now(Eastern)")
                    trades_db.update_strategy_data(self.hedge_symbol,
                        position_active=False,
                        position_shares=0,
                        used_capital=0,
                        realized_pnl=current_realized_pnl + hedge_pnl,
                        unrealized_pnl=0,
                        current_price=current_hedge_price,
                        close_time=datetime.now(pytz.timezone('US/Eastern'))
                    )
            
            print(f"[HEDGE] Close complete. Hedge P&L: ${hedge_pnl:.2f}")
            print(f"[HEDGE] Centralized position for {self.hedge_symbol} updated.")
                
        except Exception as e:
            print(f"Error closing hedge: {e}")
            traceback.print_exc()
    
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

            if self.hedge_active:
                hedge_exit_time = self.trading_hours.hedge_force_exit_time
                print(f"{hedge_exit_time} Weak Exit: Closing hedge positions")
                self.close_hedge()
        else:
            weak_exit_time = self.trading_hours.weak_exit_time
            print(f"{weak_exit_time}: Position not in weak range ({current_gain_pct*100:.1f}% gain) - keeping position")

    def safety_exit_all(self):
        # Get exit time from configuration
        safety_exit_time = self.trading_hours.safety_exit_time
        hedge_exit_time = self.trading_hours.hedge_force_exit_time
        
        print(f"Safety exit all positions at {safety_exit_time}")
        if self.position_active:
            print(f"{safety_exit_time} Safety Exit: Closing all positions")
            self.close_position('safety_exit', self.position_shares)
        
        if self.hedge_active:
            print(f"{hedge_exit_time} Safety Exit: Closing hedge positions")
            self.close_hedge()
    
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
                            # Add jitter: random value between 0.5x and 1.5x of exponential delay
                            jitter_factor = random.uniform(0.5, 1.5)
                            retry_delay = exponential_delay * jitter_factor
                            # Cap maximum delay at 60 seconds
                            retry_delay = min(retry_delay, 60)
                            
                            print(f"Failed to fetch {tf_name} data (attempt {attempt + 1}/{max_retries}) - retrying in {retry_delay:.2f}s...")
                            time.sleep(retry_delay)
                        else:
                            print(f"Failed to fetch {tf_name} data after {max_retries} attempts - data is None or empty")
                            self.data[tf_name] = pd.DataFrame()
                            break
                    else:
                        # Success - data was fetched, break out of retry loop
                        break
    
    def calculate_indicators_by_timeframe(self):
        """Calculate indicators for each timeframe separately"""
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
        print(f"Score +{self.alpha_score_config.news.weight} (News check - placeholder)")
        
        # Market Calm analysis (15%)
        if self._check_market_calm_conditions():
            self.score += self.alpha_score_config.market_calm.weight
            print(f"Score +{self.alpha_score_config.market_calm.weight} (Market Calm conditions met)")
        
        print(f"\nFinal Alpha Score: {self.score}")
        
        # Update database with alpha score
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
            price_vwap_ok = bool(self.data[vwap_tf]['close'].iloc[-1] > self.indicators[vwap_tf]['vwap'].iloc[-1])
            
            # EMA1 > EMA2 (5-min EMA > 20-min EMA)
            ema_cross_ok = bool(self.indicators[ema1_tf]['ema1'].iloc[-1] > self.indicators[ema2_tf]['ema2'].iloc[-1])
            
            return price_vwap_ok and ema_cross_ok
        except Exception as e:
            print(f"Error checking trend conditions: {e}")
            return False
    
    def _check_momentum_conditions(self):
        """Check momentum conditions"""
        # Get configured timeframe for MACD
        macd_tf = self.tf.get('macd')
        
        if not macd_tf or macd_tf not in self.indicators:
            print(f"Missing MACD timeframe or indicators: {macd_tf}")
            return False
        
        # Check if MACD indicator exists and is not empty
        if 'macd' not in self.indicators[macd_tf]:
            print(f"Missing MACD indicator key for {macd_tf}")
            return False
        
        if self.indicators[macd_tf]['macd'].empty:
            print(f"Empty MACD indicator data for {macd_tf}")
            return False
        
        try:
            return bool(self.indicators[macd_tf]['macd'].iloc[-1] > 0)
        except Exception as e:
            print(f"Error accessing MACD indicator: {e}")
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
            recent_volume = self.data[volume_avg_tf]['volume'].iloc[-1]
            avg_volume = self.indicators[volume_avg_tf]['volume_avg'].iloc[-1]
            multiplier = self.alpha_score_config.volume_volatility.conditions.volume_spike.multiplier
            volume_ok = bool(recent_volume > multiplier * avg_volume)
            
            # ADX threshold check
            adx_threshold = self.alpha_score_config.volume_volatility.conditions.adx_threshold.threshold
            adx_ok = bool(self.indicators[adx_tf]['adx'].iloc[-1] > adx_threshold)
            
            return volume_ok and adx_ok
        except Exception as e:
            print(f"Error checking volume/volatility conditions: {e}")
            return False
    
    def _check_market_calm_conditions(self):
        """Check market calm conditions (VIX)"""
        try:
            vix_timeframe = self.alpha_score_config.market_calm.conditions.vix_threshold.timeframe
            vix_df = self.get_historical_data_with_retry(stock="VIXY", bar_size=vix_timeframe)
            if vix_df.empty:
                return False
            
            vix_close = vix_df['close']
            vix_threshold = self.alpha_score_config.market_calm.conditions.vix_threshold.threshold
            
            # VIX < threshold and dropping
            vix_low = bool(vix_close.iloc[-1] < vix_threshold)
            vix_dropping = len(vix_close) >= 4 and bool(vix_close.iloc[-1] < vix_close.iloc[-4])
            
            return vix_low and vix_dropping
        except:
            return False
    
    def perform_additional_checks(self):
        """Perform additional checks after Alpha Score calculation"""
        if self.score < creds.RISK_CONFIG.alpha_score_threshold:
            self.additional_checks_passed = False
            print(f"\nAdditional Checks Passed: {self.additional_checks_passed} (Alpha Score too low)")
        else:
            # Get configured timeframes
            volume_avg_tf = self.tf.get('volume_avg')
            
            # Initialize variables
            volume_check = False
            vwap_slope_check = False
            market_conditions_check = False
            
            if not volume_avg_tf or volume_avg_tf not in self.data or volume_avg_tf not in self.indicators:
                self.additional_checks_passed = False
                print(f"\nAdditional Checks Passed: {self.additional_checks_passed} (Missing data/indicators for {volume_avg_tf})")
            elif 'volume_avg' not in self.indicators[volume_avg_tf] or self.indicators[volume_avg_tf]['volume_avg'].empty:
                self.additional_checks_passed = False
                print(f"\nAdditional Checks Passed: {self.additional_checks_passed} (Missing or empty volume_avg indicator for {volume_avg_tf})")
            elif 'volume' not in self.data[volume_avg_tf].columns or self.data[volume_avg_tf].empty:
                self.additional_checks_passed = False
                print(f"\nAdditional Checks Passed: {self.additional_checks_passed} (Missing or empty volume data for {volume_avg_tf})")
            else:
                try:
                    # Check +2x volume
                    recent_volume = self.data[volume_avg_tf]['volume'].iloc[-1]
                    avg_volume = self.indicators[volume_avg_tf]['volume_avg'].iloc[-1]
                    volume_multiplier = self.additional_checks_config.volume_multiplier
                    
                    volume_check = bool(recent_volume > volume_multiplier * avg_volume)
                    print(f"{'Passed' if volume_check else 'Failed'} +{volume_multiplier}x volume check {'passed' if volume_check else 'failed'}")
                except Exception as e:
                    print(f"Error checking volume: {e}")
                    volume_check = False
                
                # Check VWAP slope
                vwap_slope_check = self._check_vwap_slope()
                
                # Alpha Score based TRIN/TICK check
                # If Alpha Score >= bypass_threshold → enter without TRIN/TICK check
                # If base_threshold <= Alpha Score < bypass_threshold → require both TRIN <= threshold and TICK MA >= threshold
                bypass_alpha = self.additional_checks_config.trin_tick_bypass_alpha
                if self.score >= bypass_alpha:
                    market_conditions_check = True
                    print(f"Alpha Score >= {bypass_alpha} ({self.score}) - Bypassing TRIN/TICK check")
                else:
                    market_conditions_check = self._check_trin_tick_conditions()
                    print(f"Alpha Score < {bypass_alpha} ({self.score}) - TRIN/TICK check {'passed' if market_conditions_check else 'failed'}")
                
                self.additional_checks_passed = bool(volume_check and vwap_slope_check and market_conditions_check)
                
                print(f"\nAdditional Checks Passed: {self.additional_checks_passed}")
        
        # Always update database with additional checks status (regardless of pass/fail)
        trades_db.update_strategy_data(self.stock, additional_checks_passed=self.additional_checks_passed)
    
    def _check_vwap_slope(self):
        """Check VWAP slope condition"""
        # Get configured timeframe for VWAP
        vwap_tf = self.tf.get('vwap')
        
        if not vwap_tf or vwap_tf not in self.indicators:
            print(f"VWAP slope check failed: missing timeframe or indicators for {vwap_tf}")
            return False
        
        # Check if VWAP indicator exists and is not empty
        if 'vwap' not in self.indicators[vwap_tf]:
            print(f"VWAP slope check failed: missing VWAP indicator key for {vwap_tf}")
            return False
        
        vwap_series = self.indicators[vwap_tf]['vwap']
        if vwap_series.empty or len(vwap_series) < 2:
            print("VWAP slope check failed: insufficient data")
            return False
        
        try:
            current_vwap = vwap_series.iloc[-1]
            vwap_period_ago = vwap_series.iloc[-2]
            vwap_slope = (current_vwap - vwap_period_ago) / self.additional_checks_config.vwap_slope_period
            
            threshold = self.additional_checks_config.vwap_slope_threshold
            slope_ok = bool(vwap_slope > threshold)
            
            print(f"{'Passed' if slope_ok else 'Failed'} VWAP slope check {'passed' if slope_ok else 'failed'}: {vwap_slope:.3f} {'>' if slope_ok else '<='} {threshold}")
            
            return slope_ok
        except Exception as e:
            print(f"Error checking VWAP slope: {e}")
            return False
    
    def _check_trin_tick_conditions(self):
        """Check TRIN and TICK market breadth conditions"""
        # If TRIN/TICK check is disabled in config, bypass the check
        if not self.additional_checks_config.trin_tick_check_enabled:
            print("TRIN/TICK check disabled in configuration - bypassing")
            return True
        
        try:
            # Get configurable thresholds
            trin_threshold = self.additional_checks_config.trin_threshold
            tick_ma_window = self.additional_checks_config.tick_ma_window
            tick_threshold = self.additional_checks_config.tick_threshold
            
            # Check TRIN (NYSE TRIN index) - should be <= threshold
            # TRIN > 1.0 indicates bearish sentiment, TRIN < 1.0 indicates bullish
            trin_data = self.broker.get_historical_data_index(symbol="TRIN-NYSE", bar_size="1 min", duration="1 D")
            trin_check = False
            
            if trin_data is not None and not trin_data.empty and 'close' in trin_data.columns:
                current_trin = trin_data['close'].iloc[-1]
                trin_check = bool(current_trin <= trin_threshold)
                print(f"TRIN-NYSE check: {current_trin:.2f} {'<=' if trin_check else '>'} {trin_threshold} - {'Passed' if trin_check else 'Failed'}")
            else:
                print("TRIN-NYSE data unavailable - treating as failed")
                return False
            
            # Check TICK (NYSE TICK) - MA should be >= threshold
            # TICK shows net upticks minus downticks, positive = more buying pressure
            tick_data = self.broker.get_historical_data_index(symbol="TICK-NYSE", bar_size="1 min", duration="1 D")
            tick_check = False
            
            if tick_data is not None and not tick_data.empty and 'close' in tick_data.columns:
                # Calculate moving average of TICK with configurable window
                tick_ma = tick_data['close'].rolling(window=tick_ma_window).mean()
                current_tick_ma = tick_ma.iloc[-1]
                tick_check = bool(current_tick_ma >= tick_threshold)
                print(f"TICK-NYSE {tick_ma_window}-min MA check: {current_tick_ma:.0f} {'>=' if tick_check else '<'} {tick_threshold} - {'Passed' if tick_check else 'Failed'}")
            else:
                print("TICK-NYSE data unavailable - treating as failed")
                return False
            
            # Both conditions must pass
            return bool(trin_check and tick_check)
            
        except Exception as e:
            print(f"Error checking TRIN/TICK conditions: {e}")
            traceback.print_exc()
            return False 
        
    def calculate_position_size(self):
        # leverage_multiplier = self.check_leverage_conditions()
        # self.current_leverage = leverage_multiplier
        
        account_equity = creds.EQUITY
        current_price = self.get_current_price_with_retry(self.stock)
        
        # Base risk per trade
        base_risk_per_trade = creds.RISK_CONFIG.risk_per_trade
        
        # Apply leverage to risk_per_trade (NOT to shares)
        # risk_per_trade = base_risk_per_trade * leverage_multiplier
        risk_per_trade = base_risk_per_trade
        
        print(f"\n[Position Sizing] {self.stock}")
        print(f"  - Base risk: {base_risk_per_trade*100:.2f}%")
        # print(f"  - Leverage: {leverage_multiplier:.1f}x")
        # print(f"  - Adjusted risk: {risk_per_trade*100:.2f}%")
        print(f"  - Adjusted risk: {base_risk_per_trade*100:.2f}%")
        
        stop_loss_pct = self.calculate_stop_loss(current_price)
        stop_loss_price = current_price * (1 - stop_loss_pct)
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0:
            return 0, 0, 0, 0
        
        # Calculate position size based on leveraged risk
        capital_for_trade = (account_equity * risk_per_trade) / stop_loss_pct
        
        # Cap at max position equity percentage
        if capital_for_trade > (account_equity * creds.RISK_CONFIG.max_position_equity_pct):
            capital_for_trade = account_equity * creds.RISK_CONFIG.max_position_equity_pct
            print(f"  - Position capped at {creds.RISK_CONFIG.max_position_equity_pct*100:.0f}% max equity")
        
        shares = int(capital_for_trade / current_price)
        
        # # Track margin usage if leverage applied
        # if leverage_multiplier > 1.0:
        #     base_capital = (account_equity * base_risk_per_trade) / stop_loss_pct
        #     self.margin_used_leverage = capital_for_trade - base_capital
        #     print(f"  - Base capital: ${base_capital:,.0f}")
        #     print(f"  - Leveraged capital: ${capital_for_trade:,.0f}")
        #     print(f"  - Margin used: ${self.margin_used_leverage:,.0f}")
        # else:
        #     self.margin_used_leverage = 0
        #     print(f"  - No leverage applied")
        
        data_3min = self.get_historical_data_with_retry(stock=self.stock, bar_size=3)
        vwap_value = vwap.calc_vwap(data_3min).iloc[-1]
        
        if vwap_value < current_price:
            print(f"VWAP (${vwap_value:.2f}) is below current price (${current_price:.2f})")
        else:
            print(f"WARNING: VWAP (${vwap_value:.2f}) is NOT below current price (${current_price:.2f}) - skipping order")
            return 0, 0, 0, 0
        
        offset_pct = random.uniform(creds.ORDER_CONFIG.limit_offset_min, creds.ORDER_CONFIG.limit_offset_max)
        limit_price = current_price * (1 + offset_pct)
        
        print(f"Position Size: {shares} shares")
        print(f"Entry Price: ${current_price:.2f}")
        print(f"Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%)")
        print(f"VWAP: ${vwap_value:.2f}")
        print(f"Limit Order Price: ${limit_price:.2f} (VWAP + {offset_pct*100:.3f}%)")
        
        # Update database with current price and position calculations
        trades_db.update_strategy_data(self.stock, 
            current_price=current_price
        )
        
        return shares, current_price, stop_loss_price, limit_price
    
    def calculate_stop_loss(self, current_price):
        """Calculate stop loss percentage based on volatility"""
        data_3min = self.get_historical_data_with_retry(stock=self.stock, bar_size=3)
        atr14 = atr.calc_atr(data_3min, creds.STOP_LOSS_CONFIG.atr_period)
        atr14 = atr14.iloc[-1]

        base_stop = creds.STOP_LOSS_CONFIG.default_stop_loss
        
        atr_stop = (atr14 * creds.STOP_LOSS_CONFIG.atr_multiplier) / current_price
        print(f"ATR Stop: {atr_stop:.3f}")
        
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
                print(f"  - Limit Price: ${limit_price:.2f}")
                print(f"  - Stop Loss: ${stop_loss:.2f}")
                print(f"  - Exit: Market-On-Close at 4:00 PM ET")
                
                with self.manager.manager_lock:
                    if self.manager.available_capital < shares * entry_price:
                        print(f"Not enough available capital to place order for {self.stock}")
                        return
                    else:
                        print(f"Enough available capital to place order for {self.stock}")

                trade = self.broker.place_order(symbol=self.stock, qty=shares, order_type="LIMIT", price=round(limit_price, 2), side="BUY")
                if int(trade[1]) == -1:
                    print(f"Unable to place order for {self.stock}")
                    return
                else:
                    if trade[2] != shares:
                        print(f"Order partially filled for {self.stock}: {trade}")
                    print(f"Checking hedge triggers...")
                    hedge_level, hedge_beta, hedge_equity_pct = self.check_hedge_triggers()
                    if hedge_level:
                        print(f"Executing hedge...")
                        self.execute_hedge(hedge_level, hedge_beta, hedge_equity_pct)
                    shares = trade[2]
                    print(f"Order placed for {self.stock}: {trade}")
                    with self.manager.manager_lock:
                        self.used_capital += shares * trade[1]
                        self.manager.available_capital -= shares * trade[1]
                        self.manager.used_capital += shares * trade[1]
                        print(f"Used Capital: ${self.used_capital:.2f}")
                        print(f"Available Capital: ${self.manager.available_capital:.2f}")

                    self.entry_price = trade[1]
                    self.stop_loss_price = stop_loss
                    self.take_profit_price = self.entry_price * (1 + creds.PROFIT_CONFIG.profit_booking_levels[0]['gain'])  # Use 1% from profit booking levels
                    self.position_shares = shares
                    self.position_active = True
                    self.profit_booking_levels_remaining = list(creds.PROFIT_CONFIG.profit_booking_levels)
                    self.trailing_stop_levels_remaining = list(creds.STOP_LOSS_CONFIG.trailing_stop_levels)
                    print(f"[{self.stock}] Profit booking and trailing stop levels reset for new position")

                    eastern_tz = pytz.timezone('US/Eastern')
                    self.entry_time = datetime.now(eastern_tz)
                    self.current_price = trade[1]
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
                        used_capital=self.used_capital
                    )
        else:
            if self.score < creds.RISK_CONFIG.alpha_score_threshold:
                print(f"Alpha Score too low: {self.score} < {creds.RISK_CONFIG.alpha_score_threshold}")
            if not bool(self.additional_checks_passed):
                print("Additional checks failed")
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
                break

            elif current_time >= safety_exit_time:
                self.safety_exit_all()
                break
                        
            # 4:00 PM - Market-on-close for any remaining positions
            elif current_time >= market_close_time:
                self.market_on_close_exit()
                break  # End trading for the day

            if not TESTING:            
                # Check if we're in entry time windows
                if self.is_entry_time_window():
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
                self.manager.unrealized_pnl -= self.unrealized_pnl
            
            # Calculate current gain/loss percentage
            current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
            
            if self.current_price <= self.stop_loss_price:
                print(f"STOP LOSS TRIGGERED: {self.stock} - Current: ${self.current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
                self.close_position('stop_loss')
                return
            
            self.check_take_profit()
            
            if self.hedge_active:
                self.scale_down_hedge()
                # Update hedge position unrealized PnL
                self.update_hedge_position_unrealized_pnl()

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

    def update_hedge_position_unrealized_pnl(self):
        """Update unrealized PnL for the centralized hedge position"""
        try:
            if not self.hedge_active or self.hedge_shares == 0:
                return
            
            # Get current hedge price
            current_hedge_price = self.get_current_price_with_retry(self.hedge_symbol)
            if current_hedge_price is None:
                return
            
            # Get current hedge position data and update
            hedge_position_shares = trades_db.get_position_shares(self.hedge_symbol)
            hedge_entry_price = trades_db.get_entry_price(self.hedge_symbol)
            
            if hedge_position_shares > 0 and hedge_entry_price > 0:
                # Calculate unrealized PnL for the hedge position
                hedge_unrealized_pnl = (current_hedge_price - hedge_entry_price) * hedge_position_shares
                
                # Update the hedge position with current price and unrealized PnL
                trades_db.update_strategy_data(self.hedge_symbol,
                    current_price=current_hedge_price,
                    unrealized_pnl=hedge_unrealized_pnl
                )
                
        except Exception as e:
            print(f"Error updating hedge position unrealized PnL: {e}")

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
                print(f"Starting trailing exit monitoring at {gain_threshold*100:.1f}% gain")
                print(f"Monitoring for {drop_threshold*100:.1f}% price drop for {monitor_period} minutes, checking every second")
                
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
                        
                        # Calculate current price drop from peak
                        price_drop = (self.trailing_exit_start_price - self.current_price) / self.trailing_exit_start_price
                        
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
            self.used_capital -= original_cost
            print(f"Available Capital after closing position: ${self.manager.available_capital:.2f}")
            print(f"Used Capital after closing position: ${self.used_capital:.2f}")
            print(f"Used Total Capital after closing position: ${self.manager.used_capital:.2f}")

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

                # If a hedge is active for this position, close it as well
                if self.hedge_active:
                    print(f"Base position closed for {self.stock}; closing active hedge ({self.hedge_symbol})")
                    self.close_hedge()
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

    def is_connected(self):
        """Check if connected to IBKR"""
        return self.ib_broker.connected

    def place_market_order(self, symbol, quantity, action="BUY"):
        """Place market order and wait for fill"""
        # Use StrategyBroker's counter to avoid conflicts
        with self.counter_lock:
            order_id = self.request_id_counter + 2000  # Offset for orders
            self.request_id_counter += 1
        return self.ib_broker.place_market_order_with_id(symbol, quantity, action, order_id)

    def place_limit_order(self, symbol, quantity, limit_price, action="BUY"):
        """Place limit order and wait for fill"""
        # Use StrategyBroker's counter to avoid conflicts
        with self.counter_lock:
            order_id = self.request_id_counter + 2000  # Offset for orders
            self.request_id_counter += 1
        return self.ib_broker.place_limit_order_with_id(symbol, quantity, limit_price, action, order_id)

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
        self.used_capital = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.stocks_list, self.stocks_dict = [], []
        self.stop_event = threading.Event()
        
        # Semaphore to limit concurrent data requests across all strategy threads
        # This prevents thundering herd problem when multiple threads request data simultaneously
        # Limit to 5 concurrent requests to avoid overwhelming the broker API
        self.data_request_semaphore = threading.Semaphore(5)
        hedge_symbol = self.config.HEDGE_CONFIG.hedge_symbol
        print(f"Adding hedge symbol {hedge_symbol} to database for all threads")
        
        # Verify database is properly initialized before adding data
        if not trades_db.verify_database():
            print("Database verification failed, reinitializing...")
            trades_db.init_database()
            if not trades_db.verify_database():
                raise Exception("Failed to initialize database properly")
        
        trades_db.add_stocks_from_list([hedge_symbol])
        
        # Initialize centralized hedge symbol with an entry_time stamped at manager startup (US/Eastern)
        eastern_tz = pytz.timezone('US/Eastern')
        trades_db.update_strategy_data(hedge_symbol,
            position_active=False,
            position_shares=0,
            entry_price=0,
            stop_loss_price=0,
            take_profit_price=0,
            entry_time=datetime.now(eastern_tz),
            current_price=0,
            unrealized_pnl=0,
            realized_pnl=0,
            used_capital=0                                
        )
        
        # List to track stocks that pass all entry conditions
        self.qualifying_stocks = []
        self.qualifying_stocks_lock = threading.Lock()  # Thread-safe access
        
        # Start drawdown monitoring thread AFTER everything is initialized
        drawdown_thread = threading.Thread(target=self.monitor_drawdown_loop, name="DrawdownMonitor", daemon=True)
        # drawdown_thread.start()
    
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
    
    def calculate_dynamic_daily_limit(self):
        equity = creds.EQUITY
        
        limit_2pct = equity * self.config.RISK_CONFIG.lower_limit
        
        portfolio_atr_pct = self.calculate_portfolio_atr14()
        atr_multiplier = self.config.RISK_CONFIG.atr_multiplier
        limit_atr = equity * (atr_multiplier * portfolio_atr_pct)
        
        limit_3pct = equity * self.config.RISK_CONFIG.upper_limit
        
        daily_limit_dollars = min(limit_2pct, limit_atr, limit_3pct)
        daily_limit_pct = daily_limit_dollars / equity
        
        print(f"[Dynamic Daily Limit]")
        print(f"  - 2% equity: ${limit_2pct:,.0f}")
        print(f"  - {atr_multiplier} x Portfolio ATR14: ${limit_atr:,.0f} (ATR={portfolio_atr_pct*100:.2f}%)")
        print(f"  - 3% cap: ${limit_3pct:,.0f}")
        print(f"  - Selected limit: ${daily_limit_dollars:,.0f} ({daily_limit_pct*100:.2f}%)")
        
        return daily_limit_dollars
    
    def monitor_drawdown_loop(self):
            print("Starting global drawdown monitoring thread.")
            self.max_drawdown_triggered = False
            
            while not self.stop_event.is_set():
                try:
                    total_pnl = self.broker.get_total_pnl()
                    
                    threshold = self.calculate_dynamic_daily_limit()

                    print(f"[Drawdown] PnL: {total_pnl:.2f}, Threshold: {-threshold:.2f}")
                    eastern_tz = pytz.timezone('US/Eastern')
                    now = datetime.now(eastern_tz)
                    current_time_str = now.strftime("%H:%M")

                    # Update centralized hedge symbol price and unrealized PnL
                    try:
                        hedge_symbol = self.config.HEDGE_CONFIG.hedge_symbol
                        with self.manager_lock:
                            central_data = trades_db.get_latest_strategy_data(hedge_symbol) or {}
                            hedge_shares = central_data.get('position_shares', 0) or 0
                            hedge_entry_price = central_data.get('entry_price', 0) or 0
                        if hedge_shares > 0:
                            hedge_price = self.broker.get_current_price(hedge_symbol)
                            if hedge_price is not None:
                                hedge_unrealized = (hedge_price - hedge_entry_price) * hedge_shares
                                trades_db.update_strategy_data(hedge_symbol,
                                    current_price=hedge_price,
                                    unrealized_pnl=hedge_unrealized
                                )
                    except Exception as e:
                        print(f"[Drawdown] Hedge update error: {e}")
                    
                    if total_pnl <= -threshold and not self.max_drawdown_triggered:
                        print(f"Max loss threshold of ${-threshold:,.0f} hit. Stopping all strategies.")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        
                    if current_time_str >= self.config.TRADING_HOURS.market_close:
                        print(f"Exit time reached at {current_time_str} CT - Closing drawdown monitor thread")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        break
                            
                    time.sleep(5)
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
            # self.stocks_dict = []
            print(f"Stock selector returned {len(self.stocks_list)} stocks")
        except Exception as e:
            print(f"Error in stock selector: {e}")
            self.stocks_list = ["AAPL"]
            self.stocks_dict = []

        print("Start")
        for i, stock in enumerate(self.stocks_list):
            print(f"{i}:{stock} ")
            
        self.print_qualifying_stocks()
        
        for i, stock in enumerate(self.stocks_list):
            strategy = Strategy(self, stock, self.broker, self.config)
            self.strategies.append(strategy)
            
            thread_name = f"Strategy-{i}-{stock}" 
            t = threading.Thread(target=strategy.run, args=(i,), name=thread_name)
            
            t.start()
            self.threads.append(t)
        
        print(f"\nWaiting for {len(self.threads)} strategies to complete...")
        for i, thread in enumerate(self.threads):
            thread.join()
            print(f"Strategy {i+1} completed")
        
        
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
    
    def test(self):
        print("=" * 80)
        print("Testing AAPL Position Opening with Hedges and Closing After 1 Minute")
        print("(SIMULATED - Market is closed, using historical prices)")
        print("=" * 80)
        
        # Create a strategy for AAPL
        print("\n[Step 1] Creating Strategy for AAPL...")
        strategy = Strategy(self, "AAPL", self.broker, self.config)
        print("✓ Strategy created")
        
        # Get current price for AAPL (for reference, may return None if market closed)
        print("\n[Step 2] Getting reference price for AAPL...")
        current_price = strategy.get_current_price_with_retry("AAPL")
        if current_price is None:
            # Try to get last price from historical data
            print("  Market data unavailable, fetching last close from historical data...")
            hist_data = strategy.get_historical_data_with_retry("AAPL", duration="1 D", bar_size="1 day")
            if hist_data is not None and not hist_data.empty:
                current_price = hist_data['close'].iloc[-1]
                print(f"✓ Using last close price from historical data: ${current_price:.2f}")
            else:
                # Fallback: try 1-minute bars to get most recent price
                hist_data = strategy.get_historical_data_with_retry("AAPL", duration="1 D", bar_size="1 min")
                if hist_data is not None and not hist_data.empty:
                    current_price = hist_data['close'].iloc[-1]
                    print(f"✓ Using last price from 1-minute bars: ${current_price:.2f}")
                else:
                    print("✗ Unable to get any price data for AAPL")
                    return
        else:
            print(f"✓ Current AAPL price: ${current_price:.2f}")
        
        # Calculate position size manually (test only)
        print("\n[Step 3] Calculating position size (test: $10k budget)...")
        limit_price = current_price + 1
        print(f"  - Limit Price: ${limit_price:.2f} (current price + $1)")
        
        # Calculate stop loss based on limit price
        stop_loss_pct = strategy.calculate_stop_loss(limit_price)
        stop_loss = limit_price * (1 - stop_loss_pct)
        
        # Calculate shares based on $10k budget
        test_budget = 10000
        shares = int(test_budget / limit_price)
        
        if shares <= 0:
            print("✗ Unable to calculate valid position size")
            return
        
        print(f"✓ Position size calculated:")
        print(f"  - Budget: ${test_budget:,.2f}")
        print(f"  - Limit Price: ${limit_price:.2f}")
        print(f"  - Stop Loss: ${stop_loss:.2f} ({stop_loss_pct*100:.2f}%)")
        print(f"  - Shares: {shares}")
        print(f"  - Total Cost: ${shares * limit_price:.2f}")
        
        # SIMULATE: Open position at limit price (assuming order filled)
        print("\n[Step 4] Simulating AAPL position opening...")
        print("  [SIMULATED] Order would be placed at limit price")
        fill_price = limit_price  # Simulate fill at limit price
        fill_shares = shares
        
        print(f"✓ [SIMULATED] Position opened:")
        print(f"  - Order ID: SIMULATED-{int(time.time())}")
        print(f"  - Fill Price: ${fill_price:.2f} (limit price)")
        print(f"  - Fill Shares: {fill_shares}")
        
        # Initialize position tracking
        strategy.entry_price = fill_price
        strategy.stop_loss_price = stop_loss
        strategy.take_profit_price = fill_price * (1 + creds.PROFIT_CONFIG.profit_booking_levels[0]['gain'])
        strategy.position_shares = fill_shares
        strategy.position_active = True
        eastern_tz = pytz.timezone('US/Eastern')
        strategy.entry_time = datetime.now(eastern_tz)
        strategy.current_price = fill_price
        strategy.used_capital = fill_shares * fill_price
        
        print(f"  - Entry Time: {strategy.entry_time}")
        print(f"  - Stop Loss: ${strategy.stop_loss_price:.2f}")
        print(f"  - Take Profit: ${strategy.take_profit_price:.2f}")
        
        # Update database
        trades_db.update_strategy_data("AAPL",
            position_active=True,
            position_shares=fill_shares,
            shares=fill_shares,
            entry_price=fill_price,
            stop_loss_price=stop_loss,
            take_profit_price=strategy.take_profit_price,
            entry_time=strategy.entry_time,
            current_price=fill_price,
            used_capital=strategy.used_capital
        )
        
        # SIMULATE: Open hedge position
        print("\n[Step 5] Simulating hedge position opening...")
        # Use early hedge level from config
        hedge_level = "early"
        hedge_level_config = getattr(creds.HEDGE_CONFIG.hedge_levels, hedge_level)
        hedge_beta = hedge_level_config.beta
        hedge_equity_pct = hedge_level_config.equity_pct
        
        # Get hedge symbol price for simulation
        hedge_symbol = strategy.hedge_symbol
        hedge_price = strategy.get_current_price_with_retry(hedge_symbol)
        if hedge_price is None:
            # Try historical data
            hedge_hist = strategy.get_historical_data_with_retry(hedge_symbol, duration="1 D", bar_size="1 day")
            if hedge_hist is not None and not hedge_hist.empty:
                hedge_price = hedge_hist['close'].iloc[-1]
            else:
                hedge_hist = strategy.get_historical_data_with_retry(hedge_symbol, duration="1 D", bar_size="1 min")
                if hedge_hist is not None and not hedge_hist.empty:
                    hedge_price = hedge_hist['close'].iloc[-1]
                else:
                    print(f"✗ Unable to get price for {hedge_symbol}, skipping hedge")
                    hedge_price = None
        
        if hedge_price:
            # Simulate hedge execution
            account_equity = creds.EQUITY
            hedge_amount = account_equity * hedge_equity_pct
            hedge_shares = int(hedge_amount / hedge_price)
            
            print(f"  [SIMULATED] Hedge order would be placed")
            print(f"✓ [SIMULATED] Hedge opened:")
            print(f"  - Hedge Symbol: {hedge_symbol}")
            print(f"  - Hedge Shares: {hedge_shares}")
            print(f"  - Hedge Entry Price: ${hedge_price:.2f}")
            print(f"  - Hedge Level: {hedge_level}")
            
            # Update hedge status
            strategy.hedge_active = True
            strategy.hedge_shares = hedge_shares
            strategy.hedge_level = hedge_level
            strategy.hedge_entry_price = hedge_price
            
            # Update database with hedge information
            trades_db.update_strategy_data("AAPL",
                hedge_active=True,
                hedge_shares=hedge_shares,
                hedge_symbol=hedge_symbol,
                hedge_level=hedge_level,
                hedge_beta=hedge_beta,
                hedge_entry_price=hedge_price,
                hedge_entry_time=datetime.now(eastern_tz)
            )
        else:
            print("✗ Hedge opening skipped (no price data)")
        
        # Wait 1 minute
        print("\n[Step 6] Waiting 1 minute before closing position...")
        print("  Waiting 60 seconds...")
        time.sleep(60)
        print("✓ 1 minute elapsed")
        
        # Get exit price (last live market price)
        print("\n[Step 7] Getting exit price (last live market price)...")
        exit_price = strategy.get_current_price_with_retry("AAPL")
        if exit_price is None:
            # Use historical data to get last price
            hist_data = strategy.get_historical_data_with_retry("AAPL", duration="1 D", bar_size="1 min")
            if hist_data is not None and not hist_data.empty:
                exit_price = hist_data['close'].iloc[-1]
                print(f"✓ Using last price from historical data: ${exit_price:.2f}")
            else:
                # Fallback to entry price if we can't get exit price
                exit_price = fill_price
                print(f"⚠ Unable to get exit price, using entry price: ${exit_price:.2f}")
        else:
            print(f"✓ Exit price: ${exit_price:.2f}")
        
        # SIMULATE: Close hedge first
        print("\n[Step 8] Simulating hedge position closing...")
        if strategy.hedge_active:
            # Get hedge exit price
            hedge_exit_price = strategy.get_current_price_with_retry(hedge_symbol)
            if hedge_exit_price is None:
                hedge_hist = strategy.get_historical_data_with_retry(hedge_symbol, duration="1 D", bar_size="1 min")
                if hedge_hist is not None and not hedge_hist.empty:
                    hedge_exit_price = hedge_hist['close'].iloc[-1]
                else:
                    hedge_exit_price = strategy.hedge_entry_price
            
            hedge_pnl = (hedge_exit_price - strategy.hedge_entry_price) * strategy.hedge_shares
            print(f"  [SIMULATED] Hedge sell order would be placed")
            print(f"✓ [SIMULATED] Hedge closed:")
            print(f"  - Exit Price: ${hedge_exit_price:.2f}")
            print(f"  - Hedge P&L: ${hedge_pnl:.2f}")
            
            # Update hedge status
            strategy.hedge_active = False
            shares_to_sell = strategy.hedge_shares
            strategy.hedge_shares = 0
            
            # Update database
            trades_db.update_strategy_data("AAPL",
                hedge_active=False,
                hedge_shares=0,
                hedge_exit_price=hedge_exit_price,
                hedge_exit_time=datetime.now(eastern_tz),
                hedge_pnl=hedge_pnl
            )
        else:
            print("  No active hedge to close")
        
        # SIMULATE: Close position
        print("\n[Step 9] Simulating AAPL position closing...")
        if strategy.position_active and strategy.position_shares > 0:
            print(f"  [SIMULATED] Sell order would be placed at market")
            print(f"✓ [SIMULATED] Position closed:")
            print(f"  - Exit Price: ${exit_price:.2f}")
            
            # Calculate PnL
            realized_pnl = (exit_price - strategy.entry_price) * strategy.position_shares
            print(f"  - Realized P&L: ${realized_pnl:.2f}")
            print(f"  - Entry Price: ${strategy.entry_price:.2f}")
            print(f"  - Exit Price: ${exit_price:.2f}")
            print(f"  - Shares: {strategy.position_shares}")
            
            # Update capital tracking
            with self.manager_lock:
                self.available_capital += exit_price * strategy.position_shares
                original_cost = strategy.entry_price * strategy.position_shares
                self.used_capital -= original_cost
                strategy.used_capital -= original_cost
                print(f"  - Available Capital after closing: ${self.available_capital:.2f}")
                print(f"  - Used Capital after closing: ${self.used_capital:.2f}")
            
            # Update position tracking
            strategy.position_active = False
            strategy.position_shares = 0
            strategy.realized_pnl += realized_pnl
            
            # Update database
            trades_db.update_strategy_data("AAPL",
                position_active=False,
                position_shares=0,
                current_price=exit_price,
                unrealized_pnl=0,
                realized_pnl=strategy.realized_pnl,
                close_time=datetime.now(eastern_tz),
                used_capital=0
            )
        else:
            print("  No active position to close")
        
        print("\n" + "=" * 80)
        print("Test complete!")
        print("=" * 80)
        exit()


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
    print("Caching ADV and RVOL data...")
    # initialize_stock_selector()
    print("ADV and RVOL data cached")

    print("Checking for existing database to backup...")
    trades_db.backup_database()
    
    manager = StrategyManager()
    manager.run()
    print("STOPPING MANAGER")