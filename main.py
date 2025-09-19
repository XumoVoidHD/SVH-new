from simulation import ibkr_broker
import pandas as pd
from stock_selector import StockSelector
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
        self.leverage_config = creds.LEVERAGE_CONFIG
        
        self.data = {}
        self.indicators = {}
        self.position_active = True 
        self.position_shares = 5
        self.current_price = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.used_margin = 0
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        self.entry_time = None
        self.close_time = None
        self.trailing_exit_monitoring = False
        
        # Hedge and leverage tracking
        self.hedge_active = False
        self.hedge_shares = 0
        self.hedge_symbol = self.hedge_config.hedge_symbol
        self.current_leverage = 1.0
        self.margin_used_leverage = 0

        # Each stock can use risk_per_trade percentage of equity (default 0.4%)
        # No need to track total position size across stocks
        
        # Capital tracking
        self.used_capital = 0
        self.used_margin = 0
    
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
            return None, 0
        
        triggers_met = 0
        trigger_details = []
        
        try:
            # Check VIX trigger
            vix_data = self.broker.get_historical_data(stock="VIXY", bar_size=3)
            if not vix_data.empty and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                if current_vix > self.hedge_config.triggers.vix_threshold:
                    triggers_met += 1
                    trigger_details.append(f"VIX {current_vix:.1f} > {self.hedge_config.triggers.vix_threshold}")
            else:
                print(f"VIX data unavailable or empty for hedge trigger check")
            
            # Check S&P 500 drop trigger (using SPY as proxy)
            spy_data = self.broker.get_historical_data(stock="SPY", bar_size=15)  # 15-min data
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
            import traceback
            traceback.print_exc()
            return None, 0
        
        # Determine hedge level
        if triggers_met == 0:
            return None, 0
        elif triggers_met == 1:
            hedge_level = 'mild'
            beta = self.hedge_config.hedge_levels.mild.beta
        else:
            hedge_level = 'severe' 
            beta = self.hedge_config.hedge_levels.severe.beta
        
        print(f"Hedge triggers: {triggers_met} met - {', '.join(trigger_details)}")
        print(f"Hedge level: {hedge_level} (-{beta}β)")
        
        return hedge_level, beta
    
    def check_leverage_conditions(self):
        """Check if leverage conditions are met and return leverage multiplier"""
        if not self.leverage_config.enabled:
            return 1.0
        
        conditions_met = 0
        total_conditions = 3  # Alpha Score, VIX, Drawdown (VIX trend is bonus)
        condition_details = []
        
        try:
            # Check Alpha Score condition
            if self.score >= self.leverage_config.conditions.alpha_score_min:
                conditions_met += 1
                condition_details.append(f"Alpha Score {self.score} ≥ {self.leverage_config.conditions.alpha_score_min}")
            
            # Check VIX condition
            vix_data = self.broker.get_historical_data(stock="VIXY", bar_size=3)
            if not vix_data.empty and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                if current_vix < self.leverage_config.conditions.vix_max:
                    conditions_met += 1
                    condition_details.append(f"VIX {current_vix:.1f} < {self.leverage_config.conditions.vix_max}")
                
                # Check VIX trend (10-day declining)
                if len(vix_data) >= self.leverage_config.conditions.vix_trend_days:
                    vix_10d_ago = vix_data['close'].iloc[-self.leverage_config.conditions.vix_trend_days]
                    if current_vix < vix_10d_ago:
                        condition_details.append(f"VIX trending down over {self.leverage_config.conditions.vix_trend_days} days")
            else:
                print(f"VIX data unavailable or empty for leverage condition check")
             
        except Exception as e:
            print(f"Error checking leverage conditions: {e}")
            import traceback
            traceback.print_exc()
            return 1.0
        
        # Determine leverage level
        conditions_pct = conditions_met / total_conditions
        
        if conditions_pct >= 1.0:  # All conditions met
            leverage = self.leverage_config.leverage_levels.all_conditions_met
            print(f"Leverage: All conditions met ({conditions_met}/{total_conditions}) - {leverage}x leverage")
        elif conditions_pct >= 0.6:  # Most conditions met
            leverage = self.leverage_config.leverage_levels.partial_conditions 
            print(f"Leverage: Partial conditions met ({conditions_met}/{total_conditions}) - {leverage}x leverage")
        else:
            leverage = self.leverage_config.leverage_levels.default
            print(f"Leverage: Few conditions met ({conditions_met}/{total_conditions}) - {leverage}x leverage")
        
        if condition_details:
            print(f"Leverage conditions: {', '.join(condition_details)}")
        
        return leverage
    
    def execute_hedge(self, hedge_level, beta):
        """Execute hedge by shorting XLF ETF"""
        if hedge_level is None or self.hedge_active:
            return
        
        try:
            # Calculate hedge size
            account_equity = creds.EQUITY
            hedge_amount = account_equity * beta
            
            # Get XLF price
            xlf_price = self.broker.get_current_price(self.hedge_symbol)
            if xlf_price is None:
                print(f"Unable to get {self.hedge_symbol} price for hedge")
                return
            
            # Calculate shares to short
            hedge_shares = int(hedge_amount / xlf_price)

            trade = self.broker.place_order(symbol=self.hedge_symbol, qty=hedge_shares, order_type="MARKET", price=xlf_price, side="SELL")
            if trade is None:
                print(f"Unable to place hedge order for {self.hedge_symbol}")
                return
            
            print(f"Executing {hedge_level} hedge:")
            print(f"  - Hedge amount: ${hedge_amount:,.0f} ({beta*100:.1f}% of equity)")
            print(f"  - {self.hedge_symbol} price: ${trade[1]}")
            print(f"  - Shares to short: {hedge_shares}")
            
            # Update hedge status and track hedge position
            self.hedge_active = True
            self.hedge_shares = hedge_shares
            
            # Update database with hedge information
            trades_db.update_strategy_data(self.stock,
                hedge_active=True,
                hedge_shares=hedge_shares,
                hedge_symbol=self.hedge_symbol,
                hedge_level=hedge_level,
                hedge_beta=beta,
                hedge_entry_price=trade[1],
                hedge_entry_time=datetime.now(pytz.timezone('America/Chicago'))
            )
            
            print(f"Hedge executed: Short {hedge_shares} shares of {self.hedge_symbol}")
            print(f"Hedge position tracked in database for {self.stock}")
                
        except Exception as e:
            print(f"Error executing hedge: {e}")
            import traceback
            traceback.print_exc()
    
    def close_hedge(self):
        """Close hedge position by buying back XLF shares"""
        if not self.hedge_active or self.hedge_shares == 0:
            return
        
        try:
            print(f"Closing hedge: Buying back {self.hedge_shares} shares of {self.hedge_symbol}")
            
            # Get current XLF price for P&L calculation
            current_xlf_price = self.broker.get_current_price(self.hedge_symbol)
            if current_xlf_price is None:
                print(f"Unable to get current {self.hedge_symbol} price for P&L calculation")
                current_xlf_price = 0
            
            # Calculate hedge P&L (profit from short position)
            hedge_pnl = 0
            if hasattr(self, 'hedge_entry_price') and self.hedge_entry_price:
                hedge_pnl = (self.hedge_entry_price - current_xlf_price) * self.hedge_shares
            
            trade = self.broker.place_order(symbol=self.hedge_symbol, qty=self.hedge_shares, order_type="MARKET", price=current_xlf_price, side="SELL")
            if trade is None:
                print(f"Unable to place hedge order for {self.hedge_symbol}")
                return

            # Update hedge status
            self.hedge_active = False
            self.hedge_shares = 0
            
            # Update database with hedge closure
            trades_db.update_strategy_data(self.stock,
                hedge_active=False,
                hedge_shares=0,
                hedge_exit_price=current_xlf_price,
                hedge_exit_time=datetime.now(pytz.timezone('America/Chicago')),
                hedge_pnl=hedge_pnl
            )
            
            print(f"Hedge closed successfully")
            print(f"Hedge P&L: ${hedge_pnl:.2f}")
            print(f"Hedge position updated in database for {self.stock}")
                
        except Exception as e:
            print(f"Error closing hedge: {e}")
            import traceback
            traceback.print_exc()
    
    def end_of_day_weak_exit(self):
        """Exit weak positions at 3:30 PM (-0.3% to +1.2%)"""
        if not self.position_active:
            return
        
        current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
        
        # Exit if position is weak (-0.3% to +1.2%)
        if -0.003 <= current_gain_pct <= 0.012:
            print(f"3:30 PM Weak Exit: {current_gain_pct*100:.1f}% gain in weak range")
            self.close_position('end_of_day_weak', self.position_shares)
        else:
            print(f"3:30 PM: Position not in weak range ({current_gain_pct*100:.1f}% gain) - keeping position")
    
    def safety_exit_all(self):
        
        print("Safety exit all positions at 3:35 PM")
        if self.position_active:
            print("3:35 PM Safety Exit: Closing all positions")
            self.close_position('safety_exit', self.position_shares)
        
        # Close hedge if active
        if self.hedge_active:
            print("3:35 PM Safety Exit: Closing hedge positions")
            self.close_hedge()
    
    def market_on_close_exit(self):
        """Market-on-close exit at 4:00 PM"""
        if self.position_active:
            print("4:00 PM Market-On-Close: Closing remaining positions")
            self.close_position('market_on_close', self.position_shares)
        
        # Close hedge if active
        # if self.hedge_active:
        #     print("4:00 PM Market-On-Close: Closing hedge positions")
        #     self.close_hedge()
        
    
    def fetch_data_by_timeframe(self):
        """Fetch and store data for each configured timeframe with retry logic"""
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
            # Extract period number from timeframe name (e.g., '3min' -> 3)
            period = int(tf_name.replace('min', ''))
            
            # Try to fetch data with retries
            max_retries = 10
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    data = self.broker.get_historical_data(stock=self.stock, bar_size=period)
                    
                    # Check if data is valid and not empty
                    if data is not None and hasattr(data, 'empty') and not data.empty and len(data) > 0:
                        self.data[tf_name] = data
                        print(f"Fetched {tf_name} data: {len(data)} candles")
                        break
                    else:
                        if attempt < max_retries - 1:
                            print(f"Failed to fetch {tf_name} data (attempt {attempt + 1}/{max_retries}) - data is None or empty, retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            print(f"Failed to fetch {tf_name} data after {max_retries} attempts - data is None or empty")
                            self.data[tf_name] = pd.DataFrame()
                            
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error fetching {tf_name} data (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed to fetch {tf_name} data after {max_retries} attempts due to error: {e}")
                        self.data[tf_name] = pd.DataFrame()
    
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
            if '3min' not in self.data or '5min' not in self.indicators or '20min' not in self.indicators:
                return False
            
            # Price > VWAP
            price_vwap_ok = bool(self.data['3min']['close'].iloc[-1] > self.indicators['3min']['vwap'].iloc[-1])
            
            # EMA1 > EMA2 (5-min EMA > 20-min EMA)
            ema_cross_ok = bool(self.indicators['5min']['ema1'].iloc[-1] > self.indicators['20min']['ema2'].iloc[-1])
            
            return price_vwap_ok and ema_cross_ok
        except Exception as e:
            print(f"Error checking trend conditions: {e}")
            return False
    
    def _check_momentum_conditions(self):
        """Check momentum conditions"""
        if '3min' not in self.indicators:
            return False
        
        return bool(self.indicators['3min']['macd'].iloc[-1] > 0)
    
    def _check_volume_volatility_conditions(self):
        """Check volume and volatility conditions"""
        if '3min' not in self.data or '3min' not in self.indicators:
            return False
        
        # Volume spike check
        recent_volume = self.data['3min']['volume'].iloc[-1]
        avg_volume = self.indicators['3min']['volume_avg'].iloc[-1]
        multiplier = self.alpha_score_config.volume_volatility.conditions.volume_spike.multiplier
        volume_ok = bool(recent_volume > multiplier * avg_volume)
        
        # ADX threshold check
        adx_threshold = self.alpha_score_config.volume_volatility.conditions.adx_threshold.threshold
        adx_ok = bool(self.indicators['3min']['adx'].iloc[-1] > adx_threshold)
        
        return volume_ok and adx_ok
    
    def _check_market_calm_conditions(self):
        """Check market calm conditions (VIX)"""
        try:
            vix_df = self.broker.get_historical_data(stock="VIXY", bar_size=3)
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
        elif '3min' not in self.data or '3min' not in self.indicators:
            self.additional_checks_passed = False
            print(f"\nAdditional Checks Passed: {self.additional_checks_passed} (Missing data/indicators)")
        else:
            # Check +2x volume
            recent_volume = self.data['3min']['volume'].iloc[-1]
            avg_volume = self.indicators['3min']['volume_avg'].iloc[-1]
            volume_multiplier = self.additional_checks_config.volume_multiplier
            
            volume_check = bool(recent_volume > volume_multiplier * avg_volume)
            print(f"{'Passed' if volume_check else 'Failed'} +{volume_multiplier}x volume check {'passed' if volume_check else 'failed'}")
            
            # Check VWAP slope
            vwap_slope_check = self._check_vwap_slope()
            
            self.additional_checks_passed = bool(volume_check and vwap_slope_check)
            
            print(f"\nAdditional Checks Passed: {self.additional_checks_passed}")
        
        # Always update database with additional checks status (regardless of pass/fail)
        trades_db.update_strategy_data(self.stock, additional_checks_passed=self.additional_checks_passed)
    
    def _check_vwap_slope(self):
        """Check VWAP slope condition"""
        vwap_series = self.indicators['3min']['vwap']
        if len(vwap_series) < 2:
            print("VWAP slope check failed: insufficient data")
            return False
        
        current_vwap = vwap_series.iloc[-1]
        vwap_3min_ago = vwap_series.iloc[-2]
        vwap_slope = (current_vwap - vwap_3min_ago) / self.additional_checks_config.vwap_slope_period
        
        threshold = self.additional_checks_config.vwap_slope_threshold
        slope_ok = bool(vwap_slope > threshold)
        
        print(f"{'Passed' if slope_ok else 'Failed'} VWAP slope check {'passed' if slope_ok else 'failed'}: {vwap_slope:.3f} {'>' if slope_ok else '<='} {threshold}")
        
        return slope_ok 
        
    def calculate_position_size(self):
        """Calculate position size based on risk management rules with hedge and leverage"""
        
        # Step 1: Check hedge triggers and execute if needed
        hedge_level, hedge_beta = self.check_hedge_triggers()
        if hedge_level:
            self.execute_hedge(hedge_level, hedge_beta)
        
        # Step 2: Check leverage conditions
        leverage_multiplier = self.check_leverage_conditions()
        self.current_leverage = leverage_multiplier

        # leverage_multiplier = 1.0
        # self.current_leverage = leverage_multiplier
        
        # Get account equity from config
        account_equity = creds.EQUITY
        current_price = self.broker.get_current_price(self.stock)
        
        # Each stock gets exactly risk_per_trade percentage of equity
        risk_per_trade = creds.RISK_CONFIG.risk_per_trade
        
        stop_loss_pct = self.calculate_stop_loss(current_price)
        stop_loss_price = current_price * (1 - stop_loss_pct)
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0:
            return 0, 0, 0, 0
        
        # Calculate capital for trade based on risk_per_trade percentage
        capital_for_trade = ((account_equity * risk_per_trade)/(stop_loss_pct/100))
        if capital_for_trade > (account_equity * creds.RISK_CONFIG.max_position_equity_pct):
            capital_for_trade = account_equity * creds.RISK_CONFIG.max_position_equity_pct
        
        # Calculate shares based on available capital
        shares = int(capital_for_trade / current_price)
        
        # Apply leverage to all trades when conditions are met
        if leverage_multiplier > 1.0:
            shares = int(shares * leverage_multiplier)
            print(f"Applying {leverage_multiplier}x leverage: {shares} shares")
            # Calculate margin used for leverage
            base_cost = shares * current_price / leverage_multiplier
            leveraged_cost = shares * current_price
            self.margin_used_leverage = leveraged_cost - base_cost
            print(f"Margin used for leverage: ${self.margin_used_leverage:,.0f}")
        else:
            print("No leverage applied to this trade")
            self.margin_used_leverage = 0
        
        # Ensure VWAP is below current price before using it
        data_3min = self.broker.get_historical_data(stock=self.stock, bar_size=3)
        vwap_value = vwap.calc_vwap(data_3min).iloc[-1]
        
        # Check if VWAP is below current price (required condition)
        if vwap_value < current_price:
            print(f"VWAP (${vwap_value:.2f}) is below current price (${current_price:.2f})")
        else:
            print(f"WARNING: VWAP (${vwap_value:.2f}) is NOT below current price (${current_price:.2f}) - skipping order")
            return 0, 0, 0, 0
        
        offset_pct = random.uniform(creds.ORDER_CONFIG.limit_offset_min, creds.ORDER_CONFIG.limit_offset_max)  # 0.03% to 0.07%
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
        data_3min = self.broker.get_historical_data(stock=self.stock, bar_size=3)
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
        if self.score >= creds.RISK_CONFIG.alpha_score_threshold: 
            # Use this condition for now for all orders to get executed 
        # if self.score >= creds.RISK_CONFIG.alpha_score_threshold and bool(self.additional_checks_passed):
            print(f"ENTERING POSITION - Both Alpha Score >= {creds.RISK_CONFIG.alpha_score_threshold} AND additional checks passed")
            shares, entry_price, stop_loss, limit_price = self.calculate_position_size()
            if shares > 0:
                print(f"Order Details:")
                print(f"  - Symbol: {self.stock}")
                print(f"  - Shares: {shares}")
                print(f"  - Limit Price: ${limit_price:.2f}")
                print(f"  - Stop Loss: ${stop_loss:.2f}")
                print(f"  - Exit: Market-On-Close at 4:00 PM ET")
                
                trade = self.broker.place_order(symbol=self.stock, qty=shares, order_type="LIMIT", price=round(limit_price, 2), side="BUY")
                if int(trade[1]) == -1:
                    print(f"Unable to place order for {self.stock}")
                    return
                else:
                    if trade[2] != shares:
                        print(f"Order partially filled for {self.stock}: {trade}")
                    shares = trade[2]
                    print(f"Order placed for {self.stock}: {trade}")
                    with self.manager.manager_lock:
                        self.used_capital += shares * trade[1]
                        self.used_margin = shares * trade[1]
                        print(f"Used Margin: ${self.used_margin:.2f}")
                        print(f"Used Capital: ${self.used_capital:.2f}")

                    self.entry_price = trade[1]
                    self.stop_loss_price = stop_loss
                    self.take_profit_price = entry_price * (1 + creds.PROFIT_CONFIG.profit_booking_levels[0]['gain'])  # Use 1% from profit booking levels
                    self.position_shares = shares
                    self.position_active = True
                    self.profit_booking_levels_remaining = list(creds.PROFIT_CONFIG.profit_booking_levels)
                    self.trailing_stop_levels_remaining = list(creds.STOP_LOSS_CONFIG.trailing_stop_levels)
                    print(f"[{self.stock}] Profit booking and trailing stop levels reset for new position")

                    central_tz = pytz.timezone('America/Chicago')
                    self.entry_time = datetime.now(central_tz)
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
                        entry_price=self.entry_price,
                        stop_loss_price=self.stop_loss_price,
                        take_profit_price=self.take_profit_price,
                        entry_time=self.entry_time,
                        current_price=self.current_price,
                        unrealized_pnl=self.unrealized_pnl,
                        realized_pnl=self.realized_pnl,
                        used_margin=self.used_margin
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
            central_tz = pytz.timezone(self.trading_hours.timezone)
            current_time = datetime.now(central_tz).time()

            if current_time >= dtime(15, 30) and current_time < dtime(15, 35):
                self.end_of_day_weak_exit()
                break

            elif current_time >= dtime(15, 35):
                self.safety_exit_all()
                break
                        
            # # 4:00 PM - Market-on-close for any remaining positions
            # elif current_time_str >= self.trading_hours.market_close:
            #     self.market_on_close_exit()
            #     break  # End trading for the day

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
        """Monitor position for stop loss and take profit conditions"""
        try:
            # Get current price and update tracking variables
            self.current_price = self.broker.get_current_price(self.stock)
            
            # Check if current_price is None
            if self.current_price is None:
                print(f"Unable to get current price for {self.stock} in position monitoring")
                return
            
            # Calculate current unrealized PnL
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.position_shares
            with self.manager.manager_lock:
                self.manager.unrealized_pnl = self.unrealized_pnl
            
            # Calculate current gain/loss percentage
            current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
            
            # Check stop loss condition
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
            
            # Update database with current position status
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

        # Profit Booking Logic - handled by _check_profit_booking method
        self._check_profit_booking(current_gain_pct, self.current_price)
        
        # Trailing Stop Logic - handled by _check_trailing_stops method
        self._check_trailing_stops(current_gain_pct, self.current_price)

        # Trailing Exit Logic using PROFIT_CONFIG
        trailing_conditions = creds.PROFIT_CONFIG.trailing_exit_conditions
        gain_threshold = trailing_conditions.gain_threshold
        drop_threshold = trailing_conditions.drop_threshold
        monitor_period = trailing_conditions.monitor_period
        
        if current_gain_pct >= gain_threshold:
            # Start monitoring if not already started
            if not self.trailing_exit_monitoring:
                self.trailing_exit_monitoring = True
                # Get current time in USA Central Time
                central_tz = pytz.timezone('America/Chicago')
                self.trailing_exit_start_time = datetime.now(central_tz)
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
                start_monitoring_time = datetime.now(central_tz)
                
                print(f"Starting active monitoring for {monitor_period_seconds} seconds...")
                
                while True:
                    try:
                        # Check if monitoring period has expired
                        current_time = datetime.now(central_tz)
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
                        self.current_price = self.broker.get_current_price(self.stock)
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
        """
        Close position (either partial or full)
        
        Args:
            reason: Reason for closing ('stop_loss', 'profit_booking', 'trailing_exit', etc.)
            shares_to_sell: Number of shares to sell (None for all remaining shares)
        """

        current_price = self.broker.get_current_price(self.stock)
        
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
            
            # Get current time in USA Central Time
            central_tz = pytz.timezone('America/Chicago')
            self.close_time = datetime.now(central_tz)
            
            with self.manager.manager_lock:
                self.manager.realized_pnl += realized_pnl
                self.manager.unrealized_pnl = 0
                self.unrealized_pnl = 0

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
                    realized_pnl=self.manager.realized_pnl,
                    unrealized_pnl=0,
                    current_price=current_price
                )
        else:
            # Partial position closure
            print(f"Partial position closed. Remaining shares: {self.position_shares}")
            
            # Update unrealized PnL for remaining position
            self.unrealized_pnl = (trade[1] - self.entry_price) * self.position_shares
            
            with self.manager.manager_lock:
                self.manager.realized_pnl += realized_pnl
                self.manager.unrealized_pnl = self.unrealized_pnl
                
            # Update database with partial position closure
            trades_db.update_strategy_data(self.stock,
                position_shares=self.position_shares,
                unrealized_pnl=self.unrealized_pnl,
                realized_pnl=self.manager.realized_pnl,
                current_price=current_price
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
            central_tz = pytz.timezone('America/Chicago')
            self.trailing_start_time = datetime.now(central_tz)
            self.max_price_since_trailing = current_price
            print(f"TRAILING EXIT MONITORING: {self.stock} - Started monitoring for exit conditions")
            
            # Update max price
            self.max_price_since_trailing = max(self.max_price_since_trailing, current_price)
            
            # Check time and price drop
            # Get current time in USA Central Time for calculation
            central_tz = pytz.timezone('America/Chicago')
            time_since_trailing = datetime.now(central_tz) - self.trailing_start_time
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

                central_tz = pytz.timezone(self.trading_hours.timezone)
                current_time = datetime.now(central_tz).time()

                if current_time >= dtime(15, 30) and current_time < dtime(15, 35):
                    print("Weak Exit")
                    self.end_of_day_weak_exit()
                    break

                elif current_time >= dtime(15, 35):
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
            'duration': (datetime.now(pytz.timezone('America/Chicago')) - self.entry_time) if self.entry_time else None
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

    def get_historical_data(self, stock, duration="3 D", bar_size="3 mins"):
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
            bar_size = f"{bar_size} mins"
        elif isinstance(bar_size, str) and bar_size.endswith(' min'):
            bar_size = bar_size.replace(' min', ' mins')
        
        # Validate bar size
        if bar_size not in valid_bar_sizes:
            print(f"Warning: Invalid bar size '{bar_size}'. Using '5 mins' as fallback.")
            bar_size = '5 mins'
        
        with self.counter_lock:
            req_id = self.request_id_counter + 1000  # Offset for historical data
            self.request_id_counter += 1
        return self.ib_broker.get_historical_data(stock, req_id, duration, bar_size)

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
        return 0

class StrategyBrokerOld:

    def __init__(self):
        self.client = ibkr_broker.IBTWSAPI(client_id=13)
        self.client.connect()
    
    def is_connected(self):
        return self.client.is_connected()

    def get_current_price(self, stock: str):
        return self.client.get_stock_price(stock)


    def get_historical_data(self, stock: str, bar_size: str = '3', num: str = '1 D', 
                           exchange: str = 'SMART', max_retries: int = 3, retry_delay: float = 2.0):
        # Valid bar sizes according to IBKR API
        valid_bar_sizes = {
            '1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
            '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1W', '1M'
        }
        
        for attempt in range(max_retries + 1):  # +1 because we want to try max_retries times after the first attempt
            try:
                print(f"Attempting to fetch historical data for {stock} (attempt {attempt + 1}/{max_retries + 1})")
                # Convert bar_size to proper IBKR format
                if isinstance(bar_size, (int, str)) and str(bar_size).isdigit():
                    bar_size = f"{bar_size} mins"
                elif isinstance(bar_size, str) and bar_size.endswith(' min'):
                    bar_size = bar_size.replace(' min', ' mins')
                
                # Validate bar size
                if bar_size not in valid_bar_sizes:
                    print(f"Warning: Invalid bar size '{bar_size}'. Using '5 mins' as fallback.")
                    bar_size = '5 mins'
                
                data = self.client.get_historical_data(stock, num, bar_size, exchange)
                
                # Check if we got valid data
                if data is not None and not data.empty and len(data) > 0:
                    print(f"Successfully fetched {len(data)} bars of historical data for {stock}")
                    return data
                else:
                    print(f"Attempt {attempt + 1}: No data returned for {stock}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error fetching historical data for {stock}: {str(e)}")
            
            # If this wasn't the last attempt, wait before retrying
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
        
        # If we get here, all retries failed
        print(f"ERROR: Failed to fetch historical data for {stock} after {max_retries + 1} attempts")
        return pd.DataFrame()

    def place_order(self, symbol: str, qty: int, order_type: str = 'MARKET', 
               price: float = None, side: str = 'BUY', max_retries: int = 3, retry_delay: float = 2.0):

        trade = self.client.place_order(symbol=symbol, quantity=qty, order_type=order_type, price = price, side=side)
        if trade["status"] == "Filled":
            return trade
        else:
            return None

    def get_total_pnl(self):
        return 0


class StrategyManager:
    
    def __init__(self):
        self.broker = StrategyBroker("127.0.0.1", 7497, client_id=1)
        self.config = creds
        self.selector = StockSelector()
        self.manager_lock = threading.Lock()
        self.used_capital = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.stocks_list, self.stocks_dict = [], []
        # Create stop event BEFORE starting the thread
        self.stop_event = threading.Event()
        
        # Add XLF to database for hedging - all threads can use it
        hedge_symbol = self.config.HEDGE_CONFIG.hedge_symbol
        print(f"Adding hedge symbol {hedge_symbol} to database for all threads")
        trades_db.add_stocks_from_list([hedge_symbol])
        
        trades_db.update_strategy_data(hedge_symbol,
            position_active=True,
            position_shares=0,
            entry_price=0,
            stop_loss_price=0,
            take_profit_price=0,
            entry_time=0,
            current_price=0,
            unrealized_pnl=0,
            realized_pnl=0,
            used_margin=0                                
        )
        
        # List to track stocks that pass all entry conditions
        self.qualifying_stocks = []
        self.qualifying_stocks_lock = threading.Lock()  # Thread-safe access
        
        # Start drawdown monitoring thread AFTER everything is initialized
        drawdown_thread = threading.Thread(target=self.monitor_drawdown_loop, name="DrawdownMonitor", daemon=True)
        # drawdown_thread.start()
    
    def monitor_drawdown_loop(self):
            print("Starting global drawdown monitoring thread.")
            threshold = creds.EQUITY * creds.RISK_CONFIG.daily_drawdown_limit
            self.max_drawdown_triggered = False  # Initialize the attribute
            
            while not self.stop_event.is_set():
                try:
                    total_pnl = self.broker.get_total_pnl()  # Must return realized + unrealized

                    print(f"[Drawdown] PnL: {total_pnl:.2f}, Threshold: {-threshold}")
                    # Get current time in USA Central Time
                    central_tz = pytz.timezone('America/Chicago')
                    now = datetime.now(central_tz)
                    current_time_str = now.strftime("%H:%M")
                    
                    if total_pnl <= -threshold and not self.max_drawdown_triggered:
                        print(f"Max loss threshold of {-threshold} hit. Stopping all strategies.")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        
                    if current_time_str >= self.config.TRADING_HOURS.market_close:
                        print(f"Exit time reached at {current_time_str} CT")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                            
                    time.sleep(5)
                except Exception as e:
                    print(f"Error in drawdown monitoring: {e}")
                    time.sleep(5)

    def is_entry_time_window(self):
        """Check if current time is within entry time windows"""
        # Get current time in USA Eastern Time
        eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        # Get entry windows from configuration
        morning_start = self.config.TRADING_HOURS.morning_entry_start
        morning_end = self.config.TRADING_HOURS.morning_entry_end
        afternoon_start = self.config.TRADING_HOURS.afternoon_entry_start
        afternoon_end = self.config.TRADING_HOURS.afternoon_entry_end
        
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

    def run(self):
        self.threads = []
        self.strategies = []
        
        # Wait for entry time window before running stock selection
        print("Waiting for entry time window to begin stock selection...")
        while not self.is_entry_time_window() and not self.config.TESTING:
            eastern_tz = pytz.timezone(self.config.TRADING_HOURS.timezone)
            current_time = datetime.now(eastern_tz)
            current_time_str = current_time.strftime("%H:%M")
            print(f"Current time: {current_time_str} - Not in entry window, waiting...")
            time.sleep(60)  # Check every minute
        
        print("Entry time window detected! Starting stock selection...")
        
        # Only run stock selection when in entry time window
        try:
            self.stocks_list, self.stocks_dict = self.selector.run()
            # self.stocks_list = ["AAPL"]
            # self.stocks_dict = []
            print(f"Stock selector returned {len(self.stocks_list)} stocks")
        except Exception as e:
            print(f"Error in stock selector: {e}")
            # Fallback to a default list
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
        
        # Wait for all threads to complete
        print(f"\nWaiting for {len(self.threads)} strategies to complete...")
        for i, thread in enumerate(self.threads):
            thread.join()
            print(f"Strategy {i+1} completed")
        
        # Print final qualifying stocks summary

    def test(self):
        vix_df = self.broker.get_current_price("NTNX")
        print(vix_df)
        strategy = Strategy(self, "NTNX", self.broker, self.config)
        print(strategy.calculate_position_size())
        exit()


    def print_qualifying_stocks(self):
        """Print the current qualifying stocks with their details from stocks_dict"""
        if not self.qualifying_stocks:
            print("No qualifying stocks found")
            return

        print(f"\n=== QUALIFYING STOCKS ({len(self.qualifying_stocks)}) ===")
        for i, stock_data in enumerate(self.qualifying_stocks, 1):
            symbol = stock_data['symbol']
            # Try to get more info from stocks_dict if available
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
    # fetch_marketcap_csv()
    manager = StrategyManager()
    manager.run()
    print("STOPPING MANAGER")
    