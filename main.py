import pandas as pd
from stock_selector import StockSelector
from simulation.schwab_broker import SchwabBroker
from simulation.forward_broker import ForwardBroker
import random
import threading
import time
from datetime import datetime, timedelta
import pytz
from helpers import vwap, ema, macd, adx, atr
from log import setup_logger
from db.trades_db import trades_db
from fetch_marketcap_csv import fetch_marketcap_csv
import json
from types import SimpleNamespace
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
            # Keep lists as lists, but convert their items if they're dicts
            return [dict_to_obj(item) if isinstance(item, dict) else item for item in d]
        else:
            return d
    
    return dict_to_obj(config_dict)

creds = load_config('creds.json')
print(creds)
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
        self.position_active = False
        self.position_shares = 0
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
        
        # Check if current time is in either window
        in_morning_window = morning_start <= current_time_str <= morning_end
        in_afternoon_window = afternoon_start <= current_time_str <= afternoon_end
        
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
            vix_data = self.broker.get_historical_data(3, "VIXY")
            if not vix_data.empty and 'close' in vix_data.columns:
                current_vix = vix_data['close'].iloc[-1]
                if current_vix > self.hedge_config.triggers.vix_threshold:
                    triggers_met += 1
                    trigger_details.append(f"VIX {current_vix:.1f} > {self.hedge_config.triggers.vix_threshold}")
            else:
                print(f"VIX data unavailable or empty for hedge trigger check")
            
            # Check S&P 500 drop trigger (using SPY as proxy)
            spy_data = self.broker.get_historical_data(15, "SPY")  # 15-min data
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
            vix_data = self.broker.get_historical_data(3, "VIXY")
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
            
            print(f"Executing {hedge_level} hedge:")
            print(f"  - Hedge amount: ${hedge_amount:,.0f} ({beta*100:.1f}% of equity)")
            print(f"  - {self.hedge_symbol} price: ${xlf_price:.2f}")
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
                hedge_entry_price=xlf_price,
                hedge_entry_time=datetime.now(pytz.timezone('America/Chicago'))
            )

            # trades_db.update_strategy_data(self.hedge_symbol,
            #     position_active=True,
            #     position_shares=0,
            #     entry_price=0,
            #     stop_loss_price=0,
            #     take_profit_price=0,
            #     entry_time=0,
            #     current_price=0,
            #     unrealized_pnl=0,
            #     realized_pnl=0,
            #     used_margin=0
            # )
            
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
        """Safety exit all positions at 3:35 PM"""
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
                data = self.broker.get_historical_data(period, self.stock)
                
                if not data.empty:
                    self.data[tf_name] = data
                    print(f"Fetched {tf_name} data: {len(data)} candles")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"Failed to fetch {tf_name} data (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed to fetch {tf_name} data after {max_retries} attempts")
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
            price_vwap_ok = (self.data['3min']['close'].iloc[-1] > self.indicators['3min']['vwap'].iloc[-1])
            
            # EMA1 > EMA2 (5-min EMA > 20-min EMA)
            ema_cross_ok = (self.indicators['5min']['ema1'].iloc[-1] > self.indicators['20min']['ema2'].iloc[-1])
            
            return price_vwap_ok and ema_cross_ok
        except Exception as e:
            print(f"Error checking trend conditions: {e}")
            return False
    
    def _check_momentum_conditions(self):
        """Check momentum conditions"""
        if '3min' not in self.indicators:
            return False
        
        return self.indicators['3min']['macd'].iloc[-1] > 0
    
    def _check_volume_volatility_conditions(self):
        """Check volume and volatility conditions"""
        if '3min' not in self.data or '3min' not in self.indicators:
            return False
        
        # Volume spike check
        recent_volume = self.data['3min']['volume'].iloc[-1]
        avg_volume = self.indicators['3min']['volume_avg'].iloc[-1]
        multiplier = self.alpha_score_config.volume_volatility.conditions.volume_spike.multiplier
        volume_ok = recent_volume > multiplier * avg_volume
        
        # ADX threshold check
        adx_threshold = self.alpha_score_config.volume_volatility.conditions.adx_threshold.threshold
        adx_ok = self.indicators['3min']['adx'].iloc[-1] > adx_threshold
        
        return volume_ok and adx_ok
    
    def _check_market_calm_conditions(self):
        """Check market calm conditions (VIX)"""
        try:
            vix_df = self.broker.get_historical_data(3, "VIXY")
            if vix_df.empty:
                return False
            
            vix_close = vix_df['close']
            vix_threshold = self.alpha_score_config.market_calm.conditions.vix_threshold.threshold
            
            # VIX < threshold and dropping
            vix_low = vix_close.iloc[-1] < vix_threshold
            vix_dropping = len(vix_close) >= 4 and vix_close.iloc[-1] < vix_close.iloc[-4]
            
            return vix_low and vix_dropping
        except:
            return False
    
    def perform_additional_checks(self):
        """Perform additional checks after Alpha Score calculation"""
        if self.score < creds.RISK_CONFIG.alpha_score_threshold:
            self.additional_checks_passed = False
            return
        
        if '3min' not in self.data or '3min' not in self.indicators:
            self.additional_checks_passed = False
            return
        
        # Check +2x volume
        recent_volume = self.data['3min']['volume'].iloc[-1]
        avg_volume = self.indicators['3min']['volume_avg'].iloc[-1]
        volume_multiplier = self.additional_checks_config.volume_multiplier
        
        volume_check = recent_volume > volume_multiplier * avg_volume
        print(f"{'Passed' if volume_check else 'Failed'} +{volume_multiplier}x volume check {'passed' if volume_check else 'failed'}")
        
        # Check VWAP slope
        vwap_slope_check = self._check_vwap_slope()
        
        self.additional_checks_passed = volume_check and vwap_slope_check
        
        print(f"\nAdditional Checks Passed: {self.additional_checks_passed}")
        
        # Update database with additional checks status
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
        slope_ok = vwap_slope > threshold
        
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
        capital_for_trade = account_equity * risk_per_trade
        
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
        data_3min = self.broker.get_historical_data(3, self.stock)
        vwap_value = vwap.calc_vwap(data_3min).iloc[-1]
        
        # Check if VWAP is below current price (required condition)
        if vwap_value >= current_price:
            print(f"WARNING: VWAP (${vwap_value:.2f}) is NOT below current price (${current_price:.2f}) - skipping order")
            if not creds.VWAP_SHOULD_BE_BELOW_PRICE:
                vwap_value = current_price
        
        offset_pct = random.uniform(creds.ORDER_CONFIG.limit_offset_min, creds.ORDER_CONFIG.limit_offset_max)  # 0.03% to 0.07%
        limit_price = vwap_value * (1 + offset_pct)  # Slightly above VWAP for buy orders
        
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
        data_3min = self.broker.get_historical_data(3, self.stock)
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
                
                # Get current time in USA Central Time
                central_tz = pytz.timezone('America/Chicago')
                curr_time = datetime.now(central_tz)
                target_time = curr_time + timedelta(seconds=creds.ORDER_CONFIG.order_window)
                
                while curr_time < target_time:
                    self.current_price = self.broker.get_current_price(self.stock)
                    print(f"Current price: {self.current_price}")
                    
                    # Check if current_price is None before comparison
                    if self.current_price is None:
                        print(f"Unable to get current price for {self.stock}, retrying...")
                        time.sleep(1)
                        continue
                    
                    if self.current_price >= limit_price:                
                        with self.manager.manager_lock:
                            self.used_capital += shares * limit_price
                            self.used_margin = shares * limit_price
                            print(f"Used Margin: ${self.used_margin:.2f}")
                            print(f"Used Capital: ${self.used_capital:.2f}")
                        
                        # Initialize position tracking variables
                        self.entry_price = limit_price
                        self.stop_loss_price = stop_loss
                        self.take_profit_price = entry_price * (1 + creds.PROFIT_CONFIG.profit_booking_levels[0].gain)  # Use 1% from profit booking levels
                        self.position_shares = shares
                        self.position_active = True
                        
                        # Reset profit booking and trailing stop levels for new position
                        self.profit_booking_levels_remaining = list(creds.PROFIT_CONFIG.profit_booking_levels)
                        self.trailing_stop_levels_remaining = list(creds.STOP_LOSS_CONFIG.trailing_stop_levels)
                        print(f"[{self.stock}] Profit booking and trailing stop levels reset for new position")
                        # Get current time in USA Central Time
                        central_tz = pytz.timezone('America/Chicago')
                        self.entry_time = datetime.now(central_tz)
                        self.current_price = limit_price
                        self.unrealized_pnl = 0.0
                        self.realized_pnl = 0.0
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
                
                        
                        return shares, entry_price, stop_loss, limit_price
                    else:
                        print(f"Current price: {self.current_price} is not >= limit price: {limit_price}")
                        time.sleep(1)
                        continue
                else:
                    print("Order not filled on time")    
                    return -1
                
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
            current_time = datetime.now(central_tz)
            current_time_str = current_time.strftime("%H:%M")
            
            # 3:30 PM - Systematic close of weak positions
            if current_time_str == "15:30":
                self.end_of_day_weak_exit()
            
            # 3:35 PM - Safety exit all positions
            elif current_time_str == "15:35":
                self.safety_exit_all()
            
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
                    time.sleep(60*3)  # Wait 3 minutes before checking again
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
        realized_pnl = (current_price - self.entry_price) * shares_to_sell
        
        # Update remaining shares
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
            self.unrealized_pnl = (current_price - self.entry_price) * self.position_shares
            
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
    
    def __init__(self):
        self.broker = SchwabBroker()
        # Initialize ForwardBroker for order execution
        self.forward_broker = ForwardBroker(
            initial_balance=creds.EQUITY,
            spread=0.0,
            commission_fixed=0.0,
            commission_rel=0.0
        )
        self.initial_balance = creds.EQUITY
    
    def get_historical_data(self, num, stock):
        try:
            if num in [1, 5, 10, 15, 30]:            
                data = self.broker.get_price_history(
                    symbol=stock,
                    period_type="day",
                    period=1,
                    frequency_type="minute",
                    frequency=num,
                    need_extended_hours_data=False
                )
                
                print(f"DEBUG: Raw data for {stock}: {type(data)}")
                if data:
                    print(f"DEBUG: Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    if isinstance(data, dict) and "candles" in data:
                        print(f"DEBUG: Candles data type: {type(data['candles'])}, length: {len(data['candles']) if data['candles'] else 0}")
                        if data["candles"]:
                            df = pd.DataFrame(data["candles"])
                            print(f"DEBUG: DataFrame columns: {list(df.columns)}")
                            print(f"DEBUG: DataFrame shape: {df.shape}")
                            
                            # Check if datetime column exists
                            if "datetime" in df.columns:
                                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
                                # Convert UTC datetime to US Eastern Time
                                df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                                df.set_index("datetime", inplace=True)
                                return df
                            else:
                                print(f"ERROR: No 'datetime' column found. Available columns: {list(df.columns)}")
                                return pd.DataFrame()
                        else:
                            print(f"ERROR: Empty candles data for {stock}")
                            return pd.DataFrame()
                    else:
                        print(f"ERROR: No 'candles' key in data for {stock}")
                        return pd.DataFrame()
                else:
                    print(f"ERROR: No data received from broker for {stock}")
                    return pd.DataFrame()
            else:
                data = self.broker.get_price_history(
                    symbol=stock,
                    period_type="day",
                    period=1,
                    frequency_type="minute",
                    frequency=1,
                    need_extended_hours_data=False
                )

                if data and "candles" in data and data["candles"]:
                    df = pd.DataFrame(data["candles"])
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
                        # Convert UTC datetime to US Eastern Time
                        df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                        df.set_index("datetime", inplace=True)

                        # Resample to 3-minute candles
                        df_min = df.resample(f"{num}min").agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()

                        return df_min
                    else:
                        print(f"ERROR: No 'datetime' column found for resampling {stock}")
                        return pd.DataFrame()
                else:
                    print(f"ERROR: No candles data received from broker for {stock}")
                    return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: Exception in get_historical_data for {stock}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str):
        return self.broker.get_current_price(symbol)
    
    def place_order(self, symbol: str, qty: int, order_type: str = 'MARKET', 
                   price: float = None, stop_loss: float = None, take_profit: float = None, side: str = 'BUY'):
        """
        Place an order using the ForwardBroker
        
        Args:
            symbol: Stock symbol
            qty: Number of shares (positive for buy, negative for sell)
            order_type: 'MARKET' or 'LIMIT'
            price: Limit price (required for LIMIT orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            tuple: (order_id, executed_price) or (None, None) if failed
        """
        try:
            # Ensure symbol is being tracked
            self.forward_broker.add_symbol(symbol)
            
            # Use the provided side parameter
            abs_qty = abs(qty)
            
            # Create order parameters
            order_params = {
                'symbol': symbol,
                'side': side,
                'quantity': abs_qty,
                'ordertype': order_type,
                'price': price or 0,
                'stoploss': stop_loss or 0,
                'take_profit': take_profit or 0
            }
            
            # Place order through ForwardBroker
            order_id, executed_price = self.forward_broker.place_order(order_params)
            
            if order_id:
                print("ORDER PLACED SUCCESSFULLY:")
                print(f"  - Order ID: {order_id}")
                print(f"  - Symbol: {symbol}")
                print(f"  - Side: {side}")
                print(f"  - Quantity: {abs_qty}")
                print(f"  - Order Type: {order_type}")
                if price:
                    print(f"  - Limit Price: ${price:.2f}")
                if executed_price:
                    print(f"  - Executed Price: ${executed_price:.2f}")
                if stop_loss:
                    print(f"  - Stop Loss: ${stop_loss:.2f}")
                if take_profit:
                    print(f"  - Take Profit: ${take_profit:.2f}")
            
            return order_id, executed_price
            
        except Exception as e:
            print(f"ERROR PLACING ORDER for {symbol}: {str(e)}")
            return None, None
    
    def get_positions(self):
        """Get current positions from ForwardBroker"""
        return self.forward_broker.get_positions()
    
    def get_trades(self):
        """Get all trades from ForwardBroker"""
        return self.forward_broker.get_all_trades()
    
    def get_broker_summary(self):
        """Get broker summary from ForwardBroker"""
        return {
            'cash': self.forward_broker.cash,
            'equity': self.forward_broker.equity,
            'margin_used': self.forward_broker.margin_used,
            'free_margin': self.forward_broker.free_margin,
            'total_pnl': self.forward_broker.total_pnl()
        }

    def get_total_pnl(self):
        """Get total PnL from ForwardBroker"""
        return self.forward_broker.total_pnl()

    def filled_check(self, symbol: str):
        time.sleep(3)
        return self.forward_broker.filled_check(symbol)

class StrategyManager:
    
    def __init__(self):
        self.broker = StrategyBroker()
        self.config = creds
        self.selector = StockSelector()
        self.manager_lock = threading.Lock()
        self.used_capital = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        # Create stop event BEFORE starting the thread
        self.stop_event = threading.Event()

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
        # self.stocks_list = ["AAPL", "MSFT", "NVDA"]
        
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
        drawdown_thread.start()
    
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

    def run(self):
        self.threads = []
        self.strategies = []
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
        vix_df = self.broker.get_historical_data(5, "APP")
        print(vix_df)
        exit()

    
    def print_broker_summary(self):
        """Print broker summary including positions and PnL"""
        broker_summary = self.broker.get_broker_summary()
        
        print(f"\n=== BROKER SUMMARY ===")
        print(f"Cash: ${broker_summary.cash:.2f}")
        print(f"Equity: ${broker_summary.equity:.2f}")
        print(f"Margin Used: ${broker_summary.margin_used:.2f}")
        print(f"Free Margin: ${broker_summary.free_margin:.2f}")
        print(f"Total PnL: ${broker_summary.total_pnl:.2f}")
        
        # Print open positions
        positions = self.broker.get_positions()
        if positions:
            print(f"\n=== OPEN POSITIONS ({len(positions)}) ===")
            for symbol, pos in positions.items():
                if pos.qty > 0:  # Only show open positions
                    print(f"{symbol}: {pos.qty} shares @ ${pos.avg_price:.2f}")
                    print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")
                    print(f"  Realized PnL: ${pos.realized_pnl:.2f}")
                    print(f"  Stop Loss: ${pos.sl:.2f}")
                    print(f"  Take Profit: ${pos.tp:.2f}")
                    print()
        else:
            print("No open positions")
        
        # Print recent trades
        trades = self.broker.get_trades()
        if trades:
            print(f"\n=== RECENT TRADES ({len(trades)}) ===")
            for trade in trades[-5:]:  # Show last 5 trades
                print(f"Order {trade.order_id}: {trade.symbol} {trade.qty} @ ${trade.exec_price:.2f}")
                print(f"  PnL: ${trade.pnl:.2f}, Commission: ${trade.commission:.2f}")
                print(f"  Time: {trade.timestamp}")
                print()

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
    fetch_marketcap_csv()
    manager = StrategyManager()
    manager.run()
    print("STOPPING MANAGER")
    