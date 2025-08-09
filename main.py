import creds
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
setup_logger()


class Strategy:
    
    def __init__(self, manager, stock, broker, config):
        self.manager = manager
        self.stock = stock
        self.broker = broker
        self.config = config
        self.score = 0
        self.additional_checks_passed = False
        
        # Load configurations from creds.py
        self.indicators_config = creds.INDICATORS
        self.alpha_score_config = creds.ALPHA_SCORE_CONFIG
        self.additional_checks_config = creds.ADDITIONAL_CHECKS_CONFIG
        self.risk_config = creds.RISK_CONFIG
        self.stop_loss_config = creds.STOP_LOSS_CONFIG
        self.profit_config = creds.PROFIT_CONFIG
        self.order_config = creds.ORDER_CONFIG
        self.trading_hours = creds.TRADING_HOURS
        self.data_config = creds.DATA_CONFIG
        
        self.data = {}
        self.indicators = {}
        self.position_active = False
        self.position_shares = 0
        self.current_price = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.used_margin = 0
        self.pnl = 0
        self.entry_time = None
        self.close_time = None
        self.trailing_exit_monitoring = False
        
        # Load existing data from database if available
        self.load_from_database()
    
    def is_entry_time_window(self):
        """Check if current time is within entry time windows"""
        # Get current time in USA Eastern Time
        eastern_tz = pytz.timezone(self.trading_hours['timezone'])
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        # Get entry windows from configuration
        morning_start = self.trading_hours['morning_entry_start']
        morning_end = self.trading_hours['morning_entry_end']
        afternoon_start = self.trading_hours['afternoon_entry_start']
        afternoon_end = self.trading_hours['afternoon_entry_end']
        
        # Check if current time is in either window
        in_morning_window = morning_start <= current_time_str <= morning_end
        in_afternoon_window = afternoon_start <= current_time_str <= afternoon_end
        
        return in_morning_window or in_afternoon_window
    
    def get_next_entry_window(self):
        """Get information about the next entry window"""
        eastern_tz = pytz.timezone(self.trading_hours['timezone'])
        current_time = datetime.now(eastern_tz)
        current_time_str = current_time.strftime("%H:%M")
        
        # Get entry windows from configuration
        morning_start = self.trading_hours['morning_entry_start']
        morning_end = self.trading_hours['morning_entry_end']
        afternoon_start = self.trading_hours['afternoon_entry_start']
        afternoon_end = self.trading_hours['afternoon_entry_end']
        market_close = self.trading_hours['market_close']
        
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
    
    def load_from_database(self):
        """Load existing strategy data from database and ensure boolean values are properly set"""
        try:
            db_data = trades_db.get_latest_strategy_data(self.stock)
            if db_data:
                # Ensure boolean fields are properly converted
                self.position_active = bool(db_data.get('position_active', False))
                self.additional_checks_passed = bool(db_data.get('additional_checks_passed', False))
                
                # Load other fields
                self.score = db_data.get('score', 0)
                self.position_shares = db_data.get('position_shares', 0)
                self.current_price = db_data.get('current_price', 0)
                self.entry_price = db_data.get('entry_price', 0)
                self.stop_loss_price = db_data.get('stop_loss_price', 0)
                self.take_profit_price = db_data.get('take_profit_price', 0)
                self.used_margin = db_data.get('used_margin', 0)
                self.pnl = db_data.get('pnl', 0)
                
                # Convert timestamp strings back to datetime objects if needed
                if db_data.get('entry_time'):
                    try:
                        self.entry_time = datetime.fromisoformat(db_data['entry_time'])
                    except (ValueError, TypeError):
                        self.entry_time = None
                
                if db_data.get('close_time'):
                    try:
                        self.close_time = datetime.fromisoformat(db_data['close_time'])
                    except (ValueError, TypeError):
                        self.close_time = None
                
                print(f"Loaded existing data for {self.stock}: position_active={self.position_active}, additional_checks_passed={self.additional_checks_passed}")
        except Exception as e:
            print(f"Error loading data from database for {self.stock}: {e}")
            # Ensure boolean fields are properly initialized
            self.position_active = False
            self.additional_checks_passed = False
        
    
    def fetch_data_by_timeframe(self):
        """Fetch and store data for each configured timeframe"""
        # Get all unique timeframes from indicator configurations
        timeframes_needed = set()
        for indicator_name, indicator_config in self.indicators_config.items():
            timeframes_needed.update(indicator_config['timeframes'])
        
        # Fetch data for each needed timeframe
        for tf_name in timeframes_needed:
            # Extract period number from timeframe name (e.g., '3min' -> 3)
            period = int(tf_name.replace('min', ''))
            data = self.broker.get_historical_data(period, self.stock)
            
            if not data.empty:
                self.data[tf_name] = data
                print(f"Fetched {tf_name} data: {len(data)} candles")
            else:
                print(f"Failed to fetch {tf_name} data")
                self.data[tf_name] = pd.DataFrame()
    
    def calculate_indicators_by_timeframe(self):
        """Calculate indicators for each timeframe separately"""
        # Initialize indicators storage by timeframe
        for tf_name in self.data.keys():
            self.indicators[tf_name] = {}
        
        # Calculate each indicator on its specified timeframes
        for indicator_name, indicator_config in self.indicators_config.items():
            for tf_name in indicator_config['timeframes']:
                if tf_name not in self.data or self.data[tf_name].empty:
                    continue
                
                data = self.data[tf_name]
                params = indicator_config['params']
                
                # Calculate indicator based on type
                if indicator_name == 'vwap':
                    self.indicators[tf_name]['vwap'] = vwap.calc_vwap(data)
                
                elif indicator_name == 'macd':
                    self.indicators[tf_name]['macd'] = macd.calc_macd(
                        data, 
                        fast=params.get('fast', 12),
                        slow=params.get('slow', 26),
                        signal=params.get('signal', 9)
                    )
                
                elif indicator_name == 'adx':
                    self.indicators[tf_name]['adx'] = adx.calc_adx(
                        data, 
                        length=params.get('length', 14)
                    )
                
                elif indicator_name == 'ema1':
                    ema_length = params.get('length', 5)
                    self.indicators[tf_name]['ema1'] = ema.calc_ema(data, length=ema_length)
                
                elif indicator_name == 'ema2':
                    ema_length = params.get('length', 20)
                    self.indicators[tf_name]['ema2'] = ema.calc_ema(data, length=ema_length)
                
                elif indicator_name == 'volume_avg':
                    self.indicators[tf_name]['volume_avg'] = data['volume'].rolling(
                        window=params.get('window', 20)
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
            self.score += self.alpha_score_config['trend']['weight']
            print(f"Score +{self.alpha_score_config['trend']['weight']} (Trend conditions met)")
        
        # Momentum analysis (20%)
        if self._check_momentum_conditions():
            self.score += self.alpha_score_config['momentum']['weight']
            print(f"Score +{self.alpha_score_config['momentum']['weight']} (Momentum conditions met)")
        
        # Volume/Volatility analysis (20%)
        if self._check_volume_volatility_conditions():
            self.score += self.alpha_score_config['volume_volatility']['weight']
            print(f"Score +{self.alpha_score_config['volume_volatility']['weight']} (Volume/Volatility conditions met)")
        
        # News analysis (15%) - placeholder
        self.score += self.alpha_score_config['news']['weight']
        print(f"Score +{self.alpha_score_config['news']['weight']} (News check - placeholder)")
        
        # Market Calm analysis (15%)
        if self._check_market_calm_conditions():
            self.score += self.alpha_score_config['market_calm']['weight']
            print(f"Score +{self.alpha_score_config['market_calm']['weight']} (Market Calm conditions met)")
        
        print(f"\nFinal Alpha Score: {self.score}")
        
        # Update database with alpha score
        trades_db.update_strategy_data(self.stock, score=self.score)
    
    def _check_trend_conditions(self):
        """Check trend conditions"""
        if '3min' not in self.data or '5min' not in self.indicators or '20min' not in self.indicators:
            return False
        
        # Price > VWAP
        price_vwap_ok = (self.data['3min']['close'].iloc[-1] > self.indicators['3min']['vwap'].iloc[-1])
        
        # EMA1 > EMA2 (5-min EMA > 20-min EMA)
        ema_cross_ok = (self.indicators['5min']['ema1'].iloc[-1] > self.indicators['20min']['ema2'].iloc[-1])
        
        return price_vwap_ok and ema_cross_ok
    
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
        multiplier = self.alpha_score_config['volume_volatility']['conditions']['volume_spike']['multiplier']
        volume_ok = recent_volume > multiplier * avg_volume
        
        # ADX threshold check
        adx_threshold = self.alpha_score_config['volume_volatility']['conditions']['adx_threshold']['threshold']
        adx_ok = self.indicators['3min']['adx'].iloc[-1] > adx_threshold
        
        return volume_ok and adx_ok
    
    def _check_market_calm_conditions(self):
        """Check market calm conditions (VIX)"""
        try:
            vix_df = self.broker.get_historical_data(3, "VIX")
            if vix_df.empty:
                return False
            
            vix_close = vix_df['close']
            vix_threshold = self.alpha_score_config['market_calm']['conditions']['vix_threshold']['threshold']
            
            # VIX < threshold and dropping
            vix_low = vix_close.iloc[-1] < vix_threshold
            vix_dropping = len(vix_close) >= 4 and vix_close.iloc[-1] < vix_close.iloc[-4]
            
            return vix_low and vix_dropping
        except:
            return False
    
    def perform_additional_checks(self):
        """Perform additional checks after Alpha Score calculation"""
        if self.score < creds.RISK_CONFIG['alpha_score_threshold']:
            self.additional_checks_passed = False
            return
        
        if '3min' not in self.data or '3min' not in self.indicators:
            self.additional_checks_passed = False
            return
        
        # Check +2x volume
        recent_volume = self.data['3min']['volume'].iloc[-1]
        avg_volume = self.indicators['3min']['volume_avg'].iloc[-1]
        volume_multiplier = self.additional_checks_config['volume_multiplier']
        
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
        vwap_slope = (current_vwap - vwap_3min_ago) / self.additional_checks_config['vwap_slope_period']
        
        threshold = self.additional_checks_config['vwap_slope_threshold']
        slope_ok = vwap_slope > threshold
        
        print(f"{'Passed' if slope_ok else 'Failed'} VWAP slope check {'passed' if slope_ok else 'failed'}: {vwap_slope:.3f} {'>' if slope_ok else '<='} {threshold}")
        
        return slope_ok 
        
    def calculate_position_size(self):
        """Calculate position size based on risk management rules"""
        
        # Get account equity from config
        account_equity = creds.EQUITY
        
        # Risk per trade: 0.4% of equity
        risk_per_trade = creds.RISK_CONFIG['risk_per_trade']
        
        # Get current price and calculate stop loss
        current_price = self.broker.get_current_price(self.stock)
        stop_loss_pct = self.calculate_stop_loss(current_price)
        stop_loss_price = current_price * (1 - stop_loss_pct)
        
        # Calculate shares based on risk
        risk_per_share = current_price - stop_loss_price
        if risk_per_share <= 0:
            return 0, 0, 0, 0
        
        shares = int(risk_per_trade / risk_per_share)
        
        # Apply position size limits (up to 10% equity)
        max_shares_by_equity = int((account_equity * creds.RISK_CONFIG['max_position_size']) / current_price)
        shares = max(shares, max_shares_by_equity)
        
        # # Apply micro-lot sizing (4-5% chunks)
        # micro_lot_size = int(shares * 0.04)  # 4% chunk
        # if micro_lot_size < 1:
        #     micro_lot_size = 1
        
        # shares = micro_lot_size
        
        # Calculate limit order price: VWAP+/- (0.03%-0.07% random)
        # Ensure VWAP is below current price before using it
        data_3min = self.broker.get_historical_data(3, self.stock)
        vwap_value = vwap.calc_vwap(data_3min).iloc[-1]
        
        # Check if VWAP is below current price (required condition)
        if vwap_value >= current_price:
            print(f"WARNING: VWAP (${vwap_value:.2f}) is NOT below current price (${current_price:.2f}) - skipping order")
            return 0, 0, 0, 0  # Don't place order if VWAP condition not met
        
        offset_pct = random.uniform(creds.ORDER_CONFIG['limit_offset_min'], creds.ORDER_CONFIG['limit_offset_max'])  # 0.03% to 0.07%
        limit_price = vwap_value * (1 + offset_pct)  # Slightly above VWAP for buy orders
        
        print(f"Position Size: {shares} shares")
        print(f"Entry Price: ${current_price:.2f}")
        print(f"Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%)")
        print(f"VWAP: ${vwap_value:.2f}")
        print(f"Limit Order Price: ${limit_price:.2f} (VWAP + {offset_pct*100:.3f}%)")
        
        # Update database with current price and position calculations
        trades_db.update_strategy_data(self.stock, 
            current_price=current_price,
            stop_loss_price=stop_loss_price
        )
        
        return shares, current_price, stop_loss_price, limit_price
    
    def calculate_stop_loss(self, current_price):
        """Calculate stop loss percentage based on volatility"""
        data_3min = self.broker.get_historical_data(3, self.stock)
        atr14 = atr.calc_atr(data_3min, creds.STOP_LOSS_CONFIG['atr_period'])
        atr14 = atr14.iloc[-1]

        base_stop = creds.STOP_LOSS_CONFIG['default_stop_loss']
        
        atr_stop = (atr14 * creds.STOP_LOSS_CONFIG['atr_multiplier']) / current_price
        print(f"ATR Stop: {atr_stop:.3f}")
        
        stop_loss = max(base_stop, atr_stop)
        
        stop_loss = min(stop_loss, creds.STOP_LOSS_CONFIG['max_stop_loss'])
        
        return stop_loss

    def process_score(self):
        print(f"Alpha Score: {self.score}")
        print(f"Additional Checks Passed: {self.additional_checks_passed}")
        
        # Both conditions must be met to place an order
        if self.score >= creds.RISK_CONFIG['alpha_score_threshold'] and bool(self.additional_checks_passed):
            print(f"ENTERING POSITION - Both Alpha Score >= {creds.RISK_CONFIG['alpha_score_threshold']} AND additional checks passed")
            shares, entry_price, stop_loss, limit_price = self.calculate_position_size()
            if shares > 0:
                print(f"Order Details:")
                print(f"  - Symbol: {self.stock}")
                print(f"  - Shares: {shares}")
                print(f"  - Limit Price: ${limit_price:.2f}")
                print(f"  - Stop Loss: ${stop_loss:.2f}")
                print(f"  - Exit: Market-On-Close at 4:00 PM ET")
                
                # Get current time in USA Eastern Time
                eastern_tz = pytz.timezone('America/New_York')
                curr_time = datetime.now(eastern_tz)
                target_time = curr_time + timedelta(seconds=creds.ORDER_CONFIG['order_window'])
                
                while curr_time < target_time:
                    self.current_price = self.broker.get_current_price(self.stock)
                    print(f"Current price: {self.current_price}")
                    
                    # Check if current_price is None before comparison
                    if self.current_price is None:
                        print(f"Unable to get current price for {self.stock}, retrying...")
                        time.sleep(1)
                        continue
                    
                    if self.current_price >= limit_price:                
                        stock_data = {
                            'symbol': self.stock,
                            'alpha_score': self.score,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'shares': shares,
                            'limit_price': limit_price,
                            'timestamp': pd.Timestamp.now()
                        }

                        with self.manager_lock:
                            self.used_capital += shares * limit_price
                            self.used_margin = shares * limit_price
                            print(f"Used Margin: ${self.used_margin:.2f}")
                            print(f"Used Capital: ${self.used_capital:.2f}")
                        
                        # Initialize position tracking variables
                        self.entry_price = limit_price
                        self.stop_loss_price = stop_loss
                        self.take_profit_price = entry_price * (1 + creds.PROFIT_CONFIG['profit_booking_levels'][0]['gain'])  # Use 1% from profit booking levels
                        self.position_shares = shares
                        self.position_active = True
                        # Get current time in USA Eastern Time
                        eastern_tz = pytz.timezone('America/New_York')
                        self.entry_time = datetime.now(eastern_tz)
                        self.current_price = limit_price
                        self.pnl = 0.0
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
                            pnl=self.pnl,
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
            if self.score < 85:
                print(f"Alpha Score too low: {self.score} < 85")
            if not bool(self.additional_checks_passed):
                print("Additional checks failed")
        return -1
                    
    def run(self, i):
        while True:
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
                time.sleep(30)  # Wait 30 seconds before checking again
        
        # self.calculate_indicators()
        # self.process_score()
    
    def monitor_position(self):
        """Monitor position for stop loss and take profit conditions"""
        try:
            # Get current price and update tracking variables
            self.current_price = self.broker.get_current_price(self.stock)
            
            # Check if current_price is None
            if self.current_price is None:
                print(f"Unable to get current price for {self.stock} in position monitoring")
                return
            
            # Calculate current PnL
            self.pnl = (self.current_price - self.entry_price) * self.position_shares
            with self.manager.manager_lock:
                self.manager.unrealized_pnl += self.pnl
            
            # Calculate current gain/loss percentage
            current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
            
            # Check stop loss condition
            if self.current_price <= self.stop_loss_price:
                print(f"STOP LOSS TRIGGERED: {self.stock} - Current: ${self.current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
                self.close_position('stop_loss', self.current_price)
                return
            
            self.check_take_profit()

            if self.manager.stop_event.is_set():
                self.close_position('drawdown', self.current_price)

            print(f"Position Status - {self.stock}:")
            print(f"  - Current Price: ${self.current_price:.2f}")
            print(f"  - Entry Price: ${self.entry_price:.2f}")
            print(f"  - PnL: ${self.pnl:.2f} ({current_gain_pct*100:.2f}%)")
            print(f"  - Shares: {self.position_shares}")
            print(f"  - Stop Loss: ${self.stop_loss_price:.2f}")
            print(f"  - Take Profit: ${self.take_profit_price:.2f}")
            
            # Update database with current position status
            trades_db.update_strategy_data(self.stock,
                current_price=self.current_price,
                pnl=self.pnl,
                position_shares=self.position_shares,
                stop_loss_price=self.stop_loss_price,
                take_profit_price=self.take_profit_price
            )
            
        except Exception as e:
            print(f"Error monitoring position for {self.stock}: {e}")

    def check_take_profit(self):
        current_gain_pct = (self.current_price - self.entry_price) / self.entry_price

        # Profit Booking Logic using PROFIT_CONFIG
        for i, level in enumerate(creds.PROFIT_CONFIG['profit_booking_levels']):
            gain_threshold = level['gain']
            exit_pct = level['exit_pct']
            
            # Check if we haven't already booked profit at this level
            if current_gain_pct >= gain_threshold:
                shares_to_sell = int(self.position_shares * exit_pct)
                if shares_to_sell > 0:
                    self.close_position(f'profit_booking_{i+1}', shares_to_sell)
                    print(f"Profit Booking {i+1}: Sold {shares_to_sell} shares at {gain_threshold*100:.1f}% gain")
                    
                    # Update database with profit booking
                    trades_db.update_strategy_data(self.stock,
                        profit_booked_flags={f'profit_booked_{i+1}': True}
                    )
        
        # Trailing Stop Logic using STOP_LOSS_CONFIG
        for i, level in enumerate(creds.STOP_LOSS_CONFIG['trailing_stop_levels']):
            gain_threshold = level['gain']
            new_stop_pct = level['new_stop_pct']
            
            # Check if we haven't already set trailing stop at this level
            if current_gain_pct >= gain_threshold:
                new_stop_price = self.entry_price * (1 + new_stop_pct)
                if new_stop_price > self.stop_loss_price:
                    self.stop_loss_price = new_stop_price
                    print(f"Trailing Stop {i+1}: Moved stop loss to +{new_stop_pct*100:.2f}% (${self.stop_loss_price:.2f})")
                    
                    # Update database with trailing stop
                    trades_db.update_strategy_data(self.stock,
                        stop_loss_price=self.stop_loss_price,
                        trailing_stop_flags={f'trailing_stop_{i+1}_set': True}
                    )

        # Trailing Exit Logic using PROFIT_CONFIG
        trailing_conditions = creds.PROFIT_CONFIG['trailing_exit_conditions']
        gain_threshold = trailing_conditions['gain_threshold']
        drop_threshold = trailing_conditions['drop_threshold']
        monitor_period = trailing_conditions['monitor_period']
        
        if current_gain_pct >= gain_threshold:
            # Start monitoring if not already started
            if not self.trailing_exit_monitoring:
                self.trailing_exit_monitoring = True
                # Get current time in USA Eastern Time
                eastern_tz = pytz.timezone('America/New_York')
                self.trailing_exit_start_time = datetime.now(eastern_tz)
                self.trailing_exit_start_price = self.current_price
                print(f"Starting trailing exit monitoring at {gain_threshold*100:.1f}% gain")
                
                # Update database with trailing exit monitoring
                trades_db.update_strategy_data(self.stock,
                    trailing_exit_monitoring=True,
                    trailing_exit_start_time=self.trailing_exit_start_time,
                    trailing_exit_start_price=self.trailing_exit_start_price
                )
                
                # Start continuous monitoring for sustained drop - inline while loop
                monitor_period_seconds = monitor_period * 60  # Convert minutes to seconds
                print(f"Starting continuous monitoring for {monitor_period_seconds}s sustained drop of {drop_threshold*100:.1f}%")
                drop_sustained_start_time = None
                
                while True:
                    try:
                        # Update current price
                        self.current_price = self.broker.get_current_price(self.stock)
                        if self.current_price is None:
                            print(f"Unable to get current price for {self.stock}, retrying...")
                            time.sleep(1)
                            continue
                        
                        # Calculate current price drop
                        price_drop = (self.trailing_exit_start_price - self.current_price) / self.trailing_exit_start_price
                        current_time = datetime.now(eastern_tz)
                        
                        if price_drop >= drop_threshold:
                            # Price drop meets threshold
                            if drop_sustained_start_time is None:
                                # First time drop threshold is met - start tracking
                                drop_sustained_start_time = current_time
                                print(f"Price drop {price_drop*100:.2f}% >= {drop_threshold*100:.1f}% threshold - starting sustained drop timer")
                            else:
                                # Check if drop has been sustained for the required period
                                sustained_time_seconds = (current_time - drop_sustained_start_time).total_seconds()
                                if sustained_time_seconds >= monitor_period_seconds:
                                    print(f"Trailing Exit: {monitor_period_seconds}s sustained drop of {price_drop*100:.2f}% >= {drop_threshold*100:.1f}%, closing position")
                                    self.close_position('trailing_exit', self.position_shares)
                                    return  # Exit the monitoring loop
                                else:
                                    print(f"Sustained drop for {sustained_time_seconds:.0f}/{monitor_period_seconds}s - continuing to monitor")
                        else:
                            # Price drop is below threshold - reset sustained drop timer
                            if drop_sustained_start_time is not None:
                                print(f"Price drop {price_drop*100:.2f}% < {drop_threshold*100:.1f}% threshold - resetting sustained drop timer")
                                drop_sustained_start_time = None
                        
                        # Check if position is still active (in case it was closed by other means)
                        if not self.position_active:
                            print("Position no longer active - exiting sustained drop monitoring")
                            return
                        
                        # Sleep for 1 second before next check
                        time.sleep(1)
                        
                    except Exception as e:
                        print(f"Error in sustained drop monitoring for {self.stock}: {e}")
                        time.sleep(1)
                        continue
    



    
    def close_position(self, reason, shares_to_sell=None):
        """
        Close position (either partial or full)
        
        Args:
            reason: Reason for closing ('stop_loss', 'profit_booking', 'trailing_exit', etc.)
            current_price: Current market price
            shares_to_sell: Number of shares to sell (None for all remaining shares)
        """

        current_price = self.broker.get_current_price(self.stock)
        
        if shares_to_sell is None:
            shares_to_sell = self.position_shares

        # Ensure we don't sell more than we have
        shares_to_sell = min(shares_to_sell, self.position_shares)

        if shares_to_sell <= 0:
            print(f"No shares to sell for {self.stock}")
            return

        # Update remaining shares
        self.position_shares -= shares_to_sell

        if self.position_shares <= 0:
            self.position_active = False
            # Get current time in USA Eastern Time
            eastern_tz = pytz.timezone('America/New_York')
            self.close_time = datetime.now(eastern_tz)
            self.pnl = (current_price - self.entry_price) * shares_to_sell
            with self.manager.manager_lock:
                self.manager.realized_pnl += self.pnl
                self.manager.unrealized_pnl -= self.pnl

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
                print(f"  - Total PnL: ${self.pnl:.2f}")
                print(f"  - Return: {((current_price - self.entry_price) / self.entry_price * 100):.2f}%")
                print(f"  - Reason: {reason}")
                
                # Update database with position closure
                trades_db.update_strategy_data(self.stock,
                    position_active=False,
                    position_shares=0,
                    close_time=self.close_time,
                    pnl=self.pnl,
                    current_price=current_price
                )
        else:
            print(f"Partial position closed. Remaining shares: {self.position_shares}")
            # Update PnL for remaining position
            self.pnl = (current_price - self.entry_price) * self.position_shares
            with self.manager.manager_lock:
                self.manager.realized_pnl += self.pnl
                
            # Update database with partial position closure
            trades_db.update_strategy_data(self.stock,
                position_shares=self.position_shares,
                pnl=self.pnl,
                current_price=current_price
            ) 
    
    def _check_profit_booking(self, current_gain_pct, current_price):
        """Check profit booking levels"""
        for level in creds.PROFIT_CONFIG['profit_booking_levels']:
            gain_threshold = level['gain']
            exit_pct = level['exit_pct']
            
            if current_gain_pct >= gain_threshold:
                # Calculate shares to sell
                shares_to_sell = int(self.position_shares * exit_pct)
                if shares_to_sell > 0:
                    # Close partial position
                    self.close_position('profit_booking', current_price, shares_to_sell)
                    
                    # Remove this level from future checks
                    creds.PROFIT_CONFIG['profit_booking_levels'].remove(level)
                    break
    
    def _check_trailing_stops(self, current_gain_pct, current_price):
        """Check trailing stop levels"""
        for level in creds.STOP_LOSS_CONFIG['trailing_stop_levels']:
            gain_threshold = level['gain']
            new_stop_pct = level['new_stop_pct']
            
            if current_gain_pct >= gain_threshold:
                # Move stop loss to new level
                new_stop_price = self.entry_price * (1 + new_stop_pct)
                self.stop_loss_price = new_stop_price
                print(f"TRAILING STOP: {self.stock} - Moved SL to ${new_stop_price:.2f} at {gain_threshold*100:.1f}% gain")
                
                # Remove this level from future checks
                creds.STOP_LOSS_CONFIG['trailing_stop_levels'].remove(level)
                break
    
    def _check_trailing_exit(self, current_gain_pct, current_price):
        """Check trailing exit conditions after 5% gain"""
        exit_config = creds.PROFIT_CONFIG['trailing_exit_conditions']
        
        if current_gain_pct >= exit_config['gain_threshold']:
            # Check if we should exit based on price drop
            # Get current time in USA Eastern Time
            eastern_tz = pytz.timezone('America/New_York')
            self.trailing_start_time = datetime.now(eastern_tz)
            self.max_price_since_trailing = current_price
            print(f"TRAILING EXIT MONITORING: {self.stock} - Started monitoring for exit conditions")
            
            # Update max price
            self.max_price_since_trailing = max(self.max_price_since_trailing, current_price)
            
            # Check time and price drop
            # Get current time in USA Eastern Time for calculation
            eastern_tz = pytz.timezone('America/New_York')
            time_since_trailing = datetime.now(eastern_tz) - self.trailing_start_time
            if time_since_trailing.total_seconds() <= exit_config['monitor_period'] * 60:
                price_drop = (self.max_price_since_trailing - current_price) / self.max_price_since_trailing
                if price_drop >= exit_config['drop_threshold']:
                    # Exit entire position
                    self.close_position('trailing_exit', current_price)
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
            'pnl': self.pnl,
            'pnl_pct': ((self.current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0,
            'entry_time': self.entry_time,
            'duration': (datetime.now(pytz.timezone('America/New_York')) - self.entry_time) if self.entry_time else None
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
        self.pnl = 0
        self.entry_time = None
        self.close_time = None
        
        # Update database with reset
        trades_db.update_strategy_data(self.stock,
            position_active=False,
            position_shares=0,
            current_price=0,
            entry_price=0,
            stop_loss_price=0,
            take_profit_price=0,
            pnl=0,
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
        if num in [1, 5, 10, 15, 30]:            
            data = self.broker.get_price_history(
                symbol=stock,
                period_type="day",
                period=1,
                frequency_type="minute",
                frequency=num,
                need_extended_hours_data=False
            )
            
            if data and "candles" in data:
                df = pd.DataFrame(data["candles"])
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

                # Convert UTC datetime to US Eastern Time
                df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
                df.set_index("datetime", inplace=True)

                return df
            else:
                print("No data received from broker")
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

            if data and "candles" in data:
                df = pd.DataFrame(data["candles"])
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
                print("No data received from broker")
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
            print(f"Stock selector returned {len(self.stocks_list)} stocks")
        except Exception as e:
            print(f"Error in stock selector: {e}")
            # Fallback to a default list
            self.stocks_list = ["AAPL", "MSFT", "NVDA"]
            self.stocks_dict = []
        # self.stocks_list = ["AAPL", "MSFT", "NVDA"]
        
        # List to track stocks that pass all entry conditions
        self.qualifying_stocks = []
        self.qualifying_stocks_lock = threading.Lock()  # Thread-safe access
        
        # Start drawdown monitoring thread AFTER everything is initialized
        drawdown_thread = threading.Thread(target=self.monitor_drawdown_loop, name="DrawdownMonitor", daemon=True)
        drawdown_thread.start()
    
    def monitor_drawdown_loop(self):
            print("Starting global drawdown monitoring thread.")
            threshold = creds.EQUITY * creds.RISK_CONFIG['daily_drawdown_limit']
            self.max_drawdown_triggered = False  # Initialize the attribute
            
            while not self.stop_event.is_set():
                try:
                    total_pnl = self.broker.get_total_pnl()  # Must return realized + unrealized

                    print(f"[Drawdown] PnL: {total_pnl:.2f}, Threshold: {-threshold}")
                    # Get current time in USA Eastern Time
                    eastern_tz = pytz.timezone('America/New_York')
                    now = datetime.now(eastern_tz)
                    current_time_str = now.strftime("%H:%M")
                    
                    if total_pnl <= -threshold and not self.max_drawdown_triggered:
                        print(f"Max loss threshold of {-threshold} hit. Stopping all strategies.")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        
                    if current_time_str >= self.config.TRADING_HOURS['market_close']:
                        print(f"Exit time reached at {current_time_str} ET")
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
        """Test the database integration and strategy functionality"""
        print("🧪 Starting Database Integration Test...")
        
        # Test 1: Check if database is initialized with stocks
        print("\n1. Testing Database Initialization...")
        summary_data = trades_db.get_strategy_summary()
        if summary_data:
            print(f"Database initialized with {len(summary_data)} stocks")
            for stock in summary_data[:3]:  # Show first 3 stocks
                print(f"   - {stock['symbol']}: Score={stock['score']}, Active={stock['position_active']}")
        else:
            print("No data found in database")
        
        # Test 2: Create a test strategy and update database
        print("\n2. Testing Strategy Database Updates...")
        test_stock = "META"
        trades_db.add_stocks_from_list([test_stock])
        test_strategy = Strategy(self, test_stock, self.broker, self.config)
        
        # Simulate strategy calculations
        print(f"   Testing with {test_stock}...")
        
        # Test alpha score calculation
        test_strategy.calculate_indicators()
        test_strategy.calculate_alpha_score()
        test_strategy.perform_additional_checks()
        
        print(f"   Alpha Score: {test_strategy.score}")
        print(f"   Additional Checks: {bool(test_strategy.additional_checks_passed)}")
        
        # Test position size calculation
        shares, entry_price, stop_loss, limit_price = test_strategy.calculate_position_size()
        print(f"   Calculated Position: {shares} shares @ ${entry_price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}")
        print(f"   Limit Price: ${limit_price:.2f}")
        
        # Test 3: Check database updates
        print("\n3. Verifying Database Updates...")
        latest_data = trades_db.get_latest_strategy_data(test_stock)
        if latest_data:
            print(f" Database updated for {test_stock}")
            print(f"   - Score: {latest_data['score']}")
            print(f"   - Current Price: ${latest_data['current_price']:.2f}")
            print(f"   - Stop Loss: ${latest_data['stop_loss_price']:.2f}")
            print(f"   - Position Active: {bool(latest_data['position_active'])}")
        else:
            print(f"No database data found for {test_stock}")
        
        # Test 4: Place limit order and wait for price to reach limit
        print("\n4. Placing Limit Order and Waiting for Fill...")
        if shares > 0:
            print(f"   Waiting for price to reach limit price: ${limit_price:.2f}")
            print(f"   Order window: {creds.ORDER_CONFIG['order_window']} seconds")
            
            # Get current time in USA Eastern Time
            eastern_tz = pytz.timezone('America/New_York')
            curr_time = datetime.now(eastern_tz)
            target_time = curr_time + timedelta(seconds=creds.ORDER_CONFIG['order_window'])
            
            order_filled = False
            while curr_time < target_time and not order_filled:
                current_price = test_strategy.broker.get_current_price(test_stock)
                limit_price = current_price 
                print(f"   Current price: ${current_price:.2f} | Limit price: ${limit_price:.2f}")
                
                # Check if current_price is None before comparison
                if current_price is None:
                    print(f"   Unable to get current price for {test_stock}, retrying...")
                    time.sleep(1)
                    curr_time = datetime.now(eastern_tz)
                    continue
                
                if current_price >= limit_price:
                    print(f"ORDER FILLED! Current price ${current_price:.2f} >= Limit price ${limit_price:.2f}")
                    
                    # Initialize position tracking variables (same as process_score)
                    test_strategy.entry_price = limit_price
                    test_strategy.stop_loss_price = stop_loss
                    test_strategy.take_profit_price = entry_price * (1 + creds.PROFIT_CONFIG['profit_booking_levels'][0]['gain'])
                    test_strategy.position_shares = shares
                    test_strategy.position_active = True
                    test_strategy.entry_time = datetime.now(eastern_tz)
                    test_strategy.current_price = limit_price
                    test_strategy.pnl = 0.0
                    
                    # Update database with position initialization
                    trades_db.update_strategy_data(test_stock,
                        position_active=True,
                        position_shares=shares,
                        entry_price=test_strategy.entry_price,
                        stop_loss_price=test_strategy.stop_loss_price,
                        take_profit_price=test_strategy.take_profit_price,
                        entry_time=test_strategy.entry_time,
                        current_price=test_strategy.current_price,
                        pnl=test_strategy.pnl
                    )
                    
                    print(f"   Position initialized for {test_stock}:")
                    print(f"   - Entry Price: ${test_strategy.entry_price:.2f}")
                    print(f"   - Stop Loss: ${test_strategy.stop_loss_price:.2f}")
                    print(f"   - Take Profit: ${test_strategy.take_profit_price:.2f}")
                    print(f"   - Entry Time: {test_strategy.entry_time}")
                    
                    order_filled = True
                    break
                else:
                    print(f"   Waiting... Current price ${current_price:.2f} < Limit price ${limit_price:.2f}")
                    time.sleep(1)
                    curr_time = datetime.now(eastern_tz)
            
            if not order_filled:
                print("Order not filled within time window")
                return
        
        # Test 5: Check active positions
        print("\n5. Testing Active Positions Query...")
        active_positions = trades_db.get_all_active_positions()
        if active_positions:
            print(f"Found {len(active_positions)} active positions")
            for pos in active_positions:
                print(f"   - {pos['symbol']}: {pos['position_shares']} shares @ ${pos['entry_price']:.2f}")
        else:
            print("No active positions found")
        
        # Test 6: Simulate price movement and profit booking
        print("\n6. Testing Profit Booking Logic...")
        if shares > 0:
            # Simulate price increase to trigger profit booking
            test_strategy.current_price = entry_price * 1.02  # 2% gain
            test_strategy.pnl = (test_strategy.current_price - test_strategy.entry_price) * shares
            
            print(f"   Simulating 2% price increase...")
            print(f"   New Price: ${test_strategy.current_price:.2f}")
            print(f"   PnL: ${test_strategy.pnl:.2f}")
            
            # Update database with new price
            trades_db.update_strategy_data(test_stock,
                current_price=test_strategy.current_price,
                pnl=test_strategy.pnl
            )
            
            # Test profit booking logic
            test_strategy.check_take_profit()
        
        # Test 7: Final database summary
        print("\n7. Final Database Summary...")
        final_summary = trades_db.get_strategy_summary()
        if final_summary:
            test_stock_data = next((s for s in final_summary if s['symbol'] == test_stock), None)
            if test_stock_data:
                print(f" Final data for {test_stock}:")
                print(f"   - Score: {test_stock_data['score']}")
                print(f"   - Position Active: {bool(test_stock_data['position_active'])}")
                print(f"   - Current Price: ${test_stock_data['current_price']:.2f}")
                print(f"   - PnL: ${test_stock_data['pnl']:.2f}")
        
        print("\n Database Integration Test Complete!")
        print(" Run 'streamlit run db_viewer.py' to view the dashboard")
        
        # Test 8: Test stop loss and take profit functionality
        print("\n8. Testing Stop Loss and Take Profit...")
        if test_strategy.position_active:
            print(f"   Testing stop loss at ${test_strategy.stop_loss_price:.2f}")
            print(f"   Testing take profit at ${test_strategy.take_profit_price:.2f}")
            
            # Simulate price movement to test SL/TP
            test_strategy.current_price = test_strategy.stop_loss_price - 0.01  # Just below stop loss
            test_strategy.pnl = (test_strategy.current_price - test_strategy.entry_price) * shares
            
            print(f"   Simulated price: ${test_strategy.current_price:.2f}")
            print(f"   PnL: ${test_strategy.pnl:.2f}")
            
            # Update database with new price
            trades_db.update_strategy_data(test_stock,
                current_price=test_strategy.current_price,
                pnl=test_strategy.pnl
            )
            
            # Test stop loss logic
            if test_strategy.current_price <= test_strategy.stop_loss_price:
                print(f"Stop loss would trigger at this price")
            else:
                print(f"Stop loss not triggered yet")
            
            # Test take profit logic
            test_strategy.check_take_profit()
        else:
            print("No active position to test SL/TP")

    
    def print_broker_summary(self):
        """Print broker summary including positions and PnL"""
        broker_summary = self.broker.get_broker_summary()
        
        print(f"\n=== BROKER SUMMARY ===")
        print(f"Cash: ${broker_summary['cash']:.2f}")
        print(f"Equity: ${broker_summary['equity']:.2f}")
        print(f"Margin Used: ${broker_summary['margin_used']:.2f}")
        print(f"Free Margin: ${broker_summary['free_margin']:.2f}")
        print(f"Total PnL: ${broker_summary['total_pnl']:.2f}")
        
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
    manager = StrategyManager()
    manager.run()
    print("STOPPING MANAGER")
    