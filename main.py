import creds
import pandas as pd
from stock_selector import StockSelector
from simulation.schwab_broker import SchwabBroker
from simulation.forward_broker import ForwardBroker
import random
import threading
import time
from datetime import datetime, timedelta
from helpers import vwap, ema, macd, adx, atr
from log import setup_logger
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
        
        # # Save all data to CSV for analysis
        # self.save_data_to_csv()
        
        # Calculate Alpha Score
        self.calculate_alpha_score()
        
        self.perform_additional_checks()
    
    def save_data_to_csv(self):
        """Save all timeframe data and indicators to CSV files"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        for tf_name in self.data.keys():
            if not self.data[tf_name].empty:
                # Save raw data
                data_filename = f"data_{self.stock}_{tf_name}_{timestamp}.csv"
                self.data[tf_name].to_csv(data_filename, index=True)
                print(f"Raw data saved: {data_filename}")
                
                # Save indicators
                if tf_name in self.indicators:
                    indicators_df = pd.DataFrame(self.indicators[tf_name])
                    indicators_filename = f"indicators_{self.stock}_{tf_name}_{timestamp}.csv"
                    indicators_df.to_csv(indicators_filename, index=True)
                    print(f"Indicators saved: {indicators_filename}")
    
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
        max_shares_by_equity = int((account_equity * 0.10) / current_price)
        shares = max(shares, max_shares_by_equity)
        
        # # Apply micro-lot sizing (4-5% chunks)
        # micro_lot_size = int(shares * 0.04)  # 4% chunk
        # if micro_lot_size < 1:
        #     micro_lot_size = 1
        
        # shares = micro_lot_size
        
        # Calculate limit order price: VWAP+/- (0.03%-0.07% random)
        data_3min = self.broker.get_historical_data(3, self.stock)
        vwap_value = vwap.calc_vwap(data_3min).iloc[-1]
        
        offset_pct = random.uniform(0.0003, 0.0007)  # 0.03% to 0.07%
        limit_price = vwap_value * (1 + offset_pct)  # Slightly above VWAP for buy orders
        
        print(f"Position Size: {shares} shares")
        print(f"Entry Price: ${current_price:.2f}")
        print(f"Stop Loss: ${stop_loss_price:.2f} ({stop_loss_pct*100:.1f}%)")
        print(f"VWAP: ${vwap_value:.2f}")
        print(f"Limit Order Price: ${limit_price:.2f} (VWAP + {offset_pct*100:.3f}%)")
        
        return shares, current_price, stop_loss_price, limit_price
    
    def calculate_stop_loss(self, current_price):
        """Calculate stop loss percentage based on volatility"""
        data_3min = self.broker.get_historical_data(3, self.stock)
        atr14 = atr.calc_atr(data_3min)
        atr14 = atr14.iloc[-1]

        base_stop = 0.015
        
        atr_stop = (atr14 * 1.5) / current_price
        print(f"ATR Stop: {atr_stop:.3f}")
        
        stop_loss = max(base_stop, atr_stop)
        
        stop_loss = min(stop_loss, 0.04)
        
        return stop_loss

    def process_score(self):
        print(f"Alpha Score: {self.score}")
        print(f"Additional Checks Passed: {self.additional_checks_passed}")
        
        # Both conditions must be met to place an order
        if self.score >= creds.RISK_CONFIG['alpha_score_threshold'] and self.additional_checks_passed:
            print("ENTERING POSITION - Both Alpha Score >= 85 AND additional checks passed")
            shares, entry_price, stop_loss, limit_price = self.calculate_position_size()
            if shares > 0:
                print(f"Order Details:")
                print(f"  - Symbol: {self.stock}")
                print(f"  - Shares: {shares}")
                print(f"  - Limit Price: ${limit_price:.2f}")
                print(f"  - Stop Loss: ${stop_loss:.2f}")
                print(f"  - Exit: Market-On-Close at 4:00 PM ET")
                
                # Add to qualifying stocks list in manager
                stock_data = {
                    'symbol': self.stock,
                    'alpha_score': self.score,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'shares': shares,
                    'limit_price': limit_price,
                    'timestamp': pd.Timestamp.now()
                }
                self.manager.add_qualifying_stock(stock_data)
                
                # Place actual order using StrategyBroker
                try:
                    # Place limit buy order with SL and TP as 0 (will be managed by monitoring system)
                    order_id, executed_price = self.broker.place_order(
                        symbol=self.stock,
                        qty=shares,
                        order_type='LIMIT',
                        price=limit_price,  # Use limit_price for limit order
                        stop_loss=0,  # SL will be managed by monitoring system
                        take_profit=0  # TP will be managed by monitoring system
                    )
                    
                    if order_id:
                        print("ORDER PLACED SUCCESSFULLY:")
                        print(f"  - Order ID: {order_id}")
                        print(f"  - Symbol: {self.stock}")
                        print(f"  - Shares: {shares}")
                        print(f"  - Limit Price: ${limit_price:.2f}")
                        print(f"  - Stop Loss: Managed by monitoring system")
                        print(f"  - Take Profit: Managed by monitoring system")

                        with self.manager_lock:
                            self.used_capital += shares * limit_price
                            self.used_margin = shares * limit_price
                            print(f"Used Margin: ${self.used_margin:.2f}")
                            print(f"Used Capital: ${self.used_capital:.2f}")
                        
                        # Initialize position tracking variables
                        self.entry_price = limit_price
                        self.stop_loss_price = stop_loss
                        self.take_profit_price = entry_price * (1 + creds.PROFIT_CONFIG.get('take_profit_pct', 0.10))  # Default 10% TP
                        self.position_shares = shares
                        self.position_active = True
                        self.entry_time = datetime.now()
                        self.current_price = limit_price
                        self.pnl = 0.0
                        print(f"Position tracking initialized for {self.stock}:")
                        print(f"  - Entry Price: ${self.entry_price:.2f}")
                        print(f"  - Stop Loss: ${self.stop_loss_price:.2f}")
                        print(f"  - Take Profit: ${self.take_profit_price:.2f}")
                        print(f"  - Entry Time: {self.entry_time}")
                        

                    else:
                        print(f"FAILED TO PLACE ORDER for {self.stock}")
                        
                except Exception as e:
                    print(f"ERROR PLACING ORDER for {self.stock}: {str(e)}")
                
                return shares, entry_price, stop_loss, limit_price
        else:
            if self.score < 85:
                print(f"Alpha Score too low: {self.score} < 85")
            if not self.additional_checks_passed:
                print("Additional checks failed")
        return 0, 0, 0, 0
                    
    def run(self, i):
        self.calculate_indicators()
        self.process_score()
        
        # Start monitoring this individual stock's position
        if hasattr(self, 'position_active') and self.position_active:
            print(f"Starting position monitoring for {self.stock}...")
            self.start_individual_monitoring()
    
    def monitor_position(self):
        """Monitor position for stop loss and take profit conditions"""
        if not hasattr(self, 'position_active') or not self.position_active:
            return
        
        try:
            # Get current price and update tracking variables
            self.current_price = self.broker.get_current_price(self.stock)
            
            # Calculate current PnL
            self.pnl = (self.current_price - self.entry_price) * self.position_shares
            
            # Calculate current gain/loss percentage
            current_gain_pct = (self.current_price - self.entry_price) / self.entry_price
            
            # Check stop loss condition
            if self.current_price <= self.stop_loss_price:
                print(f"STOP LOSS TRIGGERED: {self.stock} - Current: ${self.current_price:.2f}, Stop: ${self.stop_loss_price:.2f}")
                self.close_position('stop_loss', self.current_price)
                return
            
            # Check take profit condition
            if self.current_price >= self.take_profit_price:
                print(f"TAKE PROFIT TRIGGERED: {self.stock} - Current: ${self.current_price:.2f}, Target: ${self.take_profit_price:.2f}")
                self.close_position('take_profit', self.current_price)
                return
            
            # Check profit booking levels
            self._check_profit_booking(current_gain_pct, self.current_price)
            
            # Check trailing stop levels
            self._check_trailing_stops(current_gain_pct, self.current_price)
            
            # Check trailing exit conditions
            self._check_trailing_exit(current_gain_pct, self.current_price)

            if self.manager.stop_event.is_set():
                self.close_position('drawdown', self.current_price)

            # Print position status every 10 iterations (every ~100 seconds)
            if hasattr(self, '_monitor_counter'):
                self._monitor_counter += 1
            else:
                self._monitor_counter = 1
                
            if self._monitor_counter % 10 == 0:
                print(f"Position Status - {self.stock}:")
                print(f"  - Current Price: ${self.current_price:.2f}")
                print(f"  - Entry Price: ${self.entry_price:.2f}")
                print(f"  - PnL: ${self.pnl:.2f} ({current_gain_pct*100:.2f}%)")
                print(f"  - Shares: {self.position_shares}")
                print(f"  - Stop Loss: ${self.stop_loss_price:.2f}")
                print(f"  - Take Profit: ${self.take_profit_price:.2f}")
            
        except Exception as e:
            print(f"Error monitoring position for {self.stock}: {e}")
    
    def close_position(self, reason, current_price, shares_to_sell=None):
        """
        Close position (either partial or full)
        
        Args:
            reason: Reason for closing ('stop_loss', 'profit_booking', 'trailing_exit', etc.)
            current_price: Current market price
            shares_to_sell: Number of shares to sell (None for all remaining shares)
        """
        try:
            # Determine shares to sell
            if shares_to_sell is None:
                shares_to_sell = self.position_shares
            
            # Ensure we don't sell more than we have
            shares_to_sell = min(shares_to_sell, self.position_shares)
            
            if shares_to_sell <= 0:
                print(f"No shares to sell for {self.stock}")
                return
            
            # Calculate PnL for this trade
            trade_pnl = (current_price - self.entry_price) * shares_to_sell
            
            # Place sell order
            order_id, executed_price = self.broker.place_order(
                symbol=self.stock,
                qty=shares_to_sell,  # Positive quantity
                order_type='MARKET',
                price=None,
                stop_loss=0,
                take_profit=0,
                side='SELL'
            )
            
            if order_id:
                print(f"{reason.upper()} EXECUTED: {self.stock} - {shares_to_sell} shares at ${current_price:.2f}")
                print(f"Order ID: {order_id}")
                print(f"Trade PnL: ${trade_pnl:.2f}")
                
                # Update remaining shares
                self.position_shares -= shares_to_sell
                
                # If all shares sold, close position completely
                if self.position_shares <= 0:
                    self.position_active = False
                    self.close_time = datetime.now()
                    self.pnl = (current_price - self.entry_price) * shares_to_sell  # Final PnL
                    
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
                else:
                    print(f"Partial position closed. Remaining shares: {self.position_shares}")
                    # Update PnL for remaining position
                    self.pnl = (current_price - self.entry_price) * self.position_shares
            else:
                print(f"Failed to place {reason} order for {self.stock}")
                
        except Exception as e:
            print(f"Error executing {reason} for {self.stock}: {e}")
    
    
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
            # This is a simplified version - you might want to track max price over time
            if not hasattr(self, 'trailing_start_time'):
                self.trailing_start_time = datetime.now()
                self.max_price_since_trailing = current_price
                print(f"TRAILING EXIT MONITORING: {self.stock} - Started monitoring for exit conditions")
            
            # Update max price
            self.max_price_since_trailing = max(self.max_price_since_trailing, current_price)
            
            # Check time and price drop
            time_since_trailing = datetime.now() - self.trailing_start_time
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
            'duration': (datetime.now() - self.entry_time) if self.entry_time else None
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

                print(df)
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

                print(df_min)
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

        drawdown_thread = threading.Thread(target=self.monitor_drawdown_loop, name="DrawdownMonitor", daemon=True)
        drawdown_thread.start()
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
    
    def monitor_drawdown_loop(self):
            print("Starting global drawdown monitoring thread.")
            threshold = creds.EQUITY * creds.RISK_CONFIG['daily_drawdown_limit']
            while not self.stop_event.is_set():
                with self.lock:
                    total_pnl = self.broker.get_total_pnl()  # Must return realized + unrealized

                    print(f"[Drawdown] PnL: {total_pnl:.2f}, Threshold: {-threshold}")
                    now = datetime.now()
                    current_time_str = now.strftime("%H:%M")
                    
                    if total_pnl <= -threshold and not self.max_drawdown_triggered:
                        print(f"Max loss threshold of {-threshold} hit. Stopping all strategies.")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        
                    if current_time_str >= self.config.EXIT_TIME:
                        print("Exit time reached")
                        self.max_drawdown_triggered = True
                        self.stop_event.set()
                        

                time.sleep(5)

    def run(self):
        self.threads = []
        self.strategies = []
        print("Start")
        for i, stock in enumerate(self.stocks_list):
            print(f"{i}:{stock} ")
        
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
        self.print_qualifying_stocks()

    def test(self):
        
        current_price = self.broker.get_current_price("AAPL")
        self.broker.place_order(
            symbol="AAPL",
            qty=100,
            order_type='LIMIT',
            price=current_price + 0.01,
            stop_loss=0,
            take_profit=0,
            side='BUY'
        )

        while not self.broker.filled_check("AAPL"):
            print("AAPL not filled")
            time.sleep(1)
        print("AAPL filled")

    
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
    

    

    
    def add_qualifying_stock(self, stock_data):
        """Add a stock that passes all entry conditions to the qualifying list"""
        with self.qualifying_stocks_lock:
            self.qualifying_stocks.append(stock_data)
            print(f"Added {stock_data['symbol']} to qualifying stocks list")
            print(f"Total qualifying stocks: {len(self.qualifying_stocks)}")
    
    def get_qualifying_stocks(self):
        """Get the current list of qualifying stocks"""
        with self.qualifying_stocks_lock:
            return self.qualifying_stocks.copy()
    
    def clear_qualifying_stocks(self):
        """Clear the qualifying stocks list (e.g., at start of new day)"""
        with self.qualifying_stocks_lock:
            self.qualifying_stocks.clear()
            print("Qualifying stocks list cleared")
    
    def print_qualifying_stocks(self):
        """Print the current qualifying stocks with their details"""
        with self.qualifying_stocks_lock:
            if not self.qualifying_stocks:
                print("No qualifying stocks found")
                return
            
            print(f"\n=== QUALIFYING STOCKS ({len(self.qualifying_stocks)}) ===")
            for i, stock_data in enumerate(self.qualifying_stocks, 1):
                print(f"{i}. {stock_data['symbol']}")
                print(f"   Alpha Score: {stock_data['alpha_score']:.1f}")
                print(f"   Entry Price: ${stock_data['entry_price']:.2f}")
                print(f"   Stop Loss: ${stock_data['stop_loss']:.2f}")
                print(f"   Shares: {stock_data['shares']}")
                print(f"   Limit Price: ${stock_data['limit_price']:.2f}")
                print()
            
if __name__ == "__main__":
    manager = StrategyManager()
    manager.test()
    print("STOPPING MANAGER")
    