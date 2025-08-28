# Market cap filter (now configured in STOCK_SELECTION)
EQUITY = 1000000
TESTING = True

STOCK_SELECTION = {
    'market_cap_min': 2_000_000_000,  # $2B minimum market cap
    'price_min': 10.0,                 # $10 minimum price
    'volume_min': 1_000_000,           # 1M minimum daily volume
    'alpha_threshold': 0.005,          # 0.5% minimum 5-day alpha
    # 'max_sector_weight': 0.30,         # Maximum 30% allocation per sector
    # 'top_sectors_count': 3             # Focus on top 3 sectors
}

# =============================================================================
# SVH ALPHA INTRADAY TRADING STRATEGY CONFIGURATION
# =============================================================================
# Modify these values to adjust strategy behavior without changing the main code.

# Indicator configurations with timeframe specifications
INDICATORS = {
    'vwap': {
        'timeframes': ['3min'],  # Calculate VWAP on 3-minute data
        'params': {}
    },
    'ema1': {
        'timeframes': ['5min'],  # Calculate EMA1 on 5-minute data
        'params': {
            'length': 5
        }
    },
    'ema2': {
        'timeframes': ['20min'],  # Calculate EMA2 on 20-minute data
        'params': {
            'length': 20
        }
    },
    'macd': {
        'timeframes': ['3min'],  # Calculate MACD on 3-minute data
        'params': {
            'fast': 12,
            'slow': 26,
            'signal': 9
        }
    },
    'adx': {
        'timeframes': ['3min'],  # Calculate ADX on 3-minute data
        'params': {
            'length': 14
        }
    },
    'volume_avg': {
        'timeframes': ['3min'],  # Calculate volume average on 3-minute data
        'params': {
            'window': 20
        }
    }
}

# Alpha Score configuration
ALPHA_SCORE_CONFIG = {
    'trend': {
        'weight': 30,
        'conditions': {
            'price_vwap': {'weight': 15},
            'ema_cross': {'weight': 15}
        }
    },
    'momentum': {
        'weight': 20,
        'conditions': {
            'macd_positive': {'weight': 20}
        }
    },
    'volume_volatility': {
        'weight': 20,
        'conditions': {
            'volume_spike': {'weight': 10, 'multiplier': 1.5},
            'adx_threshold': {'weight': 10, 'threshold': 20}
        }
    },
    'news': {
        'weight': 15,
        'conditions': {
            'no_major_news': {'weight': 15}
        }
    },
    'market_calm': {
        'weight': 15,
        'conditions': {
            'vix_threshold': {'weight': 15, 'threshold': 20}
        }
    }
}

# Additional checks configuration
ADDITIONAL_CHECKS_CONFIG = {
    'volume_multiplier': 2.0,
    'vwap_slope_threshold': 0.005,
    'vwap_slope_period': 3  # minutes
}

# Risk management configuration
RISK_CONFIG = {
    'alpha_score_threshold': 85,
    'risk_per_trade': 0.004,  # 0.4% of equity per trade
    'max_daily_trades': 10,
    'daily_drawdown_limit': 0.02,  # 2%
    'monthly_drawdown_limit': 0.08,  # 8%
    'drawdown_alert': 0.015  # 1.5%
}
# Stop loss configuration
STOP_LOSS_CONFIG = {
    'default_stop_loss': 0.015,  # 1.5%
    'volatile_stop_loss': 0.02,  # 2%
    'max_stop_loss': 0.04,  # 4%
    'atr_multiplier': 1.5,
    'atr_period': 14,
    'trailing_stop_levels': [
        {'gain': 0.02, 'new_stop_pct': 0.0075},  # At 2% gain, move SL to +0.75%
        {'gain': 0.05, 'new_stop_pct': 0.022}    # At 5% gain, move SL to +2.2%
    ]
}

# Profit taking configuration
PROFIT_CONFIG = {
    'profit_booking_levels': [
        {'gain': 0.01, 'exit_pct': 0.40},  # At 1% gain, sell 40% of position
        {'gain': 0.03, 'exit_pct': 0.30},  # At 3% gain, sell 30% of remaining
        {'gain': 0.05, 'exit_pct': 0.30}   # At 5% gain, sell 30% of remaining
    ],
    'trailing_exit_conditions': {
        'gain_threshold': 0.05,  # Only apply trailing exit logic after 5% gain
        'drop_threshold': 0.005,  # Exit if price drops 0.5%
        'monitor_period': 3,  # Monitor for 3 minutes
    }
}

# Hedge configuration (Point 9)
HEDGE_CONFIG = {
    'enabled': True,
    'hedge_symbol': 'XLF',  # Financial sector ETF to short
    'triggers': {
        'vix_threshold': 22,  # VIX > 22 triggers hedge
        'sp500_drop_threshold': 0.012,  # S&P drops >1.2% in 15min
    },
    'hedge_levels': {
        'mild': {  # 1 trigger met
            'beta': 0.15,  # -0.15β hedge
            'description': 'Mild hedge: VIX elevated but market stable'
        },
        'severe': {  # All triggers met
            'beta': 0.30,  # -0.3β hedge
            'description': 'Severe hedge: Multiple risk indicators triggered'
        }
    }
}

# Leverage configuration (Point 9) 
LEVERAGE_CONFIG = {
    'enabled': True,
    'max_leverage': 2.0,  # Maximum 2x leverage
    'conditions': {
        'alpha_score_min': 85,  # Alpha Score ≥85%
        'vix_max': 18,  # VIX <18
        'drawdown_max': 0.005,  # 10-day drawdown <0.5%
        'vix_trend_days': 10  # 10-day VIX trending down
    },
    'leverage_levels': {
        'all_conditions_met': 2.0,  # 2x leverage if all conditions pass
        'partial_conditions': 1.2,  # 1.2x if some conditions fail
        'default': 1.0  # 1x (no leverage) as fallback
    },
    'margin_alert_threshold': 1.5  # Alert if margin >150%
}

# Order configuration
ORDER_CONFIG = {
    'limit_offset_min': 0.00003,  # 0.003%
    'limit_offset_max': 0.00007,  # 0.007%
    'order_window': 60,  # seconds
}

# Trading hours
TRADING_HOURS = {
    'market_open': '09:30',
    'market_close': '16:00',
    'timezone': 'America/New_York',
    'morning_entry_start': '10:00',
    'morning_entry_end': '11:15',
    'afternoon_entry_start': '13:30',
    'afternoon_entry_end': '14:30'
}

# =============================================================================
# EXAMPLE MODIFICATIONS (uncomment and modify as needed)
# =============================================================================

# To add RSI indicator on 1-minute timeframe:
# INDICATORS['rsi'] = {
#     'timeframes': ['1min'],
#     'params': {'length': 14}
# }

# To change MACD parameters:
# INDICATORS['macd']['params']['fast'] = 8
# INDICATORS['macd']['params']['slow'] = 21

# To change EMA1 length:
# INDICATORS['ema1']['params']['length'] = 10

# To change EMA2 length:
# INDICATORS['ema2']['params']['length'] = 25

# To change Alpha Score threshold:
# RISK_CONFIG['alpha_score_threshold'] = 80

# To change volume multiplier for additional checks:
# ADDITIONAL_CHECKS_CONFIG['volume_multiplier'] = 2.5

# To modify risk parameters:
# RISK_CONFIG['risk_per_trade'] = 0.005  # 0.5% risk per trade


# To adjust stop loss parameters:
# STOP_LOSS_CONFIG['default_stop_loss'] = 0.02  # 2% default stop
# STOP_LOSS_CONFIG['volatile_stop_loss'] = 0.025  # 2.5% volatile stop
