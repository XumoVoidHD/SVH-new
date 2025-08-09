import streamlit as st
import json

st.set_page_config(page_title="SVH Trading Config", layout="wide")

st.title("📈 SVH Trading Strategy Configuration")

# Load current config
try:
    import creds as config
    current_config = {
        'MARKET_CAP': config.MARKET_CAP,
        'EQUITY': config.EQUITY,
        'INDICATORS': config.INDICATORS,
        'RISK_CONFIG': config.RISK_CONFIG,
        'STOP_LOSS_CONFIG': config.STOP_LOSS_CONFIG,
        'PROFIT_CONFIG': config.PROFIT_CONFIG,
        'TRADING_HOURS': config.TRADING_HOURS,
        'ORDER_CONFIG': config.ORDER_CONFIG
    }
except:
    st.error("Could not load config.py")
    st.stop()

# Basic Settings
st.header("Basic Settings")
col1, col2 = st.columns(2)
with col1:
    market_cap = st.number_input("Market Cap Filter ($)", value=current_config['MARKET_CAP'], step=100_000_000)
with col2:
    equity = st.number_input("Account Equity ($)", value=current_config['EQUITY'], step=1000)

# Risk Management
st.header("Risk Management")
col1, col2 = st.columns(2)
with col1:
    alpha_threshold = st.number_input("Alpha Score Threshold", value=current_config['RISK_CONFIG']['alpha_score_threshold'], min_value=0, max_value=100)
    risk_per_trade = st.slider("Risk Per Trade (%)", value=current_config['RISK_CONFIG']['risk_per_trade']*100, min_value=0.1, max_value=5.0)
with col2:
    max_daily_trades = st.number_input("Max Daily Trades", value=current_config['RISK_CONFIG']['max_daily_trades'], min_value=1, max_value=100)
    daily_drawdown = st.slider("Daily Drawdown Limit (%)", value=current_config['RISK_CONFIG']['daily_drawdown_limit']*100, min_value=0.5, max_value=10.0)

# Stop Loss
st.header("Stop Loss Settings")
col1, col2, col3 = st.columns(3)
with col1:
    default_stop = st.slider("Default Stop Loss (%)", value=current_config['STOP_LOSS_CONFIG']['default_stop_loss']*100, min_value=0.5, max_value=5.0)
with col2:
    volatile_stop = st.slider("Volatile Stop Loss (%)", value=current_config['STOP_LOSS_CONFIG']['volatile_stop_loss']*100, min_value=0.5, max_value=10.0)
with col3:
    max_stop = st.slider("Max Stop Loss (%)", value=current_config['STOP_LOSS_CONFIG']['max_stop_loss']*100, min_value=1.0, max_value=15.0)

# Trading Hours
st.header("Trading Hours")
col1, col2 = st.columns(2)
with col1:
    start_time = st.text_input("Market Open", value=current_config['TRADING_HOURS']['start'])
with col2:
    end_time = st.text_input("Market Close", value=current_config['TRADING_HOURS']['end'])

# Save button
if st.button("💾 Save Configuration", type="primary"):
    st.success("Configuration updated! (Note: This is a demo - actual file saving would be implemented)")
    
# Display current config
with st.expander("View Current Configuration"):
    st.json(current_config) 