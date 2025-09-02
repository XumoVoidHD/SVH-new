import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import pytz
import json
import os
from streamlit_autorefresh import st_autorefresh
from deldb import rename_to_creation_date

# Page configuration
st.set_page_config(
    page_title="SVH Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("SVH Trading Dashboard")
st.markdown("Configuration editor and database viewer for your trading system")
st.markdown("---")

# Function to load configuration from creds.json
def load_config():
    """Load configuration from creds.json file"""
    try:
        if os.path.exists('creds.json'):
            with open('creds.json', 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            # Create default config if file doesn't exist
            default_config = {
                "EQUITY": 200000,
                "TESTING": False,
                "VWAP_SHOULD_BE_BELOW_PRICE": True,
                "STOCK_SELECTION": {
                    "market_cap_min": 2000000000,
                    "price_min": 10.0,
                    "volume_min": 1000000,
                    "alpha_threshold": 0.005
                },
                "INDICATORS": {
                    "vwap": {"timeframes": ["3min"], "params": {}},
                    "ema1": {"timeframes": ["5min"], "params": {"length": 5}},
                    "ema2": {"timeframes": ["20min"], "params": {"length": 20}},
                    "macd": {"timeframes": ["3min"], "params": {"fast": 12, "slow": 26, "signal": 9}},
                    "adx": {"timeframes": ["3min"], "params": {"length": 14}},
                    "volume_avg": {"timeframes": ["3min"], "params": {"window": 20}}
                },
                "RISK_CONFIG": {
                    "alpha_score_threshold": 85,
                    "risk_per_trade": 0.004,
                    "max_position_equity_pct": 0.1,
                    "max_daily_trades": 10,
                    "daily_drawdown_limit": 0.02,
                    "monthly_drawdown_limit": 0.08,
                    "drawdown_alert": 0.015
                }
            }
            save_config(default_config)
            return default_config
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

# Function to save configuration to creds.json
def save_config(config):
    """Save configuration to creds.json file"""
    try:
        with open('creds.json', 'w', encoding='utf-8') as file:
            json.dump(config, file, indent=4, ensure_ascii=False)
        return True, "Configuration saved successfully!"
    except Exception as e:
        return False, f"Error saving configuration: {str(e)}"

# Function to get database connection
def get_db_connection():
    try:
        conn = sqlite3.connect('trades.db')
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to get table info
def get_table_info(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

# Function to get all data from a table
def get_all_data(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    data = cursor.fetchall()
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    
    return data, columns

def main():
    # Load configuration
    config = load_config()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Database Viewer", "Configuration Editor", "Raw Configuration"]
    )
    
    if page == "Database Viewer":
        st.header("Database Viewer")
        
        # Database management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Refresh Data", type="primary"):
                st.rerun()
        
        with col2:
            if st.button("Start", type="primary", help="Run main.py to start the trading system"):
                try:
                    import subprocess
                    import sys
                    import os
                    
                    # Get the current directory
                    current_dir = os.getcwd()
                    main_py_path = os.path.join(current_dir, "main.py")
                    
                    if os.path.exists(main_py_path):
                        # Open main.py in a new terminal window/process
                        if os.name == 'nt':  # Windows
                            # Use a simpler approach for Windows to avoid path issues
                            subprocess.Popen(['start', 'cmd', '/k', 'python', main_py_path], 
                                           shell=True, cwd=current_dir)
                        else:  # Linux/Mac
                            subprocess.Popen(['gnome-terminal', '--', 'python3', main_py_path], 
                                           cwd=current_dir)
                        
                    else:
                        st.error("❌ main.py not found!")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("Clear Database", type="secondary", help="Clear the database by calling rename_to_creation_date function"):
                try:
                    # Call the rename_to_creation_date function
                    result = rename_to_creation_date()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            st.error("Could not connect to database. Make sure `trades.db` exists in the current directory.")
            return
        
        # Check if stock_strategies table exists
        table_info = get_table_info(conn)
        if 'stock_strategies' not in table_info:
            st.error("❌ The 'stock_strategies' table does not exist in the database.")
            st.info("💡 This usually means the database hasn't been properly initialized. Try running the main trading application first.")
            conn.close()
            return
        
        # Get data from stock_strategies table (main table in trades.db)
        data, column_names = get_all_data(conn, 'stock_strategies')
        
        if data:
            # Convert to DataFrame and display
            df = pd.DataFrame(data, columns=column_names)
            
            # Fix data type issues for Streamlit compatibility
            # Convert datetime columns to proper datetime objects (fixes PyArrow serialization error)
            datetime_columns = ['entry_time', 'exit_time', 'created_at', 'updated_at', 'last_check_time']
            for col in datetime_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception:
                        # If conversion fails, keep as string but clean it
                        df[col] = df[col].astype(str)
            
            # Convert boolean columns from bytes/ints to proper booleans
            boolean_columns = ['additional_checks_passed', 'position_active', 'trailing_exit_monitoring', 'hedge_active']
            for col in boolean_columns:
                if col in df.columns:
                    # Convert bytes/ints to boolean
                    df[col] = df[col].apply(lambda x: bool(x) if x is not None else False)
            
            # Convert numeric columns to proper types
            numeric_columns = ['score', 'position_shares', 'current_price', 'entry_price', 'stop_loss_price', 
                              'take_profit_price', 'used_margin', 'unrealized_pnl', 'realized_pnl', 'hedge_shares', 
                              'hedge_level', 'hedge_beta', 'hedge_entry_price', 'hedge_exit_price', 'hedge_pnl']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Add summary row for PnL if the table has unrealized_pnl column
            if 'unrealized_pnl' in df.columns:
                # Calculate totals
                total_unrealized_pnl = df['unrealized_pnl'].sum()
                total_realized_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                total_combined_pnl = total_unrealized_pnl + total_realized_pnl
                
                # Display PnL metrics in a single row with 3 columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Unrealized PnL", f"${total_unrealized_pnl:,.2f}")
                
                with col2:
                    st.metric("Total Realized PnL", f"${total_realized_pnl:,.2f}")
                
                with col3:
                    st.metric("Total Combined PnL", f"${total_combined_pnl:,.2f}")
                
                # Display used margin below PnL metrics
                if 'used_margin' in df.columns:
                    total_used_margin = df['used_margin'].sum()
                    st.metric("Total Used Margin", f"${total_used_margin:,.2f}")
                
                # Add stock selection statistics
                st.markdown("---")
                st.subheader("🎯 Stock Selection Statistics")
                
                # Count stocks by different criteria
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Market cap analysis
                    if 'marketcap' in df.columns:
                        large_caps = len(df[df['marketcap'] >= 2_000_000_000])
                        st.metric("Large Cap (≥$2B)", large_caps)
                    else:
                        st.metric("Large Cap (≥$2B)", "N/A")
                    
                    # Price analysis
                    if 'current_price' in df.columns:
                        high_price = len(df[df['current_price'] >= 10.0])
                        st.metric("High Price (≥$10)", high_price)
                    else:
                        st.metric("High Price (≥$10)", "N/A")
                
                with col2:
                    # Volume analysis (if available)
                    if 'volume' in df.columns:
                        high_volume = len(df[df['volume'] >= 1_000_000])
                        st.metric("High Volume (≥1M)", high_volume)
                    else:
                        st.metric("High Volume (≥1M)", "N/A")
                    
                    # Alpha score analysis
                    if 'score' in df.columns:
                        high_alpha = len(df[df['score'] >= 85])
                        st.metric("High Alpha (≥85)", high_alpha)
                    else:
                        st.metric("High Alpha (≥85)", "N/A")
                
                with col3:
                    # Hedge analysis
                    if 'hedge_active' in df.columns:
                        hedged_stocks = len(df[df['hedge_active'] == True])
                        st.metric("Hedged Stocks", hedged_stocks)
                    else:
                        st.metric("Hedged Stocks", "N/A")
                    
                    # Position analysis
                    if 'position_active' in df.columns:
                        active_positions = len(df[df['position_active'] == True])
                        st.metric("Active Positions", active_positions)
                    else:
                        st.metric("Active Positions", "N/A")
                
                st.markdown("---")
            
            # Display the main data table with error handling for PyArrow serialization
            try:
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying DataFrame: {str(e)}")
                st.info("Trying alternative display method...")
                
                # Fallback: display as text with pagination
                st.subheader("Data Table (Alternative View)")
                
                # Show data in chunks to avoid overwhelming the display
                chunk_size = 20
                total_rows = len(df)
                
                if total_rows > chunk_size:
                    page = st.selectbox(f"Select page (showing {chunk_size} rows per page):", 
                                      range(1, (total_rows // chunk_size) + 2))
                    start_idx = (page - 1) * chunk_size
                    end_idx = min(start_idx + chunk_size, total_rows)
                    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
                    st.write(df.iloc[start_idx:end_idx])
                else:
                    st.write(df)
                
                st.warning("💡 The DataFrame display had issues. Consider checking your data types or restarting the dashboard.")
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"trades_db_{datetime.now(pytz.timezone('America/Chicago')).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data found in trades.db")
        
        conn.close()
        
        # Auto-refresh every 7 seconds
        st_autorefresh(interval=7000, key="database_refresh")
        
    elif page == "Configuration Editor":        
        # Create tabs for different config sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Stock Selection", "Indicators", "Alpha Score", "Risk Management", 
            "Stop Loss & Profit", "Hedge & Leverage", "Trading Hours"
        ])
        
        with tab1:
            st.subheader("Stock Selection Configuration")
            st.markdown("Configure stock filtering criteria for the trading system")
            
            stock_config = config.get('STOCK_SELECTION', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Cap Filter**")
                market_cap_min = st.number_input(
                    "Minimum Market Cap ($):", 
                    min_value=100_000_000, 
                    max_value=10_000_000_000_000, 
                    value=stock_config.get('market_cap_min', 2_000_000_000), 
                    step=100_000_000,
                    help="Minimum market capitalization in dollars"
                )
                
                st.write("**Price Filter**")
                price_min = st.number_input(
                    "Minimum Price ($):", 
                    min_value=1.0, 
                    max_value=1000.0, 
                    value=stock_config.get('price_min', 10.0), 
                    step=0.5,
                    help="Minimum stock price in dollars"
                )
                
                st.write("**Volume Filter**")
                volume_min = st.number_input(
                    "Minimum Daily Volume:", 
                    min_value=100_000, 
                    max_value=100_000_000, 
                    value=stock_config.get('volume_min', 1_000_000), 
                    step=100_000,
                    help="Minimum daily trading volume"
                )
            
            with col2:
                st.write("**Alpha Threshold**")
                stock_alpha_threshold = st.number_input(
                    "5-Day Alpha Threshold:", 
                    min_value=0.00001, 
                    max_value=0.01, 
                    value=float(stock_config.get('alpha_threshold', 0.005)), 
                    step=0.0001,
                    format="%.3f",
                    help="Minimum 5-day alpha return to qualify (as decimal, e.g., 0.005 = 0.005 = 0.5%)",
                )
                
                st.write("**Sector Allocation**")
                max_sector_weight = st.number_input(
                    "Maximum Sector Weight (%):", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=float(stock_config.get('max_sector_weight', 0.30)), 
                    step=0.01,
                    format="%.2f",
                    help="Maximum allocation per sector"
                )
                
                top_sectors_count = st.number_input(
                    "Top Sectors Count:", 
                    min_value=1, 
                    max_value=10, 
                    value=stock_config.get('top_sectors_count', 3), 
                    step=1,
                    help="Number of top sectors to focus on"
                )
        
        with tab2:
            st.subheader("Indicators Configuration")
            
            indicators_config = config.get('INDICATORS', {})
            checks_config = config.get('ADDITIONAL_CHECKS_CONFIG', {})
            
            # VWAP Settings
            st.write("**VWAP Settings**")
            vwap_timeframes = st.multiselect(
                "Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=indicators_config.get('vwap', {}).get('timeframes', ["3min"]),
                key="vwap_timeframes"
            )
            
            # EMA Settings
            st.write("**EMA Settings**")
            col1, col2, col3 = st.columns(3)
            with col1:
                ema1_length = st.number_input(
                    "EMA1 Length:", 
                    min_value=1, 
                    max_value=100, 
                    value=indicators_config.get('ema1', {}).get('params', {}).get('length', 5), 
                    key="ema1_length"
                )
                ema1_timeframes = st.multiselect(
                    "EMA1 Timeframes:", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('ema1', {}).get('timeframes', ["5min"]),
                    key="ema1_timeframes"
                )
            with col2:
                ema2_length = st.number_input(
                    "EMA2 Length:", 
                    min_value=1, 
                    max_value=100, 
                    value=indicators_config.get('ema2', {}).get('params', {}).get('length', 20), 
                    key="ema2_length"
                )
                ema2_timeframes = st.multiselect(
                    "EMA2 Timeframes:", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('ema2', {}).get('timeframes', ["20min"]),
                    key="ema2_timeframes"
                )
            with col3:
                st.write("**Current Values:**")
                st.write(f"EMA1: {ema1_length} periods")
                st.write(f"EMA2: {ema2_length} periods")
            
            # MACD Settings
            st.write("**MACD Settings**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                macd_fast = st.number_input(
                    "Fast Period:", 
                    min_value=1, 
                    max_value=50, 
                    value=indicators_config.get('macd', {}).get('params', {}).get('fast', 12), 
                    key="macd_fast"
                )
            with col2:
                macd_slow = st.number_input(
                    "Slow Period:", 
                    min_value=1, 
                    max_value=100, 
                    value=indicators_config.get('macd', {}).get('params', {}).get('slow', 26), 
                    key="macd_slow"
                )
            with col3:
                macd_signal = st.number_input(
                    "Signal Period:", 
                    min_value=1, 
                    max_value=50, 
                    value=indicators_config.get('macd', {}).get('params', {}).get('signal', 9), 
                    key="macd_signal"
                )
            with col4:
                macd_timeframes = st.multiselect(
                    "Timeframes:", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('macd', {}).get('timeframes', ["3min"]),
                    key="macd_timeframes"
                )
            
            # ADX Settings
            st.write("**ADX Settings**")
            col1, col2 = st.columns(2)
            with col1:
                adx_length = st.number_input(
                    "ADX Length:", 
                    min_value=1, 
                    max_value=100, 
                    value=indicators_config.get('adx', {}).get('params', {}).get('length', 14), 
                    key="adx_length"
                )
            with col2:
                adx_timeframes = st.multiselect(
                    "Timeframes:", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('adx', {}).get('timeframes', ["3min"]),
                    key="adx_timeframes"
                )
            
            # Volume Average Settings
            st.write("**Volume Average Settings**")
            col1, col2 = st.columns(2)
            with col1:
                volume_window = st.number_input(
                    "Volume Window:", 
                    min_value=1, 
                    max_value=100, 
                    value=indicators_config.get('volume_avg', {}).get('params', {}).get('window', 20), 
                    key="volume_window"
                )
            with col2:
                volume_timeframes = st.multiselect(
                    "Timeframes:", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('volume_avg', {}).get('timeframes', ["3min"]),
                    key="volume_timeframes"
                )
            
            # Additional Checks Configuration
            st.markdown("---")
            st.write("**Additional Checks Configuration**")
            col1, col2 = st.columns(2)
            with col1:
                volume_multiplier = st.number_input(
                    "Volume Multiplier:", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=checks_config.get('volume_multiplier', 2.0), 
                    step=0.1, 
                    key="volume_multiplier"
                )
                vwap_slope_threshold = st.number_input(
                    "VWAP Slope Threshold (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=checks_config.get('vwap_slope_threshold', 0.005), 
                    step=0.001,
                    format="%.3f",
                    key="vwap_slope_threshold"
                )
            with col2:
                vwap_slope_period = st.number_input(
                    "VWAP Slope Period (min):", 
                    min_value=1, 
                    max_value=20, 
                    value=checks_config.get('vwap_slope_period', 3), 
                    step=1, 
                    key="vwap_slope_period"
                )
                    
        with tab3:
            st.subheader("Alpha Score Configuration")
            
            alpha_config = config.get('ALPHA_SCORE_CONFIG', {})
            
            # Trend Analysis
            st.write("**Trend Analysis**")
            col1, col2 = st.columns(2)
            with col1:
                trend_weight = st.number_input(
                    "Trend Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('trend', {}).get('weight', 30), 
                    step=1, 
                    key="trend_weight"
                )
            with col2:
                st.write("**Trend Conditions:**")
                price_vwap_weight = st.number_input(
                    "Price vs VWAP Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('trend', {}).get('conditions', {}).get('price_vwap', {}).get('weight', 15), 
                    step=1, 
                    key="price_vwap_weight"
                )
                ema_cross_weight = st.number_input(
                    "EMA Cross Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('trend', {}).get('conditions', {}).get('ema_cross', {}).get('weight', 15), 
                    step=1, 
                    key="ema_cross_weight"
                )
            
            # Momentum Analysis
            st.markdown("---")
            st.write("**Momentum Analysis**")
            col1, col2 = st.columns(2)
            with col1:
                momentum_weight = st.number_input(
                    "Momentum Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('momentum', {}).get('weight', 20), 
                    step=1, 
                    key="momentum_weight"
                )
            with col2:
                st.write("**Momentum Conditions:**")
                macd_positive_weight = st.number_input(
                    "MACD Positive Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('momentum', {}).get('conditions', {}).get('macd_positive', {}).get('weight', 20), 
                    step=1, 
                    key="macd_positive_weight"
                )
            
            # Volume/Volatility Analysis
            st.markdown("---")
            st.write("**Volume/Volatility Analysis**")
            col1, col2, col3 = st.columns(3)
            with col1:
                volume_weight = st.number_input(
                    "Volume Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('volume_volatility', {}).get('weight', 20), 
                    step=1, 
                    key="volume_weight"
                )
                st.write("**Volume Conditions:**")
                volume_spike_weight = st.number_input(
                    "Volume Spike Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('volume_spike', {}).get('weight', 10), 
                    step=1, 
                    key="volume_spike_weight"
                )
            with col2:
                volume_spike_multiplier = st.number_input(
                    "Volume Spike Multiplier:", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('volume_spike', {}).get('multiplier', 1.5), 
                    step=0.1, 
                    key="volume_spike_mult"
                )
                adx_weight = st.number_input(
                    "ADX Threshold Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('adx_threshold', {}).get('weight', 10), 
                    step=1, 
                    key="adx_weight"
                )
            with col3:
                adx_threshold = st.number_input(
                    "ADX Threshold:", 
                    min_value=10, 
                    max_value=50, 
                    value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('adx_threshold', {}).get('threshold', 20), 
                    step=1, 
                    key="adx_threshold"
                )
            
            # News Analysis
            st.markdown("---")
            st.write("**News Analysis**")
            col1, col2 = st.columns(2)
            with col1:
                news_weight = st.number_input(
                    "News Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('news', {}).get('weight', 15), 
                    step=1, 
                    key="news_weight"
                )
            with col2:
                st.write("**News Conditions:**")
                no_major_news_weight = st.number_input(
                    "No Major News Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('news', {}).get('conditions', {}).get('no_major_news', {}).get('weight', 15), 
                    step=1, 
                    key="no_major_news_weight"
                )
            
            # Market Calm Analysis
            st.markdown("---")
            st.write("**Market Calm Analysis**")
            col1, col2 = st.columns(2)
            with col1:
                market_calm_weight = st.number_input(
                    "Market Calm Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('market_calm', {}).get('weight', 15), 
                    step=1, 
                    key="market_calm_weight"
                )
            with col2:
                st.write("**Market Calm Conditions:**")
                vix_threshold = st.number_input(
                    "VIX Threshold:", 
                    min_value=10, 
                    max_value=50, 
                    value=alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('threshold', 20), 
                    step=1, 
                    key="alpha_vix_threshold"
                )
                vix_threshold_weight = st.number_input(
                    "VIX Threshold Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('weight', 15), 
                    step=1, 
                    key="vix_threshold_weight"
                )
            
            # Total weight validation
            total_weight = trend_weight + momentum_weight + volume_weight + news_weight + market_calm_weight
            if total_weight != 100:
                st.warning(f"Total weight is {total_weight}% (should be 100%)")
            else:
                st.success(f"Total weight: {total_weight}%")
            
            # Detailed breakdown
            st.markdown("---")
            st.subheader("Alpha Score Breakdown")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Trend Analysis**")
                st.write(f"• Price vs VWAP: {price_vwap_weight}%")
                st.write(f"• EMA Cross: {ema_cross_weight}%")
                st.write(f"**Total Trend: {trend_weight}%**")
                
                st.write("**Momentum Analysis**")
                st.write(f"• MACD Positive: {macd_positive_weight}%")
                st.write(f"**Total Momentum: {momentum_weight}%**")
                
                st.write("**Volume/Volatility**")
                st.write(f"• Volume Spike: {volume_spike_weight}%")
                st.write(f"• ADX Threshold: {adx_weight}%")
                st.write(f"**Total Volume: {volume_weight}%**")
            
            with col2:
                st.write("**News Analysis**")
                st.write(f"• No Major News: {no_major_news_weight}%")
                st.write(f"**Total News: {news_weight}%**")
                
                st.write("**Market Calm**")
                st.write(f"• VIX Threshold: {vix_threshold_weight}%")
                st.write(f"**Total Market Calm: {market_calm_weight}%**")
                
                st.write("---")
                st.write(f"**Grand Total: {total_weight}%**")
                if total_weight == 100:
                    st.success("All conditions properly weighted")
                else:
                    st.error(f"Total weight should be 100%, currently {total_weight}%")
        
        with tab4:
            st.subheader("Risk Management Configuration")
            
            risk_config = config.get('RISK_CONFIG', {})
            order_config = config.get('ORDER_CONFIG', {})
            
            # Risk Configuration
            st.write("**Risk Management**")
            col1, col2 = st.columns(2)
            with col1:
                alpha_threshold = st.number_input(
                    "Alpha Score Threshold:", 
                    min_value=0, 
                    max_value=100, 
                    value=risk_config.get('alpha_score_threshold', 85), 
                    step=1, 
                    key="alpha_threshold"
                )
                risk_per_trade = st.number_input(
                    "Risk Per Trade (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('risk_per_trade', 0.004), 
                    step=0.001, 
                    key="risk_per_trade", 
                    format="%.3f"
                )
                max_position_equity_pct = st.number_input(
                    "Max Position Equity (%):", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=risk_config.get('max_position_equity_pct', 0.1), 
                    step=0.01, 
                    key="max_position_equity_pct", 
                    format="%.2f",
                    help="Maximum percentage of total equity that can be allocated to a single position"
                )
                max_daily_trades = st.number_input(
                    "Max Daily Trades:", 
                    min_value=1, 
                    max_value=50, 
                    value=risk_config.get('max_daily_trades', 10), 
                    step=1, 
                    key="max_daily_trades"
                )
            with col2:
                daily_drawdown = st.number_input(
                    "Daily Drawdown Limit (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('daily_drawdown_limit', 0.02), 
                    step=0.001, 
                    key="daily_drawdown", 
                    format="%.3f"
                )
                monthly_drawdown = st.number_input(
                    "Monthly Drawdown Limit (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('monthly_drawdown_limit', 0.08), 
                    step=0.001, 
                    key="monthly_drawdown", 
                    format="%.3f"
                )
                drawdown_alert = st.number_input(
                    "Drawdown Alert (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('drawdown_alert', 0.015), 
                    step=0.001, 
                    key="drawdown_alert", 
                    format="%.3f"
                )
            
            # Order Configuration
            st.markdown("---")
            st.write("**Order Configuration**")
            col1, col2, col3 = st.columns(3)
            with col1:
                limit_offset_min = st.number_input(
                    "Min Limit Offset (%):", 
                    min_value=0.00001, 
                    max_value=1.0, 
                    value=order_config.get('limit_offset_min', 0.00003), 
                    step=0.00001, 
                    key="limit_offset_min", 
                    format="%.6f", 
                    help="Minimum limit offset (as decimal, e.g., 0.00003 = 0.003%)"
                )
            with col2:
                limit_offset_max = st.number_input(
                    "Max Limit Offset (%):", 
                    min_value=0.00001, 
                    max_value=1.0, 
                    value=order_config.get('limit_offset_max', 0.00007), 
                    step=0.00001, 
                    key="limit_offset_max", 
                    format="%.6f", 
                    help="Maximum limit offset (as decimal, e.g., 0.00007 = 0.007%)"
                )
            with col3:
                order_window = st.number_input(
                    "Order Window (sec):", 
                    min_value=10, 
                    max_value=300, 
                    value=order_config.get('order_window', 60), 
                    step=10, 
                    key="order_window"
                )
                            
        with tab5:
            st.subheader("Stop Loss & Profit Configuration")
            
            stop_loss_config = config.get('STOP_LOSS_CONFIG', {})
            profit_config = config.get('PROFIT_CONFIG', {})
            
            # Stop Loss Configuration
            st.write("**Stop Loss Settings**")
            col1, col2, col3 = st.columns(3)
            with col1:
                default_stop = st.number_input(
                    "Default Stop Loss (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=stop_loss_config.get('default_stop_loss', 0.015), 
                    step=0.001, 
                    key="default_stop",
                    format="%.3f"
                )
                volatile_stop = st.number_input(
                    "Volatile Stop Loss (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=stop_loss_config.get('volatile_stop_loss', 0.02), 
                    step=0.001, 
                    key="volatile_stop",
                    format="%.3f"
                )
            with col2:
                max_stop = st.number_input(
                    "Max Stop Loss (%):", 
                    min_value=0.001, 
                    max_value=0.20, 
                    value=stop_loss_config.get('max_stop_loss', 0.04), 
                    step=0.001, 
                    key="max_stop",
                    format="%.3f"
                )
                atr_multiplier = st.number_input(
                    "ATR Multiplier:", 
                    min_value=0.1, 
                    max_value=5.0, 
                    value=stop_loss_config.get('atr_multiplier', 1.5), 
                    step=0.1, 
                    key="atr_multiplier"
                )
            with col3:
                atr_period = st.number_input(
                    "ATR Period:", 
                    min_value=5, 
                    max_value=50, 
                    value=stop_loss_config.get('atr_period', 14), 
                    step=1, 
                    key="atr_period"
                )
            
            # Trailing Stop Levels
            st.markdown("---")
            st.write("**Trailing Stop Levels**")
            col1, col2 = st.columns(2)
            with col1:
                trailing_gain1 = st.number_input(
                    "Trailing Gain 1 (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=stop_loss_config.get('trailing_stop_levels', [{}])[0].get('gain', 0.02), 
                    step=0.001, 
                    key="trailing_gain1",
                    format="%.3f"
                )
                new_stop1 = st.number_input(
                    "New Stop at Gain 1 (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=stop_loss_config.get('trailing_stop_levels', [{}])[0].get('new_stop_pct', 0.0075), 
                    step=0.001, 
                    key="new_stop1",
                    format="%.4f"
                )
            with col2:
                trailing_gain2 = st.number_input(
                    "Trailing Gain 2 (%):", 
                    min_value=0.001, 
                    max_value=0.15, 
                    value=stop_loss_config.get('trailing_stop_levels', [{}])[1].get('gain', 0.05), 
                    step=0.001, 
                    key="trailing_gain2",
                    format="%.3f"
                )
                new_stop2 = st.number_input(
                    "New Stop at Gain 2 (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=stop_loss_config.get('trailing_stop_levels', [{}])[1].get('new_stop_pct', 0.022), 
                    step=0.001, 
                    key="new_stop2",
                    format="%.3f"
                )
            
            # Profit Taking Configuration
            st.markdown("---")
            st.write("**Profit Taking Settings**")
            col1, col2, col3 = st.columns(3)
            with col1:
                profit_level1 = st.number_input(
                    "Profit Level 1 (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=profit_config.get('profit_booking_levels', [{}])[0].get('gain', 0.01), 
                    step=0.001, 
                    key="profit_level1",
                    format="%.3f"
                )
                exit_pct1 = st.number_input(
                    "Exit (%) at Level 1:", 
                    min_value=0.001, 
                    max_value=1.0,
                    step=0.001, 
                    value=profit_config.get('profit_booking_levels', [{}])[0].get('exit_pct', 0.40), 
                    key="exit_pct1",
                    format="%.3f",
                    help="Exit percentage at profit level 1"
                )
            with col2:
                profit_level2 = st.number_input(
                    "Profit Level 2 (%):", 
                    min_value=0.001, 
                    max_value=0.15, 
                    value=profit_config.get('profit_booking_levels', [{}])[1].get('gain', 0.03), 
                    step=0.001, 
                    key="profit_level2",
                    format="%.3f"
                )
                exit_pct2 = st.number_input(
                    "Exit (%) at Level 2:", 
                    min_value=0.001, 
                    max_value=1.0,
                    step=0.001, 
                    value=profit_config.get('profit_booking_levels', [{}])[1].get('exit_pct', 0.30), 
                    key="exit_pct2",
                    format="%.3f",
                    help="Exit percentage at profit level 2"
                )
            with col3:
                profit_level3 = st.number_input(
                    "Profit Level 3 (%):", 
                    min_value=0.001, 
                    max_value=0.20, 
                    value=profit_config.get('profit_booking_levels', [{}])[2].get('gain', 0.05), 
                    step=0.001, 
                    key="profit_level3",
                    format="%.3f"
                )
                exit_pct3 = st.number_input(
                    "Exit (%) at Level 3:", 
                    min_value=0.001, 
                    max_value=1.0,
                    step=0.001, 
                    value=profit_config.get('profit_booking_levels', [{}])[2].get('exit_pct', 0.30), 
                    key="exit_pct3",
                    format="%.3f",
                    help="Exit percentage at profit level 3"
                )
            
            # Trailing Exit Conditions
            st.markdown("---")
            st.write("**Trailing Exit Conditions**")
            col1, col2, col3 = st.columns(3)
            with col1:
                gain_threshold = st.number_input(
                    "Gain Threshold (%):", 
                    min_value=0.001, 
                    max_value=0.15, 
                    value=profit_config.get('trailing_exit_conditions', {}).get('gain_threshold', 0.05), 
                    step=0.001, 
                    key="gain_threshold",
                    format="%.3f"
                )
            with col2:
                drop_threshold = st.number_input(
                    "Drop Threshold (%):", 
                    min_value=0.001, 
                    max_value=0.02, 
                    value=profit_config.get('trailing_exit_conditions', {}).get('drop_threshold', 0.005), 
                    step=0.001, 
                    key="drop_threshold",
                    format="%.3f"
                )
            with col3:
                monitor_period = st.number_input(
                    "Monitor Period (min):", 
                    min_value=1, 
                    max_value=10, 
                    value=profit_config.get('trailing_exit_conditions', {}).get('monitor_period', 3), 
                    step=1, 
                    key="monitor_period"
                )
     
        with tab6:
            st.subheader("Hedge & Leverage Configuration")
            
            hedge_config = config.get('HEDGE_CONFIG', {})
            leverage_config = config.get('LEVERAGE_CONFIG', {})
            
            # Hedge Configuration
            st.write("**Hedge Settings**")
            col1, col2 = st.columns(2)
            with col1:
                hedge_enabled = st.checkbox(
                    "Hedge Enabled", 
                    value=hedge_config.get('enabled', True), 
                    key="hedge_enabled"
                )
                hedge_symbol = st.text_input(
                    "Hedge Symbol:", 
                    value=hedge_config.get('hedge_symbol', 'XLF'), 
                    key="hedge_symbol"
                )
                vix_threshold = st.number_input(
                    "VIX Threshold:", 
                    min_value=15, 
                    max_value=50, 
                    value=hedge_config.get('triggers', {}).get('vix_threshold', 22), 
                    step=1, 
                    key="hedge_vix_threshold"
                )
            with col2:
                sp500_drop_threshold = st.number_input(
                    "S&P 500 Drop Threshold (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=hedge_config.get('triggers', {}).get('sp500_drop_threshold', 0.012), 
                    step=0.001, 
                    key="sp500_drop_threshold",
                    format="%.3f"
                )
            
            # Hedge Levels
            st.markdown("---")
            st.write("**Hedge Levels**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Mild Hedge**")
                mild_beta = st.number_input(
                    "Mild Hedge Beta:", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=hedge_config.get('hedge_levels', {}).get('mild', {}).get('beta', 0.15), 
                    step=0.01, 
                    key="mild_beta",
                    format="%.2f"
                )
                mild_description = st.text_input(
                    "Mild Description:", 
                    value=hedge_config.get('hedge_levels', {}).get('mild', {}).get('description', 'Mild hedge: VIX elevated but market stable'), 
                    key="mild_description"
                )
            with col2:
                st.write("**Severe Hedge**")
                severe_beta = st.number_input(
                    "Severe Hedge Beta:", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=hedge_config.get('hedge_levels', {}).get('severe', {}).get('beta', 0.30), 
                    step=0.01, 
                    key="severe_beta",
                    format="%.2f"
                )
                severe_description = st.text_input(
                    "Severe Description:", 
                    value=hedge_config.get('hedge_levels', {}).get('severe', {}).get('description', 'Severe hedge: Multiple risk indicators triggered'), 
                    key="severe_description"
                )
            
            # Leverage Configuration
            st.markdown("---")
            st.write("**Leverage Settings**")
            col1, col2 = st.columns(2)
            with col1:
                leverage_enabled = st.checkbox(
                    "Leverage Enabled", 
                    value=leverage_config.get('enabled', True), 
                    key="leverage_enabled"
                )
                max_leverage = st.number_input(
                    "Max Leverage:", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=leverage_config.get('max_leverage', 2.0), 
                    step=0.1, 
                    key="max_leverage",
                    format="%.1f"
                )
                alpha_score_min = st.number_input(
                    "Alpha Score Min:", 
                    min_value=70, 
                    max_value=100, 
                    value=leverage_config.get('conditions', {}).get('alpha_score_min', 85), 
                    step=1, 
                    key="alpha_score_min"
                )
                vix_max = st.number_input(
                    "VIX Max:", 
                    min_value=10, 
                    max_value=30, 
                    value=leverage_config.get('conditions', {}).get('vix_max', 18), 
                    step=1, 
                    key="vix_max"
                )
            with col2:
                drawdown_max = st.number_input(
                    "Drawdown Max (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=leverage_config.get('conditions', {}).get('drawdown_max', 0.005), 
                    step=0.001, 
                    key="drawdown_max",
                    format="%.3f"
                )
                vix_trend_days = st.number_input(
                    "VIX Trend Days:", 
                    min_value=5, 
                    max_value=20, 
                    value=leverage_config.get('conditions', {}).get('vix_trend_days', 10), 
                    step=1, 
                    key="vix_trend_days"
                )
                margin_alert_threshold = st.number_input(
                    "Margin Alert Threshold:", 
                    min_value=1.0, 
                    max_value=3.0, 
                    value=leverage_config.get('margin_alert_threshold', 1.5), 
                    step=0.1, 
                    key="margin_alert_threshold",
                    format="%.1f"
                )
            
            # Leverage Levels
            st.markdown("---")
            st.write("**Leverage Levels**")
            col1, col2, col3 = st.columns(3)
            with col1:
                all_conditions_leverage = st.number_input(
                    "All Conditions Met:", 
                    min_value=1.0, 
                    max_value=5.0, 
                    value=leverage_config.get('leverage_levels', {}).get('all_conditions_met', 2.0), 
                    step=0.1, 
                    key="all_conditions_leverage",
                    format="%.1f"
                )
            with col2:
                partial_conditions_leverage = st.number_input(
                    "Partial Conditions:", 
                    min_value=1.0, 
                    max_value=3.0, 
                    value=leverage_config.get('leverage_levels', {}).get('partial_conditions', 1.2), 
                    step=0.1, 
                    key="partial_conditions_leverage",
                    format="%.1f"
                )
            with col3:
                default_leverage = st.number_input(
                    "Default Leverage:", 
                    min_value=1.0, 
                    max_value=2.0, 
                    value=leverage_config.get('leverage_levels', {}).get('default', 1.0), 
                    step=0.1, 
                    key="default_leverage",
                    format="%.1f"
                )

        with tab7:
            st.subheader("Trading Hours Configuration")
            st.info("**Note:** All times are displayed in Central Standard Time (CST)")
            
            trading_hours_config = config.get('TRADING_HOURS', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Hours**")
                market_open = st.time_input(
                    "Market Open:", 
                    value=datetime.strptime(trading_hours_config.get('market_open', '09:30'), "%H:%M").time()
                )
                market_close = st.time_input(
                    "Market Close:", 
                    value=datetime.strptime(trading_hours_config.get('market_close', '16:00'), "%H:%M").time()
                )
                timezone = st.selectbox(
                    "Timezone:", 
                    ["America/Chicago", "America/New_York", "America/Denver", "America/Los_Angeles"],
                    index=["America/Chicago", "America/New_York", "America/Denver", "America/Los_Angeles"].index(
                        trading_hours_config.get('timezone', 'America/Chicago')
                    )
                )
            
            with col2:
                st.write("**Entry Windows**")
                morning_start = st.time_input(
                    "Morning Entry Start:", 
                    value=datetime.strptime(trading_hours_config.get('morning_entry_start', '10:00'), "%H:%M").time()
                )
                morning_end = st.time_input(
                    "Morning Entry End:", 
                    value=datetime.strptime(trading_hours_config.get('morning_entry_end', '11:15'), "%H:%M").time()
                )
                afternoon_start = st.time_input(
                    "Afternoon Entry Start:", 
                    value=datetime.strptime(trading_hours_config.get('afternoon_entry_start', '13:30'), "%H:%M").time()
                )
                afternoon_end = st.time_input(
                    "Afternoon Entry End:", 
                    value=datetime.strptime(trading_hours_config.get('afternoon_entry_end', '14:30'), "%H:%M").time()
                )
        
        # Global Save Configuration Button
        st.markdown("---")
        st.subheader("Save All Configuration")
        
        # Basic Configuration Section
        st.write("**Basic Configuration**")
        col1, col2 = st.columns(2)
        
        with col1:
            equity = st.number_input(
                "Trading Equity ($):", 
                min_value=10000, 
                max_value=10000000, 
                value=config.get('EQUITY', 200000), 
                step=10000,
                key="equity_global"
            )
            
            testing = st.checkbox(
                "Testing Mode Enabled", 
                value=config.get('TESTING', False),
                key="testing_global"
            )
        
        with col2:
            vwap_below_price = st.checkbox(
                "VWAP Should Be Below Price", 
                value=config.get('VWAP_SHOULD_BE_BELOW_PRICE', True),
                key="vwap_below_price_global"
            )
        
        if st.button("Save All Configuration", type="primary", key="save_all_config"):
            # Update configuration with all values from tabs
            new_config = {
                "EQUITY": equity,
                "TESTING": testing,
                "VWAP_SHOULD_BE_BELOW_PRICE": vwap_below_price,
                "STOCK_SELECTION": {
                    "market_cap_min": market_cap_min,
                    "price_min": price_min,
                    "volume_min": volume_min,
                    "alpha_threshold": stock_alpha_threshold,
                    "max_sector_weight": max_sector_weight,
                    "top_sectors_count": top_sectors_count
                },
                "INDICATORS": {
                    "vwap": {"timeframes": vwap_timeframes, "params": {}},
                    "ema1": {"timeframes": ema1_timeframes, "params": {"length": ema1_length}},
                    "ema2": {"timeframes": ema2_timeframes, "params": {"length": ema2_length}},
                    "macd": {"timeframes": macd_timeframes, "params": {"fast": macd_fast, "slow": macd_slow, "signal": macd_signal}},
                    "adx": {"timeframes": adx_timeframes, "params": {"length": adx_length}},
                    "volume_avg": {"timeframes": volume_timeframes, "params": {"window": volume_window}}
                },
                "ADDITIONAL_CHECKS_CONFIG": {
                    "volume_multiplier": volume_multiplier,
                    "vwap_slope_threshold": vwap_slope_threshold,
                    "vwap_slope_period": vwap_slope_period
                },
                "ALPHA_SCORE_CONFIG": {
                    "trend": {"weight": trend_weight, "conditions": {"price_vwap": {"weight": price_vwap_weight}, "ema_cross": {"weight": ema_cross_weight}}},
                    "momentum": {"weight": momentum_weight, "conditions": {"macd_positive": {"weight": macd_positive_weight}}},
                    "volume_volatility": {"weight": volume_weight, "conditions": {"volume_spike": {"weight": volume_spike_weight, "multiplier": volume_spike_multiplier}, "adx_threshold": {"weight": adx_weight, "threshold": adx_threshold}}},
                    "news": {"weight": news_weight, "conditions": {"no_major_news": {"weight": no_major_news_weight}}},
                    "market_calm": {"weight": market_calm_weight, "conditions": {"vix_threshold": {"weight": vix_threshold_weight, "threshold": vix_threshold}}}
                },
                "RISK_CONFIG": {
                    "alpha_score_threshold": alpha_threshold,
                    "risk_per_trade": risk_per_trade,
                    "max_position_equity_pct": max_position_equity_pct,
                    "max_daily_trades": max_daily_trades,
                    "daily_drawdown_limit": daily_drawdown,
                    "monthly_drawdown_limit": monthly_drawdown,
                    "drawdown_alert": drawdown_alert
                },
                "STOP_LOSS_CONFIG": {
                    "default_stop_loss": default_stop,
                    "volatile_stop_loss": volatile_stop,
                    "max_stop_loss": max_stop,
                    "atr_multiplier": atr_multiplier,
                    "atr_period": atr_period,
                    "trailing_stop_levels": [
                        {"gain": trailing_gain1, "new_stop_pct": new_stop1},
                        {"gain": trailing_gain2, "new_stop_pct": new_stop2}
                    ]
                },
                "PROFIT_CONFIG": {
                    "profit_booking_levels": [
                        {"gain": profit_level1, "exit_pct": exit_pct1},
                        {"gain": profit_level2, "exit_pct": exit_pct2},
                        {"gain": profit_level3, "exit_pct": exit_pct3}
                    ],
                    "trailing_exit_conditions": {
                        "gain_threshold": gain_threshold,
                        "drop_threshold": drop_threshold,
                        "monitor_period": monitor_period
                    }
                },
                "HEDGE_CONFIG": {
                    "enabled": hedge_enabled,
                    "hedge_symbol": hedge_symbol,
                    "triggers": {"vix_threshold": vix_threshold, "sp500_drop_threshold": sp500_drop_threshold},
                    "hedge_levels": {
                        "mild": {"beta": mild_beta, "description": mild_description},
                        "severe": {"beta": severe_beta, "description": severe_description}
                    }
                },
                "LEVERAGE_CONFIG": {
                    "enabled": leverage_enabled,
                    "max_leverage": max_leverage,
                    "conditions": {"alpha_score_min": alpha_score_min, "vix_max": vix_max, "drawdown_max": drawdown_max, "vix_trend_days": vix_trend_days},
                    "leverage_levels": {"all_conditions_met": all_conditions_leverage, "partial_conditions": partial_conditions_leverage, "default": default_leverage},
                    "margin_alert_threshold": margin_alert_threshold
                },
                "ORDER_CONFIG": {
                    "limit_offset_min": limit_offset_min,
                    "limit_offset_max": limit_offset_max,
                    "order_window": order_window
                },
                "TRADING_HOURS": {
                    "market_open": market_open.strftime("%H:%M"),
                    "market_close": market_close.strftime("%H:%M"),
                    "timezone": timezone,
                    "morning_entry_start": morning_start.strftime("%H:%M"),
                    "morning_entry_end": morning_end.strftime("%H:%M"),
                    "afternoon_entry_start": afternoon_start.strftime("%H:%M"),
                    "afternoon_entry_end": afternoon_end.strftime("%H:%M")
                }
            }
            
            success, message = save_config(new_config)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    elif page == "Raw Configuration":        
        st.info("**Tip:** After saving, restart your trading application for changes to take effect.")
        
        # Read and display the current creds.json content
        try:
            if os.path.exists('creds.json'):
                with open('creds.json', 'r', encoding='utf-8') as file:
                    current_content = file.read()
                
                # Create a text area for editing
                edited_content = st.text_area(
                    "Edit creds.json content:",
                    value=current_content,
                    height=600,
                    key="raw_config_editor"
                )
                
                # Save button for raw configuration
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save Raw Configuration", type="primary", key="save_raw_config"):
                        try:
                            # Validate JSON before saving
                            json.loads(edited_content)
                            with open('creds.json', 'w', encoding='utf-8') as file:
                                file.write(edited_content)
                            st.success("Raw configuration saved successfully!")
                            st.rerun()
                        except json.JSONDecodeError as e:
                            st.error(f"Invalid JSON: {str(e)}")
                        except Exception as e:
                            st.error(f"Error saving configuration: {str(e)}")
                
                with col2:
                    if st.button("🔄 Reload from File", key="reload_raw_config"):
                        st.rerun()
                        
            else:
                st.error("creds.json does not exist. Create it first in the Configuration Editor.")
                
        except Exception as e:
            st.error(f"Error reading creds.json: {str(e)}")
            st.info("Make sure creds.json exists in the current directory.")

if __name__ == "__main__":
    main()
