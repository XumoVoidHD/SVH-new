import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
import pytz
import creds
from streamlit_autorefresh import st_autorefresh
# Page configuration
st.set_page_config(
    page_title="SVH Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SVH Trading Dashboard")
st.markdown("Configuration editor and database viewer for your trading system")
st.markdown("---")
st.write("**🧪 Testing Configuration**")

if hasattr(creds, 'TESTING'):
    testing_enabled = st.checkbox(
        "Testing Mode Enabled", 
        value=creds.TESTING,
        key="testing_enabled",
        help="Enable/disable testing mode for the trading system"
    )

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
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Database Viewer", "Configuration Editor", "Raw Configuration"]
    )
    
    if page == "Database Viewer":
        st.header("🗄️ Database Viewer")
        st.markdown("View raw data from your `trades.db` database")
        
        # Database management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", type="primary"):
                st.rerun()
        
        with col2:
            if st.button("🗄️ Initialize Database", type="secondary", help="Create the database table if it doesn't exist"):
                try:
                    # Create a simple database connection to trigger table creation
                    import sqlite3
                    conn = sqlite3.connect('trades.db')
                    cursor = conn.cursor()
                    
                    # Create the stock_strategies table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS stock_strategies (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT UNIQUE NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            score REAL DEFAULT 0,
                            additional_checks_passed BOOLEAN DEFAULT FALSE,
                            position_active BOOLEAN DEFAULT FALSE,
                            position_shares INTEGER DEFAULT 0,
                            current_price REAL DEFAULT 0,
                            entry_price REAL DEFAULT 0,
                            stop_loss_price REAL DEFAULT 0,
                            take_profit_price REAL DEFAULT 0,
                            used_margin REAL DEFAULT 0,
                            unrealized_pnl REAL DEFAULT 0,
                            realized_pnl REAL DEFAULT 0,
                            entry_time TIMESTAMP,
                            close_time TIMESTAMP,
                            hedge_active BOOLEAN DEFAULT FALSE,
                            hedge_shares INTEGER DEFAULT 0,
                            hedge_symbol TEXT,
                            hedge_level REAL DEFAULT 0,
                            hedge_beta REAL DEFAULT 0,
                            hedge_entry_price REAL DEFAULT 0,
                            hedge_entry_time TIMESTAMP,
                            hedge_exit_price REAL DEFAULT 0,
                            hedge_pnl REAL DEFAULT 0
                        )
                    ''')
                    conn.commit()
                    conn.close()
                    st.success("✅ Database table created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to create database table: {e}")
        
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
            
            # Display the main data table
            st.dataframe(df, use_container_width=True)
            
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
        st.header("⚙️ Configuration Editor")
        st.markdown("Edit trading parameters in `creds.py`")
        
        # Create tabs for different config sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Stock Selection", "Indicators", "Alpha Score", "Risk Management", 
            "Stop Loss & Profit", "Hedge & Leverage", "Trading Hours"
        ])
        
        with tab1:
            st.subheader("🎯 Stock Selection Configuration")
            st.markdown("Configure stock filtering criteria for the trading system")
            
            if hasattr(creds, 'STOCK_SELECTION'):
                stock_config = creds.STOCK_SELECTION
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Market Cap Filter**")
                    market_cap_min = st.number_input(
                        "Minimum Market Cap ($):", 
                        min_value=100_000_000, 
                        max_value=1_000_000_000_000, 
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
                    alpha_threshold = st.number_input(
                        "5-Day Alpha Threshold:", 
                        min_value=0.00001, 
                        max_value=0.01, 
                        value=stock_config.get('alpha_threshold', 0.005), 
                        step=0.0001,
                        format="%.3f",
                        help="Minimum 5-day alpha return to qualify (as decimal, e.g., 0.005 = 0.5%)",
                    )
                    
                    st.write("**Sector Allocation**")
                    max_sector_weight = st.number_input(
                        "Maximum Sector Weight (%):", 
                        min_value=10.0, 
                        max_value=100.0, 
                        value=stock_config.get('max_sector_weight', 30.0), 
                        step=5.0,
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
            st.subheader("📊 Indicators Configuration")
            
            if hasattr(creds, 'INDICATORS') and hasattr(creds, 'ADDITIONAL_CHECKS_CONFIG'):
                indicators_config = creds.INDICATORS
                checks_config = creds.ADDITIONAL_CHECKS_CONFIG
                
                # VWAP Settings - Compact layout
            st.write("**VWAP Settings**")
            vwap_timeframes = st.multiselect(
                    "Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('vwap', {}).get('timeframes', ["3min"]),
                    key="vwap_timeframes"
            )
            
            # EMA Settings - Compact layout
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
            
            # MACD Settings - Compact layout
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
            
            # ADX Settings - Compact layout
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
            
            # Volume Average Settings - Compact layout
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
            st.subheader("🎯 Alpha Score Configuration")
            
            if hasattr(creds, 'ALPHA_SCORE_CONFIG'):
                alpha_config = creds.ALPHA_SCORE_CONFIG
                
                # Trend Analysis
                st.write("**Trend Analysis**")
                trend_weight = st.number_input(
                    "Trend Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('trend', {}).get('weight', 30), 
                    step=1, 
                    key="trend_weight"
                )
                
                # Momentum Analysis
                st.write("**Momentum Analysis**")
                momentum_weight = st.number_input(
                    "Momentum Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('momentum', {}).get('weight', 20), 
                    step=1, 
                    key="momentum_weight"
                )
                
                # Volume/Volatility Analysis
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
                with col2:
                    volume_spike_multiplier = st.number_input(
                        "Volume Spike Multiplier:", 
                        min_value=0.5, 
                        max_value=5.0, 
                        value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('volume_spike', {}).get('multiplier', 1.5), 
                        step=0.1, 
                        key="volume_spike_mult"
                    )
                with col3:
                    adx_threshold = st.number_input(
                        "ADX Threshold:", 
                        min_value=10, 
                        max_value=50, 
                        value=alpha_config.get('volume_volatility', {}).get('conditions', {}).get('adx', {}).get('threshold', 20), 
                        step=1, 
                        key="adx_threshold"
                    )
                
                # News Analysis
                st.write("**News Analysis**")
                news_weight = st.number_input(
                    "News Weight (%):", 
                    min_value=0, 
                    max_value=100, 
                    value=alpha_config.get('news', {}).get('weight', 15), 
                    step=1, 
                    key="news_weight"
                )
                
                # Market Calm Analysis
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
                    vix_threshold = st.number_input(
                        "VIX Threshold:", 
                        min_value=10, 
                        max_value=50, 
                        value=alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('threshold', 20), 
                        step=1, 
                        key="alpha_vix_threshold"
                    )
                
                # Total weight validation
                total_weight = trend_weight + momentum_weight + volume_weight + news_weight + market_calm_weight
                if total_weight != 100:
                    st.warning(f"⚠️ Total weight is {total_weight}% (should be 100%)")
                else:
                    st.success(f"✅ Total weight: {total_weight}%")
        
        with tab4:
            st.subheader("⚠️ Risk Management Configuration")
            
            if hasattr(creds, 'RISK_CONFIG') and hasattr(creds, 'ORDER_CONFIG'):
                risk_config = creds.RISK_CONFIG
                order_config = creds.ORDER_CONFIG
                
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
            st.subheader("🛑 Stop Loss & Profit Configuration")
            
            if hasattr(creds, 'STOP_LOSS_CONFIG') and hasattr(creds, 'PROFIT_CONFIG'):
                stop_loss_config = creds.STOP_LOSS_CONFIG
                profit_config = creds.PROFIT_CONFIG
                
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
            st.subheader("🛡️ Hedge & Leverage Configuration")
            
            if hasattr(creds, 'HEDGE_CONFIG') and hasattr(creds, 'LEVERAGE_CONFIG'):
                hedge_config = creds.HEDGE_CONFIG
                leverage_config = creds.LEVERAGE_CONFIG
                
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
            st.subheader("⏰ Trading Hours Configuration")
            st.info("🕐 **Note:** All times are displayed in Central Standard Time (CST)")
            
            if hasattr(creds, 'TRADING_HOURS'):
                trading_hours_config = creds.TRADING_HOURS
                
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

    
    elif page == "Raw Configuration":
        st.header("📝 Raw Configuration Editor")
        st.markdown("Edit `creds.py` directly as text")
        
        # Show file info
        st.info("💡 **Tip:** After saving, restart your trading application for changes to take effect.")

if __name__ == "__main__":
    main()
