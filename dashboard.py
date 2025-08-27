import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
# Page configuration
st.set_page_config(
    page_title="SVH Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SVH Trading Dashboard")
st.markdown("Configuration editor and database viewer for your trading system")

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

# Function to read creds.py
def read_creds_file():
    try:
        with open('creds.py', 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        st.error(f"Error reading creds.py: {e}")
        return None

# Function to write creds.py
def write_creds_file(content):
    try:
        with open('creds.py', 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        st.error(f"Error writing creds.py: {e}")
        return False

# Main app
def main():
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Database Viewer", "Configuration Editor", "Raw Configuration"]
    )
    
    if page == "Database Viewer":
        st.header("🗄️ Database Viewer")
        st.markdown("View raw data from your `trades.db` database")
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
        
        # Connect to database
        conn = get_db_connection()
        if not conn:
            st.error("Could not connect to database. Make sure `trades.db` exists in the current directory.")
            return
        
        # Get data from stock_strategies table (main table in trades.db)
        data, column_names = get_all_data(conn, 'stock_strategies')
        
        if data:
            # Convert to DataFrame and display
            df = pd.DataFrame(data, columns=column_names)
            
            # Add summary row for PnL if the table has unrealized_pnl column
            if 'unrealized_pnl' in df.columns:
                # Calculate total unrealized PnL
                total_unrealized_pnl = df['unrealized_pnl'].sum()
                
                # Display total unrealized PnL
                st.metric("Total Unrealized PnL", f"${total_unrealized_pnl:,.2f}")
                
                # Show realized PnL if available
                if 'realized_pnl' in df.columns:
                    total_realized_pnl = df['realized_pnl'].sum()
                    st.metric("Total Realized PnL", f"${total_realized_pnl:,.2f}")
                    
                    # Show combined PnL
                    total_combined_pnl = total_unrealized_pnl + total_realized_pnl
                    st.metric("Total Combined PnL", f"${total_combined_pnl:,.2f}")
                
                st.markdown("---")
            
            # Display the main data table
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"trades_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Indicators", "Alpha Score", "Risk Management", 
            "Stop Loss & Profit", "Trading Hours"
        ])
        
        with tab1:
            st.subheader("📊 Indicators Configuration")
            st.write("**VWAP Settings**")
            vwap_timeframes = st.multiselect(
                "VWAP Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["3min"]
            )
            
            st.write("**EMA Settings**")
            ema1_length = st.number_input("EMA1 Length:", min_value=1, max_value=100, value=5)
            ema1_timeframes = st.multiselect(
                "EMA1 Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["5min"]
            )
            
            ema2_length = st.number_input("EMA2 Length:", min_value=1, max_value=100, value=20)
            ema2_timeframes = st.multiselect(
                "EMA2 Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["20min"]
            )
            
            st.write("**MACD Settings**")
            macd_fast = st.number_input("MACD Fast Period:", min_value=1, max_value=50, value=12)
            macd_slow = st.number_input("MACD Slow Period:", min_value=1, max_value=100, value=26)
            macd_signal = st.number_input("MACD Signal Period:", min_value=1, max_value=50, value=9)
            macd_timeframes = st.multiselect(
                "MACD Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["3min"]
            )
            
            st.write("**ADX Settings**")
            adx_length = st.number_input("ADX Length:", min_value=1, max_value=100, value=14)
            adx_timeframes = st.multiselect(
                "ADX Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["3min"]
            )
            
            st.write("**Volume Average Settings**")
            volume_window = st.number_input("Volume Window:", min_value=1, max_value=100, value=20)
            volume_timeframes = st.multiselect(
                "Volume Timeframes:", 
                ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                default=["3min"]
            )
        
        with tab2:
            st.subheader("🎯 Alpha Score Configuration")
            st.write("**Trend Analysis (30%)**")
            trend_weight = st.slider("Trend Weight:", min_value=0, max_value=100, value=30)
            
            st.write("**Momentum Analysis (20%)**")
            momentum_weight = st.slider("Momentum Weight:", min_value=0, max_value=100, value=20)
            
            st.write("**Volume/Volatility Analysis (20%)**")
            volume_weight = st.slider("Volume Weight:", min_value=0, max_value=100, value=20)
            volume_spike_multiplier = st.number_input("Volume Spike Multiplier:", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            adx_threshold = st.number_input("ADX Threshold:", min_value=10, max_value=50, value=20)
            
            st.write("**News Analysis (15%)**")
            news_weight = st.slider("News Weight:", min_value=0, max_value=100, value=15)
            
            st.write("**Market Calm Analysis (15%)**")
            market_calm_weight = st.slider("Market Calm Weight:", min_value=0, max_value=100, value=15)
            vix_threshold = st.number_input("VIX Threshold:", min_value=10, max_value=50, value=20)
        
        with tab3:
            st.subheader("⚠️ Risk Management Configuration")
            alpha_threshold = st.number_input("Alpha Score Threshold:", min_value=0, max_value=100, value=1)
            risk_per_trade = st.number_input("Risk Per Trade (%):", min_value=0.1, max_value=10.0, value=0.4, step=0.1)
            max_daily_trades = st.number_input("Max Daily Trades:", min_value=1, max_value=100, value=10)
            daily_drawdown = st.number_input("Daily Drawdown Limit (%):", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
            monthly_drawdown = st.number_input("Monthly Drawdown Limit (%):", min_value=1.0, max_value=20.0, value=8.0, step=1.0)
            drawdown_alert = st.number_input("Drawdown Alert (%):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)
        
        with tab4:
            st.subheader("🛑 Stop Loss & Profit Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Stop Loss Settings**")
                default_stop = st.number_input("Default Stop Loss (%):", min_value=0.5, max_value=10.0, value=1.5, step=0.5)
                volatile_stop = st.number_input("Volatile Stop Loss (%):", min_value=1.0, max_value=15.0, value=2.0, step=0.5)
                max_stop = st.number_input("Max Stop Loss (%):", min_value=2.0, max_value=20.0, value=4.0, step=0.5)
                atr_multiplier = st.number_input("ATR Multiplier:", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
                atr_period = st.number_input("ATR Period:", min_value=5, max_value=50, value=14)
            
            with col2:
                st.write("**Profit Taking Settings**")
                profit_level1 = st.number_input("Profit Level 1 (%):", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
                exit_pct1 = st.number_input("Exit % at Level 1:", min_value=10, max_value=100, value=40)
                profit_level2 = st.number_input("Profit Level 2 (%):", min_value=1.0, max_value=15.0, value=3.0, step=0.5)
                exit_pct2 = st.number_input("Exit % at Level 2:", min_value=10, max_value=100, value=30)
                profit_level3 = st.number_input("Profit Level 3 (%):", min_value=2.0, max_value=20.0, value=5.0, step=0.5)
                exit_pct3 = st.number_input("Exit % at Level 3:", min_value=10, max_value=100, value=30)
        
        with tab5:
            st.subheader("⏰ Trading Hours Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Hours**")
                market_open = st.time_input("Market Open:", value=datetime.strptime("09:30", "%H:%M").time())
                market_close = st.time_input("Market Close:", value=datetime.strptime("16:00", "%H:%M").time())
                timezone = st.selectbox("Timezone:", ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"])
            
            with col2:
                st.write("**Entry Windows**")
                morning_start = st.time_input("Morning Entry Start:", value=datetime.strptime("10:00", "%H:%M").time())
                morning_end = st.time_input("Morning Entry End:", value=datetime.strptime("11:15", "%H:%M").time())
                afternoon_start = st.time_input("Afternoon Entry Start:", value=datetime.strptime("13:30", "%H:%M").time())
                afternoon_end = st.time_input("Afternoon Entry End:", value=datetime.strptime("14:30", "%H:%M").time())
        
        # Save button
        st.markdown("---")
        if st.button("💾 Save Configuration", type="primary"):
            st.info("Configuration editor is in development. Use the Raw Configuration page to edit and save changes.")
            st.write("**Note:** The interactive editor above shows current values. To make changes, use the Raw Configuration page to edit creds.py directly.")
    
    elif page == "Raw Configuration":
        st.header("📝 Raw Configuration Editor")
        st.markdown("Edit `creds.py` directly as text")
        
        # Read current creds.py
        creds_content = read_creds_file()
        if not creds_content:
            return
        
        # Text area for editing
        edited_content = st.text_area(
            "Edit creds.py:",
            value=creds_content,
            height=600,
            help="Make your changes here and click Save to update the file"
        )
        
        # Save button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                if write_creds_file(edited_content):
                    st.success("Configuration saved successfully!")
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Original"):
                st.rerun()
        
        # Show file info
        st.info("💡 **Tip:** After saving, restart your trading application for changes to take effect.")

if __name__ == "__main__":
    main()
