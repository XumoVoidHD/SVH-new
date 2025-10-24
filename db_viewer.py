import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json
import os
from streamlit_autorefresh import st_autorefresh
from helpers.deldb import rename_to_creation_date
from helpers.reset_positions import close_all_open_positions
from db.trades_db import trades_db
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="SVH Capital - Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("SVH Capital - Dashboard")
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
                    "ADV_large_length": 20,
                    "ADV_small_length": 10,
                    "ADV_large": 500000,
                    "ADV_small": 750000,
                    "RVOL_length": 10,
                    "RVOL_filter": 1.3,
                    "alpha_threshold": 0.005,
                    "max_sector_weight": 0.3,
                    "top_sectors_count": 3
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
                },
                "TRADING_HOURS": {
                    "market_open": "09:30",
                    "market_close": "16:00",
                    "timezone": "US/Eastern",
                    "morning_entry_start": "9:00",
                    "morning_entry_end": "11:15",
                    "afternoon_entry_start": "13:30",
                    "afternoon_entry_end": "14:30",
                    "weak_exit_time": "15:30",
                    "hedge_force_exit_time": "15:55",
                    "safety_exit_time": "15:55"
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
        ["Database Viewer", "Historical Data", "Configuration Editor", "Raw Configuration"]
    )
    
    if page == "Database Viewer":
        st.header("Database Viewer")
        
        # Create tabs for different database views
        tab1, tab2 = st.tabs(["Simplified DB", "Raw Database"])
        
        with tab1:
            st.subheader("Simplified DB View")
            
            # Initialize session state for database save tracking
            if 'db_saved_simplified' not in st.session_state:
                st.session_state.db_saved_simplified = False
            
            # Check if database has tables
            def has_database_tables():
                try:
                    if not os.path.exists('trades.db'):
                        return False
                    conn = sqlite3.connect('trades.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    return len(tables) > 0
                except:
                    return False
            
            db_has_tables = has_database_tables()
            
            # # Show appropriate warnings
            # if db_has_tables:
            #     st.warning("⚠️ Clear Database before starting! Old data exists.")
            
            # Database management buttons
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("Refresh Data", type="primary"):
                    st.rerun()
            
            with col2:
                if st.button("Start", type="primary", help="Run main.py to start the trading system"):
                    try:
                        # Auto-backup if database has data (backup_database also deletes the original)
                        if db_has_tables:
                            st.info("📦 Backing up existing database...")
                            backup_result = trades_db.backup_database()
                            if backup_result:
                                st.success("✅ Database backed up and cleared! Starting system...")
                            else:
                                st.error("❌ Failed to backup database. Not starting.")
                                st.stop()
                        
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
                if st.button("Save Database", type="secondary", help="Backup the current database", key="save_db_raw"):
                    try:
                        result = trades_db.backup_database()
                        if result:
                            st.session_state.db_saved_simplified = True
                            st.success("✅ Database backed up successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to backup database")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col4:
                if st.button("Clear Database", type="secondary", help="Delete the database file", key="clear_db_raw"):
                    try:
                        if os.path.exists('trades.db'):
                            # Close any existing connections first
                            try:
                                conn = get_db_connection()
                                if conn:
                                    conn.close()
                            except:
                                pass
                            
                            # Wait a moment for connections to close
                            import time
                            time.sleep(0.1)
                            
                            # Try to delete the file
                            os.remove('trades.db')
                            st.session_state.db_saved_simplified = False
                            
                            # Verify deletion
                            if not os.path.exists('trades.db'):
                                st.success("✅ Database deleted successfully!")
                            else:
                                st.warning("⚠️ Database file still exists - may be locked by another process")
                            st.rerun()
                        else:
                            st.warning("⚠️ Database file does not exist")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col5:
                if st.button("Reset Positions", type="secondary", help="Close all open positions on IBKR and update database", key="reset_positions_raw"):
                    try:
                        
                        with st.spinner("Closing all open positions on IBKR..."):
                            count, positions = close_all_open_positions()
                        
                        if count > 0:
                            total_pnl = sum(p['pnl'] for p in positions)
                            st.success(f"Successfully closed {count} positions! Total PnL: ${total_pnl:.2f}")
                            
                            # Show details of closed positions
                            if positions:
                                st.write("**Closed Positions:**")
                                for pos in positions:
                                    st.write(f"- {pos['symbol']}: {pos['shares']} shares, PnL: ${pos['pnl']:.2f}")
                            
                            st.rerun()
                        else:
                            st.info("No open positions found to close")
                            
                    except Exception as e:
                        st.error(f"Error closing positions: {str(e)}")
            
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
                                'take_profit_price', 'used_capital', 'unrealized_pnl', 'realized_pnl', 'hedge_shares', 
                                'hedge_level', 'hedge_beta', 'hedge_entry_price', 'hedge_exit_price', 'hedge_pnl']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Add summary row for PnL if the table has unrealized_pnl column
                if 'unrealized_pnl' in df.columns:
                    # Filter for active positions only for unrealized PnL calculation
                    active_df = df[df['position_active'] == True] if 'position_active' in df.columns else df
                    
                    # Calculate unrealized PnL from raw database fields (active positions only)
                    calculated_unrealized = (active_df['current_price'] - active_df['entry_price']) * active_df['position_shares']
                    total_unrealized_pnl = calculated_unrealized.sum()
                    
                    # Calculate realized PnL from ALL positions (active + closed)
                    total_realized_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                    total_combined_pnl = total_unrealized_pnl + total_realized_pnl
                    active_positions = len(active_df)
                    
                    # Display PnL metrics in a single row with 5 columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Unrealized PnL", f"${total_unrealized_pnl:,.2f}")
                    
                    with col2:
                        st.metric("Total Realized PnL", f"${total_realized_pnl:,.2f}")
                    
                    with col3:
                        st.metric("Total Combined PnL", f"${total_combined_pnl:,.2f}")
                    
                    with col4:
                        st.metric("Active Positions", active_positions)
                    
                    with col5:
                        # Trades Executed = count of trades where shares > 0
                        if 'shares' in df.columns:
                            trades_executed = len(df[df['shares'] > 0])
                            st.metric("Trades Executed", trades_executed)
                        else:
                            st.metric("Trades Executed", 0)
                    
                    # Additional metrics in a separate row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        if 'used_capital' in active_df.columns:
                            total_used_capital = active_df['used_capital'].sum()
                            st.metric("Total Capital Deployed", f"${total_used_capital:,.2f}")
                        else:
                            total_used_capital = 0
                    
                    with col2:
                        # Calculate total cost basis (sum of all positions)
                        total_cost_basis = (df['shares'] * df['entry_price']).sum()
                        if total_cost_basis > 0:
                            # Calculate intraday return %
                            intraday_return_pct = ((total_realized_pnl + total_unrealized_pnl) / total_cost_basis) * 100
                            st.metric("Intraday Return %", f"{intraday_return_pct:.2f}%")
                        else:
                            st.metric("Intraday Return %", "0.00%")
                    
                    with col3:
                        # Total Portfolio Value = used_capital + unrealized_pnl
                        total_portfolio_value = total_used_capital + total_unrealized_pnl
                        st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
                    
                    with col4:
                        # Available Cash = EQUITY - used_capital
                        equity = config.get('EQUITY', 0)
                        available_cash = equity - total_used_capital
                        st.metric("Cash Available", f"${available_cash:,.2f}")
                    
                    with col5:
                        # Average Trade Duration (for closed positions)
                        if 'entry_time' in df.columns and 'close_time' in df.columns:
                            # Filter for closed positions with valid times
                            closed_with_times = df[(df['position_shares'] == 0) & 
                                                   (df['entry_time'].notna()) & 
                                                   (df['close_time'].notna())].copy()
                            if len(closed_with_times) > 0:
                                # Convert to datetime and calculate durations
                                closed_with_times['entry_dt'] = pd.to_datetime(closed_with_times['entry_time'])
                                closed_with_times['close_dt'] = pd.to_datetime(closed_with_times['close_time'])
                                closed_with_times['duration'] = closed_with_times['close_dt'] - closed_with_times['entry_dt']
                                
                                # Calculate average duration
                                avg_duration = closed_with_times['duration'].mean()
                                
                                # Format duration
                                total_seconds = int(avg_duration.total_seconds())
                                hours = total_seconds // 3600
                                minutes = (total_seconds % 3600) // 60
                                seconds = total_seconds % 60
                                
                                if hours > 0:
                                    duration_str = f"{hours}h {minutes}m"
                                elif minutes > 0:
                                    duration_str = f"{minutes}m {seconds}s"
                                else:
                                    duration_str = f"{seconds}s"
                                
                                st.metric("Avg Trade Duration", duration_str)
                            else:
                                st.metric("Avg Trade Duration", "N/A")
                        else:
                            st.metric("Avg Trade Duration", "N/A")
                    
                    # Trading statistics in a third row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        # Exposure Ratio = (total_used_capital / equity) * 100
                        if equity > 0:
                            exposure_ratio = (total_used_capital / equity) * 100
                            st.metric("Exposure Ratio", f"{exposure_ratio:.2f}%")
                        else:
                            st.metric("Exposure Ratio", "0.00%")
                    
                    with col2:
                        # Avg Position Size = total_used_capital / number of active positions
                        if active_positions > 0:
                            avg_position_size = total_used_capital / active_positions
                            st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                        else:
                            st.metric("Avg Position Size", "$0.00")
                    
                    with col3:
                        # Win Rate % = (profitable closed positions / total closed positions) * 100
                        # Closed positions are those where position_shares = 0
                        closed_positions = df[df['position_shares'] == 0] if 'position_shares' in df.columns else pd.DataFrame()
                        if len(closed_positions) > 0:
                            profitable_closed = len(closed_positions[closed_positions['realized_pnl'] > 0])
                            losing_closed = len(closed_positions[closed_positions['realized_pnl'] < 0])
                            total_closed = profitable_closed + losing_closed
                            if total_closed > 0:
                                win_rate = (profitable_closed / total_closed) * 100
                                st.metric("Win Rate %", f"{win_rate:.2f}%")
                            else:
                                st.metric("Win Rate %", "N/A")
                        else:
                            st.metric("Win Rate %", "N/A")
                    
                    with col4:
                        # Profit Factor = gross profit / gross loss
                        gross_profit = df[df['realized_pnl'] > 0]['realized_pnl'].sum()
                        gross_loss = abs(df[df['realized_pnl'] < 0]['realized_pnl'].sum())
                        if gross_loss > 0:
                            profit_factor = gross_profit / gross_loss
                            st.metric("Profit Factor", f"{profit_factor:.2f}")
                        else:
                            st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                    
                    with col5:
                        # Average PnL per Trade
                        total_trades = len(df)
                        if total_trades > 0:
                            avg_pnl_per_trade = (total_realized_pnl + total_unrealized_pnl) / total_trades
                            st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                        else:
                            st.metric("Avg PnL/Trade", "$0.00")
                    
                    st.markdown("---")
                
                # Add filter options in a single row
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_live_only = st.checkbox("Live position", value=False, key="show_live_positions_simplified")
                with col2:
                    qualified_stocks_only = st.checkbox("Qualified Stocks", value=False, key="qualified_stocks_only_simplified")
                with col3:
                    executed_trades_only = st.checkbox("Executed Trades", value=False, key="executed_trades_only_simplified")
                
                # Filter positions based on checkbox combinations
                simplified_df = df.copy()
                
                # Apply filters
                if show_live_only:
                    if 'position_active' in simplified_df.columns:
                        simplified_df = simplified_df[simplified_df['position_active'] == True]
                
                if qualified_stocks_only:
                    if 'shares' in simplified_df.columns:
                        simplified_df = simplified_df[simplified_df['shares'] == 0]
                
                if executed_trades_only:
                    if 'shares' in simplified_df.columns:
                        simplified_df = simplified_df[simplified_df['shares'] > 0]
                
                simplified_df = simplified_df.copy()
                
                if not simplified_df.empty:
                    # Create simplified DataFrame with calculated columns
                    simplified_display = pd.DataFrame({
                        'Symbol': simplified_df['symbol'],
                        'Active': simplified_df['position_active'] if 'position_active' in simplified_df.columns else True,
                        'Actual Quantity': simplified_df['position_shares'] if 'position_shares' in simplified_df.columns else simplified_df['shares'],
                        'Initial Quantity': simplified_df['shares'],
                        'Entry Price': simplified_df['entry_price'],
                        'Current Price': simplified_df['current_price'],
                        'Unrealized PnL': simplified_df['unrealized_pnl'],
                        'Realized PnL': simplified_df['realized_pnl'],
                        'Score': simplified_df['score'] if 'score' in simplified_df.columns else 0,
                        'Entry Time': simplified_df['entry_time'] if 'entry_time' in simplified_df.columns else 'N/A',
                        'Close Time': simplified_df['close_time'] if 'close_time' in simplified_df.columns else 'N/A'
                    })
                    
                    # Ensure numeric types for calculations
                    simplified_display['Actual Quantity'] = pd.to_numeric(simplified_display['Actual Quantity'], errors='coerce')
                    simplified_display['Initial Quantity'] = pd.to_numeric(simplified_display['Initial Quantity'], errors='coerce')
                    simplified_display['Entry Price'] = pd.to_numeric(simplified_display['Entry Price'], errors='coerce')
                    simplified_display['Current Price'] = pd.to_numeric(simplified_display['Current Price'], errors='coerce')
                    simplified_display['Unrealized PnL'] = pd.to_numeric(simplified_display['Unrealized PnL'], errors='coerce')
                    simplified_display['Realized PnL'] = pd.to_numeric(simplified_display['Realized PnL'], errors='coerce')
                    
                    # Calculate additional columns using Actual Quantity
                    simplified_display['Cost Basis'] = simplified_display['Initial Quantity'] * simplified_display['Entry Price']
                    simplified_display['Market Value'] = simplified_display['Actual Quantity'] * simplified_display['Current Price']
                    
                    # Calculate PnL from raw database fields (same as raw database view)
                    calculated_unrealized_pnl = simplified_display['Actual Quantity'] * (simplified_display['Current Price'] - simplified_display['Entry Price'])
                    calculated_unrealized_pnl = calculated_unrealized_pnl.fillna(0)
                    
                    # Use calculated unrealized PnL + stored realized PnL
                    total_pnl = calculated_unrealized_pnl + simplified_display['Realized PnL']
                    simplified_display['PnL ($)'] = total_pnl
                    
                    # Calculate percentage based on total PnL relative to cost basis
                    simplified_display['PnL (%)'] = (total_pnl / simplified_display['Cost Basis'] * 100).round(2)
                    simplified_display['PnL (%)'] = simplified_display['PnL (%)'].fillna(0)
                    
                    # Rename columns for display
                    simplified_display = simplified_display.rename(columns={
                        'Actual Quantity': 'Actual Qty',
                        'Initial Quantity': 'Initial Qty'
                    })
                    
                    # Select and reorder columns for display
                    display_columns = [
                        'Symbol', 'Active', 'Score', 'Actual Qty', 'Initial Qty', 'Entry Price', 'Current Price', 
                        'Cost Basis', 'Market Value', 'PnL ($)', 'PnL (%)', 'Entry Time', 'Close Time'
                    ]
                    simplified_display = simplified_display[display_columns]
                    
                    # Format the data with commas and dollar signs
                    def format_currency(value):
                        if pd.isna(value):
                            return "$0.00"
                        return f"${value:,.2f}"
                    
                    def format_percentage(value):
                        if pd.isna(value):
                            return "0.00%"
                        return f"{value:.2f}%"
                    
                    # Apply formatting to the display dataframe
                    formatted_display = simplified_display.copy()
                    formatted_display['Entry Price'] = formatted_display['Entry Price'].apply(format_currency)
                    formatted_display['Current Price'] = formatted_display['Current Price'].apply(format_currency)
                    formatted_display['Cost Basis'] = formatted_display['Cost Basis'].apply(format_currency)
                    formatted_display['Market Value'] = formatted_display['Market Value'].apply(format_currency)
                    formatted_display['PnL ($)'] = formatted_display['PnL ($)'].apply(format_currency)
                    formatted_display['PnL (%)'] = formatted_display['PnL (%)'].apply(format_percentage)
                    
                    # Format time columns to show only HH:MM:SS
                    def format_time(time_value):
                        if pd.isna(time_value) or time_value == 'N/A' or str(time_value) in ['None', 'nan', '']:
                            return 'N/A'
                        try:
                            # Try to parse as datetime and extract time
                            time_obj = pd.to_datetime(time_value)
                            return time_obj.strftime('%H:%M:%S')
                        except:
                            return 'N/A'
                    
                    formatted_display['Entry Time'] = formatted_display['Entry Time'].apply(format_time)
                    formatted_display['Close Time'] = formatted_display['Close Time'].apply(format_time)
                    
                    # Display the formatted table
                    st.dataframe(formatted_display, use_container_width=True)
                else:
                    st.info("No trades found in the database.")
                
                # Download button for simplified data
                csv = simplified_df.to_csv(index=False)
                st.download_button(
                    label="Download Simplified Data as CSV",
                    data=csv,
                    file_name=f"simplified_trades_{datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                if df.empty:
                    st.info("No data found in trades.db")
            
            conn.close()
            
            # Auto-refresh every 7 seconds
            st_autorefresh(interval=7000, key="simplified_database_refresh")
        
        with tab2:
            st.subheader("Raw Database View")
            
            # Initialize session state for database save tracking
            if 'db_saved_raw' not in st.session_state:
                st.session_state.db_saved_raw = False
            
            # Check if database has tables
            def has_database_tables_raw():
                try:
                    if not os.path.exists('trades.db'):
                        return False
                    conn = sqlite3.connect('trades.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    return len(tables) > 0
                except:
                    return False
            
            db_has_tables_raw = has_database_tables_raw()
            
            # # Show appropriate warnings
            # if db_has_tables_raw:
            #     st.warning("⚠️ Clear Database before starting! Old data exists.")
            
            # Database management buttons
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("Refresh Data", type="primary", key="refresh_raw"):
                    st.rerun()
            
            with col2:
                if st.button("Start", type="primary", help="Run main.py to start the trading system", key="start_raw"):
                    try:
                        # Auto-backup if database has data (backup_database also deletes the original)
                        if db_has_tables_raw:
                            st.info("📦 Backing up existing database...")
                            backup_result = trades_db.backup_database()
                            if backup_result:
                                st.success("✅ Database backed up and cleared! Starting system...")
                            else:
                                st.error("❌ Failed to backup database. Not starting.")
                                st.stop()
                        
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
                            st.error("main.py not found!")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col3:
                if st.button("Save Database", type="secondary", help="Backup the current database", key="save_db_simplified"):
                    try:
                        result = trades_db.backup_database()
                        if result:
                            st.session_state.db_saved_raw = True
                            st.success("Database backed up successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to backup database")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col4:
                if st.button("Clear Database", type="secondary", help="Delete the database file", key="clear_db_simplified"):
                    try:
                        if os.path.exists('trades.db'):
                            # Close any existing connections first
                            try:
                                conn = get_db_connection()
                                if conn:
                                    conn.close()
                            except:
                                pass
                            
                            # Wait a moment for connections to close
                            import time
                            time.sleep(0.1)
                            
                            # Try to delete the file
                            os.remove('trades.db')
                            st.session_state.db_saved_raw = False
                            
                            # Verify deletion
                            if not os.path.exists('trades.db'):
                                st.success("Database deleted successfully!")
                            else:
                                st.warning("Database file still exists - may be locked by another process")
                            st.rerun()
                        else:
                            st.warning("Database file does not exist")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col5:
                if st.button("Reset Positions", type="secondary", help="Close all open positions on IBKR and update database", key="reset_positions_simplified"):
                    try:
                        
                        
                        with st.spinner("Closing all open positions on IBKR..."):
                            count, positions = close_all_open_positions()
                        
                        if count > 0:
                            total_pnl = sum(p['pnl'] for p in positions)
                            st.success(f"Successfully closed {count} positions! Total PnL: ${total_pnl:.2f}")
                            
                            # Show details of closed positions
                            if positions:
                                st.write("**Closed Positions:**")
                                for pos in positions:
                                    st.write(f"- {pos['symbol']}: {pos['shares']} shares, PnL: ${pos['pnl']:.2f}")
                            
                            st.rerun()
                        else:
                            st.info("No open positions found to close")
                            
                    except Exception as e:
                        st.error(f"Error closing positions: {str(e)}")
            
            # Connect to database for raw view
            conn = get_db_connection()
            if not conn:
                st.error("Could not connect to database. Make sure `trades.db` exists in the current directory.")
                return
            
            # Check if stock_strategies table exists
            table_info = get_table_info(conn)
            if 'stock_strategies' not in table_info:
                st.error("The 'stock_strategies' table does not exist in the database.")
                st.info("This usually means the database hasn't been properly initialized. Try running the main trading application first.")
                conn.close()
                return
            
            # Get data from stock_strategies table (raw database view)
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
                                'take_profit_price', 'used_capital', 'unrealized_pnl', 'realized_pnl', 'hedge_shares', 
                                'hedge_level', 'hedge_beta', 'hedge_entry_price', 'hedge_exit_price', 'hedge_pnl']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                # Calculate PnL metrics (raw database view)
                if 'unrealized_pnl' in df.columns:
                    total_unrealized_pnl = df['unrealized_pnl'].sum()
                    total_realized_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                    total_combined_pnl = total_unrealized_pnl + total_realized_pnl
                    active_positions = len(df[df['position_active'] == True]) if 'position_active' in df.columns else 0
                    
                    # Display PnL metrics in a single row with 4 columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Unrealized PnL", f"${total_unrealized_pnl:,.2f}")
                    
                    with col2:
                        st.metric("Total Realized PnL", f"${total_realized_pnl:,.2f}")
                    
                    with col3:
                        st.metric("Total Combined PnL", f"${total_combined_pnl:,.2f}")
                    
                    with col4:
                        st.metric("Active Positions", active_positions)
                    
                    # Additional metrics in a second row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Filter for active positions
                    active_df = df[df['position_active'] == True] if 'position_active' in df.columns else df
                    
                    with col1:
                        if 'used_capital' in active_df.columns:
                            total_used_capital = active_df['used_capital'].sum()
                            st.metric("Total Capital Deployed", f"${total_used_capital:,.2f}")
                        else:
                            total_used_capital = 0
                    
                    with col2:
                        # Calculate total cost basis (sum of all positions)
                        total_cost_basis = (df['shares'] * df['entry_price']).sum()
                        if total_cost_basis > 0:
                            # Calculate intraday return %
                            intraday_return_pct = ((total_realized_pnl + total_unrealized_pnl) / total_cost_basis) * 100
                            st.metric("Intraday Return %", f"{intraday_return_pct:.2f}%")
                        else:
                            st.metric("Intraday Return %", "0.00%")
                    
                    with col3:
                        # Total Portfolio Value = used_capital + unrealized_pnl
                        total_portfolio_value = total_used_capital + total_unrealized_pnl
                        st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
                    
                    with col4:
                        # Available Cash = EQUITY - used_capital
                        equity = config.get('EQUITY', 0)
                        available_cash = equity - total_used_capital
                        st.metric("Cash Available", f"${available_cash:,.2f}")
                    
                    # Trading statistics in a third row
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        # Exposure Ratio = (total_used_capital / equity) * 100
                        if equity > 0:
                            exposure_ratio = (total_used_capital / equity) * 100
                            st.metric("Exposure Ratio", f"{exposure_ratio:.2f}%")
                        else:
                            st.metric("Exposure Ratio", "0.00%")
                    
                    with col2:
                        # Avg Position Size = total_used_capital / number of active positions
                        if active_positions > 0:
                            avg_position_size = total_used_capital / active_positions
                            st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                        else:
                            st.metric("Avg Position Size", "$0.00")
                    
                    with col3:
                        # Win Rate % = (profitable closed positions / total closed positions) * 100
                        # Closed positions are those where position_shares = 0
                        closed_positions = df[df['position_shares'] == 0] if 'position_shares' in df.columns else pd.DataFrame()
                        if len(closed_positions) > 0:
                            profitable_closed = len(closed_positions[closed_positions['realized_pnl'] > 0])
                            losing_closed = len(closed_positions[closed_positions['realized_pnl'] < 0])
                            total_closed = profitable_closed + losing_closed
                            if total_closed > 0:
                                win_rate = (profitable_closed / total_closed) * 100
                                st.metric("Win Rate %", f"{win_rate:.2f}%")
                            else:
                                st.metric("Win Rate %", "N/A")
                        else:
                            st.metric("Win Rate %", "N/A")
                    
                    with col4:
                        # Profit Factor = gross profit / gross loss
                        gross_profit = df[df['realized_pnl'] > 0]['realized_pnl'].sum()
                        gross_loss = abs(df[df['realized_pnl'] < 0]['realized_pnl'].sum())
                        if gross_loss > 0:
                            profit_factor = gross_profit / gross_loss
                            st.metric("Profit Factor", f"{profit_factor:.2f}")
                        else:
                            st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                    
                    with col5:
                        # Average PnL per Trade
                        total_trades = len(df)
                        if total_trades > 0:
                            avg_pnl_per_trade = (total_realized_pnl + total_unrealized_pnl) / total_trades
                            st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                        else:
                            st.metric("Avg PnL/Trade", "$0.00")
                    
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
                    file_name=f"trades_db_{datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                if df.empty:
                    st.info("No data found in trades.db")
            
            conn.close()
        
        # Auto-refresh every 7 seconds
            st_autorefresh(interval=7000, key="raw_database_refresh")
        
    
    elif page == "Historical Data":
        st.header("Historical Data Analysis")
        st.markdown("View and analyze historical trading data from previous sessions")
        
        # Get available historical databases
        historical_files = trades_db.get_historical_databases()
        
        if not historical_files:
            st.warning("No historical data found. Run the trading system first to generate historical data.")
            st.info("Historical data is automatically backed up when you start main.py")
        else:
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Symbol Performance", "File Explorer", "Database Viewer"])
            
            with tab1:
                st.subheader("Summary Statistics")
                
                # Date range selector
                col1, col2, col3 = st.columns(3)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7), key="summary_start_date")
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now(), key="summary_end_date")
                
                # Get aggregated data for the full range first
                # Convert date objects to datetime objects for comparison
                start_datetime = datetime.combine(start_date, datetime.min.time()) if start_date else None
                end_datetime = datetime.combine(end_date, datetime.max.time()) if end_date else None
                
                data = trades_db.get_aggregated_data(start_date=start_datetime, end_date=end_datetime)
                st.write(f"Found {len(data) if data else 0} records")
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Calculate key metrics
                    total_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                    
                    # Calculate total cost basis for all positions
                    total_cost_basis = (df['shares'] * df['entry_price']).sum() if 'shares' in df.columns and 'entry_price' in df.columns else 0
                    
                    # Row 1: Primary Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total PnL", f"${total_pnl:,.2f}")
                    
                    with col2:
                        # Overall Return % = realized_pnl / cost_basis
                        if total_cost_basis > 0:
                            overall_return_pct = (total_pnl / total_cost_basis) * 100
                            st.metric("Overall Return %", f"{overall_return_pct:.2f}%")
                        else:
                            st.metric("Overall Return %", "0.00%")
                    
                    with col3:
                        # Win Rate % = (trades with +ve realized_pnl) / (total positions with shares > 0)
                        positions_with_shares = df[df['shares'] > 0] if 'shares' in df.columns else pd.DataFrame()
                        if len(positions_with_shares) > 0:
                            winning_positions = len(df[df['realized_pnl'] > 0])
                            win_rate = (winning_positions / len(positions_with_shares)) * 100
                            st.metric("Win Rate %", f"{win_rate:.2f}%")
                        else:
                            st.metric("Win Rate %", "N/A")
                    
                    # Row 2: Trading Statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Avg Position Size = total cost basis / number of positions
                        if len(df) > 0:
                            avg_position_size = total_cost_basis / len(df)
                            st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                        else:
                            st.metric("Avg Position Size", "$0.00")
                    
                    with col2:
                        # Profit Factor = gross profit / gross loss
                        gross_profit = df[df['realized_pnl'] > 0]['realized_pnl'].sum()
                        gross_loss = abs(df[df['realized_pnl'] < 0]['realized_pnl'].sum())
                        if gross_loss > 0:
                            profit_factor = gross_profit / gross_loss
                            st.metric("Profit Factor", f"{profit_factor:.2f}")
                        else:
                            st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                    
                    with col3:
                        # Average PnL per Trade
                        if len(df) > 0:
                            avg_pnl_per_trade = total_pnl / len(df)
                            st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                        else:
                            st.metric("Avg PnL/Trade", "$0.00")
                    
                    # Row 3: Win/Loss Analysis
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Average Win
                        winning_trades = df[df['realized_pnl'] > 0]
                        if len(winning_trades) > 0:
                            avg_win = winning_trades['realized_pnl'].mean()
                            st.metric("Avg Win", f"${avg_win:,.2f}")
                        else:
                            st.metric("Avg Win", "$0.00")
                    
                    with col2:
                        # Average Loss
                        losing_trades = df[df['realized_pnl'] < 0]
                        if len(losing_trades) > 0:
                            avg_loss = losing_trades['realized_pnl'].mean()
                            st.metric("Avg Loss", f"${avg_loss:,.2f}")
                        else:
                            st.metric("Avg Loss", "$0.00")
                    
                    
                    st.markdown("---")
                    
                    # Performance chart
                    st.subheader("Performance Over Time")
                    
                    # Group by date and calculate daily metrics
                    if 'data_date' in df.columns:
                        daily_metrics = df.groupby('data_date').agg({
                            'realized_pnl': 'sum'
                        }).reset_index()
                        
                        daily_metrics['pnl'] = daily_metrics['realized_pnl']
                        daily_metrics['cumulative_pnl'] = daily_metrics['pnl'].cumsum()
                        daily_metrics['data_date'] = pd.to_datetime(daily_metrics['data_date'].str.split('_').str[0])
                        
                        # Create two columns for charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Daily PnL")
                            st.line_chart(daily_metrics.set_index('data_date')[['pnl']])
                        
                        with col2:
                            st.subheader("Cumulative PnL")
                            st.line_chart(daily_metrics.set_index('data_date')[['cumulative_pnl']])
                    
                    # All Trades View
                    st.subheader("All Trades")
                    
                    # Add filter options in a single row
                    col1, col2 = st.columns(2)
                    with col1:
                        executed_trades_only = st.checkbox("Executed Trades", value=True, key="executed_trades_filter")
                    with col2:
                        qualified_stocks_only_summary = st.checkbox("Qualified Stocks", value=False, key="qualified_stocks_filter_summary")
                    
                    # Filter based on checkboxes
                    df_filtered = df.copy()
                    
                    if executed_trades_only:
                        if 'shares' in df_filtered.columns:
                            df_filtered = df_filtered[df_filtered['shares'] > 0]
                    
                    if qualified_stocks_only_summary:
                        if 'shares' in df_filtered.columns:
                            df_filtered = df_filtered[df_filtered['shares'] == 0]
                    
                    df_filtered = df_filtered.copy()
                    
                    # Create simplified display format for all trades
                    all_trades_display = pd.DataFrame({
                        'Date': df_filtered['data_date'].fillna('N/A') if 'data_date' in df_filtered.columns else 'N/A',
                        'Symbol': df_filtered['symbol'].fillna('N/A') if 'symbol' in df_filtered.columns else 'N/A',
                        'Quantity': df_filtered['shares'] if 'shares' in df_filtered.columns else 0,
                        'Entry Price': df_filtered['entry_price'] if 'entry_price' in df_filtered.columns else 0,
                        'Exit Price': df_filtered['current_price'] if 'current_price' in df_filtered.columns else 0,
                        'Realized PnL': df_filtered['realized_pnl'] if 'realized_pnl' in df_filtered.columns else 0
                    })
                    
                    # Ensure numeric types for calculations and fill missing values
                    all_trades_display['Quantity'] = pd.to_numeric(all_trades_display['Quantity'], errors='coerce').fillna(0)
                    all_trades_display['Entry Price'] = pd.to_numeric(all_trades_display['Entry Price'], errors='coerce').fillna(0)
                    all_trades_display['Exit Price'] = pd.to_numeric(all_trades_display['Exit Price'], errors='coerce').fillna(0)
                    all_trades_display['Realized PnL'] = pd.to_numeric(all_trades_display['Realized PnL'], errors='coerce').fillna(0)
                    
                    # Clean up date format - extract just the date part
                    if 'Date' in all_trades_display.columns:
                        all_trades_display['Date'] = all_trades_display['Date'].astype(str).str.split('_').str[0]
                    
                    # Calculate additional columns
                    all_trades_display['Cost Basis'] = all_trades_display['Quantity'] * all_trades_display['Entry Price']
                    all_trades_display['Market Value'] = all_trades_display['Quantity'] * all_trades_display['Exit Price']
                    
                    # Calculate PnL percentage
                    all_trades_display['PnL (%)'] = all_trades_display.apply(
                        lambda row: (row['Realized PnL'] / row['Cost Basis'] * 100) if row['Cost Basis'] != 0 else 0,
                        axis=1
                    ).round(2)
                    
                    # Select and reorder columns for display
                    display_columns = [
                        'Date', 'Symbol', 'Quantity', 'Entry Price', 'Exit Price',
                        'Cost Basis', 'Market Value', 'Realized PnL', 'PnL (%)'
                    ]
                    all_trades_display = all_trades_display[display_columns]
                    
                    # Sort by date (most recent first)
                    all_trades_display = all_trades_display.sort_values('Date', ascending=False)
                    
                    # Format the data with commas and dollar signs
                    def format_currency(value):
                        if pd.isna(value):
                            return "$0.00"
                        return f"${value:,.2f}"
                    
                    def format_percentage(value):
                        if pd.isna(value):
                            return "0.00%"
                        return f"{value:.2f}%"
                    
                    # Apply formatting to the display dataframe
                    formatted_all_trades = all_trades_display.copy()
                    formatted_all_trades['Entry Price'] = formatted_all_trades['Entry Price'].apply(format_currency)
                    formatted_all_trades['Exit Price'] = formatted_all_trades['Exit Price'].apply(format_currency)
                    formatted_all_trades['Cost Basis'] = formatted_all_trades['Cost Basis'].apply(format_currency)
                    formatted_all_trades['Market Value'] = formatted_all_trades['Market Value'].apply(format_currency)
                    formatted_all_trades['Realized PnL'] = formatted_all_trades['Realized PnL'].apply(format_currency)
                    formatted_all_trades['PnL (%)'] = formatted_all_trades['PnL (%)'].apply(format_percentage)
                    
                    # Display the formatted table
                    st.dataframe(formatted_all_trades, use_container_width=True)
                    
                    # Download button for all trades
                    csv = all_trades_display.to_csv(index=False)
                    st.download_button(
                        label="Download All Trades as CSV",
                        data=csv,
                        file_name=f"all_trades_{datetime.now(pytz.timezone('US/Eastern')).strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with tab2:
                st.subheader("Symbol Performance Analysis")
                
                # Date range selector for Symbol Performance
                col1, col2, col3 = st.columns(3)
                with col1:
                    symbol_start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=7), key="symbol_start_date")
                with col2:
                    symbol_end_date = st.date_input("End Date", value=datetime.now(), key="symbol_end_date")

                
                # Get all symbols from historical data
                # Convert date objects to datetime objects for comparison
                start_datetime = datetime.combine(symbol_start_date, datetime.min.time()) if symbol_start_date else None
                end_datetime = datetime.combine(symbol_end_date, datetime.max.time()) if symbol_end_date else None
                all_data = trades_db.get_aggregated_data(start_date=start_datetime, end_date=end_datetime)
                st.write(f"Found {len(all_data) if all_data else 0} records")
                if all_data:
                    df = pd.DataFrame(all_data)
                    symbols = sorted(df['symbol'].unique())
                    
                    # Initialize session state for selected symbol if it doesn't exist
                    if 'selected_symbol_perf' not in st.session_state:
                        st.session_state.selected_symbol_perf = symbols[0] if symbols else None
                    
                    # Check if previously selected symbol still exists in current data
                    default_index = 0
                    if st.session_state.selected_symbol_perf in symbols:
                        default_index = symbols.index(st.session_state.selected_symbol_perf)
                    else:
                        # If previous symbol not in list, reset to first symbol
                        st.session_state.selected_symbol_perf = symbols[0] if symbols else None
                    
                    selected_symbol = st.selectbox(
                        "Select Symbol:", 
                        symbols, 
                        index=default_index,
                        key="symbol_selectbox_perf"
                    )
                    
                    # Update session state with current selection
                    st.session_state.selected_symbol_perf = selected_symbol
                    
                    if selected_symbol:
                        symbol_data = df[df['symbol'] == selected_symbol].copy()
                        
                        # Calculate key metrics for this symbol
                        total_realized_pnl = symbol_data['realized_pnl'].sum() if 'realized_pnl' in symbol_data.columns else 0
                        total_cost_basis = (symbol_data['shares'] * symbol_data['entry_price']).sum() if 'shares' in symbol_data.columns and 'entry_price' in symbol_data.columns else 0
                        
                        # Row 1: Core Metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total PnL", f"${total_realized_pnl:,.2f}")
                        
                        with col2:
                            # Overall Return % = realized_pnl / cost_basis
                            if total_cost_basis > 0:
                                overall_return_pct = (total_realized_pnl / total_cost_basis) * 100
                                st.metric("Overall Return %", f"{overall_return_pct:.2f}%")
                            else:
                                st.metric("Overall Return %", "0.00%")
                        
                        # Row 2: Trading Statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Avg Position Size = total cost basis / number of positions
                            if len(symbol_data) > 0:
                                avg_position_size = total_cost_basis / len(symbol_data)
                                st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                            else:
                                st.metric("Avg Position Size", "$0.00")
                        
                        with col2:
                            # Win Rate % = (trades with +ve realized_pnl) / (total positions with shares > 0)
                            positions_with_shares = symbol_data[symbol_data['shares'] > 0] if 'shares' in symbol_data.columns else pd.DataFrame()
                            if len(positions_with_shares) > 0:
                                winning_positions = len(symbol_data[symbol_data['realized_pnl'] > 0])
                                win_rate = (winning_positions / len(positions_with_shares)) * 100
                                st.metric("Win Rate %", f"{win_rate:.2f}%")
                            else:
                                st.metric("Win Rate %", "N/A")
                        
                        with col3:
                            # Profit Factor = gross profit / gross loss
                            gross_profit = symbol_data[symbol_data['realized_pnl'] > 0]['realized_pnl'].sum()
                            gross_loss = abs(symbol_data[symbol_data['realized_pnl'] < 0]['realized_pnl'].sum())
                            if gross_loss > 0:
                                profit_factor = gross_profit / gross_loss
                                st.metric("Profit Factor", f"{profit_factor:.2f}")
                            else:
                                st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                        
                        # Row 3: Win/Loss Analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Average Win
                            winning_trades = symbol_data[symbol_data['realized_pnl'] > 0]
                            if len(winning_trades) > 0:
                                avg_win = winning_trades['realized_pnl'].mean()
                                st.metric("Avg Win", f"${avg_win:,.2f}")
                            else:
                                st.metric("Avg Win", "$0.00")
                        
                        with col2:
                            # Average Loss
                            losing_trades = symbol_data[symbol_data['realized_pnl'] < 0]
                            if len(losing_trades) > 0:
                                avg_loss = losing_trades['realized_pnl'].mean()
                                st.metric("Avg Loss", f"${avg_loss:,.2f}")
                            else:
                                st.metric("Avg Loss", "$0.00")
                        
                        with col3:
                            # Average PnL per Trade
                            if len(symbol_data) > 0:
                                avg_pnl_per_trade = total_realized_pnl / len(symbol_data)
                                st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                            else:
                                st.metric("Avg PnL/Trade", "$0.00")
                        
                        st.markdown("---")
                        
                        # Symbol performance over time
                        st.subheader(f"{selected_symbol} Performance Over Time")
                        
                        symbol_data.loc[:, 'data_date_clean'] = pd.to_datetime(symbol_data['data_date'].str.split('_').str[0])
                        symbol_data_sorted = symbol_data.sort_values('data_date_clean')
                        
                        # Create two columns for charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"{selected_symbol} PnL Over Time")
                            st.line_chart(
                                symbol_data_sorted.set_index('data_date_clean')[['realized_pnl']]
                            )
                        
                        with col2:
                            st.subheader(f"{selected_symbol} Position Size Over Time")
                            st.line_chart(
                                symbol_data_sorted.set_index('data_date_clean')[['shares']]
                            )
                        
                        # Detailed symbol data - exclude datetime columns that cause Arrow issues
                        st.subheader(f"{selected_symbol} Detailed Data")
                        
                        # Add view toggle
                        view_mode = st.radio(
                            "Select View Mode:",
                            ["Simplified View", "Raw Data"],
                            key=f"symbol_view_{selected_symbol}"
                        )
                        
                        if view_mode == "Simplified View":
                            # Create simplified display format
                            simplified_display = pd.DataFrame({
                                'Date': symbol_data['data_date'],
                                'Symbol': symbol_data['symbol'],
                                'Quantity': symbol_data['position_shares'],
                                'Entry Price': symbol_data['entry_price'],
                                'Exit Price': symbol_data['current_price'],
                                'Unrealized PnL': symbol_data['unrealized_pnl'],
                                'Realized PnL': symbol_data['realized_pnl']
                            })
                            
                            # Calculate additional columns
                            simplified_display['Cost Basis'] = simplified_display['Quantity'] * simplified_display['Entry Price']
                            simplified_display['Market Value'] = simplified_display['Quantity'] * simplified_display['Exit Price']
                            
                            # Calculate total PnL (unrealized + realized)
                            total_pnl = simplified_display['Unrealized PnL'] + simplified_display['Realized PnL']
                            simplified_display['PnL ($)'] = total_pnl
                            
                            # Calculate percentage based on total PnL relative to cost basis
                            simplified_display['PnL (%)'] = (total_pnl / simplified_display['Cost Basis'] * 100).round(2)
                            
                            # Select and reorder columns for display
                            display_columns = [
                                'Date', 'Symbol', 'Quantity', 'Entry Price', 'Exit Price',
                                'Cost Basis', 'Market Value', 'PnL ($)', 'PnL (%)'
                            ]
                            simplified_display = simplified_display[display_columns]
                            
                            # Format the data with commas and dollar signs
                            def format_currency(value):
                                if pd.isna(value):
                                    return "$0.00"
                                return f"${value:,.2f}"
                            
                            def format_percentage(value):
                                if pd.isna(value):
                                    return "0.00%"
                                return f"{value:.2f}%"
                            
                            # Apply formatting to the display dataframe
                            formatted_display = simplified_display.copy()
                            formatted_display['Entry Price'] = formatted_display['Entry Price'].apply(format_currency)
                            formatted_display['Exit Price'] = formatted_display['Exit Price'].apply(format_currency)
                            formatted_display['Cost Basis'] = formatted_display['Cost Basis'].apply(format_currency)
                            formatted_display['Market Value'] = formatted_display['Market Value'].apply(format_currency)
                            formatted_display['PnL ($)'] = formatted_display['PnL ($)'].apply(format_currency)
                            formatted_display['PnL (%)'] = formatted_display['PnL (%)'].apply(format_percentage)
                            
                            # Display the formatted table
                            st.dataframe(formatted_display, use_container_width=True)
                        else:
                            # Raw data view
                            display_data = symbol_data[['data_date', 'position_active', 'position_shares', 'shares',
                                                    'entry_price', 'current_price', 'realized_pnl',
                                                    'score']].copy()
                            
                            # Convert datetime columns to strings to avoid Arrow conversion issues
                            if 'entry_time' in symbol_data.columns:
                                display_data['entry_time'] = symbol_data['entry_time'].astype(str)
                            if 'close_time' in symbol_data.columns:
                                display_data['close_time'] = symbol_data['close_time'].astype(str)
                            
                            st.dataframe(display_data, use_container_width=True)
                else:
                    st.warning("No historical data available for symbol analysis")
            
            with tab3:
                st.subheader("Historical File Explorer")
                
                # Display all historical files
                st.write("Available historical database files:")
                
                if historical_files:
                    # Create a file selector for deletion
                    file_options = [f"{f['filename']} ({f['date']})" for f in historical_files]
                    selected_file_idx = st.selectbox(
                        "Select Database File to Delete:",
                        range(len(file_options)),
                        format_func=lambda x: file_options[x],
                        key="delete_file_selector"
                    )
                    
                    if selected_file_idx is not None:
                        selected_file = historical_files[selected_file_idx]
                        
                        # Show file details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Selected File", selected_file['filename'])
                        with col2:
                            st.metric("Size", f"{round(selected_file['size'] / 1024, 2)} KB")
                        with col3:
                            st.metric("Modified", selected_file['modified'].strftime('%Y-%m-%d'))
                        
                        # Delete button
                        if st.button("Delete Selected Database", type="secondary", help=f"Delete {selected_file['filename']}"):
                            try:
                                if os.path.exists(selected_file['path']):
                                    os.remove(selected_file['path'])
                                    
                                    # Verify deletion
                                    if not os.path.exists(selected_file['path']):
                                        st.success(f"Successfully deleted {selected_file['filename']}")
                                        st.rerun()
                                    else:
                                        st.warning("File still exists - may be locked by another process")
                                else:
                                    st.warning("File does not exist")
                            except Exception as e:
                                st.error(f"Error deleting file: {str(e)}")
                    
                # Display table of all files
                file_data = []
                for file_info in historical_files:
                    file_data.append({
                        'Filename': file_info['filename'],
                        'Date': file_info['date'],
                        'Size (KB)': round(file_info['size'] / 1024, 2),
                        'Modified': file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                df_files = pd.DataFrame(file_data)
                st.dataframe(df_files, use_container_width=True)
                
                # File actions
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Refresh File List", type="primary"):
                        st.rerun()
                
                with col2:
                    if st.button("Load All Data"):
                        # Use default date range for file explorer (last 30 days)
                        file_start_date = datetime.now() - timedelta(days=30)
                        file_end_date = datetime.now()
                        start_datetime = datetime.combine(file_start_date, datetime.min.time())
                        end_datetime = datetime.combine(file_end_date, datetime.max.time())
                        all_data = trades_db.get_aggregated_data(start_date=start_datetime, end_date=end_datetime)
                        if all_data:
                            st.success(f"Loaded {len(all_data)} records from all historical files")
                            
                            # Export option
                            csv_data = pd.DataFrame(all_data).to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name=f"historical_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No data found in historical files")
            
            with tab4:
                st.subheader("Database Viewer")
                st.markdown("View raw and simplified data from any historical database file")
                
                # Database file selector
                if historical_files:
                    file_options = [f"{f['filename']} ({f['date']})" for f in historical_files]
                    selected_file_idx = st.selectbox(
                        "Select Database File:",
                        range(len(file_options)),
                        format_func=lambda x: file_options[x]
                    )
                    
                    if selected_file_idx is not None:
                        selected_file = historical_files[selected_file_idx]
                        
                        # Load data from selected file
                        data = trades_db.load_historical_data(selected_file['path'])
                        
                        if data:
                            df = pd.DataFrame(data)
                            
                            # File info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("File", selected_file['filename'])
                            with col2:
                                st.metric("Records", len(df))
                            with col3:
                                st.metric("Symbols", df['symbol'].nunique())
                            
                            st.markdown("---")
                            
                            # View type selector
                            view_type = st.radio(
                                "Select View Type:",
                                ["Simplified View", "Raw Database"],
                                horizontal=True
                            )
                            
                            if view_type == "Simplified View":
                                st.subheader("Simplified View")
                                
                                # Calculate key metrics
                                total_realized_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                                total_cost_basis = (df['shares'] * df['entry_price']).sum() if 'shares' in df.columns and 'entry_price' in df.columns else 0
                                
                                # Row 1: Core Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total PnL", f"${total_realized_pnl:,.2f}")
                                
                                with col2:
                                    # Overall Return % = realized_pnl / cost_basis
                                    if total_cost_basis > 0:
                                        overall_return_pct = (total_realized_pnl / total_cost_basis) * 100
                                        st.metric("Overall Return %", f"{overall_return_pct:.2f}%")
                                    else:
                                        st.metric("Overall Return %", "0.00%")
                                
                                
                                # Row 2: Trading Statistics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Avg Position Size = total cost basis / number of positions
                                    if len(df) > 0:
                                        avg_position_size = total_cost_basis / len(df)
                                        st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                                    else:
                                        st.metric("Avg Position Size", "$0.00")
                                
                                with col2:
                                    # Win Rate % = (trades with +ve realized_pnl) / (total positions with shares > 0)
                                    positions_with_shares = df[df['shares'] > 0] if 'shares' in df.columns else pd.DataFrame()
                                    if len(positions_with_shares) > 0:
                                        winning_positions = len(df[df['realized_pnl'] > 0])
                                        win_rate = (winning_positions / len(positions_with_shares)) * 100
                                        st.metric("Win Rate %", f"{win_rate:.2f}%")
                                    else:
                                        st.metric("Win Rate %", "N/A")
                                
                                with col3:
                                    # Profit Factor = gross profit / gross loss
                                    gross_profit = df[df['realized_pnl'] > 0]['realized_pnl'].sum()
                                    gross_loss = abs(df[df['realized_pnl'] < 0]['realized_pnl'].sum())
                                    if gross_loss > 0:
                                        profit_factor = gross_profit / gross_loss
                                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                                    else:
                                        st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                                
                                # Row 3: Win/Loss Analysis
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Average Win
                                    winning_trades = df[df['realized_pnl'] > 0]
                                    if len(winning_trades) > 0:
                                        avg_win = winning_trades['realized_pnl'].mean()
                                        st.metric("Avg Win", f"${avg_win:,.2f}")
                                    else:
                                        st.metric("Avg Win", "$0.00")
                                
                                with col2:
                                    # Average Loss
                                    losing_trades = df[df['realized_pnl'] < 0]
                                    if len(losing_trades) > 0:
                                        avg_loss = losing_trades['realized_pnl'].mean()
                                        st.metric("Avg Loss", f"${avg_loss:,.2f}")
                                    else:
                                        st.metric("Avg Loss", "$0.00")
                                
                                with col3:
                                    # Average PnL per Trade
                                    if len(df) > 0:
                                        avg_pnl_per_trade = total_realized_pnl / len(df)
                                        st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                                    else:
                                        st.metric("Avg PnL/Trade", "$0.00")
                                
                                # Row 4: Additional Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Trades Executed = count of trades where shares > 0
                                    if 'shares' in df.columns:
                                        trades_executed = len(df[df['shares'] > 0])
                                        st.metric("Trades Executed", trades_executed)
                                    else:
                                        st.metric("Trades Executed", 0)
                                
                                with col2:
                                    # Average Trade Duration (for closed positions)
                                    if 'entry_time' in df.columns and 'close_time' in df.columns:
                                        # Filter for closed positions with valid times
                                        closed_with_times = df[(df['position_shares'] == 0) & 
                                                               (df['entry_time'].notna()) & 
                                                               (df['close_time'].notna())].copy()
                                        if len(closed_with_times) > 0:
                                            # Convert to datetime and calculate durations
                                            closed_with_times['entry_dt'] = pd.to_datetime(closed_with_times['entry_time'])
                                            closed_with_times['close_dt'] = pd.to_datetime(closed_with_times['close_time'])
                                            closed_with_times['duration'] = closed_with_times['close_dt'] - closed_with_times['entry_dt']
                                            
                                            # Calculate average duration
                                            avg_duration = closed_with_times['duration'].mean()
                                            
                                            # Format duration
                                            total_seconds = int(avg_duration.total_seconds())
                                            hours = total_seconds // 3600
                                            minutes = (total_seconds % 3600) // 60
                                            seconds = total_seconds % 60
                                            
                                            if hours > 0:
                                                duration_str = f"{hours}h {minutes}m"
                                            elif minutes > 0:
                                                duration_str = f"{minutes}m {seconds}s"
                                            else:
                                                duration_str = f"{seconds}s"
                                            
                                            st.metric("Avg Trade Duration", duration_str)
                                        else:
                                            st.metric("Avg Trade Duration", "N/A")
                                    else:
                                        st.metric("Avg Trade Duration", "N/A")
                                
                                st.markdown("---")
                                
                                # Add filter options in a single row
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    qualified_stocks_only_hist_simp = st.checkbox("Qualified Stocks", value=False, key="qualified_stocks_only_hist_simp")
                                with col2:
                                    executed_trades_only_hist_simp = st.checkbox("Executed Trades", value=False, key="executed_trades_only_hist_simp")
                                
                                # Apply filters to display dataframe only (not affecting metrics)
                                simplified_df = df.copy()
                                
                                if qualified_stocks_only_hist_simp:
                                    if 'shares' in simplified_df.columns:
                                        simplified_df = simplified_df[simplified_df['shares'] == 0]
                                
                                if executed_trades_only_hist_simp:
                                    if 'shares' in simplified_df.columns:
                                        simplified_df = simplified_df[simplified_df['shares'] > 0]
                                
                                simplified_df = simplified_df.copy()
                                
                                if not simplified_df.empty:
                                    # Create simplified display format (same as Top Performing Symbols)
                                    simplified_display = pd.DataFrame({
                                        'Symbol': simplified_df['symbol'],
                                        'Quantity': simplified_df['shares'],
                                        'Entry Price': simplified_df['entry_price'],
                                        'Exit Price': simplified_df['current_price'],
                                        'Unrealized PnL': simplified_df['unrealized_pnl'],
                                        'Realized PnL': simplified_df['realized_pnl'],
                                        'Entry Time': simplified_df['entry_time'] if 'entry_time' in simplified_df.columns else 'N/A',
                                        'Close Time': simplified_df['close_time'] if 'close_time' in simplified_df.columns else 'N/A'
                                    })
                                    
                                    # Ensure numeric types for calculations
                                    simplified_display['Quantity'] = pd.to_numeric(simplified_display['Quantity'], errors='coerce').fillna(0)
                                    simplified_display['Entry Price'] = pd.to_numeric(simplified_display['Entry Price'], errors='coerce').fillna(0)
                                    simplified_display['Exit Price'] = pd.to_numeric(simplified_display['Exit Price'], errors='coerce').fillna(0)
                                    simplified_display['Unrealized PnL'] = pd.to_numeric(simplified_display['Unrealized PnL'], errors='coerce').fillna(0)
                                    simplified_display['Realized PnL'] = pd.to_numeric(simplified_display['Realized PnL'], errors='coerce').fillna(0)
                                    
                                    # Calculate additional columns
                                    simplified_display['Cost Basis'] = simplified_display['Quantity'] * simplified_display['Entry Price']
                                    simplified_display['Market Value'] = simplified_display['Quantity'] * simplified_display['Exit Price']
                                    
                                    # Calculate total PnL (unrealized + realized)
                                    total_pnl = simplified_display['Unrealized PnL'] + simplified_display['Realized PnL']
                                    simplified_display['PnL ($)'] = total_pnl
                                    
                                    # Calculate percentage based on total PnL relative to cost basis (handle division by zero)
                                    simplified_display['PnL (%)'] = simplified_display.apply(
                                        lambda row: (row['PnL ($)'] / row['Cost Basis'] * 100) if row['Cost Basis'] != 0 else 0, 
                                        axis=1
                                    ).round(2)
                                    
                                    # Select and reorder columns for display (same as Top Performing Symbols)
                                    display_columns = [
                                        'Symbol', 'Quantity', 'Entry Price', 'Exit Price',
                                        'Cost Basis', 'Market Value', 'PnL ($)', 'PnL (%)', 'Entry Time', 'Close Time'
                                    ]
                                    simplified_display = simplified_display[display_columns]
                                    
                                    # Format the data with commas and dollar signs
                                    def format_currency(value):
                                        if pd.isna(value):
                                            return "$0.00"
                                        return f"${value:,.2f}"
                                    
                                    def format_percentage(value):
                                        if pd.isna(value):
                                            return "0.00%"
                                        return f"{value:.2f}%"
                                    
                                    def format_time(time_value):
                                        if pd.isna(time_value) or time_value == 'N/A' or str(time_value) in ['None', 'nan', '']:
                                            return 'N/A'
                                        try:
                                            # Try to parse as datetime and extract time
                                            time_obj = pd.to_datetime(time_value)
                                            return time_obj.strftime('%H:%M:%S')
                                        except:
                                            return 'N/A'
                                    
                                    # Apply formatting to the display dataframe
                                    formatted_display = simplified_display.copy()
                                    formatted_display['Entry Price'] = formatted_display['Entry Price'].apply(format_currency)
                                    formatted_display['Exit Price'] = formatted_display['Exit Price'].apply(format_currency)
                                    formatted_display['Cost Basis'] = formatted_display['Cost Basis'].apply(format_currency)
                                    formatted_display['Market Value'] = formatted_display['Market Value'].apply(format_currency)
                                    formatted_display['PnL ($)'] = formatted_display['PnL ($)'].apply(format_currency)
                                    formatted_display['PnL (%)'] = formatted_display['PnL (%)'].apply(format_percentage)
                                    formatted_display['Entry Time'] = formatted_display['Entry Time'].apply(format_time)
                                    formatted_display['Close Time'] = formatted_display['Close Time'].apply(format_time)
                                    
                                    # Display the formatted table
                                    st.dataframe(formatted_display, use_container_width=True)
                                else:
                                    st.info("No active positions found in the database.")
                                
                                # Download simplified data
                                csv_data = simplified_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Simplified Data",
                                    data=csv_data,
                                    file_name=f"simplified_{selected_file['filename'].replace('.db', '.csv')}",
                                    mime="text/csv"
                                )
                                
                            
                            else:  # Raw Database
                                st.subheader("Raw Database View")
                                st.markdown("Complete data from the selected database file")
                                
                                # Calculate key metrics
                                total_realized_pnl = df['realized_pnl'].sum() if 'realized_pnl' in df.columns else 0
                                total_cost_basis = (df['position_shares'] * df['entry_price']).sum() if 'position_shares' in df.columns and 'entry_price' in df.columns else 0
                                
                                # Row 1: Core Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total PnL", f"${total_realized_pnl:,.2f}")
                                
                                with col2:
                                    # Overall Return % = realized_pnl / cost_basis
                                    if total_cost_basis > 0:
                                        overall_return_pct = (total_realized_pnl / total_cost_basis) * 100
                                        st.metric("Overall Return %", f"{overall_return_pct:.2f}%")
                                    else:
                                        st.metric("Overall Return %", "0.00%")
                                
                                
                                # Row 2: Trading Statistics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Avg Position Size = total cost basis / number of positions
                                    if len(df) > 0:
                                        avg_position_size = total_cost_basis / len(df)
                                        st.metric("Avg Position Size", f"${avg_position_size:,.2f}")
                                    else:
                                        st.metric("Avg Position Size", "$0.00")
                                
                                with col2:
                                    # Win Rate % = (trades with +ve realized_pnl) / (total positions with shares > 0)
                                    positions_with_shares = df[df['position_shares'] > 0] if 'position_shares' in df.columns else pd.DataFrame()
                                    if len(positions_with_shares) > 0:
                                        winning_positions = len(df[df['realized_pnl'] > 0])
                                        win_rate = (winning_positions / len(positions_with_shares)) * 100
                                        st.metric("Win Rate %", f"{win_rate:.2f}%")
                                    else:
                                        st.metric("Win Rate %", "N/A")
                                
                                with col3:
                                    # Profit Factor = gross profit / gross loss
                                    gross_profit = df[df['realized_pnl'] > 0]['realized_pnl'].sum()
                                    gross_loss = abs(df[df['realized_pnl'] < 0]['realized_pnl'].sum())
                                    if gross_loss > 0:
                                        profit_factor = gross_profit / gross_loss
                                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                                    else:
                                        st.metric("Profit Factor", "N/A" if gross_profit == 0 else "∞")
                                
                                # Row 3: Win/Loss Analysis
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Average Win
                                    winning_trades = df[df['realized_pnl'] > 0]
                                    if len(winning_trades) > 0:
                                        avg_win = winning_trades['realized_pnl'].mean()
                                        st.metric("Avg Win", f"${avg_win:,.2f}")
                                    else:
                                        st.metric("Avg Win", "$0.00")
                                
                                with col2:
                                    # Average Loss
                                    losing_trades = df[df['realized_pnl'] < 0]
                                    if len(losing_trades) > 0:
                                        avg_loss = losing_trades['realized_pnl'].mean()
                                        st.metric("Avg Loss", f"${avg_loss:,.2f}")
                                    else:
                                        st.metric("Avg Loss", "$0.00")
                                
                                with col3:
                                    # Average PnL per Trade
                                    if len(df) > 0:
                                        avg_pnl_per_trade = total_realized_pnl / len(df)
                                        st.metric("Avg PnL/Trade", f"${avg_pnl_per_trade:,.2f}")
                                    else:
                                        st.metric("Avg PnL/Trade", "$0.00")
                                
                                # Row 4: Additional Metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    # Trades Executed = count of trades where shares > 0
                                    if 'shares' in df.columns:
                                        trades_executed = len(df[df['shares'] > 0])
                                        st.metric("Trades Executed", trades_executed)
                                    else:
                                        st.metric("Trades Executed", 0)
                                
                                with col2:
                                    # Average Trade Duration (for closed positions)
                                    if 'entry_time' in df.columns and 'close_time' in df.columns:
                                        # Filter for closed positions with valid times
                                        closed_with_times = df[(df['position_shares'] == 0) & 
                                                               (df['entry_time'].notna()) & 
                                                               (df['close_time'].notna())].copy()
                                        if len(closed_with_times) > 0:
                                            # Convert to datetime and calculate durations
                                            closed_with_times['entry_dt'] = pd.to_datetime(closed_with_times['entry_time'])
                                            closed_with_times['close_dt'] = pd.to_datetime(closed_with_times['close_time'])
                                            closed_with_times['duration'] = closed_with_times['close_dt'] - closed_with_times['entry_dt']
                                            
                                            # Calculate average duration
                                            avg_duration = closed_with_times['duration'].mean()
                                            
                                            # Format duration
                                            total_seconds = int(avg_duration.total_seconds())
                                            hours = total_seconds // 3600
                                            minutes = (total_seconds % 3600) // 60
                                            seconds = total_seconds % 60
                                            
                                            if hours > 0:
                                                duration_str = f"{hours}h {minutes}m"
                                            elif minutes > 0:
                                                duration_str = f"{minutes}m {seconds}s"
                                            else:
                                                duration_str = f"{seconds}s"
                                            
                                            st.metric("Avg Trade Duration", duration_str)
                                        else:
                                            st.metric("Avg Trade Duration", "N/A")
                                    else:
                                        st.metric("Avg Trade Duration", "N/A")
                                
                                st.markdown("---")
                                
                                # Add filter options in a single row
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    show_live_only_hist_raw = st.checkbox("live position", value=False, key="show_live_positions_hist_raw")
                                with col2:
                                    qualified_stocks_only_hist_raw = st.checkbox("Qualified Stocks", value=False, key="qualified_stocks_only_hist_raw")
                                with col3:
                                    executed_trades_only_hist_raw = st.checkbox("Executed Trades", value=False, key="executed_trades_only_hist_raw")
                                
                                # Apply filters to display dataframe only (not affecting metrics)
                                df_display = df.copy()
                                
                                if show_live_only_hist_raw:
                                    if 'position_active' in df_display.columns:
                                        df_display = df_display[df_display['position_active'] == True]
                                
                                if qualified_stocks_only_hist_raw:
                                    if 'shares' in df_display.columns:
                                        df_display = df_display[df_display['shares'] == 0]
                                
                                if executed_trades_only_hist_raw:
                                    if 'shares' in df_display.columns:
                                        df_display = df_display[df_display['shares'] > 0]
                                
                                df_display = df_display.copy()
                                
                                # Convert datetime columns to strings to avoid Arrow conversion issues
                                datetime_columns = ['entry_time', 'close_time', 'trailing_exit_start_time', 'hedge_entry_time']
                                for col in datetime_columns:
                                    if col in df_display.columns:
                                        df_display[col] = df_display[col].astype(str)
                                
                                # Show all columns
                                st.dataframe(df_display, use_container_width=True)
                                
                                # Download raw data
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="Download Raw Data",
                                    data=csv_data,
                                    file_name=f"raw_{selected_file['filename'].replace('.db', '.csv')}",
                                    mime="text/csv"
                                )
                        else:
                            st.error("Failed to load data from selected file")
                else:
                    st.warning("No historical database files found")
        
    elif page == "Configuration Editor":        
        # Create tabs for different config sections
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Stock Selection", "Entry Rules", "Risk Management", "Drawdown Limits",
            "Stop Loss & Profit", "Hedge & Leverage", "Trading Hours"
        ])
        
        with tab1:
            st.subheader("Stock Selection Configuration")
            
            stock_config = config.get('STOCK_SELECTION', {})
            
            # Market Cap and Price Filter
            st.markdown("---")
            st.write("**Basic Filters**")
            col1, col2 = st.columns(2)
            with col1:
                market_cap_min = st.number_input(
                    "Minimum Market Cap ($):", 
                    min_value=100_000_000, 
                    max_value=10_000_000_000_000, 
                    value=stock_config.get('market_cap_min', 2_000_000_000), 
                    step=100_000_000,
                    help="Minimum market capitalization in dollars"
                )
            with col2:
                price_min = st.number_input(
                    "Minimum Price ($):", 
                    min_value=1.0, 
                    max_value=1000.0, 
                    value=stock_config.get('price_min', 10.0), 
                    step=0.5,
                    help="Minimum stock price in dollars"
                )
            
            st.markdown("---")
            # ADV Configuration
            st.write("**ADV Configuration**")
            col1, col2 = st.columns(2)
            with col1:
                adv_large_length = st.number_input(
                    "ADV Large Length (days):", 
                    min_value=5, 
                    max_value=50, 
                    value=stock_config.get('ADV_large_length', 20), 
                    step=1,
                    help="Number of days for large ADV calculation"
                )
                adv_large = st.number_input(
                    "ADV Large Threshold:", 
                    min_value=100_000, 
                    max_value=10_000_000, 
                    value=stock_config.get('ADV_large', 500_000), 
                    step=50_000,
                    help="Minimum average daily volume for large cap stocks"
                )
            with col2:
                adv_small_length = st.number_input(
                    "ADV Small Length (days):", 
                    min_value=5, 
                    max_value=50, 
                    value=stock_config.get('ADV_small_length', 10), 
                    step=1,
                    help="Number of days for small ADV calculation"
                )
                adv_small = st.number_input(
                    "ADV Small Threshold:", 
                    min_value=100_000, 
                    max_value=10_000_000, 
                    value=stock_config.get('ADV_small', 750_000), 
                    step=50_000,
                    help="Minimum average daily volume for small cap stocks"
                )
            
            st.markdown("---")
            # RVOL Configuration
            st.write("**RVOL Configuration**")
            col1, col2 = st.columns(2)
            with col1:
                rvol_length = st.number_input(
                    "RVOL Length (days):", 
                    min_value=5, 
                    max_value=50, 
                    value=stock_config.get('RVOL_length', 10), 
                    step=1,
                    help="Number of days for relative volume calculation"
                )
            with col2:
                rvol_filter = st.number_input(
                    "RVOL Filter Threshold:", 
                    min_value=0.5, 
                    max_value=5.0, 
                    value=float(stock_config.get('RVOL_filter', 1.3)), 
                    step=0.1,
                    format="%.1f",
                    help="Minimum relative volume multiplier (e.g., 1.3 = 130% of average volume)"
                )
            
            st.markdown("---")
            # Alpha Threshold
            st.write("**Alpha Threshold**")
            col1, col2 = st.columns(2)
            with col1:
                stock_alpha_threshold = st.number_input(
                    "5-Day Alpha Threshold:", 
                    min_value=0.00001, 
                    max_value=0.01, 
                    value=float(stock_config.get('alpha_threshold', 0.005)), 
                    step=0.0001,
                    format="%.3f",
                    help="Minimum 5-day alpha return to qualify (as decimal, e.g., 0.005 = 0.5%)"
                )
            
            # Sector Allocation
            st.write("**Sector Allocation**")
            col1, col2 = st.columns(2)
            with col1:
                max_sector_weight = st.number_input(
                    "Maximum Sector Weight (%):", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=float(stock_config.get('max_sector_weight', 0.30)), 
                    step=0.01,
                    format="%.2f",
                    help="Maximum allocation per sector"
                )
            with col2:
                top_sectors_count = st.number_input(
                    "Top Sectors Count:", 
                    min_value=1, 
                    max_value=10, 
                    value=stock_config.get('top_sectors_count', 3), 
                    step=1,
                    help="Number of top sectors to focus on"
                )
        
        with tab2:
            st.subheader("Entry Rules - Alpha Score Configuration")
            checks_config = config.get('ADDITIONAL_CHECKS_CONFIG', {})
            
            indicators_config = config.get('INDICATORS', {})
            alpha_config = config.get('ALPHA_SCORE_CONFIG', {})
            
            # Trend Analysis
            st.write("**1. Trend Analysis (30% weight)**")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_weight = st.number_input("Trend Weight (%):", 0, 100, alpha_config.get('trend', {}).get('weight', 30), 1, key="trend_weight")            
            # VWAP Configuration
            st.write("VWAP Settings")
            col1, col2 = st.columns(2)
            with col1:
                vwap_timeframes = st.multiselect("VWAP Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('vwap', {}).get('timeframes', ["3min"]), key="vwap_timeframes")
            with col2:
                vwap_below_price = st.checkbox("VWAP Should Be Below Price", value=config.get('VWAP_SHOULD_BE_BELOW_PRICE', True), key="vwap_below_price_global")
            
            # EMA Configuration
            st.write("EMA Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                ema1_length = st.number_input("EMA1 Length (Fast):", 1, 100, indicators_config.get('ema1', {}).get('params', {}).get('length', 5), key="ema1_length")
                ema1_timeframes = st.multiselect("EMA1 Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('ema1', {}).get('timeframes', ["5min"]), key="ema1_timeframes")
            with col2:
                ema2_length = st.number_input("EMA2 Length (Slow):", 1, 100, indicators_config.get('ema2', {}).get('params', {}).get('length', 20), key="ema2_length")
                ema2_timeframes = st.multiselect("EMA2 Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('ema2', {}).get('timeframes', ["20min"]), key="ema2_timeframes")
            with col3:
                ema_cross_weight = st.number_input("EMA Cross Weight (%):", 0, 100, alpha_config.get('trend', {}).get('conditions', {}).get('ema_cross', {}).get('weight', 15), 1, key="ema_cross_weight")
            
            # Momentum Analysis
            st.markdown("---")
            st.write("**2. Momentum Analysis (20% weight)**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                momentum_weight = st.number_input("Momentum Weight (%):", 0, 100, alpha_config.get('momentum', {}).get('weight', 20), 1, key="momentum_weight")
                macd_positive_weight = st.number_input("MACD > 0 Weight (%):", 0, 100, alpha_config.get('momentum', {}).get('conditions', {}).get('macd_positive', {}).get('weight', 20), 1, key="macd_positive_weight")
            with col2:
                macd_fast = st.number_input("MACD Fast Period:", 1, 50, indicators_config.get('macd', {}).get('params', {}).get('fast', 12), key="macd_fast")
                macd_timeframes = st.multiselect("MACD Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('macd', {}).get('timeframes', ["3min"]), key="macd_timeframes")
            with col3:
                macd_slow = st.number_input("MACD Slow Period:", 1, 100, indicators_config.get('macd', {}).get('params', {}).get('slow', 26), key="macd_slow")
            with col4:
                macd_signal = st.number_input("MACD Signal Period:", 1, 50, indicators_config.get('macd', {}).get('params', {}).get('signal', 9), key="macd_signal")
            

            
            # Volume/Volatility Analysis
            st.markdown("---")
            st.write("**3. Volume/Volatility Analysis (20% weight)**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                volume_weight = st.number_input("Volume/Volatility Weight (%):", 0, 100, alpha_config.get('volume_volatility', {}).get('weight', 20), 1, key="volume_weight")
                volume_spike_weight = st.number_input("Volume Spike Weight (%):", 0, 100, alpha_config.get('volume_volatility', {}).get('conditions', {}).get('volume_spike', {}).get('weight', 10), 1, key="volume_spike_weight")
                volume_spike_multiplier = st.number_input("Volume Multiplier:", 0.5, 5.0, alpha_config.get('volume_volatility', {}).get('conditions', {}).get('volume_spike', {}).get('multiplier', 1.5), 0.1, key="volume_spike_mult")
            with col2:
                adx_weight = st.number_input("ADX Weight (%):", 0, 100, alpha_config.get('volume_volatility', {}).get('conditions', {}).get('adx_threshold', {}).get('weight', 10), 1, key="adx_weight")
                adx_threshold = st.number_input("ADX Threshold:", 10, 50, alpha_config.get('volume_volatility', {}).get('conditions', {}).get('adx_threshold', {}).get('threshold', 20), 1, key="adx_threshold")
                adx_length = st.number_input("ADX Length:", 1, 100, indicators_config.get('adx', {}).get('params', {}).get('length', 14), key="adx_length")
            with col3:
                volume_window = st.number_input("Volume Window:", 1, 100, indicators_config.get('volume_avg', {}).get('params', {}).get('window', 20), key="volume_window")
                adx_timeframes = st.multiselect("ADX Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('adx', {}).get('timeframes', ["3min"]), key="adx_timeframes")
                volume_timeframes = st.multiselect("Volume Timeframes:", ["1min", "3min", "5min", "10min", "15min", "20min", "30min"],
                    default=indicators_config.get('volume_avg', {}).get('timeframes', ["3min"]), key="volume_timeframes")
            
            # News Analysis
            st.markdown("---")
            st.write("**4. News Analysis (15% weight)**")       
            
            col1, col2 = st.columns(2)
            with col1:
                news_weight = st.number_input("News Weight (%):", 0, 100, alpha_config.get('news', {}).get('weight', 15), 1, key="news_weight")
            with col2:
                no_major_news_weight = st.number_input("No Major News Weight (%):", 0, 100, alpha_config.get('news', {}).get('conditions', {}).get('no_major_news', {}).get('weight', 15), 1, key="no_major_news_weight")
            
            # Market Calm Analysis
            st.markdown("---")
            st.write("**5. Market Calm Analysis (15% weight)**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                market_calm_weight = st.number_input("Market Calm Weight (%):", 0, 100, alpha_config.get('market_calm', {}).get('weight', 15), 1, key="market_calm_weight")
            with col2:
                vix_threshold = st.number_input("VIX Threshold:", 10, 50, alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('threshold', 20), 1, key="alpha_vix_threshold")
            with col3:
                vix_threshold_weight = st.number_input("VIX Weight (%):", 0, 100, alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('weight', 15), 1, key="vix_threshold_weight")
                vix_timeframe_market_calm = st.selectbox(
                    "VIX Timeframe (Market Calm):", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"],
                    index=["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"].index(alpha_config.get('market_calm', {}).get('conditions', {}).get('vix_threshold', {}).get('timeframe', '3min')),
                    key="vix_timeframe_market_calm",
                    help="Timeframe for VIX data in market calm analysis"
                )
            
            # Additional Entry Checks
            st.markdown("---")
            st.write("**Additional Entry Checks**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                volume_multiplier = st.number_input("Volume Multiplier (xAvg):", 1.0, 5.0, checks_config.get('volume_multiplier', 2.0), 0.1, key="volume_multiplier")
            with col2:
                vwap_slope_threshold = st.number_input("VWAP Slope Threshold:", 0.001, 1.0, checks_config.get('vwap_slope_threshold', 0.005), 0.001, format="%.3f", key="vwap_slope_threshold")
            with col3:
                vwap_slope_period = st.number_input("VWAP Slope Period (min):", 1, 20, checks_config.get('vwap_slope_period', 3), 1, key="vwap_slope_period")
            
            # TRIN/TICK Market Breadth Checks
            st.markdown("---")
            st.write("**TRIN/TICK Market Breadth Checks**")
            bypass_alpha = checks_config.get('trin_tick_bypass_alpha', 85)
            
            # Enable/Disable checkbox
            trin_tick_check_enabled = st.checkbox("Enable TRIN/TICK Check", value=checks_config.get('trin_tick_check_enabled', True), 
                key="trin_tick_check_enabled")
                     
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                trin_tick_bypass_alpha = st.number_input("Bypass Alpha Score:", 0, 100, bypass_alpha, 1, key="trin_tick_bypass_alpha",
                    help="Alpha score threshold to bypass TRIN/TICK checks", disabled=not trin_tick_check_enabled)
            with col2:
                trin_threshold = st.number_input("TRIN Threshold:", 0.5, 2.0, checks_config.get('trin_threshold', 1.1), 0.1, format="%.1f", key="trin_threshold", 
                    help="Market not overly bearish if TRIN ≤ threshold", disabled=not trin_tick_check_enabled)
            with col3:
                tick_ma_window = st.number_input("TICK MA Window (min):", 1, 10, checks_config.get('tick_ma_window', 1), 1, key="tick_ma_window",
                    help="Moving average window for TICK indicator", disabled=not trin_tick_check_enabled)
            with col4:
                tick_threshold = st.number_input("TICK Threshold:", -1000, 1000, checks_config.get('tick_threshold', 0), 50, key="tick_threshold",
                    help="Net uptick buying pressure if TICK MA ≥ threshold", disabled=not trin_tick_check_enabled)
            
            # Weight Validation
            st.markdown("---")
            total_weight = trend_weight + momentum_weight + volume_weight + news_weight + market_calm_weight
            if total_weight != 100:
                st.error(f"Total weight is {total_weight}% (should be 100%)")
            else:
                st.success(f"Total weight: {total_weight}%")
        
        with tab3:
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
            with col2:
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
        
        with tab4:
            st.subheader("Drawdown Limits Configuration")
            
            risk_config = config.get('RISK_CONFIG', {})
            
            st.write("**Fixed Drawdown Limits**")
            col1, col2, col3 = st.columns(3)
            with col1:
                daily_drawdown = st.number_input(
                    "Daily Drawdown Limit (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('daily_drawdown_limit', 0.02), 
                    step=0.001, 
                    key="daily_drawdown", 
                    format="%.3f"
                )
            with col2:
                monthly_drawdown = st.number_input(
                    "Monthly Drawdown Limit (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('monthly_drawdown_limit', 0.08), 
                    step=0.001, 
                    key="monthly_drawdown", 
                    format="%.3f"
                )
            with col3:
                drawdown_alert = st.number_input(
                    "Drawdown Alert (%):", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=risk_config.get('drawdown_alert', 0.015), 
                    step=0.001, 
                    key="drawdown_alert", 
                    format="%.3f"
                )
            
            # Dynamic Drawdown Limits
            st.markdown("---")
            st.write("**Dynamic Daily Limit Configuration**")
            
            atr_multiplier_dd = risk_config.get('atr_multiplier', 1.5)
            st.markdown(f"*Dynamic limit = min(lower_limit, {atr_multiplier_dd} x Portfolio ATR14, upper_limit)*")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                lower_limit = st.number_input(
                    "Lower Limit (%):", 
                    min_value=0.001, 
                    max_value=0.05, 
                    value=risk_config.get('lower_limit', 0.02), 
                    step=0.001, 
                    key="lower_limit", 
                    format="%.3f",
                    help="Minimum daily drawdown limit (e.g., 0.02 = 2% of equity)"
                )
            with col2:
                upper_limit = st.number_input(
                    "Upper Limit (%):", 
                    min_value=0.001, 
                    max_value=0.10, 
                    value=risk_config.get('upper_limit', 0.03), 
                    step=0.001, 
                    key="upper_limit", 
                    format="%.3f",
                    help="Maximum daily drawdown limit cap (e.g., 0.03 = 3% of equity)"
                )
            with col3:
                atr_multiplier_dd = st.number_input(
                    "ATR Multiplier:", 
                    min_value=0.5, 
                    max_value=3.0, 
                    value=risk_config.get('atr_multiplier', 1.5), 
                    step=0.1, 
                    key="atr_multiplier_dd",
                    format="%.1f",
                    help="Multiplier for Portfolio ATR14 in dynamic limit calculation"
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
                hedge_options = hedge_config.get('hedge_options', ['SQQQ', 'SPXU'])
                current_hedge = hedge_config.get('hedge_symbol', 'SQQQ')
                if current_hedge not in hedge_options:
                    hedge_options.insert(0, current_hedge)
                
                hedge_symbol = st.selectbox(
                    "Hedge Symbol:", 
                    options=hedge_options,
                    index=hedge_options.index(current_hedge),
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
                vix_timeframe_triggers = st.selectbox(
                    "VIX Timeframe (Triggers):", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"],
                    index=["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"].index(hedge_config.get('triggers', {}).get('vix_timeframe', '3min')),
                    key="vix_timeframe_triggers",
                    help="Timeframe for VIX data in hedge trigger conditions"
                )
            
            # Hedge Exit Conditions
            st.markdown("---")
            st.write("**Hedge Exit Conditions**")
            col1, col2 = st.columns(2)
            with col1:
                vix_exit_threshold = st.number_input(
                    "VIX Exit Threshold:", 
                    min_value=10, 
                    max_value=30, 
                    value=hedge_config.get('exit_conditions', {}).get('vix_exit_threshold', 20), 
                    step=1, 
                    key="vix_exit_threshold",
                    help="VIX level below which hedge can be scaled down"
                )
                vix_slope_period = st.number_input(
                    "VIX Slope Period (min):", 
                    min_value=5, 
                    max_value=30, 
                    value=hedge_config.get('exit_conditions', {}).get('vix_slope_period', 10), 
                    step=1, 
                    key="vix_slope_period",
                    help="Time period to check VIX falling trend"
                )
            with col2:
                sp500_recovery_threshold = st.number_input(
                    "S&P 500 Recovery Threshold (%):", 
                    min_value=0.001, 
                    max_value=0.02, 
                    value=hedge_config.get('exit_conditions', {}).get('sp500_recovery_threshold', 0.006), 
                    step=0.001, 
                    key="sp500_recovery_threshold",
                    format="%.3f",
                    help="S&P 500 gain threshold to trigger hedge reduction"
                )
                sqqq_vwap_consecutive_bars = st.number_input(
                    "SQQQ VWAP Consecutive Bars:", 
                    min_value=1, 
                    max_value=5, 
                    value=hedge_config.get('exit_conditions', {}).get('sqqq_vwap_consecutive_bars', 2), 
                    step=1, 
                    key="sqqq_vwap_consecutive_bars",
                    help="Number of consecutive bars SQQQ must trade above VWAP"
                )
                vix_timeframe_exit = st.selectbox(
                    "VIX Timeframe (Exit):", 
                    ["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"],
                    index=["1min", "3min", "5min", "10min", "15min", "20min", "30min", "1 hour", "1 day"].index(hedge_config.get('exit_conditions', {}).get('vix_timeframe', '3min')),
                    key="vix_timeframe_exit",
                    help="Timeframe for VIX data in hedge exit conditions"
                )
            
            # Hedge Levels
            st.markdown("---")
            st.write("**Hedge Levels**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Early Hedge**")
                early_beta = st.number_input(
                    "Beta Offset:", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=hedge_config.get('hedge_levels', {}).get('early', {}).get('beta', 0.1), 
                    step=0.01, 
                    key="early_beta",
                    format="%.2f",
                    help="Portfolio beta offset (e.g., 0.1 = -0.1β)"
                )
                early_equity_pct = st.number_input(
                    "Equity % for Hedge:", 
                    min_value=0.001, 
                    max_value=0.5, 
                    value=hedge_config.get('hedge_levels', {}).get('early', {}).get('equity_pct', 0.033), 
                    step=0.001, 
                    key="early_equity_pct",
                    format="%.3f",
                    help="Percentage of equity to allocate (e.g., 0.033 = 3.3%)"
                )
            with col2:
                st.write("**Mild Hedge**")
                mild_beta = st.number_input(
                    "Beta Offset:", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=hedge_config.get('hedge_levels', {}).get('mild', {}).get('beta', 0.15), 
                    step=0.01, 
                    key="mild_beta",
                    format="%.2f",
                    help="Portfolio beta offset (e.g., 0.15 = -0.15β)"
                )
                mild_equity_pct = st.number_input(
                    "Equity % for Hedge:", 
                    min_value=0.001, 
                    max_value=0.5, 
                    value=hedge_config.get('hedge_levels', {}).get('mild', {}).get('equity_pct', 0.05), 
                    step=0.001, 
                    key="mild_equity_pct",
                    format="%.3f",
                    help="Percentage of equity to allocate (e.g., 0.05 = 5%)"
                )
            with col3:
                st.write("**Severe Hedge**")
                severe_beta = st.number_input(
                    "Beta Offset:", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=hedge_config.get('hedge_levels', {}).get('severe', {}).get('beta', 0.30), 
                    step=0.01, 
                    key="severe_beta",
                    format="%.2f",
                    help="Portfolio beta offset (e.g., 0.3 = -0.3β)"
                )
                severe_equity_pct = st.number_input(
                    "Equity % for Hedge:", 
                    min_value=0.001, 
                    max_value=0.5, 
                    value=hedge_config.get('hedge_levels', {}).get('severe', {}).get('equity_pct', 0.10), 
                    step=0.001, 
                    key="severe_equity_pct",
                    format="%.3f",
                    help="Percentage of equity to allocate (e.g., 0.10 = 10%)"
                )
            
            # # Leverage Configuration
            # st.markdown("---")
            # st.write("**Leverage Settings**")
            # col1, col2 = st.columns(2)
            # with col1:
            #     leverage_enabled = st.checkbox(
            #         "Leverage Enabled", 
            #         value=leverage_config.get('enabled', True), 
            #         key="leverage_enabled"
            #     )
            #     max_leverage = st.number_input(
            #         "Max Leverage:", 
            #         min_value=1.0, 
            #         max_value=5.0, 
            #         value=leverage_config.get('max_leverage', 2.0), 
            #         step=0.1, 
            #         key="max_leverage",
            #         format="%.1f"
            #     )
            #     alpha_score_min = st.number_input(
            #         "Alpha Score Min:", 
            #         min_value=70, 
            #         max_value=100, 
            #         value=leverage_config.get('conditions', {}).get('alpha_score_min', 85), 
            #         step=1, 
            #         key="alpha_score_min"
            #     )
            #     vix_max = st.number_input(
            #         "VIX Max:", 
            #         min_value=10, 
            #         max_value=30, 
            #         value=leverage_config.get('conditions', {}).get('vix_max', 18), 
            #         step=1, 
            #         key="vix_max"
            #     )
            # with col2:
            #     drawdown_max = st.number_input(
            #         "Drawdown Max (%):", 
            #         min_value=0.001, 
            #         max_value=0.10, 
            #         value=leverage_config.get('conditions', {}).get('drawdown_max', 0.005), 
            #         step=0.001, 
            #         key="drawdown_max",
            #         format="%.3f"
            #     )
            #     vix_trend_days = st.number_input(
            #         "VIX Trend Days:", 
            #         min_value=5, 
            #         max_value=20, 
            #         value=leverage_config.get('conditions', {}).get('vix_trend_days', 10), 
            #         step=1, 
            #         key="vix_trend_days"
            #     )
            #     margin_alert_threshold = st.number_input(
            #         "Margin Alert Threshold:", 
            #         min_value=1.0, 
            #         max_value=3.0, 
            #         value=leverage_config.get('margin_alert_threshold', 1.5), 
            #         step=0.1, 
            #         key="margin_alert_threshold",
            #         format="%.1f"
            #     )
            
            # # Leverage Levels
            # st.markdown("---")
            # st.write("**Leverage Levels**")
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     all_conditions_leverage = st.number_input(
            #         "All Conditions Met:", 
            #         min_value=1.0, 
            #         max_value=5.0, 
            #         value=leverage_config.get('leverage_levels', {}).get('all_conditions_met', 2.0), 
            #         step=0.1, 
            #         key="all_conditions_leverage",
            #         format="%.1f"
            #     )
            # with col2:
            #     partial_conditions_leverage = st.number_input(
            #         "Partial Conditions:", 
            #         min_value=1.0, 
            #         max_value=3.0, 
            #         value=leverage_config.get('leverage_levels', {}).get('partial_conditions', 1.2), 
            #         step=0.1, 
            #         key="partial_conditions_leverage",
            #         format="%.1f"
            #     )
            # with col3:
            #     default_leverage = st.number_input(
            #         "Default Leverage:", 
            #         min_value=1.0, 
            #         max_value=2.0, 
            #         value=leverage_config.get('leverage_levels', {}).get('default', 1.0), 
            #         step=0.1, 
            #         key="default_leverage",
            #         format="%.1f"
            #                     )

        with tab7:
            st.subheader("Trading Hours Configuration")
            
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
                # Timezone options with US/Eastern as primary option
                timezone_options = ["US/Eastern", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]
                timezone_mapping = {
                    'US/Eastern': 'US/Eastern',
                    'US/Central': 'America/Chicago',
                    'US/Mountain': 'America/Denver',
                    'US/Pacific': 'America/Los_Angeles',
                    'America/New_York': 'America/New_York',
                    'America/Chicago': 'America/Chicago',
                    'America/Denver': 'America/Denver',
                    'America/Los_Angeles': 'America/Los_Angeles'
                }
                
                current_timezone = trading_hours_config.get('timezone', 'US/Eastern')
                mapped_timezone = timezone_mapping.get(current_timezone, 'US/Eastern')
                
                timezone = st.selectbox(
                    "Timezone:", 
                    timezone_options,
                    index=timezone_options.index(mapped_timezone)
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
            
            # Exit Times Configuration
            st.markdown("---")
            st.write("**Exit Times**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weak_exit_time = st.time_input(
                    "Weak Exit Time:", 
                    value=datetime.strptime(trading_hours_config.get('weak_exit_time', '15:30'), "%H:%M").time(),
                    help="Time to exit weak positions (-0.3% to +1.2%)"
                )
            
            with col2:
                hedge_force_exit_time = st.time_input(
                    "Hedge Force Exit Time:", 
                    value=datetime.strptime(trading_hours_config.get('hedge_force_exit_time', '15:55'), "%H:%M").time(),
                    help="Time to force close all hedge positions"
                )
            
            with col3:
                safety_exit_time = st.time_input(
                    "Safety Exit Time:", 
                    value=datetime.strptime(trading_hours_config.get('safety_exit_time', '15:55'), "%H:%M").time(),
                    help="Time to force close all positions"
                )
            
            # Weak Position Configuration
            st.markdown("---")
            st.write("**Weak Position Exit Configuration**")
            col1, col2 = st.columns(2)
            
            with col1:
                weak_min_gain = st.number_input(
                    "Min Gain % (Weak Exit):", 
                    min_value=-0.10, 
                    max_value=0.0, 
                    value=config.get('WEAK_POSITION_CONFIG', {}).get('min_gain_pct', -0.003), 
                    step=0.001, 
                    key="weak_min_gain",
                    format="%.3f",
                    help="Minimum gain percentage for weak exit (e.g., -0.003 = -0.3%)"
                )
            
            with col2:
                weak_max_gain = st.number_input(
                    "Max Gain % (Weak Exit):", 
                    min_value=0.0, 
                    max_value=0.10, 
                    value=config.get('WEAK_POSITION_CONFIG', {}).get('max_gain_pct', 0.012), 
                    step=0.001, 
                    key="weak_max_gain",
                    format="%.3f",
                    help="Maximum gain percentage for weak exit (e.g., 0.012 = 1.2%)"
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
        with col2:
            testing = st.checkbox(
                "Testing Mode Enabled", 
                value=config.get('TESTING', False),
                key="testing_global"
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
                    "ADV_large_length": adv_large_length,
                    "ADV_small_length": adv_small_length,
                    "ADV_large": adv_large,
                    "ADV_small": adv_small,
                    "RVOL_length": rvol_length,
                    "RVOL_filter": rvol_filter,
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
                    "vwap_slope_period": vwap_slope_period,
                    "trin_tick_check_enabled": trin_tick_check_enabled,
                    "trin_tick_bypass_alpha": trin_tick_bypass_alpha,
                    "trin_threshold": trin_threshold,
                    "tick_ma_window": tick_ma_window,
                    "tick_threshold": tick_threshold
                },
                "ALPHA_SCORE_CONFIG": {
                    "trend": {"weight": trend_weight, "conditions": {"ema_cross": {"weight": ema_cross_weight}}},
                    "momentum": {"weight": momentum_weight, "conditions": {"macd_positive": {"weight": macd_positive_weight}}},
                    "volume_volatility": {"weight": volume_weight, "conditions": {"volume_spike": {"weight": volume_spike_weight, "multiplier": volume_spike_multiplier}, "adx_threshold": {"weight": adx_weight, "threshold": adx_threshold}}},
                    "news": {"weight": news_weight, "conditions": {"no_major_news": {"weight": no_major_news_weight}}},
                    "market_calm": {"weight": market_calm_weight, "conditions": {"vix_threshold": {"weight": vix_threshold_weight, "threshold": vix_threshold, "timeframe": vix_timeframe_market_calm}}}
                },
                "RISK_CONFIG": {
                    "alpha_score_threshold": alpha_threshold,
                    "risk_per_trade": risk_per_trade,
                    "max_position_equity_pct": max_position_equity_pct,
                    "max_daily_trades": max_daily_trades,
                    "daily_drawdown_limit": daily_drawdown,
                    "monthly_drawdown_limit": monthly_drawdown,
                    "drawdown_alert": drawdown_alert,
                    "lower_limit": lower_limit,
                    "upper_limit": upper_limit,
                    "atr_multiplier": atr_multiplier_dd
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
                    "hedge_options": hedge_options,
                    "triggers": {
                        "vix_threshold": vix_threshold, 
                        "sp500_drop_threshold": sp500_drop_threshold,
                        "vix_timeframe": vix_timeframe_triggers
                    },
                    "exit_conditions": {
                        "vix_exit_threshold": vix_exit_threshold,
                        "vix_slope_period": vix_slope_period,
                        "sp500_recovery_threshold": sp500_recovery_threshold,
                        "sqqq_vwap_consecutive_bars": sqqq_vwap_consecutive_bars,
                        "vix_timeframe": vix_timeframe_exit
                    },
                    "hedge_levels": {
                        "early": {"beta": early_beta, "equity_pct": early_equity_pct},
                        "mild": {"beta": mild_beta, "equity_pct": mild_equity_pct},
                        "severe": {"beta": severe_beta, "equity_pct": severe_equity_pct}
                    }
                },
                # "LEVERAGE_CONFIG": {
                #     "enabled": leverage_enabled,
                #     "max_leverage": max_leverage,
                #     "conditions": {"alpha_score_min": alpha_score_min, "vix_max": vix_max, "drawdown_max": drawdown_max, "vix_trend_days": vix_trend_days},
                #     "leverage_levels": {"all_conditions_met": all_conditions_leverage, "partial_conditions": partial_conditions_leverage, "default": default_leverage},
                #     "margin_alert_threshold": margin_alert_threshold
                # },
                # "ORDER_CONFIG": {
                #     "limit_offset_min": limit_offset_min,
                #     "limit_offset_max": limit_offset_max,
                #     "order_window": order_window
                # },
                # "TRADING_HOURS": {
                #     "market_open": market_open.strftime("%H:%M"),
                #     "market_close": market_close.strftime("%H:%M"),
                #     "timezone": timezone,
                #     "morning_entry_start": morning_start.strftime("%H:%M"),
                #     "morning_entry_end": morning_end.strftime("%H:%M"),
                #     "afternoon_entry_start": afternoon_start.strftime("%H:%M"),
                #     "afternoon_entry_end": afternoon_end.strftime("%H:%M"),
                #     "weak_exit_time": weak_exit_time.strftime("%H:%M"),
                #     "hedge_force_exit_time": hedge_force_exit_time.strftime("%H:%M"),
                #     "safety_exit_time": safety_exit_time.strftime("%H:%M")
                # },
                "WEAK_POSITION_CONFIG": {
                    "min_gain_pct": weak_min_gain,
                    "max_gain_pct": weak_max_gain
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
                    if st.button("Reload from File", key="reload_raw_config"):
                        st.rerun()
                        
            else:
                st.error("creds.json does not exist. Create it first in the Configuration Editor.")
                
        except Exception as e:
            st.error(f"Error reading creds.json: {str(e)}")
            st.info("Make sure creds.json exists in the current directory.")

if __name__ == "__main__":
    main()
