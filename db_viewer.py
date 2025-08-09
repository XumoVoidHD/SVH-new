import streamlit as st
import pandas as pd
from db.trades_db import trades_db
import json
from datetime import datetime

st.set_page_config(page_title="Trading Strategy Monitor", layout="wide")

st.title("📊 Trading Strategy Database Monitor")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Strategy Summary", "Active Positions", "Stock History", "Database Stats"]
)

if page == "Strategy Summary":
    st.header("Strategy Summary")
    
    # Get strategy summary
    summary_data = trades_db.get_strategy_summary()
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Format the data
        df['pnl_pct'] = df.apply(lambda row: 
            ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100) 
            if row['entry_price'] > 0 else 0, axis=1)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stocks = len(df)
            st.metric("Total Stocks", total_stocks)
        
        with col2:
            # Ensure position_active is treated as boolean
            df['position_active'] = df['position_active'].astype(bool)
            active_positions = len(df[df['position_active'] == True])
            st.metric("Active Positions", active_positions)
        
        with col3:
            total_pnl = df['total_pnl'].sum()
            st.metric("Total PnL", f"${total_pnl:.2f}")
        
        with col4:
            avg_score = df['score'].mean()
            st.metric("Avg Alpha Score", f"{avg_score:.1f}")
        
        # Display the data table
        st.subheader("All Strategies")
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}" if x != 0 else "N/A")
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%" if x != 0 else "N/A")
        display_df['total_pnl'] = display_df['total_pnl'].apply(lambda x: f"${x:.2f}" if x != 0 else "N/A")
        display_df['position_active'] = display_df['position_active'].apply(lambda x: "✅" if bool(x) else "❌")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Filter by active positions
        if active_positions > 0:
            st.subheader("Active Positions Only")
            # Ensure position_active is treated as boolean for filtering
            active_df = df[df['position_active'].astype(bool) == True]
            st.dataframe(active_df[['symbol', 'position_shares', 'entry_price', 'current_price', 'pnl', 'pnl_pct']], use_container_width=True)
    
    else:
        st.info("No strategy data available yet. Start your trading strategies to see data here.")

elif page == "Active Positions":
    st.header("Active Positions")
    
    active_positions = trades_db.get_all_active_positions()
    
    if active_positions:
        df = pd.DataFrame(active_positions)
        
        # Calculate additional metrics
        df['pnl_pct'] = df.apply(lambda row: 
            ((row['current_price'] - row['entry_price']) / row['entry_price'] * 100) 
            if row['entry_price'] > 0 else 0, axis=1)
        
        df['position_value'] = df['position_shares'] * df['current_price']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_positions = len(df)
            st.metric("Total Positions", total_positions)
        
        with col2:
            total_value = df['position_value'].sum()
            st.metric("Total Position Value", f"${total_value:.2f}")
        
        with col3:
            total_pnl = df['pnl'].sum()
            st.metric("Total PnL", f"${total_pnl:.2f}")
        
        with col4:
            avg_pnl_pct = df['pnl_pct'].mean()
            st.metric("Avg PnL %", f"{avg_pnl_pct:.2f}%")
        
        # Display positions table
        st.subheader("Position Details")
        
        # Format for display
        display_df = df.copy()
        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['stop_loss_price'] = display_df['stop_loss_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['take_profit_price'] = display_df['take_profit_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:.2f}%")
        display_df['position_value'] = display_df['position_value'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Charts
        st.subheader("Position Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PnL by position
            fig = st.bar_chart(df.set_index('symbol')['pnl'])
            st.caption("PnL by Position")
        
        with col2:
            # PnL % by position
            fig = st.bar_chart(df.set_index('symbol')['pnl_pct'])
            st.caption("PnL % by Position")
    
    else:
        st.info("No active positions found.")

elif page == "Stock History":
    st.header("Stock History")
    
    # Get list of stocks
    summary_data = trades_db.get_strategy_summary()
    if summary_data:
        stock_list = [row['symbol'] for row in summary_data]
        
        selected_stock = st.selectbox("Select Stock", stock_list)
        
        if selected_stock:
            # Get history for selected stock
            history_data = trades_db.get_stock_history(selected_stock, limit=50)
            
            if history_data:
                df = pd.DataFrame(history_data)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Display latest data
                latest = df.iloc[0]
                
                st.subheader(f"Latest Data for {selected_stock}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Alpha Score", f"{latest['score']:.1f}")
                    st.metric("Position Active", "✅" if bool(latest['position_active']) else "❌")
                
                with col2:
                    st.metric("Current Price", f"${latest['current_price']:.2f}")
                    st.metric("Entry Price", f"${latest['entry_price']:.2f}" if latest['entry_price'] > 0 else "N/A")
                
                with col3:
                    st.metric("Position Shares", latest['position_shares'])
                    st.metric("PnL", f"${latest['pnl']:.2f}")
                
                with col4:
                    st.metric("Stop Loss", f"${latest['stop_loss_price']:.2f}" if latest['stop_loss_price'] > 0 else "N/A")
                    st.metric("Take Profit", f"${latest['take_profit_price']:.2f}" if latest['take_profit_price'] > 0 else "N/A")
                
                # Display history table
                st.subheader("Recent History")
                
                # Select columns to display
                display_columns = ['timestamp', 'score', 'position_active', 'current_price', 
                                 'entry_price', 'position_shares', 'pnl', 'stop_loss_price', 'take_profit_price']
                
                display_df = df[display_columns].copy()
                
                # Format the data
                display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}" if x != 0 else "N/A")
                display_df['stop_loss_price'] = display_df['stop_loss_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                display_df['take_profit_price'] = display_df['take_profit_price'].apply(lambda x: f"${x:.2f}" if x > 0 else "N/A")
                display_df['position_active'] = display_df['position_active'].apply(lambda x: "✅" if bool(x) else "❌")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Charts
                st.subheader("Price and Score History")
                
                # Create charts
                chart_df = df[['timestamp', 'current_price', 'entry_price', 'score']].copy()
                chart_df = chart_df.set_index('timestamp')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.line_chart(chart_df[['current_price', 'entry_price']])
                    st.caption("Price History")
                
                with col2:
                    st.line_chart(chart_df[['score']])
                    st.caption("Alpha Score History")
            else:
                st.info(f"No history data available for {selected_stock}")
    else:
        st.info("No stocks available in database.")

elif page == "Database Stats":
    st.header("Database Statistics")
    
    # Get summary data
    summary_data = trades_db.get_strategy_summary()
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Calculate statistics
        total_stocks = len(df)
        # Ensure position_active is treated as boolean
        df['position_active'] = df['position_active'].astype(bool)
        active_positions = len(df[df['position_active'] == True])
        total_pnl = df['total_pnl'].sum()
        avg_score = df['score'].mean()
        
        # Performance metrics
        total_trades = df['total_trades'].sum()
        winning_trades = df['winning_trades'].sum()
        losing_trades = df['losing_trades'].sum()
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Display stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("General Stats")
            st.metric("Total Stocks", total_stocks)
            st.metric("Active Positions", active_positions)
            st.metric("Total PnL", f"${total_pnl:.2f}")
            st.metric("Average Alpha Score", f"{avg_score:.1f}")
        
        with col2:
            st.subheader("Performance Stats")
            st.metric("Total Trades", total_trades)
            st.metric("Winning Trades", winning_trades)
            st.metric("Losing Trades", losing_trades)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Database cleanup
        st.subheader("Database Maintenance")
        
        if st.button("Clean Old Data (Keep Last 30 Days)"):
            deleted_count = trades_db.cleanup_old_data(days_to_keep=30)
            st.success(f"Cleaned up {deleted_count} old records")
        
        # Export data
        st.subheader("Export Data")
        
        if st.button("Export Strategy Summary to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"strategy_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("No data available for statistics.")

# Footer
st.markdown("---")
st.markdown("*Database updated in real-time from trading strategies*") 