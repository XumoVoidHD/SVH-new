import sqlite3
import pandas as pd

def get_orders(db_path='trading.db'):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM order_book ORDER BY timestamp DESC", conn)

def get_trades(db_path='trading.db'):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM trade_book ORDER BY timestamp DESC", conn)

def get_positions(db_path='trading.db'):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM positions_book ORDER BY timestamp DESC", conn)

def get_metrics(db_path='trading.db'):
    positions_df = get_positions(db_path)
    if positions_df.empty:
        return {
            "total_positions": 0,
            "total_realized_pnl": 0.0,
            "total_unrealized_pnl": 0.0,
            "total_pnl": 0.0
        }
    latest_positions = positions_df.sort_values('timestamp', ascending=False)
    return {
        "total_positions": len(latest_positions[latest_positions['qty'] != 0]),
        "total_realized_pnl": latest_positions['realized_pnl'].sum(),
        "total_unrealized_pnl": latest_positions['unrealized_pnl'].sum(),
        "total_pnl": latest_positions['realized_pnl'].sum() + latest_positions['unrealized_pnl'].sum()
    } 