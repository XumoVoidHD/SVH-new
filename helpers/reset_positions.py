import sqlite3
from datetime import datetime
import pytz
import sys
import os

# Add parent directory to path to import broker
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.ib_broker import IBBroker

def close_all_open_positions():
    """Close all open positions on IBKR and in the database"""
    
    broker = None
    
    try:
        # Connect to IBKR
        print("\nConnecting to IBKR...")
        broker = IBBroker()
        broker.connect_to_ibkr(host="127.0.0.1", port=7497, client_id=999)  # Use unique client ID
        print("✓ Connected to IBKR")
        
        # Connect to database
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        
        # Find all open positions
        cursor.execute("""
            SELECT symbol, position_shares, entry_price, current_price, realized_pnl, shares
            FROM stock_strategies 
            WHERE position_active = 1 AND position_shares > 0
        """)
        
        open_positions = cursor.fetchall()
        
        if not open_positions:
            print("No open positions found")
            conn.close()
            if broker:
                broker.disconnect()
            return 0, []
        
        print(f"\n{'='*60}")
        print(f"Found {len(open_positions)} open positions to close")
        print(f"{'='*60}")
        
        # Get current Eastern time
        eastern_tz = pytz.timezone('US/Eastern')
        close_time = datetime.now(eastern_tz)
        close_time_str = close_time.isoformat()
        
        closed_positions = []
        total_pnl = 0
        
        for position in open_positions:
            symbol, position_shares, entry_price, current_price, existing_realized_pnl, shares = position
            
            print(f"\nClosing {symbol} on IBKR:")
            print(f"  Position Shares: {position_shares}")
            print(f"  Entry Price: ${entry_price:.2f}")
            
            # Place SELL order on IBKR
            try:
                # Get unique order ID
                order_id = broker.get_next_order_id_from_ibkr()
                if order_id is None:
                    order_id = 10000  # Fallback
                
                # Place market sell order
                trade_result = broker.place_market_order_with_id(symbol, position_shares, "SELL", order_id)
                
                if trade_result and len(trade_result) >= 2:
                    fill_order_id, fill_price = trade_result
                    if fill_price > 0:
                        actual_exit_price = fill_price
                        print(f"Sold on IBKR at ${actual_exit_price:.2f}")
                    else:
                        actual_exit_price = current_price
                        print(f"IBKR order failed, using current price ${actual_exit_price:.2f}")
                else:
                    actual_exit_price = current_price
                    print(f"IBKR order failed, using current price ${actual_exit_price:.2f}")
                    
            except Exception as e:
                print(f"Error selling on IBKR: {e}")
                actual_exit_price = current_price
                print(f"  Using current price ${actual_exit_price:.2f}")
            
            # Calculate PnL using actual exit price
            closure_pnl = (actual_exit_price - entry_price) * position_shares
            new_realized_pnl = existing_realized_pnl + closure_pnl
            total_pnl += new_realized_pnl
            
            print(f"  Exit Price: ${actual_exit_price:.2f}")
            print(f"  Closure PnL: ${closure_pnl:.2f}")
            print(f"  Total Realized PnL: ${new_realized_pnl:.2f}")
            
            # Update the database
            update_query = """
                UPDATE stock_strategies 
                SET 
                    position_active = 0,
                    position_shares = 0,
                    current_price = ?,
                    unrealized_pnl = 0,
                    realized_pnl = ?,
                    close_time = ?
                WHERE symbol = ?
            """
            
            cursor.execute(update_query, (actual_exit_price, new_realized_pnl, close_time_str, symbol))
            
            closed_positions.append({
                'symbol': symbol,
                'shares': position_shares,
                'pnl': closure_pnl,
                'total_realized_pnl': new_realized_pnl
            })
        
        # Commit all changes
        conn.commit()
        conn.close()
        
        # Disconnect from IBKR
        if broker:
            broker.disconnect()
            print("\n✓ Disconnected from IBKR")
        
        print(f"\n{'='*60}")
        print(f"Successfully closed {len(closed_positions)} positions")
        print(f"Total PnL from closures: ${sum(p['pnl'] for p in closed_positions):.2f}")
        print(f"Total Realized PnL: ${total_pnl:.2f}")
        print(f"{'='*60}\n")
        
        return len(closed_positions), closed_positions
        
    except Exception as e:
        print(f"Error closing positions: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to disconnect broker on error
        if broker:
            try:
                broker.disconnect()
            except:
                pass
        
        return 0, []

if __name__ == "__main__":
    count, positions = close_all_open_positions()
    print(f"\nClosed {count} positions")

