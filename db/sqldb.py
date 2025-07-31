import sqlite3
import threading
from datetime import datetime
import pytz

class SQLDB:
    _global_lock = threading.Lock()
    def __init__(self, db_name='trading.db'):
        self.db_name = db_name
        self._local = threading.local()
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Initialize database schema in the main thread
        conn = self._get_connection()
        cursor = conn.cursor()
        self._create_tables(cursor)
        conn.commit()
        conn.close()

    def reset_database(self):
            with SQLDB._global_lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute('DROP TABLE IF EXISTS order_book')
                    cursor.execute('DROP TABLE IF EXISTS trade_book')
                    cursor.execute('DROP TABLE IF EXISTS positions_book')
                    self._create_tables(cursor)
                    conn.commit()
                    print("Trading database has been reset successfully.")
                except Exception as e:
                    print(f"Error resetting database: {e}")
                    conn.rollback()
                finally:
                    conn.close()

    def _get_connection(self):
        """Get a thread-local database connection with proper configuration."""
        conn = sqlite3.connect(self.db_name)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        return conn
        
    def _create_tables(self, cursor):
        """Create tables if they don't exist."""
        # Create order book table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS order_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                symbol TEXT,
                qty REAL CHECK (qty != 0),
                order_type TEXT CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP-MARKET', 'STOP-LIMIT')),
                price REAL,
                sl REAL,
                tp REAL,
                triggerprice REAL,
                status TEXT CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED')),
                timestamp DATETIME DEFAULT (datetime('now', '+5 hours 30 minutes')),
                filled_qty REAL DEFAULT 0,
                fill_price REAL,
                fill_time DATETIME
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_symbol ON order_book(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_status ON order_book(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_id ON order_book(order_id)')

        # Create trade book table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                symbol TEXT,
                qty REAL CHECK (qty != 0),
                exec_price REAL CHECK (exec_price > 0),
                commission REAL CHECK (commission >= 0),
                pnl REAL,
                timestamp DATETIME DEFAULT (datetime('now', '+5 hours 30 minutes')),
                FOREIGN KEY (order_id) REFERENCES order_book(order_id)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade_book(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_order ON trade_book(order_id)')

        # Create positions book table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                symbol TEXT,
                qty REAL CHECK (qty >= 0),
                direction INTEGER CHECK (direction IN (-1, 0, 1)),
                open_qty REAL CHECK (open_qty >= 0),
                close_qty REAL CHECK (close_qty >= 0),
                avg_price REAL CHECK (avg_price >= 0),
                unrealized_pnl REAL,
                realized_pnl REAL,
                sl REAL,
                tp REAL,
                status TEXT CHECK (status IN ('OPEN', 'CLOSED')),
                open_time DATETIME,
                close_time DATETIME,
                open_price REAL CHECK (open_price >= 0),
                close_price REAL CHECK (close_price >= 0),
                trade_ids TEXT,
                order_ids TEXT,
                timestamp DATETIME DEFAULT (datetime('now', '+5 hours 30 minutes')),
                CHECK (qty = 0 OR direction != 0),
                CHECK (close_qty <= open_qty),
                CHECK ((status = 'OPEN' AND qty > 0) OR (status = 'CLOSED' AND qty = 0)),
                UNIQUE(session_id, symbol, open_time)
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_status ON positions_book(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_symbol ON positions_book(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_session ON positions_book(session_id)')

    def _get_current_ist_time(self):
        """Get current time in IST."""
        return datetime.now(self.ist)

    def insert_order(self, order_id, symbol, qty, order_type, price, sl, tp, triggerprice, status, filled_qty, fill_price, fill_time):
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if fill_time:
                fill_time = fill_time.astimezone(self.ist)
            cursor.execute('''
                INSERT INTO order_book (order_id, symbol, qty, order_type, price, sl, tp, triggerprice, status, filled_qty, fill_price, fill_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, symbol, qty, order_type, price, sl, tp, triggerprice, status, filled_qty, fill_price, fill_time))
            conn.commit()
            conn.close()

    def insert_trade(self, order_id, symbol, qty, exec_price, commission, pnl, timestamp):
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if timestamp:
                timestamp = timestamp.astimezone(self.ist)
            cursor.execute('''
                INSERT INTO trade_book (order_id, symbol, qty, exec_price, commission, pnl, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (order_id, symbol, qty, exec_price, commission, pnl, timestamp))
            conn.commit()
            conn.close()

    def insert_position(self, symbol: str, qty: float, direction: int, open_qty: float, close_qty: float,
                      avg_price: float, unrealized_pnl: float, realized_pnl: float,
                      sl: float, tp: float, status: str,
                      open_time: datetime = None, close_time: datetime = None,
                      open_price: float = None, close_price: float = None,
                      trade_ids: str = '', order_ids: str = '', session_id: str = None):
        """Insert or update a position in the database."""
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                # Convert timestamps to IST
                if open_time:
                    open_time = open_time.astimezone(self.ist)
                if close_time:
                    close_time = close_time.astimezone(self.ist)
                
                # Check if position exists and is OPEN
                cursor.execute('SELECT id, status FROM positions_book WHERE symbol = ? AND session_id = ?', (symbol, session_id))
                r = self.get_orders()
                existing = cursor.fetchone()
                
                if existing and str(existing[1]) == str('OPEN'):  # Only update if status is OPEN
                    # Update existing position
                    cursor.execute('''
                        UPDATE positions_book SET
                            qty = ?, direction = ?, open_qty = ?, close_qty = ?,
                            avg_price = ?, unrealized_pnl = ?, realized_pnl = ?,
                            sl = ?, tp = ?, status = ?, open_time = ?, close_time = ?,
                            open_price = ?, close_price = ?, trade_ids = ?, order_ids = ?
                        WHERE symbol = ? AND session_id = ? AND status = 'OPEN'
                    ''', (
                        qty, direction, open_qty, close_qty,
                        avg_price, unrealized_pnl, realized_pnl,
                        sl, tp, status, open_time, close_time,
                        open_price, close_price, trade_ids, order_ids,
                        symbol, session_id
                    ))
                else:
                    # Insert new position
                    cursor.execute('''
                        INSERT INTO positions_book (
                            session_id, symbol, qty, direction, open_qty, close_qty,
                            avg_price, unrealized_pnl, realized_pnl, sl, tp, status,
                            open_time, close_time, open_price, close_price,
                            trade_ids, order_ids
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id, symbol, qty, direction, open_qty, close_qty,
                        avg_price, unrealized_pnl, realized_pnl, sl, tp, status,
                        open_time, close_time, open_price, close_price,
                        trade_ids, order_ids
                    ))
                conn.commit()
            except sqlite3.Error as e:
                print(f"Database error: {e}")
                conn.rollback()
                raise
            except Exception as e:
                print(f"Error handling position: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()

    def get_orders(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM order_book')
        results = cursor.fetchall()
        conn.close()
        return results

    def get_orders_by_symbol(self, symbol: str):
        """
        Get all orders for a specific symbol.
        
        Args:
            symbol: Stock symbol to query
            
        Returns:
            list: List of order records for the symbol
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM order_book WHERE symbol = ?', (symbol,))
        results = cursor.fetchall()
        conn.close()
        return results

    def get_trades(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM trade_book')
        results = cursor.fetchall()
        conn.close()
        return results

    def get_positions(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM positions_book')
        results = cursor.fetchall()
        conn.close()
        return results

    def update_order(self, order_id, **kwargs):
        conn = self._get_connection()
        cursor = conn.cursor()
        columns = ', '.join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values())
        values.append(order_id)
        cursor.execute(f"UPDATE order_book SET {columns} WHERE order_id = ?", values)
        conn.commit()
        conn.close()

    def update_trade(self, trade_id, **kwargs):
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            columns = ', '.join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values())
            values.append(trade_id)
            cursor.execute(f"UPDATE trade_book SET {columns} WHERE id = ?", values)
            conn.commit()
            conn.close()

    def update_position(self, position_id: str, **kwargs):
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            columns = ', '.join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values())
            values.append(position_id)
            cursor.execute(f"UPDATE positions_book SET {columns} WHERE symbol = ? AND status = 'OPEN'", values)
            conn.commit()
            conn.close()

            
    def update_closed_position(self, position_id: str, **kwargs):
        with SQLDB._global_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            columns = ', '.join(f"{k} = ?" for k in kwargs.keys())
            values = list(kwargs.values())
            values.append(position_id)
            cursor.execute(f"UPDATE positions_book SET {columns} WHERE symbol = ? AND status = 'CLOSED'", values)
            conn.commit()
            conn.close()

    def total_unrealized_pnl(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT SUM(unrealized_pnl) FROM positions_book")
            result = cursor.fetchone()[0]
            return result if result is not None else 0.0
        except Exception as e:
            print(f"Error calculating total unrealized PnL: {e}")
            return 0.0
        finally:
            conn.close()
    
    def total_realized_pnl(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT SUM(realized_pnl) FROM positions_book")
            result = cursor.fetchone()[0]
            return result if result is not None else 0.0
        except Exception as e:
            print(f"Error calculating total realized PnL: {e}")
            return 0.0
        finally:
            conn.close()
    
    def get_unrealized_pnl_by_symbol(self, symbol: str) -> float:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT SUM(unrealized_pnl) FROM positions_book WHERE symbol = ? AND status = 'OPEN'"
            cursor.execute(query, (symbol,))
            result = cursor.fetchone()[0]
            return result if result is not None else 0.0
        except Exception as e:
            print(f"Error fetching unrealized PnL for {symbol}: {e}")
            return 0.0
        finally:
            conn.close()



    def close(self):
        # Nothing to close since we create a new connection for each operation
        pass 
    
if __name__ == "__main__":
    db = SQLDB("trading.db")
    x = db.update_closed_position("BANKNIFTY26JUN2556300CE", unrealized_pnl=5)

    
