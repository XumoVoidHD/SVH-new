import sqlite3
import threading
from datetime import datetime, timedelta
import json
import os
import shutil
import time
import random

class TradesDatabase:
    def __init__(self, db_path='trades.db'):
        # Ensure we use absolute path to avoid issues with working directory changes
        # If relative path, use project root (parent of db folder) or current working directory
        if not os.path.isabs(db_path):
            # Get the directory where this script is located (db folder)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Get the project root (parent directory of db folder)
            project_root = os.path.dirname(script_dir)
            
            # Use project root if it's writable, otherwise use current working directory
            if os.access(project_root, os.W_OK):
                self.db_path = os.path.join(project_root, db_path)
            else:
                self.db_path = os.path.abspath(db_path)
        else:
            self.db_path = db_path
        
        self.lock = threading.Lock()
        print(f"TradesDatabase initializing with path: {self.db_path}")
        self.init_database()
    
    def _get_connection(self, timeout=30.0):
        """Get a database connection with proper configuration and error handling"""
        max_retries = 3
        base_delay = 1.0
        
        # Ensure the directory exists and is writable
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created database directory: {db_dir}")
            except Exception as e:
                raise Exception(f"Cannot create database directory {db_dir}: {e}")
        
        # Check if directory is writable
        if db_dir and not os.access(db_dir, os.W_OK):
            raise PermissionError(f"Database directory {db_dir} is not writable. Please check file permissions.")
        
        for attempt in range(max_retries):
            try:
                # Check if database file exists and is writable
                if os.path.exists(self.db_path):
                    if not os.access(self.db_path, os.W_OK):
                        # Try to fix permissions (read-only flag on Windows)
                        try:
                            import stat
                            current_permissions = os.stat(self.db_path).st_mode
                            os.chmod(self.db_path, current_permissions | stat.S_IWRITE)
                            print(f"Fixed write permissions for database file: {self.db_path}")
                        except Exception as perm_error:
                            raise PermissionError(f"Database file {self.db_path} is not writable and cannot be fixed: {perm_error}")
                    
                    # Also check if the file is actually writable by trying to open it
                    try:
                        test_file = open(self.db_path, 'r+b')
                        test_file.close()
                    except Exception as test_error:
                        raise PermissionError(f"Database file {self.db_path} cannot be opened for writing: {test_error}")
                    
                    print(f"Database file exists and is writable: {self.db_path}")
                else:
                    print(f"Database file does not exist, will be created: {self.db_path}")
                
                conn = sqlite3.connect(self.db_path, timeout=timeout, check_same_thread=False)
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA foreign_keys=ON')
                print(f"Database connection established successfully")
                return conn
                
            except sqlite3.OperationalError as e:
                error_str = str(e).lower()
                if "readonly" in error_str:
                    # More detailed error message for readonly database
                    error_msg = (
                        f"Database is read-only: {self.db_path}\n"
                        f"Possible causes:\n"
                        f"  1. File permissions: Check if the file/directory is writable\n"
                        f"  2. File is locked by another process\n"
                        f"  3. Database file is in a read-only location\n"
                        f"  4. Insufficient user permissions\n"
                        f"Please check file permissions and ensure you have write access to: {os.path.dirname(self.db_path) or '.'}"
                    )
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Database readonly error, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                        print(f"Error details: {e}")
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception(error_msg)
                elif "locked" in error_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Database locked, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        raise Exception(f"Database remains locked after {max_retries} attempts: {e}")
                else:
                    raise e
            except PermissionError as e:
                # Don't retry permission errors, they won't fix themselves
                raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Database connection failed, retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
                    continue
                else:
                    raise e
        
        raise Exception(f"Failed to connect to database after {max_retries} attempts")
    
    def init_database(self):
        """Initialize the database with a single table for all stock data"""
        try:
            with self.lock:
                print(f"Initializing database at: {self.db_path}")
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Create a single table for all stock strategy data
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
                        shares INTEGER DEFAULT 0,
                        current_price REAL DEFAULT 0,
                        entry_price REAL DEFAULT 0,
                        stop_loss_price REAL DEFAULT 0,
                        take_profit_price REAL DEFAULT 0,
                        used_capital REAL DEFAULT 0,
                        unrealized_pnl REAL DEFAULT 0,
                        realized_pnl REAL DEFAULT 0,
                        entry_time TIMESTAMP,
                        close_time TIMESTAMP,
                        profit_booking_levels TEXT,
                        trailing_exit_conditions TEXT,
                        trailing_stop_levels TEXT,
                        profit_booked_flags TEXT,
                        trailing_stop_flags TEXT,
                        trailing_exit_monitoring BOOLEAN DEFAULT FALSE,
                        trailing_exit_start_time TIMESTAMP,
                        trailing_exit_start_price REAL DEFAULT 0,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        avg_trade_duration REAL DEFAULT 0,
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
                print("Database table 'stock_strategies' created successfully")
                
                # Verify table was created
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_strategies'")
                table_exists = cursor.fetchone()
                if table_exists:
                    print("✓ Database initialization completed successfully")
                else:
                    print("✗ Database initialization failed - table not found after creation")
                
                conn.close()
        except Exception as e:
            print(f"Error during database initialization: {e}")
            import traceback
            traceback.print_exc()
            raise

    def verify_database(self):
        """Verify that the database and table exist"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_strategies'")
                table_exists = cursor.fetchone()
                
                if table_exists:
                    print("✓ Database verification passed - stock_strategies table exists")
                    conn.close()
                    return True
                else:
                    print("✗ Database verification failed - stock_strategies table not found")
                    conn.close()
                    return False
        except Exception as e:
            print(f"Error during database verification: {e}")
            return False
    
    def backup_database(self):
        """Backup the current database to historical_data folder with date-based naming using last modified date"""
        if not os.path.exists(self.db_path):
            return False
            
        # Create historical_data directory if it doesn't exist
        historical_dir = "historical_data"
        if not os.path.exists(historical_dir):
            os.makedirs(historical_dir)
        
        # Get database last modified date
        db_modified_time = os.path.getmtime(self.db_path)
        modified_date = datetime.fromtimestamp(db_modified_time).strftime("%Y-%m-%d")
        
        # Create backup filename using last modified date
        backup_filename = f"trades_{modified_date}.db"
        backup_path = os.path.join(historical_dir, backup_filename)
        
        # If backup already exists for this modified date, add timestamp
        if os.path.exists(backup_path):
            timestamp = datetime.fromtimestamp(db_modified_time).strftime("%Y-%m-%d_%H-%M-%S")
            backup_filename = f"trades_{timestamp}.db"
            backup_path = os.path.join(historical_dir, backup_filename)
        
        try:
            # Copy the database file
            shutil.copy2(self.db_path, backup_path)
            print(f"Database backed up to: {backup_path}")
            
            # Delete the original trades.db file after successful backup
            os.remove(self.db_path)
            print(f"Original database file {self.db_path} deleted after successful backup")
            
            return True
        except Exception as e:
            print(f"Error backing up database: {e}")
            return False
    
    def get_historical_databases(self):
        """Get list of all historical database files"""
        historical_dir = "historical_data"
        if not os.path.exists(historical_dir):
            return []
        
        historical_files = []
        for filename in os.listdir(historical_dir):
            if filename.startswith("trades_") and filename.endswith(".db"):
                file_path = os.path.join(historical_dir, filename)
                file_stat = os.stat(file_path)
                historical_files.append({
                    'filename': filename,
                    'path': file_path,
                    'date': filename.replace("trades_", "").replace(".db", ""),
                    'size': file_stat.st_size,
                    'modified': datetime.fromtimestamp(file_stat.st_mtime)
                })
        
        # Sort by date (newest first)
        historical_files.sort(key=lambda x: x['modified'], reverse=True)
        return historical_files
    
    def load_historical_data(self, historical_db_path):
        """Load data from a historical database file"""
        if not os.path.exists(historical_db_path):
            return None
            
        try:
            conn = sqlite3.connect(historical_db_path)
            cursor = conn.cursor()
            
            # Get all strategy data
            cursor.execute('SELECT * FROM stock_strategies')
            rows = cursor.fetchall()
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                # Convert boolean fields from SQLite integers to Python booleans
                boolean_fields = ['additional_checks_passed', 'position_active', 'trailing_exit_monitoring', 'hedge_active']
                for field in boolean_fields:
                    if field in row_dict and row_dict[field] is not None:
                        row_dict[field] = bool(row_dict[field])
                data.append(row_dict)
            
            return data
        except Exception as e:
            print(f"Error loading historical data from {historical_db_path}: {e}")
            return None
    
    def get_aggregated_data(self, start_date=None, end_date=None):
        """Get aggregated data from historical databases within a date range
        
        Args:
            start_date: Start date (datetime object or string in YYYY-MM-DD format)
            end_date: End date (datetime object or string in YYYY-MM-DD format)
        
        Returns:
            List of aggregated data records
        """
        historical_files = self.get_historical_databases()
        
        if not historical_files:
            return []
        
        # Convert string dates to datetime objects if needed
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                print(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")
                return []
                
        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                print(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD")
                return []
        
        # If no dates provided, return all data
        if start_date is None and end_date is None:
            filtered_files = historical_files
        else:
            filtered_files = []
            for file_info in historical_files:
                # Extract date from filename (trades_YYYY-MM-DD_HH-MM-SS.db or trades_YYYY-MM-DD.db)
                filename_date_str = file_info['date']  # This is already extracted in get_historical_databases
                
                # Parse the date part (handle both formats)
                try:
                    # Try parsing with time first (YYYY-MM-DD_HH-MM-SS)
                    if '_' in filename_date_str and len(filename_date_str.split('_')[1]) > 0:
                        file_date = datetime.strptime(filename_date_str.split('_')[0], '%Y-%m-%d')
                    else:
                        # Just date (YYYY-MM-DD)
                        file_date = datetime.strptime(filename_date_str, '%Y-%m-%d')
                    
                    # Check if file date is within range
                    include_file = True
                    if start_date is not None and file_date < start_date:
                        include_file = False
                    if end_date is not None and file_date > end_date:
                        include_file = False
                    
                    if include_file:
                        filtered_files.append(file_info)
                        
                except ValueError:
                    # Skip files with invalid date formats
                    print(f"Skipping file with invalid date format: {file_info['filename']}")
                    continue
        
        all_data = []
        for file_info in filtered_files:
            data = self.load_historical_data(file_info['path'])
            if data:
                # Add date info to each record
                for record in data:
                    record['data_date'] = file_info['date']
                    record['data_file'] = file_info['filename']
                all_data.extend(data)
        
        return all_data
    
    def add_stocks_from_list(self, stock_list):
        """Add multiple stocks from a list"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                for symbol in stock_list:
                    cursor.execute('''
                        INSERT OR IGNORE INTO stock_strategies (symbol, additional_checks_passed, updated_at) 
                        VALUES (?, 0, CURRENT_TIMESTAMP)
                    ''', (symbol,))
                
                conn.commit()
                print(f"Added {len(stock_list)} stocks to database")
            except Exception as e:
                print(f"Error adding stocks to database: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def update_strategy_data(self, symbol, **kwargs):
        """Update strategy data for a specific stock"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare the update query dynamically
            update_fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in [
                    'score', 'additional_checks_passed', 'position_active',
                    'position_shares', 'shares', 'current_price', 'entry_price', 
                    'stop_loss_price', 'take_profit_price', 'used_capital', 
                    'unrealized_pnl', 'realized_pnl', 'entry_time', 'close_time',
                    'profit_booking_levels',
                    'trailing_exit_conditions', 'trailing_stop_levels',
                    'profit_booked_flags', 'trailing_stop_flags',
                    'trailing_exit_monitoring', 'trailing_exit_start_time',
                    'trailing_exit_start_price', 'total_trades', 'winning_trades',
                    'losing_trades', 'total_pnl', 'avg_trade_duration',
                    'hedge_active', 'hedge_shares', 'hedge_symbol', 'hedge_level',
                    'hedge_beta', 'hedge_entry_price', 'hedge_entry_time',
                    'hedge_exit_price', 'hedge_pnl'
                ]:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if update_fields:
                            # Convert datetime objects and complex types
                for i, value in enumerate(values):
                    if isinstance(value, datetime):
                        values[i] = value.isoformat()
                    elif isinstance(value, (dict, list)):
                        values[i] = json.dumps(value)
                    elif isinstance(value, bool):
                        # Ensure boolean values are stored as integers (0/1) in SQLite
                        values[i] = 1 if value else 0

                # Add symbol and updated_at to the update
                update_fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(symbol)
                
                query = f'''
                    UPDATE stock_strategies 
                    SET {', '.join(update_fields)}
                    WHERE symbol = ?
                '''
                
                cursor.execute(query, values)
                
                # If no rows were updated, insert a new record using the same connection
                if cursor.rowcount == 0:
                    self._insert_strategy_data_with_connection(cursor, symbol, **kwargs)
            
            conn.commit()
            conn.close()
    
    def _insert_strategy_data_with_connection(self, cursor, symbol, **kwargs):
        """Insert new strategy data record using existing connection (internal method)"""
        # Prepare the insert query
        fields = ['symbol']
        values = [symbol]
        
        for key, value in kwargs.items():
            if key in [
                'score', 'additional_checks_passed', 'position_active',
                'position_shares', 'shares', 'current_price', 'entry_price', 
                'stop_loss_price', 'take_profit_price', 'used_capital', 
                'unrealized_pnl', 'realized_pnl', 'entry_time', 'close_time',
                'profit_booking_levels',
                'trailing_exit_conditions', 'trailing_stop_levels',
                'profit_booked_flags', 'trailing_stop_flags',
                'trailing_exit_monitoring', 'trailing_exit_start_time',
                'trailing_exit_start_price', 'total_trades', 'winning_trades',
                'losing_trades', 'total_pnl', 'avg_trade_duration',
                'hedge_active', 'hedge_shares', 'hedge_symbol', 'hedge_level',
                'hedge_beta', 'hedge_entry_price', 'hedge_entry_time',
                'hedge_exit_price', 'hedge_pnl'
            ]:
                fields.append(key)
                # Convert datetime objects to string if needed
                if isinstance(value, datetime):
                    values.append(value.isoformat())
                elif isinstance(value, (dict, list)):
                    values.append(json.dumps(value))
                elif isinstance(value, bool):
                    # Ensure boolean values are stored as integers (0/1) in SQLite
                    values.append(1 if value else 0)
                else:
                    values.append(value)
        
        placeholders = ', '.join(['?' for _ in fields])
        field_names = ', '.join(fields)
        
        query = f'''
            INSERT OR REPLACE INTO stock_strategies ({field_names}, updated_at)
            VALUES ({placeholders}, CURRENT_TIMESTAMP)
        '''
        
        cursor.execute(query, values)
    
    def insert_strategy_data(self, symbol, **kwargs):
        """Insert new strategy data record (standalone method with lock)"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                self._insert_strategy_data_with_connection(cursor, symbol, **kwargs)
                conn.commit()
            except Exception as e:
                print(f"Error inserting strategy data: {e}")
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def get_latest_strategy_data(self, symbol):
        """Get the strategy data for a specific stock"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT * FROM stock_strategies 
                    WHERE symbol = ?
                ''', (symbol,))
                
                row = cursor.fetchone()
                
                if row:
                    columns = [description[0] for description in cursor.description]
                    data = dict(zip(columns, row))
                    
                    # Convert boolean fields from SQLite integers to Python booleans
                    boolean_fields = ['additional_checks_passed', 'position_active', 'trailing_exit_monitoring', 'hedge_active']
                    for field in boolean_fields:
                        if field in data and data[field] is not None:
                            data[field] = bool(data[field])
                    
                    return data
                return None
            except Exception as e:
                print(f"Error getting strategy data: {e}")
                return None
            finally:
                conn.close()
    
    def get_all_active_positions(self):
        """Get all currently active positions"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT symbol, position_shares, shares, entry_price, current_price, 
                           stop_loss_price, take_profit_price, unrealized_pnl, realized_pnl, entry_time
                    FROM stock_strategies 
                    WHERE position_active = 1
                ''')
                
                rows = cursor.fetchall()
                
                columns = ['symbol', 'position_shares', 'shares', 'entry_price', 'current_price', 
                          'stop_loss_price', 'take_profit_price', 'unrealized_pnl', 'realized_pnl', 'entry_time']
                
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                print(f"Error getting active positions: {e}")
                return []
            finally:
                conn.close()
    
    def get_strategy_summary(self):
        """Get summary of all strategies"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    SELECT 
                        symbol,
                        score,
                        position_active,
                        position_shares,
                        shares,
                        current_price,
                        entry_price,
                        unrealized_pnl,
                        realized_pnl,
                        total_trades,
                        winning_trades,
                        losing_trades,
                        total_pnl
                    FROM stock_strategies 
                    ORDER BY symbol
                ''')
                
                rows = cursor.fetchall()
                
                columns = ['symbol', 'score', 'position_active', 'position_shares', 'shares',
                          'current_price', 'entry_price', 'unrealized_pnl', 'realized_pnl', 'total_trades', 
                          'winning_trades', 'losing_trades', 'total_pnl']
                
                # Convert boolean fields from SQLite integers to Python booleans
                result = []
                for row in rows:
                    data = dict(zip(columns, row))
                    # Convert position_active from integer to boolean
                    if 'position_active' in data and data['position_active'] is not None:
                        data['position_active'] = bool(data['position_active'])
                    result.append(data)
                
                return result
            except Exception as e:
                print(f"Error getting strategy summary: {e}")
                return []
            finally:
                conn.close()
    
    def get_position_shares(self, symbol):
        """Get current position shares for a symbol"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT position_shares FROM stock_strategies WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result and result[0] is not None else 0
        except Exception as e:
            print(f"Error getting position shares for {symbol}: {e}")
            return 0
    
    def get_realized_pnl(self, symbol):
        """Get current realized PnL for a symbol"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT realized_pnl FROM stock_strategies WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result and result[0] is not None else 0
        except Exception as e:
            print(f"Error getting realized PnL for {symbol}: {e}")
            return 0
    
    def get_entry_price(self, symbol):
        """Get entry price for a symbol"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT entry_price FROM stock_strategies WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result and result[0] is not None else 0
        except Exception as e:
            print(f"Error getting entry price for {symbol}: {e}")
            return 0
    
    def get_shares(self, symbol):
        """Get cumulative shares (total bought) for a symbol"""
        try:
            with self.lock:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT shares FROM stock_strategies WHERE symbol = ?", (symbol,))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result and result[0] is not None else 0
        except Exception as e:
            print(f"Error getting shares for {symbol}: {e}")
            return 0

# Global database instance
trades_db = TradesDatabase()
