import sqlite3
import threading
from datetime import datetime, timedelta
import json
import os
import shutil

class TradesDatabase:
    def __init__(self, db_path='trades.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
    
    def init_database(self):
        """Initialize the database with a single table for all stock data"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
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
            conn.close()
    
    def backup_database(self):
        """Backup the current database to historical_data folder with date-based naming using creation date"""
        if not os.path.exists(self.db_path):
            return False
            
        # Create historical_data directory if it doesn't exist
        historical_dir = "historical_data"
        if not os.path.exists(historical_dir):
            os.makedirs(historical_dir)
        
        # Get database creation date
        db_creation_time = os.path.getctime(self.db_path)
        creation_date = datetime.fromtimestamp(db_creation_time).strftime("%Y-%m-%d")
        
        # Create backup filename using creation date
        backup_filename = f"trades_{creation_date}.db"
        backup_path = os.path.join(historical_dir, backup_filename)
        
        # If backup already exists for this creation date, add timestamp
        if os.path.exists(backup_path):
            timestamp = datetime.fromtimestamp(db_creation_time).strftime("%Y-%m-%d_%H-%M-%S")
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for symbol in stock_list:
                cursor.execute('''
                    INSERT OR IGNORE INTO stock_strategies (symbol, additional_checks_passed, updated_at) 
                    VALUES (?, 0, CURRENT_TIMESTAMP)
                ''', (symbol,))
            
            conn.commit()
            conn.close()
            print(f"Added {len(stock_list)} stocks to database")
    
    def update_strategy_data(self, symbol, **kwargs):
        """Update strategy data for a specific stock"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
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
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            self._insert_strategy_data_with_connection(cursor, symbol, **kwargs)
            
            conn.commit()
            conn.close()
    
    def get_latest_strategy_data(self, symbol):
        """Get the strategy data for a specific stock"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM stock_strategies 
                WHERE symbol = ?
            ''', (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
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
    
    def get_all_active_positions(self):
        """Get all currently active positions"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, position_shares, shares, entry_price, current_price, 
                       stop_loss_price, take_profit_price, unrealized_pnl, realized_pnl, entry_time
                FROM stock_strategies 
                WHERE position_active = 1
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            columns = ['symbol', 'position_shares', 'shares', 'entry_price', 'current_price', 
                      'stop_loss_price', 'take_profit_price', 'unrealized_pnl', 'realized_pnl', 'entry_time']
            
            return [dict(zip(columns, row)) for row in rows]
    
    def get_strategy_summary(self):
        """Get summary of all strategies"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
            conn.close()
            
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

# Global database instance
trades_db = TradesDatabase()
