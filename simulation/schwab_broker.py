import os
from datetime import datetime
from schwabdev import Client
import sys
sys.path.append('..')

class SchwabBroker:
    """
    Schwab Broker implementation using the schwabdev library.
    Provides core market data functions.
    """
    
    def __init__(self, tokens_file="tokens.json", timeout=10):
        """
        Initialize the Schwab broker client.
        
        Args:
            tokens_file (str): Path to the tokens.json file
            timeout (int): Request timeout in seconds
        """
        self.tokens_file = tokens_file
        self.timeout = timeout
        
        # Hard-coded API credentials
        self.api_key = "18TX6PVgTzC3k3iaWgmiGwff2Gb9CiEv"
        self.secret_key = "pUGJDAm3oSqjx5Aq"
        
        # Initialize the Schwab client with connection pool management
        self.client = Client(
            app_key=self.api_key,
            app_secret=self.secret_key,
            callback_url="https://127.0.0.1",
            tokens_file=self.tokens_file,
            timeout=self.timeout,
            capture_callback=False,
            use_session=True
        )
        
        # Add connection pool management
        import threading
        self._connection_semaphore = threading.Semaphore(8)  # Limit to 8 concurrent connections
        self._max_connections = 8  # Keep below the 10 limit
    
    def _make_api_call(self, api_method, *args, **kwargs):
        """
        Make an API call with connection pool management.
        
        Args:
            api_method: The API method to call
            *args: Arguments for the API method
            **kwargs: Keyword arguments for the API method
            
        Returns:
            The API response or None if failed
        """
        try:
            with self._connection_semaphore:
                import time
                # Add a small delay to prevent overwhelming the API
                time.sleep(0.1)
                return api_method(*args, **kwargs)
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    def account_details(self):
        """
        Get account details for all linked accounts.
        
        Returns:
            dict: Account details response
        """
        try:
            response = self._make_api_call(self.client.account_details_all)
            if response and response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get account details: {response.status_code if response else 'No response'} - {response.text if response else 'No response'}")
                return None
        except Exception as e:
            print(f"Error getting account details: {e}")
            return None
    
    def get_quote_on_symbol(self, symbol: str, fields: str = None):
        """
        Get quote for a single symbol.
        
        Args:
            symbol (str): Ticker symbol (e.g., "AAPL")
            fields (str, optional): Data fields to return (e.g., "all", "quote", "fundamental")
            
        Returns:
            dict: Quote data for the symbol
        """
        try:
            response = self._make_api_call(self.client.quote, symbol, fields=fields)
            if response and response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get quote for {symbol}: {response.status_code if response else 'No response'} - {response.text if response else 'No response'}")
                return None
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_bulk_quotes(self, symbols: list[str], fields: str = "quote,reference"):
        """
        Fetch quotes in bulk from Schwab API.
        
        Args:
            symbols (list): List of stock symbols
            fields (str): Optional fields to include (default: quote,reference)
            
        Returns:
            dict: Symbol-wise quote data
        """
        try:
            response = self._make_api_call(self.client.quotes, symbols, fields=fields)
            if response and response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get bulk quotes: {response.status_code if response else 'No response'} - {response.text if response else 'No response'}")
                return {}
        except Exception as e:
            print(f"Error getting bulk quotes: {e}")
            return {}
    
    def get_price_history(
        self,
        symbol: str,
        period_type: str = "day",
        period: int = 5,
        frequency_type: str = "minute",
        frequency: int = 1,
        start_date: datetime = None,
        end_date: datetime = None,
        need_extended_hours_data: bool = False,
        need_previous_close: bool = False
    ):
        """
        Fetches historical price data from Charles Schwab's /pricehistory endpoint.
        
        Args:
            symbol (str): Stock symbol
            period_type (str): Type of period (day, month, year, ytd)
            period (int): Number of periods
            frequency_type (str): Type of frequency (minute, daily, weekly, monthly)
            frequency (int): Frequency value
            start_date (datetime): Start date for historical data
            end_date (datetime): End date for historical data
            need_extended_hours_data (bool): Include extended hours data
            need_previous_close (bool): Include previous close data
            
        Returns:
            dict: Historical price data
        """
        try:
            response = self._make_api_call(
                self.client.price_history,
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                frequency=frequency,
                startDate=start_date,
                endDate=end_date,
                needExtendedHoursData=need_extended_hours_data,
                needPreviousClose=need_previous_close
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get price history for {symbol}: {response.status_code if response else 'No response'} - {response.text if response else 'No response'}")
                return None
        except Exception as e:
            print(f"Error getting price history for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str):
        """
        Get current price for a single symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            float: Current price or None if error
        """
        try:
            quote_data = self.get_quote_on_symbol(symbol, fields="quote")
            if quote_data and symbol in quote_data:
                return quote_data[symbol]["quote"]["lastPrice"]
            elif quote_data and "quote" in quote_data:
                return quote_data["quote"]["lastPrice"]
            else:
                print(f"Unexpected quote data structure for {symbol}")
                return None
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_connection_pool_status(self):
        """
        Get the current status of the connection pool.
        
        Returns:
            dict: Connection pool status information
        """
        return {
            "max_connections": self._max_connections,
            "available_connections": self._connection_semaphore._value,
            "used_connections": self._max_connections - self._connection_semaphore._value
        }
    
    def get_bulk_quotes_with_retry(self, symbols: list[str], max_retries: int = 3, fields: str = "quote,reference"):
        """
        Fetch quotes in bulk with retry logic and connection pool management.
        
        Args:
            symbols (list): List of stock symbols
            max_retries (int): Maximum number of retry attempts
            fields (str): Optional fields to include
            
        Returns:
            dict: Symbol-wise quote data
        """
        for attempt in range(max_retries):
            try:
                result = self.get_bulk_quotes(symbols, fields=fields)
                if result:
                    return result
                else:
                    print(f"Bulk quotes attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(1)  # Wait before retry
            except Exception as e:
                print(f"Bulk quotes attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)  # Wait longer before retry
        
        print(f"All {max_retries} attempts to get bulk quotes failed")
        return {}
    



if __name__ == "__main__":
    # Example usage
    try:
        broker = SchwabBroker()
        
        # Test account details
        print("Getting account details...")
        accounts = broker.account_details()
        if accounts:
            print("Account details retrieved successfully")
        
        # Test quote
        print("\nGetting quote for AAPL...")
        quote = broker.get_quote_on_symbol("AAPL", fields="quote")
        if quote:
            print("Quote retrieved successfully")
        
        # Test bulk quotes
        print("\nGetting bulk quotes...")
        symbols = ["AAPL", "MSFT", "AMZN", "BAC", "GOOG"]
        bulk_quotes = broker.get_bulk_quotes(symbols)
        if bulk_quotes:
            print(f"Bulk quotes retrieved for {len(symbols)} symbols")
        
        # Test current price
        print("\nGetting current price for AAPL...")
        price = broker.get_current_price("AAPL")
        if price:
            print(f"AAPL current price: ${price}")
        
    except Exception as e:
        print(f"Error: {e}") 