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
        
        # Initialize the Schwab client
        self.client = Client(
            app_key=self.api_key,
            app_secret=self.secret_key,
            callback_url="https://127.0.0.1",
            tokens_file=self.tokens_file,
            timeout=self.timeout,
            capture_callback=False,
            use_session=True
        )
    
    def account_details(self):
        """
        Get account details for all linked accounts.
        
        Returns:
            dict: Account details response
        """
        try:
            response = self.client.account_details_all()
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get account details: {response.status_code} - {response.text}")
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
            response = self.client.quote(symbol, fields=fields)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get quote for {symbol}: {response.status_code} - {response.text}")
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
            response = self.client.quotes(symbols, fields=fields)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get bulk quotes: {response.status_code} - {response.text}")
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
            response = self.client.price_history(
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
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get price history for {symbol}: {response.status_code} - {response.text}")
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