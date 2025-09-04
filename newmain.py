from simulation import ibkr_broker 
import pandas as pd
import time
import asyncio

class StrategyBroker:

    def __init__(self):
        self.client = ibkr_broker.IBTWSAPI()
        self.client.connect()
    
    def is_connected(self):
        return self.client.is_connected()

    def get_current_price(self, stock: str):
        return self.client.get_stock_price(stock)


    def get_historical_data(self, stock: str, bar_size: str = '3 mins', num: str = '1 D', 
                           exchange: str = 'SMART', max_retries: int = 3, retry_delay: float = 2.0):
        for attempt in range(max_retries + 1):  # +1 because we want to try max_retries times after the first attempt
            try:
                print(f"Attempting to fetch historical data for {stock} (attempt {attempt + 1}/{max_retries + 1})")
                
                data = self.client.get_historical_data(stock, num, bar_size, exchange)
                
                # Check if we got valid data
                if data is not None and not data.empty and len(data) > 0:
                    print(f"Successfully fetched {len(data)} bars of historical data for {stock}")
                    return data
                else:
                    print(f"Attempt {attempt + 1}: No data returned for {stock}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: Error fetching historical data for {stock}: {str(e)}")
            
            # If this wasn't the last attempt, wait before retrying
            if attempt < max_retries:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
        
        # If we get here, all retries failed
        print(f"ERROR: Failed to fetch historical data for {stock} after {max_retries + 1} attempts")
        return pd.DataFrame()

    def place_order(self, symbol: str, qty: int, order_type: str = 'MARKET', 
               price: float = None, side: str = 'BUY', max_retries: int = 3, retry_delay: float = 2.0):

        trade = self.client.place_order(symbol=symbol, quantity=qty, order_type=order_type, price = price, side=side)
        
        if trade["status"] == "Filled":
            return trade
        else:
            return None
        # for attempt in range(max_retries+1):
        #     try:
                
                    
        #     except Exception as e:
        #         pass
            
        #     # If this wasn't the last attempt, wait before retrying
        #     if attempt < max_retries:
        #         print(f"Waiting {retry_delay} seconds before retry...")
        #         time.sleep(retry_delay)
        
        # # If we get here, all retries failed
        # print(f"ERROR: Failed to place order for  {symbol} after {max_retries + 1} attempts")
        # return pd.DataFrame()

    def get_bulk_quotes(self, symbols: list, max_retries: int = 3, retry_delay: float = 2.0):
        result = {}
        for i in symbols:
            quote = self.client.get_quote(i)
            print(quote)
            result[i] = quote
        return result


if __name__ == "__main__":
    strategy_broker = StrategyBroker()
    print(f"Connected: {strategy_broker.is_connected()}")
    
    # Test bulk quotes with batching
    symbols = [
        "AAPL", 
        "MSFT", "GOOG", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX",
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SPGI", "V",
        "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
        "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS", "CMCSA",
        "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO", "PSX", "KMI",
        "BA", "CAT", "GE", "HON", "MMM", "UPS", "FDX", "LMT", "RTX", "NOC",
        "LIN", "APD", "SHW", "ECL", "DD", "DOW", "FCX", "NEM", "PPG", "EMN",
        "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "PEG", "WEC", "ES"
    ]

    df = pd.read_csv("companies_by_marketcap.csv")
    df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
    df["price (USD)"] = pd.to_numeric(df["price (USD)"], errors="coerce")
    
    # Filter by market cap > $200B and price > $10
    df = df[(df["marketcap"] > 2_000_000_000) & (df["price (USD)"] > 10)]
    tic = df["Symbol"].tolist()
    print(f"Loaded {len(tic)} symbols with market cap > $2B and price > $10")
    
    print(f"Testing bulk quotes for {len(tic)} symbols...")
    start_time = time.time()
    
    # Process symbols in batches of 50
    batch_size = 50
    all_quotes = {}
    
    for i in range(0, len(tic), batch_size):
        batch = tic[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(tic) + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} with {len(batch)} symbols...")
        
        # Get quotes for this batch
        batch_quotes = strategy_broker.get_bulk_quotes(batch)
        
        # Store results in all_quotes
        all_quotes.update(batch_quotes)
        
        print(f"Batch {batch_num} completed. Total quotes collected: {len(all_quotes)}")
    
    end_time = time.time()
    print(f"All batches completed. Time taken: {end_time - start_time} seconds")
    print(f"Total quotes collected: {len(all_quotes)}")
    
    # Store the final results in quotes variable
    quotes = all_quotes
    print(quotes)

    # # Test current price
    # current_price = strategy_broker.get_current_price("AAPL")
    # print(f"Current AAPL price: {current_price}")
    
    # # Test historical data with retry logic
    # print("\n--- Testing Historical Data with Retry Logic ---")
    # historical_data = strategy_broker.get_historical_data("AAPL", num='1 D', bar_size='5 mins')
    
    # if not historical_data.empty:
    #     print(f"Historical data shape: {historical_data.shape}")
    #     print("First few rows:")
    #     print(historical_data.head())
    # else:
    #     print("No historical data retrieved")