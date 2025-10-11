import pandas as pd
import time
from collections import defaultdict
import yfinance as yf
from simulation.ibkr_broker import IBTWSAPI
import json
import pickle
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from helpers.adv import calc_adv
from helpers.rvol import calc_rvol
from helpers.fetch_marketcap_csv import fetch_marketcap_csv

def load_config(json_file='creds.json'):
    """Load configuration from JSON file and make it accessible like a module"""
    with open(json_file, 'r') as f:
        config_dict = json.load(f)
    
    # Convert dict to object with dot notation access
    def dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_obj(item) for item in d]
        else:
            return d
    
    return dict_to_obj(config_dict)

creds = load_config('creds.json')


class StockSelector:
    def __init__(self, csv_path='companies_by_marketcap.csv', client_id=None):
        self.csv_path = csv_path
        self.market_cap_filter = creds.STOCK_SELECTION.market_cap_min
        self.price_filter = creds.STOCK_SELECTION.price_min
        self.volume_filter_ADV_large_filter = creds.STOCK_SELECTION.ADV_large
        self.volume_filter_ADV_large_length = creds.STOCK_SELECTION.ADV_large_length 
        self.volume_filter_ADV_small_filter = creds.STOCK_SELECTION.ADV_small
        self.volume_filter_ADV_small_length = creds.STOCK_SELECTION.ADV_small_length
        self.rvol_filter = creds.STOCK_SELECTION.RVOL_filter
        self.rvol_length = creds.STOCK_SELECTION.RVOL_length
        # self.client = SchwabBroker()
        self.client = IBTWSAPI(client_id=client_id) if client_id is not None else IBTWSAPI()
        self.client.connect()
        self.batch_size = 50

        self.symbols = []
        self.filtered = []
        self.qualified = []
        self.sector_returns = defaultdict(list)
        self.top_sectors = []
        
        # RVOL data caching
        self.rvol_cache_file = "rvol_cache.pkl"
        self.rvol_data_cache = {}
        
        # ADV data caching
        self.adv_cache_file = "adv_cache.pkl"
        self.adv_data_cache = {}

    def disconnect(self):
        """Disconnect from IBKR broker"""
        if self.client and hasattr(self.client, 'client') and self.client.client:
            try:
                self.client.client.disconnect()
                print("Disconnected from IBKR broker")
            except Exception as e:
                print(f"Error disconnecting from IBKR broker: {e}")

    def load_and_filter_market_cap(self):
        
        df = pd.read_csv(self.csv_path)
        df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
        df = df[df["marketcap"] > self.market_cap_filter]
        self.symbols = df["Symbol"].tolist()
        
        print(f"Loaded {len(self.symbols)} symbols with market cap > {self.market_cap_filter}")
        
        self._cache_adv_data()
        self.qualified_symbols = self._filter_symbols_by_adv_criteria()
        self._cache_rvol_data()

    def _cache_rvol_data(self):

        # Check if cache file exists and is recent (less than 1 day old)
        if os.path.exists(self.rvol_cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(self.rvol_cache_file))
            if datetime.now() - cache_time < timedelta(days=1):
                print("Loading existing RVOL cache...")
                self._load_rvol_cache()
                return
        
        # Fetch historical data for qualified symbols
        start_time = time.time()
        for i, symbol in enumerate(self.qualified_symbols):
            try:
                print(f"Fetching RVOL data for {symbol} ({i+1}/{len(self.qualified_symbols)})")
                # Fetch historical data (excluding today)
                historical_df = self.client.get_volume(
                    symbol=symbol, 
                    duration=f"{self.rvol_length + 1} D", 
                    bar_size="15 mins"
                )
                
                if not historical_df.empty:
                    # Remove last entry (today's incomplete data)
                    historical_df = historical_df.iloc[:-1]
                    self.rvol_data_cache[symbol] = historical_df
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        # Save cache to file
        self._save_rvol_cache()
        
        end_time = time.time()
        print(f"RVOL data caching completed in {end_time - start_time:.2f} seconds")
        print(f"Cached data for {len(self.rvol_data_cache)} symbols")

    def _save_rvol_cache(self):
        """Save RVOL data cache to file"""
        try:
            with open(self.rvol_cache_file, 'wb') as f:
                pickle.dump(self.rvol_data_cache, f)
            print(f"RVOL cache saved to {self.rvol_cache_file}")
        except Exception as e:
            print(f"Error saving RVOL cache: {e}")

    def _load_rvol_cache(self):
        """Load RVOL data cache from file"""
        try:
            with open(self.rvol_cache_file, 'rb') as f:
                self.rvol_data_cache = pickle.load(f)
            print(f"Loaded RVOL cache with {len(self.rvol_data_cache)} symbols")
        except Exception as e:
            print(f"Error loading RVOL cache: {e}")
            self.rvol_data_cache = {}

    def _get_rvol_data_with_today(self, symbol):
        """Get RVOL data by combining cached historical data with today's data"""
        # Load cache if not already loaded
        if not self.rvol_data_cache:
            self._load_rvol_cache()
            
        if symbol not in self.rvol_data_cache:
            print(f"No cached data for {symbol}")
            return None
        
        try:
            # Get today's data
            today_df = self.client.get_volume(
                symbol=symbol,
                duration="1 D",
                bar_size="15 mins"
            )
            
            if today_df.empty:
                print(f"No today's data for {symbol}")
                return self.rvol_data_cache[symbol]
            
            # Get historical data and remove any existing today's data
            historical_df = self.rvol_data_cache[symbol].copy()
            today_date = today_df.index.max().date()
            
            # Remove any data from today that might already be in historical data
            historical_df = historical_df[historical_df.index.date < today_date]
            
            # Combine historical data with today's data
            combined_df = pd.concat([historical_df, today_df], ignore_index=False)
            
            return combined_df
            
        except Exception as e:
            print(f"Error getting today's data for {symbol}: {e}")
            return self.rvol_data_cache[symbol]

    def _cache_adv_data(self):
        """Pre-fetch and cache ADV historical data for all symbols"""
        print(f"Pre-fetching ADV data for {len(self.symbols)} symbols...")
        
        # Check if cache file exists and is recent (less than 1 day old)
        if os.path.exists(self.adv_cache_file):
            cache_time = datetime.fromtimestamp(os.path.getmtime(self.adv_cache_file))
            if datetime.now() - cache_time < timedelta(days=1):
                print("Loading existing ADV cache...")
                self._load_adv_cache()
                return
        
        # Fetch historical data for all symbols
        start_time = time.time()
        for i, symbol in enumerate(self.symbols):
            try:
                print(f"Fetching ADV data for {symbol} ({i+1}/{len(self.symbols)})")
                # Fetch historical data (excluding today)
                historical_df = self.client.get_volume(
                    symbol=symbol, 
                    duration=f"{max(self.volume_filter_ADV_large_length, self.volume_filter_ADV_small_length) + 5} D",
                    bar_size="1 day"
                )
                
                if not historical_df.empty:
                    # Remove last entry (today's incomplete data)
                    historical_df = historical_df.iloc[:-1]
                    self.adv_data_cache[symbol] = historical_df
                    
            except Exception as e:
                print(f"Error fetching ADV data for {symbol}: {e}")
                continue
        
        # Save cache to file
        self._save_adv_cache()
        
        end_time = time.time()
        print(f"ADV data caching completed in {end_time - start_time:.2f} seconds")
        print(f"Cached ADV data for {len(self.adv_data_cache)} symbols")

    def _filter_symbols_by_adv_criteria(self):

        print(f"Filtering symbols by ADV criteria...")
        
        qualified_symbols = []
        start_time = time.time()
        
        for i, symbol in enumerate(self.symbols):
            try:
                print(f"Checking ADV criteria for {symbol} ({i+1}/{len(self.symbols)})")
                
                # Get cached ADV data
                volume_df_adv = self._get_adv_data(symbol)
                if volume_df_adv is not None:
                    adv_large = calc_adv(volume_df_adv, days=self.volume_filter_ADV_large_length)    
                    adv_small = calc_adv(volume_df_adv, days=self.volume_filter_ADV_small_length)
                    
                    # Check if symbol passes ADV criteria
                    if adv_large >= self.volume_filter_ADV_large_filter and adv_small >= self.volume_filter_ADV_small_filter:
                        qualified_symbols.append(symbol)
                        print(f"{symbol} passed ADV criteria (Large: {adv_large:,.0f}, Small: {adv_small:,.0f})")
                    else:
                        print(f"{symbol} failed ADV criteria (Large: {adv_large:,.0f}, Small: {adv_small:,.0f})")
                else:
                    print(f"{symbol} - No ADV data available")
                    
            except Exception as e:
                print(f"Error checking ADV criteria for {symbol}: {e}")
                continue
        
        end_time = time.time()
        print(f"ADV filtering completed in {end_time - start_time:.2f} seconds")
        print(f"{len(qualified_symbols)} symbols passed ADV criteria out of {len(self.symbols)} total symbols")
        
        return qualified_symbols

    def _save_adv_cache(self):
        """Save ADV data cache to file"""
        try:
            with open(self.adv_cache_file, 'wb') as f:
                pickle.dump(self.adv_data_cache, f)
            print(f"ADV cache saved to {self.adv_cache_file}")
        except Exception as e:
            print(f"Error saving ADV cache: {e}")

    def _load_adv_cache(self):
        """Load ADV data cache from file"""
        try:
            with open(self.adv_cache_file, 'rb') as f:
                self.adv_data_cache = pickle.load(f)
            print(f"Loaded ADV cache with {len(self.adv_data_cache)} symbols")
        except Exception as e:
            print(f"Error loading ADV cache: {e}")
            self.adv_data_cache = {}

    def _get_adv_data(self, symbol):
        """Get cached ADV historical data (no need for today's data)"""
        # Load cache if not already loaded
        if not self.adv_data_cache:
            self._load_adv_cache()
            
        if symbol not in self.adv_data_cache:
            print(f"No cached ADV data for {symbol}")
            return None
        
        return self.adv_data_cache[symbol]

    def process_volume_batch(self, batch):
        result = {}
        for _, row in batch.iterrows():
            symbol = row["Symbol"]
            price = row["price (USD)"]

            # Use cached ADV data (no need for today's data)
            volume_df_adv = self._get_adv_data(symbol)
            if volume_df_adv is not None:
                adv_large = calc_adv(volume_df_adv, days=self.volume_filter_ADV_large_length)    
                adv_small = calc_adv(volume_df_adv, days=self.volume_filter_ADV_small_length)
            else:
                # Fallback: fetch ADV data manually if cache fails
                print(f"No cached ADV data for {symbol}, fetching manually...")
                try:
                    volume_df_adv = self.client.get_volume(
                        symbol=symbol, 
                        duration=f"{max(self.volume_filter_ADV_large_length, self.volume_filter_ADV_small_length)} D", 
                        bar_size="1 day"
                    )
                    if not volume_df_adv.empty:
                        adv_large = calc_adv(volume_df_adv, days=self.volume_filter_ADV_large_length)    
                        adv_small = calc_adv(volume_df_adv, days=self.volume_filter_ADV_small_length)
                    else:
                        print(f"No ADV data available for {symbol} even after manual fetch")
                        continue
                except Exception as e:
                    print(f"Error manually fetching ADV data for {symbol}: {e}")
                    continue
        
            if adv_large >= self.volume_filter_ADV_large_filter and adv_small >= self.volume_filter_ADV_small_filter:
                # Use cached RVOL data with today's data appended
                volume_df_rvol = self._get_rvol_data_with_today(symbol)
                if volume_df_rvol is not None:
                    rvol = calc_rvol(volume_df_rvol, days=self.rvol_length)
                else:
                    # Fallback: fetch RVOL data manually if cache fails
                    print(f"No cached RVOL data for {symbol}, fetching manually...")
                    try:
                        volume_df_rvol = self.client.get_volume(
                            symbol=symbol, 
                            duration=f"{self.rvol_length} D", 
                            bar_size="15 mins"
                        )
                        if not volume_df_rvol.empty:
                            rvol = calc_rvol(volume_df_rvol, days=self.rvol_length)
                        else:
                            print(f"No RVOL data available for {symbol} even after manual fetch")
                            continue
                    except Exception as e:
                        print(f"Error manually fetching RVOL data for {symbol}: {e}")
                        continue
                
                if rvol >= self.rvol_filter:
                    result[symbol] = {
                        "symbol": symbol,
                        "price": price,
                        "adv_large": adv_large,
                        "adv_small": adv_small,
                        "rvol": rvol
                    }

            else:
                continue
        return result

    def filter_by_price_volume(self):
        start_time = time.time()
        df = pd.read_csv("companies_by_marketcap.csv")
        df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
        df["price (USD)"] = pd.to_numeric(df["price (USD)"], errors="coerce")

        # Apply filters
        df = df[(df["marketcap"] > self.market_cap_filter) & (df["price (USD)"] > self.price_filter)]
        print(f"Filtered {len(df)} symbols with market cap > {self.market_cap_filter} and price > {self.price_filter}")
        # Process in batches
        batch_size = 50
        all_quotes = {}
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_quotes = self.process_volume_batch(batch)
            all_quotes.update(batch_quotes)
            print(f"Batch {i//batch_size + 1} completed. Total quotes collected: {len(all_quotes)}")

        for i in all_quotes:
            self.filtered.append(all_quotes[i])
            print(f"Added {all_quotes[i]} to filtered")

        # Final results
        print(f"Total quotes collected: {len(all_quotes)}")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")


        return all_quotes
        
    # def filter_by_price_volume(self):
    #     df = pd.read_csv("companies_by_marketcap.csv")
    #     df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
    #     df["price (USD)"] = pd.to_numeric(df["price (USD)"], errors="coerce")
    #     df = df[(df["marketcap"] > self.market_cap_filter) & (df["price (USD)"] > self.price_filter)]
    #     tic = df["Symbol"].tolist()
    #     print(f"Loaded {len(tic)} symbols with market cap > {self.market_cap_filter} and price > {self.price_filter}")

    #     print(f"Testing bulk quotes for {len(tic)} symbols...")
    #     start_time = time.time()
    
    #     # Process symbols in batches of 50
    #     batch_size = 50
    #     all_quotes = {}
        
    #     for i in range(0, len(tic), batch_size):
    #         batch = tic[i:i + batch_size]
    #         batch_num = (i // batch_size) + 1
    #         total_batches = (len(tic) + batch_size - 1) // batch_size
            
    #         print(f"Processing batch {batch_num}/{total_batches} with {len(batch)} symbols...")
            
    #         # Get quotes for this batch
    #         batch_quotes = self.process_volume_batch(batch)
            
    #         # Store results in all_quotes
    #         all_quotes.update(batch_quotes)
            
    #         print(f"Batch {batch_num} completed. Total quotes collected: {len(all_quotes)}")
        
    #     end_time = time.time()
    #     print(f"All batches completed. Time taken: {end_time - start_time} seconds")
    #     print(f"Total quotes collected: {len(all_quotes)}")
        
    #     # Store the final results in quotes variable
    #     quotes = all_quotes
    #     print("Original quotes:")
    #     print(quotes)
        
    #     # Filter out stocks with volume < 100,000 and price < 10
    #     filtered_quotes = {}
    #     for symbol, data in quotes.items():
    #         if data and 'volume' in data and 'price' in data:
    #             volume = data['volume']
    #             price = data['price']
                
    #             # Keep stocks with volume >= 100,000 AND price >= 10
    #             if volume >= self.volume_filter and price >= self.price_filter:
    #                 filtered_quotes[symbol] = data
    #             else:
    #                 print(f"Removing {symbol}: volume={volume}, price={price}")
        
    #     print(f"\nFiltered quotes (removed {len(quotes) - len(filtered_quotes)} stocks):")
    #     print(filtered_quotes)
    #     print(f"Remaining stocks: {len(filtered_quotes)}")
        
    #     # Add filtered stocks to self.filtered
    #     for symbol, data in filtered_quotes.items():
    #         self.filtered.append({
    #             "symbol": symbol,
    #             "price": data['price'],
    #             "volume": data['volume']
    #         })
        
    #     print(f"Added {len(filtered_quotes)} stocks to self.filtered")

    # def filter_by_price_volume(self):
    #     for i in range(0, len(self.symbols), self.batch_size):
    #         batch = self.symbols[i:i + self.batch_size]
    #         try:
    #             quotes = self.client.get_bulk_quotes(batch)
    #             for symbol in batch:
    #                 q = quotes.get(symbol, {}).get("quote", {})
    #                 price = q.get("mark") or q.get("lastPrice")
    #                 volume = q.get("totalVolume")

    #                 if price and price >= creds.STOCK_SELECTION.price_min and volume and volume >= creds.STOCK_SELECTION.volume_min:
    #                     self.filtered.append({
    #                         "symbol": symbol,
    #                         "price": price,
    #                         "volume": volume
    #                     })
    #         except Exception as e:
    #             print(f"Skipping batch {batch}: {e}")

    #     print(f"{len(self.filtered)} stocks passed price & volume filter")

    def enrich_with_alpha_and_sector(self):
        for stock in self.filtered:
            symbol = stock["symbol"]
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector = info.get("sector", "Unknown")

                hist = ticker.history(period="6d")
                if len(hist) < 6:
                    continue

                price_now = hist["Close"].iloc[-1]
                price_5d_ago = hist["Close"].iloc[0]
                alpha_5d = (price_now - price_5d_ago) / price_5d_ago

                stock.update({
                    "sector": sector,
                    "alpha_5d": alpha_5d
                })

                if alpha_5d > creds.STOCK_SELECTION.alpha_threshold:
                    self.qualified.append(stock)
                    self.sector_returns[sector].append(alpha_5d)
            except Exception as e:
                print(f"Skipping {symbol} (yfinance): {e}")

        print(f"{len(self.qualified)} stocks qualified with alpha > 0.5%")

    def identify_top_sectors(self):
        if not self.sector_returns:
            print("\nNo sectors found - no stocks qualified")
            self.top_sectors = []
            return

        self.top_sectors = sorted(
            self.sector_returns.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )[:3]

        print("\nTop 3 Sectors by Avg 5-Day Alpha:")
        for sector, alphas in self.top_sectors:
            avg_alpha = sum(alphas) / len(alphas)
            count = len([s for s in self.qualified if s["sector"] == sector])
            print(f"- {sector}: {avg_alpha:.2%} ({count} stocks)")

    def get_top_sector_stocks(self):
        top_sector_names = {s[0] for s in self.top_sectors}
        return [s for s in self.qualified if s["sector"] in top_sector_names]

    def run(self):
        start_time = time.time()

        self.load_and_filter_market_cap()
        self.filter_by_price_volume()
        self.enrich_with_alpha_and_sector()
        self.identify_top_sectors()

        # Check if we have any qualified stocks before creating DataFrames
        if not self.qualified:
            print("No stocks qualified - returning empty lists")
            return [], []

        all_df = pd.DataFrame(self.qualified).sort_values(by="alpha_5d", ascending=False)
        # all_df.to_csv("filtered_all_valid.csv", index=False)

        top_sector_stocks = self.get_top_sector_stocks()
        if not top_sector_stocks:
            print("No top sector stocks found - returning empty lists")
            return [], []

        top_df = pd.DataFrame(top_sector_stocks).sort_values(by="alpha_5d", ascending=False)
        # top_df.to_csv("filtered_top_sectors.csv", index=False)

        end_time = time.time()
        print(f"\nTime Taken: {end_time - start_time:.2f} seconds")
        return top_df["symbol"].tolist(), top_df.to_dict(orient="records")  # Final result as list of dicts


def initialize_stock_selector():
    fetch_marketcap_csv()
    print("Deleted existing cache files")
    cache_files = ['adv_cache.pkl', 'rvol_cache.pkl']
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"Deleted existing cache file: {cache_file}")
            except Exception as e:
                print(f"Warning: Could not delete cache file {cache_file}: {e}")
    selector = StockSelector()
    start_time = time.time()
    selector.load_and_filter_market_cap()
    end_time = time.time()
    print(f"Time taken {end_time-start_time}")
    
    # Disconnect from IBKR to free up the client ID
    selector.disconnect()
    print("Disconnected from IBKR broker after initialization")

if __name__ == "__main__":
    initialize_stock_selector()

