import pandas as pd
import time
from collections import defaultdict
import yfinance as yf
from simulation.ibkr_broker import IBTWSAPI
import json
from types import SimpleNamespace
from helpers.adv import calc_adv

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
    def __init__(self, csv_path='companies_by_marketcap.csv'):
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
        self.client = IBTWSAPI()
        self.client.connect()
        self.batch_size = 50

        self.symbols = []
        self.filtered = []
        self.qualified = []
        self.sector_returns = defaultdict(list)
        self.top_sectors = []

    def load_and_filter_market_cap(self):
        df = pd.read_csv(self.csv_path)
        df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
        df = df[df["marketcap"] > self.market_cap_filter]
        self.symbols = df["Symbol"].tolist()
        print(f"Loaded {len(self.symbols)} symbols with market cap > {self.market_cap_filter}")

    def process_volume_batch(self, batch):
        result = {}
        for _, row in batch.iterrows():
            symbol = row["Symbol"]
            price = row["price (USD)"]

            volume_df_adv = self.client.get_volume(symbol=symbol, duration=f"{self.volume_filter_ADV_large_length} D", bar_size="1 day")
            adv_large = calc_adv(volume_df_adv, days=self.volume_filter_ADV_large_length)    
            adv_small = calc_adv(volume_df_adv, days=self.volume_filter_ADV_small_length)
        
            if adv_large >= self.volume_filter_ADV_large_filter and adv_small >= self.volume_filter_ADV_small_filter:
                result[symbol] = {
                    "symbol": symbol,
                    "price": price,
                    "adv_large": adv_large,
                    "adv_small": adv_small
                }
                print(f"Added {symbol} to result")
            else:
                continue
            volume_df_rvol = self.client.get_volume(symbol=symbol, duration=f"{self.rvol_length} D", bar_size="15 mins")


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
        top_df.to_csv("filtered_top_sectors.csv", index=False)

        end_time = time.time()
        print(f"\nTime Taken: {end_time - start_time:.2f} seconds")
        return top_df["symbol"].tolist(), top_df.to_dict(orient="records")  # Final result as list of dicts


if __name__ == "__main__":
    selector = StockSelector()
    top_sector_stocks = selector.run()
    print(top_sector_stocks)
