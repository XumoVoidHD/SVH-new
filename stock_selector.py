import pandas as pd
import time
from collections import defaultdict
import yfinance as yf
from simulation.schwab_broker import SchwabBroker
import creds


class StockSelector:
    def __init__(self, csv_path='companies_by_marketcap.csv'):
        self.csv_path = csv_path
        self.market_cap_filter = creds.MARKET_CAP
        self.client = SchwabBroker()
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

    def filter_by_price_volume(self):
        for i in range(0, len(self.symbols), self.batch_size):
            batch = self.symbols[i:i + self.batch_size]
            try:
                quotes = self.client.get_bulk_quotes(batch, fields="quote")
                for symbol in batch:
                    q = quotes.get(symbol, {}).get("quote", {})
                    price = q.get("mark") or q.get("lastPrice")
                    volume = q.get("totalVolume")

                    if price and price >= 10 and volume and volume >= 1_000_000:
                        self.filtered.append({
                            "symbol": symbol,
                            "price": price,
                            "volume": volume
                        })
            except Exception as e:
                print(f"Skipping batch {batch}: {e}")

        print(f"{len(self.filtered)} stocks passed price & volume filter")

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

                if alpha_5d > 0.005:
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


if __name__ == "__main__":
    selector = StockSelector()
    top_sector_stocks = selector.run()
    print(top_sector_stocks)
