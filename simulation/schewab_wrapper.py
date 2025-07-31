import os
import base64
import requests
import json
import threading
import time
import urllib.parse
from datetime import datetime, timedelta

market_data_endpoint = "https://api.schwabapi.com/marketdata/v1"

class SchwabBroker:
    def __init__(self):
        self.token_file = "tokens.json"
        tokens = self.load_tokens()
        self.api_key = tokens.get("SCHWAB_API_KEY")
        self.secret_key = tokens.get("SCHWAB_SECRET_KEY")
        self.refresh_token = tokens.get("SCHWAB_REFRESH_TOKEN")
        self.access_token = tokens.get("SCHWAB_ACCESS_TOKEN")
        self.created_at = tokens.get("SCHWAB_REFRESH_TOKEN_CREATED_ON")
        self.refresh_access_token(self.refresh_token)
        # Start background thread
        self.refresh_thread = threading.Thread(target=self.auto_refresh_token, daemon=True, name="Token Referesher")
        self.refresh_thread.start()

    def load_tokens(self):
        if not os.path.exists(self.token_file):
            raise FileNotFoundError(f"{self.token_file} not found.")
        with open(self.token_file, "r") as f:
            return json.load(f)

    def save_tokens(self, access_token: str):
        with open(self.token_file, "r") as f:
            tokens = json.load(f)

        tokens["SCHWAB_ACCESS_TOKEN"] = access_token
        tokens["SCHWAB_REFRESH_TOKEN_CREATED_ON"] = datetime.now().isoformat()

        with open(self.token_file, "w") as f:
            json.dump(tokens, f, indent=2)

        self.access_token = access_token
        self.created_at = tokens["SCHWAB_REFRESH_TOKEN_CREATED_ON"]
        print("tokens.json updated with new access token and timestamp.")

    def refresh_access_token(self, refresh_token):
        auth_bytes = f"{self.api_key}:{self.secret_key}".encode("utf-8")
        base64_credentials = base64.b64encode(auth_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }

        res = requests.post("https://api.schwabapi.com/v1/oauth/token", headers=headers, data=payload)

        print("\n>>> REFRESH RESPONSE:")
        print(res.status_code, res.text)

        if res.status_code == 200:
            token_data = res.json()
            access_token = token_data.get("access_token")
            self.save_tokens(access_token)
            print("\n>>> NEW ACCESS TOKEN:", access_token)
            return access_token
        else:
            print(">>> Failed to refresh token.")
            return None

    def auto_refresh_token(self):
        while True:
            try:
                if self.created_at:
                    created_dt = datetime.fromisoformat(self.created_at)
                    age = datetime.now() - created_dt
                    if age >= timedelta(minutes=30):
                        print("Auto-refreshing token...")
                        self.refresh_access_token(self.refresh_token)
                else:
                    print("No token creation timestamp found.")

            except Exception as e:
                print("Auto-refresh thread error:", e)

            time.sleep(60)  # Check every 60 seconds
        

    def account_details(self):
        url = "https://api.schwabapi.com/trader/v1/accounts"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        print("\n>>> Making test API call to /trader/v1/accounts ...")
        res = requests.get(url, headers=headers)
        print(">>> API Response:", res.status_code, res.text)

    def get_quote_on_symbol(self, symbol: str, fields: str | None = None) -> requests.Response:
        """
        Get quote for a single symbol

        Args:
            symbol (str): Ticker symbol (e.g., "AAPL")
            fields (str, optional): Data fields to return (e.g., "all", "quote", "fundamental")

        Returns:
            requests.Response: Full response object from Schwab
        """
        encoded_symbol = urllib.parse.quote(symbol, safe="")  # Ensure URL-safe symbol

        endpoint = f"{market_data_endpoint}/{encoded_symbol}/quotes"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        params = {}
        if fields:
            params["fields"] = fields

        response = requests.get(endpoint, headers=headers, params=params)
        return response
    
    def get_bulk_quotes(self, symbols: list[str], fields: str = "quote,reference") -> dict:
        """
        Fetch quotes in bulk from Schwab API

        Args:
            symbols (list): List of stock symbols
            access_token (str): OAuth access token
            fields (str): Optional fields to include (default: quote,reference)

        Returns:
            dict: Symbol-wise quote data
        """

        # Join symbols as comma-separated string (limit is likely 300–500 symbols per request)
        symbol_string = ",".join(symbols)

        params = {
            "symbols": symbol_string,
            "fields": fields
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        response = requests.get(f"{market_data_endpoint}/quotes", headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return {}

    def get_price_history(
        self,
        symbol: str,
        period_type: str = "day",
        period: int = 5,
        frequency_type: str = "minute",
        frequency: int = 1,
        start_date: int = None,
        end_date: int = None,
        need_extended_hours_data: bool = False,
        need_previous_close: bool = False
    ):
        """
        Fetches historical price data from Charles Schwab's /pricehistory endpoint.
        """

        url = f"https://api.schwabapi.com/marketdata/v1/pricehistory"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        params = {
            "symbol": symbol,
            "periodType": period_type,
            "period": period,
            "frequencyType": frequency_type,
            "frequency": frequency,
            "needExtendedHoursData": str(need_extended_hours_data).lower(),
            "needPreviousClose": str(need_previous_close).lower()
        }

        # Optional timestamp params
        if start_date:
            params["startDate"] = int(start_date)
        if end_date:
            params["endDate"] = int(end_date)

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    def get_current_price(self, symbol: str):
        """
        Get current price for a single symbol
        """
        response = self.get_quote_on_symbol(symbol, fields="quote")
        response_data = response.json()
        print(response_data)
        
        # Handle the nested structure where symbol is the key
        if symbol in response_data:
            return response_data[symbol]["quote"]["lastPrice"]
        else:
            # Fallback for direct quote access
            return response_data["quote"]["lastPrice"]

if __name__ == "__main__":
    client = SchwabBroker()
    # res = client.get_quote_on_symbol("AAPL", fields="quote")
    # print(res.status_code)
    # print(json.dumps(res.json(), indent=2))
    symbols = ["AAPL", "MSFT", "AMZN", "BAC", "GOOG", "MRAD", "AAAIX"]
    data = client.get_bulk_quotes(symbols)
    print(data)
