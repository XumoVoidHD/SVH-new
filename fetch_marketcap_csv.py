import os
import shutil
import requests

# Path to Downloads folder
def fetch_marketcap_csv():
    url = "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/?download=csv"

    # Step 1: Download the file to a temp location (Downloads-like path)
    downloaded_file = "temp_marketcap.csv"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(downloaded_file, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print("File downloaded successfully.")
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

    # Step 2: Move & rename to current directory
    current_dir = os.getcwd()
    final_path = os.path.join(current_dir, "companies_by_marketcap.csv")

    shutil.move(downloaded_file, final_path)

    print(f"File moved to: {final_path}")
