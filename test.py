"""Test TRIN-NYSE historical data fetch using ib_insync (same as main.py additional checks)."""
from ib_insync import IB, Index, util

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=199)
print("Connected. Requesting TRIN-NYSE 1 min, 1 D...")

# TRIN-NYSE: full symbol, exchange NYSE, secType IND (per IBKR)
contract = Index("TICK-NYSE", "NYSE", "USD")
bars = ib.reqHistoricalData(
    contract,
    endDateTime="",
    durationStr="1 D",
    barSizeSetting="1 min",
    whatToShow="TRADES",
    useRTH=False,
    formatDate=1,
    timeout=30,
)

if bars:
    df = util.df(bars)
    if df is not None and not df.empty:
        print(f"TRIN-NYSE: got {len(df)} bars")
        print(df.tail(10))
        if "close" in df.columns:
            print(f"Latest TRIN close: {df['close'].iloc[-1]}")
    else:
        print("TRIN-NYSE: no bars in response")
else:
    print("TRIN-NYSE: no data returned (timeout or symbol not available)")

ib.disconnect()
