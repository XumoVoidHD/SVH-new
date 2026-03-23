"""Tests for historical-data + indicator logic.

Currently includes:
1) TRIN-NYSE / TICK-NYSE fetch sanity check
2) Volume "recent vs rolling average" sanity check to debug volume_spike logic
"""

import argparse
import os
import re
import time
from datetime import datetime

import pytz
import pandas as pd

from main import StrategyBroker  # uses simulation.ib_broker.IBBroker under the hood
from main import load_config


def print_index_summary(name, df):
    if df is None:
        print(f"{name}: NO DATA (None returned)")
        return
    if getattr(df, "empty", True):
        print(f"{name}: NO ROWS (empty DataFrame)")
        return
    print(f"{name}: got {len(df)} bars")
    print(df.tail(5))
    if "close" in df.columns:
        print(f"{name}: latest close = {df['close'].iloc[-1]}")


def _parse_tf_to_minutes(tf_value: str) -> int:
    """Parse strings like '3 mins'/'3 min'/'5 mins' into integer minutes."""
    if tf_value is None:
        return 0
    s = str(tf_value).strip().lower()
    # Extract leading integer (e.g., '3 mins' -> 3)
    m = re.search(r"(\d+)", s)
    if not m:
        return 0
    return int(m.group(1))

def _parse_hhmm(hhmm: str):
    if hhmm is None:
        raise ValueError("hhmm cannot be None")
    return datetime.strptime(hhmm, "%H:%M").time()

def _parse_yyyy_mm_dd(date_str: str, tz) -> datetime.date:
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    # default: today's date in the configured timezone
    return datetime.now(tz).date()


def test_volume_recent_vs_avg(symbol: str, host: str, port: int, client_id: int):
    creds = load_config("creds.json")

    # Mirror the code path:
    # - fetch timeframe data for INDICATORS.volume_avg.timeframes
    # - volume_avg = data['volume'].rolling(window=params.window).mean()
    volume_tf = creds.INDICATORS.volume_avg.timeframes[0]  # e.g. "3 mins"
    window = int(getattr(creds.INDICATORS.volume_avg.params, "window", 20))
    volume_multiplier = float(getattr(creds.ADDITIONAL_CHECKS_CONFIG, "volume_multiplier", 1.0))

    bar_size_minutes = _parse_tf_to_minutes(volume_tf)
    if bar_size_minutes <= 0:
        raise ValueError(f"Could not parse volume_avg timeframe minutes from: {volume_tf}")

    broker = StrategyBroker(host=host, port=port, client_id=client_id)
    try:
        print(f"Connected. Fetching {symbol} historical data for volume_avg debug...")
        print(f"  volume_avg timeframe: {volume_tf} (bar_size={bar_size_minutes} mins)")
        print(f"  rolling window: {window} bars")
        print(f"  volume_multiplier (required factor): {volume_multiplier}")

        # Mirror main.py: fetch_data_by_timeframe uses get_historical_data_with_retry(stock=self.stock, bar_size=period)
        # where duration defaults to "3 D" and bar_size is the minute count (int) which StrategyBroker converts.
        df = broker.get_historical_data_stock(
            symbol,
            duration="3 D",
            bar_size=bar_size_minutes,  # StrategyBroker converts int -> "<n> mins"
        )

        if df is None or df.empty:
            print("No candle data returned; aborting volume debug.")
            return
        if "volume" not in df.columns:
            print(f"Returned data missing 'volume' column. Columns: {list(df.columns)}")
            return

        # Ensure deterministic indexing (main.py uses iloc[-1])
        df = df.copy()
        recent_volume = float(df["volume"].iloc[-1])

        rolling_mean = df["volume"].rolling(window=window).mean()
        avg_volume = rolling_mean.iloc[-1]

        print(f"Fetched {len(df)} candles. Latest candle timestamp (index): {df.index[-1] if len(df.index) else 'N/A'}")
        print(f"Recent volume (last bar): {recent_volume:,.0f}")

        if pd.isna(avg_volume):
            print(f"Average volume (rolling mean over last {window} bars): NaN (not enough bars yet)")
            return

        avg_volume = float(avg_volume)
        required_volume = volume_multiplier * avg_volume
        volume_check = recent_volume > required_volume

        print(f"Average volume (rolling mean): {avg_volume:,.0f}")
        print(f"Required volume: recent > {volume_multiplier} * {avg_volume:,.0f} = {required_volume:,.0f}")
        print(f"Volume check result: {volume_check}")

        # Helpful context: show the last few volumes and the rolling mean
        tail_n = min(10, len(df))
        context = pd.DataFrame(
            {
                "volume": df["volume"].tail(tail_n).astype(float).values,
                "vol_avg": rolling_mean.tail(tail_n).astype(float).values,
            },
            index=df.tail(tail_n).index,
        )
        print("\nLast few bars (volume + rolling avg):")
        print(context)
    finally:
        broker.disconnect()

def _compute_volume_recent_vs_avg_once(broker: StrategyBroker, creds, symbol: str):
    """Compute recent vs rolling average volume the same way main.py does it."""
    volume_tf = creds.INDICATORS.volume_avg.timeframes[0]  # e.g. "3 mins"
    window = int(getattr(creds.INDICATORS.volume_avg.params, "window", 20))
    volume_multiplier = float(getattr(creds.ADDITIONAL_CHECKS_CONFIG, "volume_multiplier", 1.0))

    bar_size_minutes = _parse_tf_to_minutes(volume_tf)
    if bar_size_minutes <= 0:
        raise ValueError(f"Could not parse volume_avg timeframe minutes from: {volume_tf}")

    df = broker.get_historical_data_stock(
        symbol,
        duration="3 D",
        bar_size=bar_size_minutes,
    )
    if df is None or df.empty:
        return None
    if "volume" not in df.columns:
        raise ValueError(f"Returned data missing 'volume' column. Columns: {list(df.columns)}")

    df = df.copy()
    recent_idx = -2 if len(df) >= 2 else -1
    recent_volume = float(df["volume"].iloc[recent_idx])

    rolling_mean = df["volume"].rolling(window=window).mean()
    avg_volume = rolling_mean.iloc[recent_idx]
    if pd.isna(avg_volume):
        return {
            "recent_volume": recent_volume,
            "avg_volume": None,
            "required_volume": None,
            "volume_check": None,
            "window": window,
            "volume_tf": volume_tf,
            "bar_size_minutes": bar_size_minutes,
            "volume_multiplier": volume_multiplier,
        }

    avg_volume = float(avg_volume)
    required_volume = volume_multiplier * avg_volume
    volume_check = bool(recent_volume > required_volume)

    return {
        "recent_volume": recent_volume,
        "avg_volume": avg_volume,
        "required_volume": required_volume,
        "volume_check": volume_check,
        "window": window,
        "volume_tf": volume_tf,
        "bar_size_minutes": bar_size_minutes,
        "volume_multiplier": volume_multiplier,
    }


def simulate_volume_recent_vs_avg_until(
    symbol: str,
    host: str,
    port: int,
    client_id: int,
    until_hhmm: str,
    interval_seconds: int = 60,
):
    """
    Repeatedly fetch and print recent vs rolling average volume until a given HH:MM time.
    Intended for quick "simulate what bot would see over time" debugging.
    """
    creds = load_config("creds.json")
    tz_name = getattr(getattr(creds, "TRADING_HOURS", None), "timezone", None) or "US/Eastern"
    tz = pytz.timezone(tz_name)
    until_time = datetime.strptime(until_hhmm, "%H:%M").time()

    broker = StrategyBroker(host=host, port=port, client_id=client_id)
    try:
        print(f"Connected. Simulating volume checks for {symbol} until {until_hhmm} ({tz_name}).")
        print(f"Fetch interval: {interval_seconds}s")

        while True:
            now = datetime.now(tz)
            if now.time() > until_time:
                print(f"Reached target time {until_hhmm}. Stopping simulation.")
                break

            result = _compute_volume_recent_vs_avg_once(broker, creds, symbol)
            if result is None:
                print(f"[{now.strftime('%H:%M:%S')}] {symbol}: No data returned.")
            else:
                recent = result["recent_volume"]
                avg = result["avg_volume"]
                required = result["required_volume"]
                ok = result["volume_check"]
                vol_tf = result["volume_tf"]
                window = result["window"]
                mult = result["volume_multiplier"]

                if avg is None:
                    print(
                        f"[{now.strftime('%H:%M:%S')}] {symbol} recent={recent:,.0f}, "
                        f"avg=NaN (need {window} bars for rolling mean @ {vol_tf})"
                    )
                else:
                    print(
                        f"[{now.strftime('%H:%M:%S')}] {symbol} recent={recent:,.0f} vs "
                        f"avg={avg:,.0f} (required>{mult}x={required:,.0f}) -> {'PASSED' if ok else 'FAILED'}"
                    )

            time.sleep(max(1, int(interval_seconds)))
    finally:
        broker.disconnect()

def print_volume_recent_vs_avg_range(
    symbol: str,
    host: str,
    port: int,
    client_id: int,
    date_str: str,
    start_hhmm: str,
    end_hhmm: str,
):
    """
    Fetch candles once per configured volume_avg timeframe, then compute:
      - recent volume = volume at each bar in [start, end]
      - average volume = rolling mean(window) of volume at each bar in [start, end]
    Prints values for every bar unit of the timeframe (e.g. every 3-min candle).
    """
    creds = load_config("creds.json")
    tz_name = getattr(getattr(creds, "TRADING_HOURS", None), "timezone", None) or "US/Eastern"
    tz = pytz.timezone(tz_name)

    sim_date = _parse_yyyy_mm_dd(date_str, tz)
    start_t = _parse_hhmm(start_hhmm)
    end_t = _parse_hhmm(end_hhmm)

    volume_tfs = getattr(creds.INDICATORS.volume_avg, "timeframes", [])
    window = int(getattr(creds.INDICATORS.volume_avg.params, "window", 20))
    volume_multiplier = float(getattr(creds.ADDITIONAL_CHECKS_CONFIG, "volume_multiplier", 1.0))

    if not volume_tfs:
        raise ValueError("creds.json: INDICATORS.volume_avg.timeframes is empty")

    broker = StrategyBroker(host=host, port=port, client_id=client_id)
    try:
        print(
            f"Connected. Printing recent vs avg volume for {symbol} "
            f"on {sim_date.isoformat()} from {start_hhmm} to {end_hhmm} ({tz_name})"
        )
        print(f"Rolling window: {window} bars | volume_multiplier(required factor): {volume_multiplier}")

        start_dt = tz.localize(datetime.combine(sim_date, start_t))
        end_dt = tz.localize(datetime.combine(sim_date, end_t))

        for volume_tf in volume_tfs:
            bar_size_minutes = _parse_tf_to_minutes(volume_tf)
            if bar_size_minutes <= 0:
                print(f"\n{symbol} {volume_tf}: could not parse bar size; skipping")
                continue

            print(f"\n--- volume_avg timeframe: {volume_tf} (bar_size={bar_size_minutes} mins) ---")
            df = broker.get_historical_data_stock(
                symbol,
                duration="3 D",
                bar_size=bar_size_minutes,
            )

            if df is None or getattr(df, "empty", True):
                print("No candle data returned.")
                continue
            if "volume" not in df.columns:
                print(f"Returned data missing 'volume' column. Columns: {list(df.columns)}")
                continue

            # Broker returns timestamps in a `date` column (not the df index).
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date").sort_index()

            # Ensure timezone-aware index so filtering uses correct window.
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz)
            else:
                df.index = df.index.tz_convert(tz)

            # Rolling mean computed on the full series (same as main.py behavior)
            rolling_mean = df["volume"].rolling(window=window).mean()

            mask = (df.index >= start_dt) & (df.index <= end_dt)
            sub = df.loc[mask]
            sub_avg = rolling_mean.loc[mask]

            print(f"Bars in window: {len(sub)} (of {len(df)} fetched)")
            if sub.empty:
                continue

            # Avoid last candle when it's still forming: drop the last row in the simulated window.
            if len(sub) >= 2:
                sub = sub.iloc[:-1]
                sub_avg = sub_avg.loc[sub.index]

            # Print each candle's values
            for ts, row in sub.iterrows():
                recent_volume = float(row["volume"])
                avg_volume = sub_avg.loc[ts]
                if pd.isna(avg_volume):
                    print(f"{ts.strftime('%H:%M:%S')} recent={recent_volume:,.0f} avg=NaN (insufficient history for rolling window)")
                    continue
                avg_volume_f = float(avg_volume)
                required_volume = volume_multiplier * avg_volume_f
                volume_check = recent_volume > required_volume

                status = "PASSED" if volume_check else "FAILED"
                print(
                    f"{ts.strftime('%H:%M:%S')} recent={recent_volume:,.0f} avg={avg_volume_f:,.0f} "
                    f"required>{volume_multiplier}x={required_volume:,.0f} -> {status}"
                )
    finally:
        broker.disconnect()


def test_trin_tick(host: str, port: int, client_id: int):
    broker = StrategyBroker(host=host, port=port, client_id=client_id)
    try:
        print("Connected via StrategyBroker. Requesting TRIN-NYSE and TICK-NYSE (1 D, 1 min)...")
        trin_df = broker.get_trin_tick_data("TRIN-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")
        tick_df = broker.get_trin_tick_data("TICK-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")
        print_index_summary("TRIN-NYSE", trin_df)
        print_index_summary("TICK-NYSE", tick_df)
    finally:
        broker.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trin_tick", "volume"], default="volume")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client_id", type=int, default=199)
    parser.add_argument("--until", default=None, help="If set, runs simulation until HH:MM (e.g. 11:25). Uses current clock; not historical backtest.")
    parser.add_argument("--start", default=None, help="Historical window start time HH:MM (e.g. 11:00).")
    parser.add_argument("--end", default=None, help="Historical window end time HH:MM (e.g. 11:30).")
    parser.add_argument("--date", default=None, help="Historical date YYYY-MM-DD (e.g. 2026-03-20). Defaults to today's date in creds timezone.")
    parser.add_argument("--interval", type=int, default=60, help="Simulation interval in seconds.")
    args = parser.parse_args()

    if args.mode == "trin_tick":
        test_trin_tick(args.host, args.port, args.client_id)
    else:
        if args.start and args.end:
            print_volume_recent_vs_avg_range(
                symbol=args.symbol,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                date_str=args.date,
                start_hhmm=args.start,
                end_hhmm=args.end,
            )
        elif args.until:
            simulate_volume_recent_vs_avg_until(
                symbol=args.symbol,
                host=args.host,
                port=args.port,
                client_id=args.client_id,
                until_hhmm=args.until,
                interval_seconds=args.interval,
            )
        else:
            test_volume_recent_vs_avg(args.symbol, args.host, args.port, args.client_id)


if __name__ == "__main__":
    main()
