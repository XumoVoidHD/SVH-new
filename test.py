"""Simple test to verify TRIN-NYSE and TICK-NYSE historical data using StrategyBroker/IBBroker (ibapi only)."""

from main import StrategyBroker  # uses simulation.ib_broker.IBBroker under the hood


def print_index_summary(name, df):
    if df is None:
        print(f"{name}: NO DATA (None returned)")
        return
    if df.empty:
        print(f"{name}: NO ROWS (empty DataFrame)")
        return
    print(f"{name}: got {len(df)} bars")
    print(df.tail(5))
    if "close" in df.columns:
        print(f"{name}: latest close = {df['close'].iloc[-1]}")


def main():
    broker = StrategyBroker(host="127.0.0.1", port=7497, client_id=199)
    print("Connected via StrategyBroker. Requesting TRIN-NYSE and TICK-NYSE (1 D, 1 min)...")

    # Use specialized TRIN/TICK helper to mirror ib_insync Index(symbol='TICK-NYSE', exchange='NYSE')
    trin_df = broker.get_trin_tick_data("TRIN-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")
    tick_df = broker.get_trin_tick_data("TICK-NYSE", exchange="NYSE", duration="1 D", bar_size="1 min")

    print_index_summary("TRIN-NYSE", trin_df)
    print_index_summary("TICK-NYSE", tick_df)

    broker.disconnect()


if __name__ == "__main__":
    main()
