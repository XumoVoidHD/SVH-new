# System Flow – How Everything Fits Together

This document explains **how the whole system runs step by step** during a trading day.

## Main pieces

- **`main.py`**
  - `StrategyManager`: Manages all stocks, capital, hedge, and global risk.
  - `Strategy`: Logic for **one stock** (enter, manage, exit).
- **`creds.json`**
  - Configuration file: account size, risk per trade, indicator settings, profit/stop levels, trading hours, hedge rules, etc.
- **`helpers/`**
  - Indicator calculators (MACD, VWAP, ATR, EMA, ADX, etc.).
- **`db/trades_db.py`**
  - Reads/writes the latest status of each stock and the hedge to a database.

You usually **do not edit code**. You change behavior mostly by editing `creds.json`.

---

## Daily timeline (conceptual)

### 1. Startup (before market)

1. The program loads `creds.json` into a config object (`creds`).
2. `StrategyManager` starts:
   - Connects to the broker.
   - Verifies/initializes the database.
   - Sets up:
     - Capital tracking (`available_capital`, `used_capital`, `stock_invested_capital`).
     - Sector capital tracking (`sector_used_capital`).
     - Hedge symbol (e.g. `SPXU`) in the database.
   - Starts background threads:
     - **Hedge monitoring loop** (thread 0).
     - **Global drawdown monitor** (stops all trading if losses exceed limits).

3. For each stock you want to trade, a `Strategy` instance is created:
   - Knows which stock it is trading.
   - Knows the broker, the config, and its sector.

---

### 2. During the day – per-stock loop

Each `Strategy` runs a loop:

1. **Check time windows**
   - Only opens **new** positions inside:
     - Morning window: 10:00–11:15 (US/Eastern).
     - Afternoon window: 13:30–14:30.
   - Outside these windows:
     - It still manages open positions (take profit, stops).
     - It does **not** open new trades.

2. **If inside an entry window**
   - Calls `calculate_indicators()`:
     - Fetches recent intraday data for needed timeframes (e.g. 3-min, 5-min, 20-min).
     - Calculates indicators (VWAP, EMAs, Adjusted MACD, ADX, volume averages).
     - Calculates **Alpha Score** (trend, momentum, volume/volatility, market calm, news placeholder).
     - Runs **additional checks** (volume spike, VWAP slope, TRIN/TICK).
   - If Alpha Score and additional checks both pass:
     - Calls `calculate_position_size()` to decide how many shares to buy.
     - If size > 0:
       - Places a **limit BUY order** anchored to VWAP.
       - On fill, initializes all tracking for the new position.

3. **If there is an active position**
   - Continuously monitors:
     - Current gain/loss %.
     - Profit booking levels (partial exits).
     - Trailing stops (move stop up when gains increase).
     - Trailing exit (exit if price drops a certain amount from the peak after big gains).
     - Time-based exits (weak exit, safety exit, market-on-close).

All important state is pushed to the database so you can see it later and restart safely.

---

### 3. Hedge monitoring (portfolio-level)

In parallel, `StrategyManager` runs `hedge_monitoring_loop()`:

1. Watches **VIX** and **SPY** (S&P 500 proxy) to see if conditions justify adding a hedge (e.g. `SPXU`).
2. Looks at your **total stock invested capital** to size the hedge:
   - Early, Mild, or Severe level based on VIX and S&P drop.
3. Keeps the hedge position in the database (single, centralized hedge).
4. Monitors for **recovery conditions**:
   - VIX down and below a threshold.
   - S&P 500 up over a 15‑minute window.
   - Nasdaq (QQQ) trading above 5‑minute VWAP for several bars.
5. If recovery conditions are met:
   - Scales down hedge: Severe → Mild → Early → Exit completely.

At **15:55 ET**, any active hedge is forcibly closed, regardless of signals.

---

### 4. Global drawdown monitoring

A dedicated loop:

1. Periodically asks the database for total realized + unrealized P&L across stock positions.
2. Computes:
   - A fixed daily limit (e.g. 2% of equity).
   - A dynamic limit based on portfolio ATR (“how volatile the portfolio is”).
3. If total loss exceeds the more conservative of these two:
   - Sets a global `stop_event`.
   - Each `Strategy` sees this and exits positions accordingly.

This prevents one bad day from blowing up the account.

---

### 5. End-of-day clean-up

Towards the end of the session:

- **Weak exit time** – closes “weak” trades with small gains/losses.
- **Safety exit time** – closes any remaining stock positions.
- **Market close** – final safeguard to ensure all positions are flat.
- **Hedge force exit time** (15:55) – guarantees the hedge is flat intraday.

After this, the system is ready to restart cleanly the next day.