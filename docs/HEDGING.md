# Hedging Logic – Portfolio Protection with Inverse ETFs

This document explains how the system uses an inverse ETF (like `SPXU`) to hedge the portfolio.

## 1. Goal of the hedge

The hedge is a **separate central position** (not per stock) used to:

- Reduce portfolio exposure when the market looks dangerous.
- Scale down (and eventually remove) hedge exposure when the market recovers.
- Always close hedges **before the end of the day** (intraday hedge only).

The main parameters live in `HEDGE_CONFIG` in `creds.json`.

---

## 2. Hedge instruments

- **Default hedge symbol**: `hedge_symbol` (currently `SPXU`).
- **Available options**: `hedge_options` (e.g. `["SQQQ", "SPXU"]`).

The system always uses whichever symbol is set as `hedge_symbol`.

---

## 3. Hedge levels and sizes

There are **three hedge levels**, each with:

- A target **beta offset** (`beta`): 0.1, 0.15, 0.3.
- A hedge **size as a percent of stock invested capital** (`equity_pct`):
  - Early: 0.033 → 3.3%
  - Mild: 0.05 → 5%
  - Severe: 0.10 → 10%

**Important:**  
The size is **NOT** a percent of total account equity directly. It is a percent of **the capital currently invested in stocks** (`stock_invested_capital`). For example:

- If equity is $100k but only $50k is currently invested in stocks, **Severe** hedge is:
  - `hedge_amount = 50k × 10% = $5k` in SPXU.

This makes the hedge scale with how much of your equity is actually deployed.

---

## 4. Hedge entry – when a hedge is opened

The function `check_hedge_triggers()` runs periodically:

1. **VIX data**
   - Uses VIX index with timeframe from `HEDGE_CONFIG.triggers.vix_timeframe` (e.g. 3‑min).
   - Reads current VIX level.

2. **S&P 500 drop over 15 minutes**
   - Uses SPY 15‑minute bars as a proxy for the S&P 500.
   - Compares the last close to the previous one to find **15‑min % drop**.

3. **Thresholds per level**

From `HEDGE_CONFIG.hedge_levels`:

- **Severe hedge**:
  - `VIX ≥ severe.vix_threshold` (e.g. 25)
  - `15‑min S&P drop ≥ severe.spx_drop_threshold` (e.g. 2.0%)
- **Mild hedge**:
  - `VIX ≥ mild.vix_threshold` (e.g. 22)
  - `15‑min S&P drop ≥ mild.spx_drop_threshold` (e.g. 1.2%)
- **Early hedge**:
  - `VIX ≥ early.vix_threshold` (e.g. 20)
  - `15‑min S&P drop ≥ early.spx_drop_threshold` (e.g. 0.75%)

The system checks **severe first**, then mild, then early.  
If a level’s conditions are met, it returns that level and its parameters.

There is **no** Polymarket odds integration yet.

---

## 5. Executing the hedge

When `execute_hedge(level, beta, equity_pct)` is called:

1. It ensures no hedge is already active.
2. It reads **stock invested capital** (`stock_invested_capital`).
3. It computes the **hedge amount**:
   \[
   hedge\_amount = stock\_invested\_capital \times equity\_pct
   \]
4. It gets the current hedge ETF price and calculates:
   \[
   hedge\_shares = \left\lfloor \frac{hedge\_amount}{hedge\_price} \right\rfloor
   \]
5. It sends a **MARKET BUY** order for `hedge_shares` in the hedge symbol.
6. It updates:
   - The hedge position in the database (shares, entry price, used capital for hedge only, etc.).
   - `available_capital` is reduced by the hedge amount.
   - `used_capital` for **stocks** is **not** changed (hedge is tracked separately).

Once done:

- The portfolio has some **negative beta** exposure (through SPXU or SQQQ) to offset losses if the market keeps dropping.

---

## 6. Hedge exit conditions – when to reduce hedge

Function: `check_hedge_exit_conditions()`:

It considers **three recovery signals**, all of which must be true:

1. **VIX < threshold and falling**
   - Uses VIX data again (same exit timeframe).
   - Condition:
     - Current VIX < `vix_exit_threshold` (e.g. 20).
     - Current VIX < VIX ~10 minutes ago.
   - Means “fear is lower and decreasing”.

2. **S&P 500 up > X% in 15 minutes**
   - Uses SPY 15‑minute bars again.
   - Condition:
     - Last 15‑min gain > `sp500_recovery_threshold` (e.g. 0.6%).

3. **Nasdaq (QQQ) above 5‑min VWAP for consecutive bars**
   - Fetches QQQ 5‑minute data.
   - Calculates 5‑min VWAP.
   - Checks:
     - Last two bars have close > VWAP at each bar.
   - Means “growth/tech is recovering strongly”.

There is **no** Polymarket retrace condition in code yet.

If all three are met, the function returns `True` (recovery signals present).

---

## 7. Scaling down the hedge

Function: `scale_down_hedge()`:

1. If no hedge is active, it returns.
2. It calls `check_hedge_exit_conditions()`; if signals are not all met, it returns.
3. It reads current hedge data from the database:
   - Current hedge level (`early`, `mild`, or `severe`).
   - Current shares and used capital.
4. Based on the level:
   - **Severe → Mild**
   - **Mild → Early**
   - **Early → Exit fully (close all hedge shares)**

If scaling is needed (Severe or Mild):

1. It computes new target hedge amount:
   \[
   new\_hedge\_amount = stock\_invested\_capital \times new\_equity\_pct
   \]
2. Calculates new target shares and the difference vs current.
3. Closes just enough shares to move from the old to the new level.
4. Updates hedge position and available capital accordingly.

If the level is **Early**, it calls `close_hedge()` and fully exits the hedge.

---

## 8. Forced intraday hedge exit

In `hedge_monitoring_loop()`:

- The system reads a `hedge_force_exit_time` from `TRADING_HOURS` (e.g. 15:55).
- During trading hours, if the clock passes that time:
  - If a hedge is active, it is **fully closed** by a market order.
  - The hedge monitoring loop then ends for the day.

This guarantees **no hedge positions remain overnight**, matching the intraday hedge design.