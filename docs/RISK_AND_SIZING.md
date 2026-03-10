# Risk Management & Position Sizing

This document explains **how much** capital is put into each trade and how sector limits are enforced.

## 1. Key concepts

- **Total account equity (`EQUITY`)** – set in `creds.json`.
- **Risk per trade (`risk_per_trade`)** – fraction of equity you are willing to risk on a single trade (e.g. 0.004 = 0.4%).
- **Max position size (`max_position_equity_pct`)** – maximum fraction of equity any single position can hold (e.g. 0.10 = 10%).
- **Max sector weight (`max_sector_weight`)** – maximum fraction of equity that can be invested in a single sector (e.g. 0.30 = 30%).
- **Sector** – each stock belongs to a sector; the system tracks capital used per sector.

---

## 2. Sector capital caps

The system calculates:

- **Sector cap**:
  \[
  \text{sector\_cap} = \text{EQUITY} \times \text{max\_sector\_weight}
  \]

- **Sector used**:
  - Sum of capital invested in all **active positions** in that sector (from the database).
  - Stored in `sector_used_capital[sector]`.

- **Available for sector**:
  \[
  available\_for\_sector = \max(0, sector\_cap - sector\_used)
  \]

During position sizing, the system **never** allocates more than `available_for_sector` to a new trade in that sector.

---

## 3. Stop-loss percentage (volatility-aware)

The stop-loss distance (in %) is based on **ATR (Average True Range)** and volatility:

1. Fetch intraday data using the same bar size as the VWAP timeframe (e.g. 3‑min).
2. Compute **ATR** over a number of bars (e.g. `atr_period = 10`).
3. Compute **ATR%**:
   \[
   atr\_pct = \frac{ATR}{\text{current price}}
   \]

4. Decide if the stock is “volatile”:
   - If `atr_pct ≥ atr_volatility_threshold` (e.g. 2%), it is volatile.

5. Choose a **base stop**:
   - If volatile → `volatile_stop_loss` (e.g. 2%).
   - If not volatile → `default_stop_loss` (e.g. 1.5%).

6. Compute an ATR-based stop:
   \[
   atr\_stop = \frac{ATR \times atr\_multiplier}{\text{current price}}
   \]

7. Final stop %:
   - At least the (volatility-adjusted) base stop.
   - At most `max_stop_loss` (e.g. 4%).

So the stop-loss as a % of entry price is bounded within a sensible range (e.g. 1.5–4%), and scales with actual price volatility.

---

## 4. Position sizing formula

Given:

- Equity `E`
- Risk per trade `r` (fraction of equity)
- Stop-loss % `SL` (fraction)

The **risk-based capital** for a trade is:

\[
capital\_for\_trade = \frac{E \times r}{SL}
\]

Then, two important caps are applied:

1. **Sector cap**:
   - Do not exceed `available_for_sector` for that stock’s sector.
2. **Max position cap**:
   - Do not exceed `E × max_position_equity_pct`.

So:

1. Start with `capital_for_trade` from the risk formula.
2. Limit it by:
   \[
   capital\_for\_trade = \min(capital\_for\_trade, available\_for\_sector, E \times max\_position\_equity\_pct)
   \]
3. Convert to **shares**:
   \[
   \text{shares} = \left\lfloor \frac{capital\_for\_trade}{\text{current price}} \right\rfloor
   \]

If the result is ≤ 0 (e.g. sector cap is fully used), **no new position is opened**.

---

## 5. Capital tracking

The system tracks three main values:

- `available_capital` – how much account equity is still free.
- `used_capital` – how much is currently locked in active stock positions.
- `stock_invested_capital` – total dollar amount invested in stocks.

When a trade is entered:

- `used_capital` and `stock_invested_capital` increase.
- `available_capital` decreases.
- `sector_used_capital[sector]` increases by the cost of the new position.

When a position is closed (fully or partially):

- `available_capital` increases by the sale proceeds.
- `used_capital`, `stock_invested_capital`, and `sector_used_capital[sector]` are reduced accordingly.

This ensures **position sizing and hedging** always reflect the latest state of the portfolio.