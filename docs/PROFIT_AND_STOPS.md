# Profit Taking & Stop-Loss Logic

This document explains how the bot locks in profits and cuts losses, in a way a non-coder can follow.

## 1. Initial stop-loss (“Start Stop”)

When a position is opened:

1. The system calculates **ATR%** and decides if the stock is **volatile**.
2. It chooses:
   - A **default base stop** for calm stocks.
   - A **higher base stop** for volatile stocks.
3. It computes an ATR-based stop and then ensures:
   - Stop distance is at least the base stop.
   - And at most a configured maximum (e.g. 4%).

This gives an initial stop-loss price:

\[
stop\_loss\_price = entry\_price \times (1 - stop\_loss\_pct)
\]

---

## 2. Profit booking (taking partial profits)

Your profit plan:

- **40%** of shares at **+1%** gain.
- **30%** of shares at **+3%** gain.
- **30%** of shares at **+5%** gain.

This is implemented using `PROFIT_CONFIG.profit_booking_levels` in `creds.json`.

During position monitoring:

1. The current gain % is:
   \[
   gain\_pct = \frac{\text{current price} - \text{entry price}}{\text{entry price}}
   \]

2. For each **profit booking level**:
   - If `gain_pct` is greater than or equal to that level’s gain threshold (1%, 3%, or 5%):
     - The system sells that fraction (`exit_pct`) of the **current** shares.
     - Marks that level as “used” so it triggers only once.
     - Logs the partial sale and updates the database.

Result:

- At around 1% gain, you lock in profit on 40% of the position.
- At around 3%, you lock in another 30%.
- At around 5%, you lock in the final 30% (if any shares remain).

---

## 3. Trailing stops (moving the stop up as price rises)

There is a separate set of “trailing stop levels” defined in `STOP_LOSS_CONFIG.trailing_stop_levels`.

Example configuration:

- At **+2% gain** → move stop to **+1.0%** above entry.
- At **+5% gain** → move stop to **+2.2%** above entry.

When current gain % reaches one of these levels:

1. The system computes a new stop price:
   \[
   new\_stop\_price = entry\_price \times (1 + new\_stop\_pct)
   \]
2. Updates `stop_loss_price` to this higher value.
3. Marks that trailing level as used.

Effect:

- As the trade moves further into profit, your **worst-case loss** becomes a **locked-in gain**.
- E.g., after the second trailing level, your stop might be 2.2% above entry: if hit, you still exit with a gain.

---

## 4. “3-minute drop” trailing exit (after big gains)

To protect against sharp reversals after a strong gain:

1. When gain reaches a configured threshold (e.g. **+5%**), the system:
   - Starts a **monitoring window** (e.g. 3 minutes).
   - Records the **peak price** seen during this window.

2. While monitoring:
   - It updates the peak price whenever a new high is reached.
   - Computes the price drop from that **peak**:
     \[
     price\_drop\_pct = \frac{\text{peak price} - \text{current price}}{\text{peak price}}
     \]

3. If the drop exceeds a threshold (e.g. **0.5%**) **within the 3-minute window**:
   - The system **exits all remaining shares** immediately (trailing exit).

If the 3-minute window ends without a large enough drop:

- The monitoring stops, and the position continues under normal trailing stops and profit booking.

---

## 5. Time-based exits

In addition to price-based rules, there are **time-of-day** exits:

- **Weak exit time** (e.g. 15:30 ET):
  - Closes “weak” positions that are in a small gain/loss range.
- **Safety exit time** (e.g. 15:55 ET):
  - Closes any remaining positions as a safety check.
- **Market-on-close exit** (e.g. 16:00 ET):
  - Ensures no positions remain open past the official market close.

These rules ensure you are **flattened by end-of-day**, even if stops and profit targets did not trigger.

---

## 6. Global drawdown protection

Separately from individual stops:

- A **global drawdown monitor** watches your total daily P&L.
- It compares losses against:
  - A fixed limit (e.g. 2% of equity).
  - A dynamic limit based on the portfolio ATR (how volatile the portfolio is).
- If losses exceed the more conservative of these two:
  - Sets a global stop flag.
  - All strategies exit their positions and stop opening new ones for the day.

This is the “circuit breaker” for the whole system.