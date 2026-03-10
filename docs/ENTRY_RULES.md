# Entry Rules – When a New Trade Is Opened

This document describes **when and why** the bot decides to open a new long position in a stock.

## 1. Time windows (when entries are allowed)

New trades are only opened during two daily windows (US/Eastern):

- **Morning window:** 10:00–11:15  
- **Afternoon window:** 13:30–14:30

Code reference (concept only):

- `StrategyManager.is_entry_time_window()`: compares the current time to:
  - `TRADING_HOURS.morning_entry_start` / `morning_entry_end`
  - `TRADING_HOURS.afternoon_entry_start` / `afternoon_entry_end`

If we are **outside** these windows:

- No new entries.
- Existing positions are still monitored.

---

## 2. Indicators used for entry

Before making an entry decision, the bot calculates indicators on different timeframes:

- **VWAP** (Volume Weighted Average Price) on a short intraday timeframe (e.g. 3-min bars).
- **EMA5** and **EMA20** on 5‑min and 20‑min bars.
- **Adjusted MACD** on a short timeframe (e.g. 3‑min):
  - MACD histogram divided by price → scale-free momentum measure.
- **ADX** (Average Directional Index) – measures trend strength.
- **Volume Average** – rolling average of volume on the 3‑min timeframe.
- **VIX** (volatility index) – used for “Market Calm” part of the score.

The timeframes and parameters are all defined in `creds.json` under `INDICATORS`.

---

## 3. Alpha Score – “how good is this setup?”

The Alpha Score is a single number (0–100) that summarizes several conditions:

1. **Trend (e.g. 35 points)**
   - Price must be **above VWAP** on the VWAP timeframe.
   - Short EMA > Long EMA (e.g. EMA5 > EMA20).
   - If both hold, the Trend weight is added to the score.

2. **Momentum (e.g. 25 points)**
   - Adjusted MACD must be **above 0**:
     - Means short EMA−long EMA is larger than its signal and scaled by price.
   - If so, the Momentum weight is added.

3. **Volume & Volatility (e.g. 25 points)**
   - The latest 3‑min volume must be greater than a multiple of its average (e.g. 1.5×).
   - ADX must be above a threshold (trend is strong enough).
   - If both are true, the Volume/Volatility weight is added.

4. **News (currently a placeholder)**
   - The code currently marks this as **passed** and adds its weight.
   - Real news integration can be added later.

5. **Market Calm (e.g. 15 points)**
   - Uses VIX data:
     - VIX is below a threshold (market not too fearful).
     - VIX has dropped over the last ~10 minutes.
   - If both are true, this weight is added.

The Alpha Score is then written to the database.

---

## 4. Additional checks – stricter filters

After computing Alpha Score, the system runs **additional checks**:

### 4.1. 3‑min volume spike

- Uses the same 3‑min data and volume average as above.
- Checks if:
  \[
  \text{recent volume} > \text{volume\_multiplier} \times \text{average volume}
  \]
- This ensures there is **fresh interest** and not just a slow drift.

### 4.2. VWAP slope (rising at a minimum pace)

- Looks at VWAP now vs VWAP some minutes ago (based on:
  - The VWAP timeframe (e.g. 3‑min bars).
  - `vwap_slope_period` in minutes.
- Computes:
  \[
  \text{slope in $/min} = \frac{\text{VWAP now} - \text{VWAP earlier}}{\text{minutes between them}}
  \]
- Requires slope > threshold (e.g. 0.005 $/min).

This says: **not only is price above VWAP**, but VWAP itself is **rising fast enough**.

### 4.3. TRIN/TICK breadth (market health)

- **TRIN-NYSE (TRIN):**
  - Measures up-volume vs down-volume on NYSE.
  - Value ≤ threshold (e.g. 1.1) means the market is not extremely bearish.
- **TICK-NYSE (TICK):**
  - Short moving average of TICK must be ≥ threshold (e.g. 0).
  - Indicates net uptick buying pressure.

If both are good, market breadth is supportive.

### 4.4. Bypass condition

There is a **bypass Alpha level** (e.g. 88):

- If Alpha Score is **above** this value, the system **may skip** TRIN/TICK.
- If below, it requires TRIN and TICK conditions to pass.

All results (volumes, slopes, TRIN/TICK, pass/fail flags) are logged and saved.

---

## 5. Final decision to enter

A new position is only entered when:

1. **We are in an entry time window** (morning or afternoon slot).
2. **Alpha Score ≥ threshold** (from `RISK_CONFIG.alpha_score_threshold` in `creds.json`).
3. **All required additional checks pass**:
   - Volume spike,
   - VWAP slope,
   - TRIN/TICK (unless bypassed).

If all three are true:

- The system calculates a risk-based, sector-limited position size.
- It then places a **limit BUY** order **anchored to VWAP**.
- On a successful fill, the position becomes **active** and is handed over to the monitoring logic described in `PROFIT_AND_STOPS.md`.