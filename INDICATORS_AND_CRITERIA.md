# Indicators and Criteria for Entry (Alpha Score + Additional Checks)

From `main.py`: `calculate_indicators` → `calculate_alpha_score` and `perform_additional_checks`.  
This doc lists what is calculated, what each criterion uses, and at what level it passes.

---

## Entry condition (both must be true)

1. **Alpha score** ≥ `alpha_score_threshold` (from config).
2. **Additional checks passed** = True (volume check AND VWAP slope AND TRIN/TICK all pass, unless TRIN/TICK is bypassed by score).

---

## Part 1: Alpha score (`calculate_indicators` → `calculate_alpha_score`)

Indicators are computed per timeframe; then each block below adds to the score if its criteria pass. Weights come from `alpha_score_config` (e.g. trend 30%, momentum 20%, volume_volatility 20%, news 15%, market_calm 15%).

### 1. Trend (e.g. 30% weight)


| Indicator | Source                  | Criteria that use it           | Pass condition |
| --------- | ----------------------- | ------------------------------ | -------------- |
| **close** | Last close price (data) | price_above_vwap, trend_passed | close > vwap   |
| **vwap**  | VWAP indicator          | price_above_vwap, trend_passed | (see above)    |
| **ema1**  | EMA1 indicator          | ema_cross, trend_passed        | ema1 > ema2    |
| **ema2**  | EMA2 indicator          | ema_cross, trend_passed        | (see above)    |


- **price_above_vwap**: True when **close > vwap**.
- **ema_cross**: True when **ema1 > ema2**.
- **trend_passed**: True only when **both** price_above_vwap and ema_cross are true.

### 2. Momentum (e.g. 20% weight)


| Indicator | Source                      | Criteria that use it | Pass condition |
| --------- | --------------------------- | -------------------- | -------------- |
| **macd**  | MACD indicator (last value) | momentum_passed      | macd > 0       |


- **momentum_passed**: True when **macd > 0**.

### 3. Volume / volatility (e.g. 20% weight)


| Indicator      | Source                   | Criteria that use it                          | Pass condition                     |
| -------------- | ------------------------ | --------------------------------------------- | ---------------------------------- |
| **volume**     | Last bar volume (data)   | volume_spike, volume_volatility_passed        | volume > (multiplier × volume_avg) |
| **volume_avg** | Volume average indicator | volume_spike, volume_volatility_passed        | (threshold for volume)             |
| **adx**        | ADX indicator            | adx_above_threshold, volume_volatility_passed | adx > adx_threshold                |


- **volume_spike**: True when **recent volume > multiplier × volume_avg** (multiplier from `alpha_score_config.volume_volatility.conditions.volume_spike.multiplier`).
- **adx_above_threshold**: True when **adx > adx_threshold** (from `alpha_score_config.volume_volatility.conditions.adx_threshold.threshold`).
- **volume_volatility_passed**: True only when **both** volume_spike and adx_above_threshold are true.

### 4. News (e.g. 15% weight)

- No real indicator; score is always added and **news_passed** is always set to **True** (placeholder).

### 5. Market calm (e.g. 15% weight)


| Indicator | Source                   | Criteria that use it                      | Pass condition                       |
| --------- | ------------------------ | ----------------------------------------- | ------------------------------------ |
| **vix**   | VIX index close (broker) | vix_low, vix_dropping, market_calm_passed | vix < vix_threshold AND vix dropping |


- **vix_low**: True when **vix < vix_threshold** (from `alpha_score_config.market_calm.conditions.vix_threshold.threshold`).
- **vix_dropping**: True when **current VIX close < VIX close 4 bars ago** (needs at least 4 bars).
- **market_calm_passed**: True only when **both** vix_low and vix_dropping are true.

---

## Part 2: Additional checks (`perform_additional_checks`)

Only run if alpha score ≥ `alpha_score_threshold`. **additional_checks_passed** = True only if **all three** below pass (or TRIN/TICK is bypassed when score ≥ bypass_alpha).

### 1. Volume check (additional)


| Indicator           | Source                                         | Criteria that use it | Pass condition                                      |
| ------------------- | ---------------------------------------------- | -------------------- | --------------------------------------------------- |
| **volume_recent**   | Last bar volume (same timeframe as volume_avg) | volume_check_passed  | volume_recent > volume_multiplier × volume_avg_addl |
| **volume_avg_addl** | Volume average indicator (same timeframe)      | volume_check_passed  | (threshold for volume_recent)                       |


- **volume_check_passed**: True when **volume_recent > volume_multiplier × volume_avg_addl** (`additional_checks_config.volume_multiplier`, e.g. 2 → “2× volume” check).

### 2. VWAP slope


| Indicator      | Source                                               | Criteria that use it | Pass condition                    |
| -------------- | ---------------------------------------------------- | -------------------- | --------------------------------- |
| **vwap_slope** | (VWAP now − VWAP one period ago) / vwap_slope_period | vwap_slope_passed    | vwap_slope > vwap_slope_threshold |


- **vwap_slope_passed**: True when **vwap_slope > vwap_slope_threshold** (and **vwap_slope_period** is the divisor in the slope calculation, e.g. 3 minutes).

### 3. TRIN / TICK (market breadth)

- If **trin_tick_check_enabled** is False → **trin_tick_passed** is True (check skipped).
- If **alpha score ≥ trin_tick_bypass_alpha** → **trin_tick_passed** is True (bypass).
- Otherwise both TRIN and TICK must pass:


| Indicator   | Source                                              | Criteria that use it | Pass condition           |
| ----------- | --------------------------------------------------- | -------------------- | ------------------------ |
| **trin**    | TRIN-NYSE index close (broker)                      | trin_tick_passed     | trin ≤ trin_threshold    |
| **tick_ma** | TICK-NYSE close, then MA over `tick_ma_window` bars | trin_tick_passed     | tick_ma ≥ tick_threshold |


- **trin_tick_passed**: True only when **both** TRIN ≤ trin_threshold and tick_ma ≥ tick_threshold (config: `trin_threshold`, `tick_threshold`, `tick_ma_window`).

---

## Summary: what must be true for entry

- **Alpha score** ≥ threshold.
- **Trend**: close > vwap and ema1 > ema2.
- **Momentum**: macd > 0.
- **Volume/volatility**: volume > multiplier×volume_avg and adx > adx_threshold.
- **Market calm**: vix < vix_threshold and vix dropping (current < 4 bars ago).
- **Additional checks**:
  - volume_recent > volume_multiplier × volume_avg_addl,
  - vwap_slope > vwap_slope_threshold,
  - (TRIN ≤ trin_threshold and tick_ma ≥ tick_threshold) or TRIN/TICK disabled or score ≥ bypass_alpha.

News is a fixed add to score and always “passed” in code; it doesn’t depend on any real indicator.