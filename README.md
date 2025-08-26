# SPY Options Volatility Arbitrage Strategy

This project implements and evaluates a **systematic options trading strategy** on SPY, based on discrepancies between **forecasted realized volatility** and **implied volatility (IV)**.  
It is designed to mimic the workflow of a quantitative researcher or options trader on a prop desk: forecasting, signal generation, strategy implementation with realistic costs, and statistical evaluation.

---

## 🔎 Strategy Overview

- **Idea:**  
  Implied volatility in index options often contains a *volatility risk premium*.  
  By forecasting realized volatility (RV) with a HAR-RV model and comparing it to implied volatility (IV, proxied by VIX), we take systematic positions in SPY straddles:
  - **Long straddle** if RV forecast > IV (expect realized vol to exceed implied).  
  - **Short straddle** if RV forecast < IV (expect implied vol overpriced).  

- **Execution:**  
  - ATM straddles (strike = spot at entry).  
  - Expiry: 30 calendar days.  
  - Holding period: 5 trading days.  
  - Daily mark-to-market with Black–Scholes pricing.  
  - Optional **delta hedging** with transaction costs.  
  - Realistic **bid–ask spread costs** on options and **bps costs** on hedge adjustments.

---

## 📊 Results (SPY 2022)

**Backtest parameters:**  
- Horizon: Jan 2022 – Dec 2022  
- Holding: 5 trading days per trade  
- Option maturity: 30 days  
- Spread: 1% bid–ask on straddles  
- Hedge cost: 2 bps per rebalance  
- Forecast model: Walk-forward HAR-RV  

**Performance summary:**

| Metric | Value |
|--------|-------|
| Total trades | 34 |
| Win rate | 94.12% |
| Avg net PnL / trade | 16.20 |
| Total net PnL | 550.77 |
| Sharpe ratio | 9.89 |
| Max Drawdown | -44.91 |
| Net PnL (Long straddles) | 0.00 |
| Net PnL (Short straddles) | 550.77 |

---

## 🧑‍🔬 Interpretation

- **Alpha source:**  
  The strategy exploits the tendency of SPY implied volatility (proxied by VIX) to trade above forecasted realized volatility.  
  The strong performance is driven almost entirely by **systematic short straddles**.  

- **Risk profile:**  
  - High win rate suggests the signal is mostly correct in identifying overpriced IV.  
  - The **drawdown (-44.91)** indicates tail risk: short volatility strategies can suffer during volatility spikes.  
  - Sharpe ratio of 9.89 is likely inflated due to limited sample (2022 only).  

- **Caveats:**  
  - Results depend on **proxy choice (VIX)** — true ATM IV for the traded expiry may differ.  
  - No modeling of execution slippage or margin requirements.  
  - Risk of large losses in unhedged short vol positions remains significant.  

---

## 🛠️ Project Structure

