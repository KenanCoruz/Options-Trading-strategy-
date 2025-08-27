# SPY Options Volatility Arbitrage Strategy

This project implements and evaluates a systematic options trading strategy on S&P 500 index, based on discrepancies between forecasted realized volatility and implied volatility (IV).  

---

## Strategy Overview

- **Idea:**  
  Implied volatility in index options often contains a volatility risk premium.  
  By forecasting realized volatility (RV) with a HAR-RV model and comparing it to implied volatility (IV, proxied by VIX), we take systematic positions in SPY straddles:
  - **Long straddle** if RV forecast > IV (expect realized vol to exceed implied).  
  - **Short straddle** if RV forecast < IV (expect implied vol overpriced).  

- **Execution:**  
  - ATM straddles (strike = spot at entry).  
  - Expiry: 30 calendar days.  
  - Holding period: 5 trading days.  
  - Daily mark-to-market with Black–Scholes pricing.  
  

---

## Results 

**Backtest parameters:**  
- Horizon: Jan 2024 – Jan 2025  
- Holding: 5 trading days per trade  
- Option maturity: 30 days  
- Spread: 1% bid–ask on straddles  
- Hedge cost: 2 bps per rebalance  
- Forecast model: Walk-forward HAR-RV
- Implied volatility approximated by VIX 

**Performance summary:**

| Metric | Value |
|--------|-------|
| Total trades | 34 |
| Win rate | 97.06% |
| Avg net PnL / trade | 14.0222 |
| Total net PnL | 476.75 |
| Sharpe ratio | 21.58 |
| Max Drawdown | -6.53 |
| Net PnL (Long straddles) | 0.00 |
| Net PnL (Short straddles) | 476.75 |

---

## Interpretation

- **Alpha source:**  
  The strategy exploits the tendency of SPY implied volatility (proxied by VIX) to trade above forecasted realized volatility.  
  The strong performance is driven entirely by **systematic short straddles**.  

- **Risk profile:**  
  - High win rate suggests the signal is mostly correct in identifying overpriced IV.  
  - The high maximum drawdown of **(-6.53)** indicates medium tail risk, i.e. the strategie can suffer during volatility spikes.  
  - Sharpe ratio of 21.58 is most likely inflated due to limited sample.

- **Caveats:**  
  - Results depend on **proxy choice (VIX)** — true ATM IV for the traded expiry may differ.  
  - No modeling of execution slippage or margin requirements.  
  - Risk of large losses in unhedged short vol positions remains significant.  

---

## Suggestions for future projects 
- Obtain true IV from data of ATM call options
- Include slippage
- Implement strategy to mitigate tail risks 

---
