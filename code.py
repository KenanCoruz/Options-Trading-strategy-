import yfinance as yf
import numpy as np
import pandas as pd
from datetime import timedelta

def get_spy_iv_series(start_date="2024-01-01", end_date="2025-01-01"):
    ticker = yf.Ticker("SPY")
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    iv_series = {}

    for date in all_dates:
        try:
            options_dates = ticker.options
            if not options_dates:
                continue

            selected_expiry = None
            for expiry in options_dates:
                days_to_expiry = (pd.to_datetime(expiry) - date).days
                if days_to_expiry >= 25:
                    selected_expiry = expiry
                    break

            if selected_expiry is None:
                continue

            opt_chain = ticker.option_chain(selected_expiry)
            calls = opt_chain.calls
            puts = opt_chain.puts

            spot = ticker.history(start=date.strftime('%Y-%m-%d'), end=(date + timedelta(days=1)).strftime('%Y-%m-%d'))['Close']
            if spot.empty:
                continue

            spot_price = spot.iloc[0]
            calls['distance'] = abs(calls['strike'] - spot_price)
            atm_strike = calls.sort_values(by='distance').iloc[0]['strike']

            atm_call_iv = calls[calls['strike'] == atm_strike]['impliedVolatility'].values
            atm_put_iv = puts[puts['strike'] == atm_strike]['impliedVolatility'].values

            if len(atm_call_iv) == 0 or len(atm_put_iv) == 0:
                continue

            avg_iv = 0.5 * (atm_call_iv[0] + atm_put_iv[0])
            iv_series[date] = avg_iv

        except Exception as e:
            continue

    return pd.Series(iv_series).sort_index()


def realized_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window).std() * np.sqrt(252)
    return vol
from sklearn.linear_model import LinearRegression

def prepare_har_features(rv: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=rv.index)
    df['RV'] = rv
    df['RV_d'] = rv.shift(1)
    df['RV_w'] = rv.shift(1).rolling(5).mean()
    df['RV_m'] = rv.shift(1).rolling(22).mean()
    df.dropna(inplace=True)
    return df

def fit_har_model(rv_series: pd.Series):
    data = prepare_har_features(rv_series)
    X = data[['RV_d', 'RV_w', 'RV_m']]
    y = data['RV']

    model = LinearRegression()
    model.fit(X, y)

    forecast = pd.Series(model.predict(X), index=data.index, name='RV_forecast')
    return model, forecast
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def generate_trade_signals(rv_forecast: pd.Series, iv_series: pd.Series, threshold: float = 0.02):
    diff = rv_forecast.shift(1) - iv_series.shift(1)
    signal = pd.Series(0, index=rv_forecast.index)
    signal[diff > threshold] = 1
    signal[diff < -threshold] = -1
    return signal
def to_scalar(x):
    if isinstance(x, pd.Series) or isinstance(x, pd.Index):
        return float(x.iloc[0])
    return float(x)

def run_straddle_backtest(signals, spot_prices, iv_series, forecast_series,
                          r=0.01, T=30/252, hold_days=5, opt_spread_pct=0.01, hedge_cost_bps=2,
                          non_overlapping=True):
    trade_log = []
    equity_curve = pd.Series(dtype=float)
    open_trade_until = None

    for date in signals.index[:-hold_days]:
        signal = signals.loc[date]
        if signal == 0:
            continue
        # If non-overlapping, only take a new trade if previous is closed
        if non_overlapping and open_trade_until is not None and date <= open_trade_until:
            continue

        try:
            entry_price = to_scalar(spot_prices.loc[date])
            iv = to_scalar(iv_series.loc[date])
            forecast = to_scalar(forecast_series.loc[date])

            
            call_entry = black_scholes_price(S=entry_price, K=entry_price, T=T, r=r, sigma=iv, option_type='call')
            put_entry = black_scholes_price(S=entry_price, K=entry_price, T=T, r=r, sigma=iv, option_type='put')
            straddle_cost_raw = call_entry + put_entry

            straddle_cost = straddle_cost_raw * (1 + opt_spread_pct / 2) if signal > 0 else straddle_cost_raw * (1 - opt_spread_pct / 2)
            hedge_cost = 2 * hedge_cost_bps / 10000 * entry_price

            entry_day = date
            exit_idx = signals.index.get_loc(date) + hold_days
            if exit_idx >= len(signals.index):
                break
            exit_day = signals.index[exit_idx]

            exit_price = to_scalar(spot_prices.loc[exit_day])
            iv_exit = to_scalar(iv_series.loc[exit_day])

            # Always price the *same strike* (entry price), same expiry reduced by days held
            remaining_T = T - hold_days / 252
            if remaining_T <= 0:
                continue  # Avoid negative time to expiry

            call_exit = black_scholes_price(S=exit_price, K=entry_price, T=remaining_T, r=r, sigma=iv_exit, option_type='call')
            put_exit = black_scholes_price(S=exit_price, K=entry_price, T=remaining_T, r=r, sigma=iv_exit, option_type='put')

            straddle_end_raw = call_exit + put_exit

            straddle_end = straddle_end_raw * (1 - opt_spread_pct / 2) if signal > 0 else straddle_end_raw * (1 + opt_spread_pct / 2)
            gross_pnl = signal * (straddle_end - straddle_cost)
            net_pnl = gross_pnl - hedge_cost

            trade_log.append({
                'entry_day': entry_day,
                'exit_day': exit_day,
                'signal': signal,
                'forecast': forecast,
                'iv': iv,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'straddle_cost': straddle_cost,
                'straddle_end': straddle_end,
                'gross_pnl': gross_pnl,
                'hedge_cost': hedge_cost,
                'net_pnl': net_pnl
            })
            equity_curve.loc[exit_day] = equity_curve.get(exit_day, 0) + net_pnl
            # Mark the trade as open for N days
            if non_overlapping:
                open_trade_until = exit_day

        except Exception as e:
            print(f"Error on {date}: {e}")
            continue

    if not trade_log:
        print("No trades executed.")
        return pd.DataFrame(), pd.Series(dtype=float)
    trade_df = pd.DataFrame(trade_log).set_index('exit_day')
    equity_curve = equity_curve.sort_index().cumsum()
    return trade_df, equity_curve
import matplotlib.pyplot as plt

def evaluate_strategy(trade_df: pd.DataFrame, equity_curve: pd.Series):
    print("STRATEGY PERFORMANCE OVERVIEW")
    
    if trade_df.empty or 'net_pnl' not in trade_df.columns:
        print("No valid trades")
        return

    trade_df = trade_df.dropna(subset=['net_pnl'])

    total_trades = len(trade_df)
    print(f"Total trades: {total_trades}")

    if total_trades == 0:
        print("No trades executed.")
        return

    win_rate = (trade_df['net_pnl'].astype(float) > 0).mean()
    print(f"Win rate: {win_rate:.2%}")

    avg_pnl = float(trade_df['net_pnl'].mean())
    total_pnl = float(trade_df['net_pnl'].sum())
    print(f"Average net PnL per trade: {avg_pnl:.4f}")
    print(f"Total net PnL: {total_pnl:.2f}")

    if not equity_curve.empty:
        daily_returns = equity_curve.diff().fillna(0)
        if daily_returns.std() > 0:
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            print(f"Sharpe Ratio: {sharpe:.2f}")
        else:
            print("Sharpe Ratio undefined")

        cum_max = equity_curve.cummax()
        drawdown = equity_curve - cum_max
        max_dd = drawdown.min()
        print(f"Max Drawdown: {max_dd:.2f}")
    else:
        print("Equity curve is empty")
        return

    long_pnl = float(trade_df[trade_df['signal'] > 0]['net_pnl'].sum())
    short_pnl = float(trade_df[trade_df['signal'] < 0]['net_pnl'].sum())
    print(f"Net PnL (Long Straddles):  {long_pnl:.2f}")
    print(f"Net PnL (Short Straddles): {short_pnl:.2f}")

    plot_equity_curve(equity_curve)
    plot_rolling_sharpe(daily_returns)
    plot_pnl_hist(trade_df)

def plot_equity_curve(equity_curve: pd.Series):
    plt.figure(figsize=(10, 4))
    equity_curve.plot()
    plt.title("Cumulative PnL")
    plt.ylabel("PnL ($)")
    plt.grid(True)
    plt.show()

def plot_rolling_sharpe(daily_returns: pd.Series, window=30):
    rolling_sharpe = daily_returns.rolling(window).mean() / daily_returns.rolling(window).std() * np.sqrt(252)
    plt.figure(figsize=(10, 4))
    rolling_sharpe.plot()
    plt.title("30-Day Rolling Sharpe Ratio")
    plt.grid(True)
    plt.show()

def plot_pnl_hist(trade_df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    trade_df['net_pnl'].hist(bins=30)
    plt.title("Histogram of Net PnL per Trade")
    plt.xlabel("Net PnL")
    plt.grid(True)
    plt.show()
def main():
    spy_data = yf.download("SPY", start="2024-01-01", end="2025-01-01")
    spy_prices = spy_data["Close"]

    rv = realized_vol(spy_prices, window=21)

    model, rv_forecast = fit_har_model(rv)

    iv_series = get_spy_iv_series(start_date="2024-01-01", end_date="2025-01-01")

    all_index = spy_prices.index.intersection(rv_forecast.index).intersection(iv_series.index)
    rv_forecast = rv_forecast.loc[all_index]
    iv_series = iv_series.loc[all_index]
    spy_prices = spy_prices.loc[all_index]

    signals = generate_trade_signals(rv_forecast, iv_series, threshold=0.02)
    trade_df, equity_curve = run_straddle_backtest(
    signals=signals,
    spot_prices=spy_prices,
    iv_series=iv_series,
    forecast_series=rv_forecast,
    hold_days=5,
    opt_spread_pct=0.01,
    hedge_cost_bps=2,
    non_overlapping=True
)

    evaluate_strategy(trade_df, equity_curve)


main()
