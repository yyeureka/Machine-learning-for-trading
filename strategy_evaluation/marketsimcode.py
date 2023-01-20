import numpy as np
import pandas as pd

import sys
sys.path.append("..")
import util as ut


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jyu497"


def compute_portvals(df_trades, start_val=100000, commission=0.00, impact=0.00):
    symbol = df_trades.columns[0]
    df = ut.get_data([symbol], pd.date_range(df_trades.index[0], df_trades.index[-1]))
    df = df.ffill().bfill()
    df = df.rename(columns={'SPY': 'Portval'})
    dates = df.index

    cash = start_val
    shares = 0

    for date in dates:
        price = df.loc[date, symbol]
        trade = int(df_trades.loc[date])

        if 0 != trade:
            cash -= commission + trade * impact * price + trade * price
            shares += trade

        df.loc[date, 'Portval'] = cash + shares * price

    return df['Portval']


def get_benchmark(symbol, dates, sv=100000):
    df_trades = pd.DataFrame(0, index=dates, columns=[symbol])
    df_trades.iloc[0] = 1000
    portvals = compute_portvals(df_trades=df_trades, start_val=sv, commission=0.00, impact=0.00)

    return portvals


def access(prices):
    # Daily returns
    dr = prices / prices.shift(1) - 1
    dr[0] = 0

    # Cumulative returns
    cr = prices[-1] / prices[0] - 1

    # Average daily returns
    adr = dr[1:].mean()

    # Standard deviation of daily returns
    sddr = dr[1:].std()

    # Sharpe Ratio
    sr = np.sqrt(252) * (adr - 0) / sddr

    return cr, adr, sddr, sr