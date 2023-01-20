import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from util import get_data


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jyu497"


def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices.ffill().bfill()
    dates = df_prices.index
    df_trades = pd.DataFrame(np.zeros(len(dates)), index=dates, columns=[symbol])

    shares = 0

    for i in range(len(dates) - 1):
        if df_prices.loc[dates[i], symbol] < df_prices.loc[dates[i + 1], symbol]:
            df_trades.loc[dates[i]] = 1000 - shares
            shares = 1000
        elif df_prices.loc[dates[i], symbol] > df_prices.loc[dates[i + 1], symbol]:
            df_trades.loc[dates[i]] = -1000 - shares
            shares = -1000

    return df_trades


def compute_portvals(df_trades, start_val=100000, commission=0.00, impact=0.00):
    symbol = df_trades.columns[0]
    df = get_data([symbol], pd.date_range(df_trades.index[0], df_trades.index[-1]))
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
    df_trades = pd.DataFrame(np.zeros(len(dates)), index=dates, columns=[symbol])
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

    return cr, adr, sddr


def run():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    df_trades = testPolicy(symbol="JPM", sd=sd, ed=ed, sv=sv)
    portvals = compute_portvals(df_trades=df_trades, start_val=sv, commission=0.00, impact=0.00)
    benchmark = get_benchmark(symbol="JPM", dates=df_trades.index, sv=sv)

    cum_ret_tos, avg_daily_ret_tos, std_daily_ret_tos = access(portvals)
    cum_ret, avg_daily_ret, std_daily_ret = access(benchmark)

    f = open('p6_results.txt', 'w')
    f.write('Portfolio:\n')
    f.write('Cumulative return: {}\n'.format(cum_ret_tos))
    f.write('Mean of daily returns: {}\n'.format(avg_daily_ret_tos))
    f.write('Stdev of daily returns: {}\n'.format(std_daily_ret_tos))
    f.write('\n')
    f.write('Benchmark:\n')
    f.write('Cumulative return: {}\n'.format(cum_ret))
    f.write('Mean of daily returns: {}\n'.format(avg_daily_ret))
    f.write('Stdev of daily returns: {}\n'.format(std_daily_ret))
    f.close()

    # Normalization
    portvals = portvals / portvals[0]
    benchmark = benchmark / benchmark[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(portvals, label="Theoretically optimal portfolio", color="red")
    plt.title("Theoretically Optimal Strategy")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/TOS.png")
    plt.clf()

