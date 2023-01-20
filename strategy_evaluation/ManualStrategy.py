import datetime as dt
import pandas as pd

import sys

from matplotlib import pyplot as plt

sys.path.append("..")
import util as ut
import indicators
from marketsimcode import compute_portvals, get_benchmark, access


class ManualStrategy(object):
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        self.commission = commission
        self.impact = impact
        self.verbose = verbose

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        look_back = 14

        syms = [symbol]
        prices = ut.get_data(syms, pd.date_range(sd, ed))
        prices = prices.ffill().bfill()
        prices = prices[syms]

        rsi = indicators.rsi(prices, look_back)
        ema = indicators.ema(prices, look_back)
        bbp = indicators.bollinger(prices, look_back, 2)

        dates = prices.index
        trades = pd.DataFrame(0, index=dates, columns=[symbol])
        shares = 0

        for i in range(len(dates)):
            e = ema.iloc[i, 0]
            e_pre = ema.iloc[i - 1, 0]
            b = bbp.iloc[i, 0]
            r = rsi.iloc[i, 0]

            if (e < 0.95) and (b < 0.15) and (r < 30):
                trades.loc[dates[i]] = 1000 - shares
                shares = 1000
            elif (e > 1.05) and (b > 0.85) and (r > 70):
                trades.loc[dates[i]] = -1000 - shares
                shares = -1000
            # elif (e >= 1) and (e_pre < 1) and (shares > 0):
            #     trades.loc[dates[i]] = -shares
            #     shares = 0
            # elif (e <= 1) and (e_pre > 1) and (shares < 0):
            #     trades.loc[dates[i]] = -shares
            #     shares = 0

        return trades

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jyu497"


def run():
    symbol = "JPM"
    impact = 0.005
    commission = 9.95

    # in sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    learner_ms = ManualStrategy(impact=impact, commission=commission)
    trades_ms = learner_ms.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_ms = compute_portvals(trades_ms, start_val=sv, commission=commission, impact=impact)

    benchmark = get_benchmark(symbol="JPM", dates=trades_ms.index, sv=sv)

    # Normalization
    portvals_ms = portvals_ms / portvals_ms[0]
    benchmark = benchmark / benchmark[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(portvals_ms, label="Manual strategy", color="red")
    for date in trades_ms.index:
        shares = int(trades_ms.loc[date])
        if shares > 0:
            plt.axvline(date, color="blue")
        elif shares < 0:
            plt.axvline(date, color="black")
    plt.title("Manual strategy - in sample")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/manual_in.png")
    plt.clf()

    # out of sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000

    trades_ms = learner_ms.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_ms = compute_portvals(trades_ms, start_val=sv, commission=commission, impact=impact)

    benchmark = get_benchmark(symbol="JPM", dates=trades_ms.index, sv=sv)

    # Normalization
    portvals_ms = portvals_ms / portvals_ms[0]
    benchmark = benchmark / benchmark[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(portvals_ms, label="Manual strategy", color="red")
    for date in trades_ms.index:
        shares = int(trades_ms.loc[date])
        if shares > 0:
            plt.axvline(date, color="blue")
        elif shares < 0:
            plt.axvline(date, color="black")
    plt.title("Manual strategy - out of sample")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/manual_out.png")
    plt.clf()
