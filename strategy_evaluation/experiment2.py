import datetime as dt
import numpy as np
from matplotlib import pyplot as plt

import StrategyLearner as sl
from marketsimcode import compute_portvals, access


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "jyu497"


def run():
    symbol = "JPM"
    impacts = np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.2])
    commission = 0.0
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    f = open('p8_results.txt', 'a+')

    plt.figure(figsize=(14, 8))

    for impact in impacts:
        learner = sl.StrategyLearner(impact=impact, commission=commission)
        learner.add_evidence(symbol=symbol, sd=sd, ed=ed)
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed)
        portvals = compute_portvals(trades, start_val=sv, commission=commission, impact=impact)
        cr, adr, sddr, _ = access(portvals)

        f.write('\n')
        f.write('Impact={}: {}\n'.format(impact, learner.transaction_num))

        portvals = portvals / portvals[0]
        plt.plot(portvals, label="Impact={}".format(impact))

    f.close()

    plt.title("How impact affects the Strategy learner")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/impact.png")
    plt.clf()
