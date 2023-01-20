import datetime as dt
from matplotlib import pyplot as plt

import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals, get_benchmark, access


def author():
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

    learner_ms = ms.ManualStrategy(impact=impact, commission=commission)
    trades_ms = learner_ms.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_ms = compute_portvals(trades_ms, start_val=sv, commission=commission, impact=impact)

    learner_sl = sl.StrategyLearner(impact=impact, commission=commission)
    learner_sl.add_evidence(symbol=symbol, sd=sd, ed=ed)
    trades_sl = learner_sl.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_sl = compute_portvals(trades_sl, start_val=sv, commission=commission, impact=impact)

    benchmark = get_benchmark(symbol="JPM", dates=trades_ms.index, sv=sv)

    cr_ms, adr_ms, sddr_ms, _ = access(portvals_ms)
    cr_sl, adr_sl, sddr_sl, _ = access(portvals_sl)
    cr, adr, sddr, _ = access(benchmark)

    f = open('p8_results.txt', 'w')
    f.write('In sample:\n')
    f.write('Manual strategy:\n')
    f.write('Cumulative return: {}\n'.format(cr_ms))
    f.write('Mean of daily returns: {}\n'.format(adr_ms))
    f.write('Stdev of daily returns: {}\n'.format(sddr_ms))
    f.write('Strategy learner:\n')
    f.write('Cumulative return: {}\n'.format(cr_sl))
    f.write('Mean of daily returns: {}\n'.format(adr_sl))
    f.write('Stdev of daily returns: {}\n'.format(sddr_sl))
    f.write('Benchmark:\n')
    f.write('Cumulative return: {}\n'.format(cr))
    f.write('Mean of daily returns: {}\n'.format(adr))
    f.write('Stdev of daily returns: {}\n'.format(sddr))

    # Normalization
    portvals_ms = portvals_ms / portvals_ms[0]
    portvals_sl = portvals_sl / portvals_sl[0]
    benchmark = benchmark / benchmark[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(portvals_ms, label="Manual strategy", color="red")
    plt.plot(portvals_sl, label="Strategy learner")
    plt.title("Strategy learner vs. Manual strategy - in sample")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/in.png")
    plt.clf()

    # out of sample
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000

    trades_ms = learner_ms.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_ms = compute_portvals(trades_ms, start_val=sv, commission=commission, impact=impact)

    trades_sl = learner_sl.testPolicy(symbol=symbol, sd=sd, ed=ed)
    portvals_sl = compute_portvals(trades_sl, start_val=sv, commission=commission, impact=impact)

    benchmark = get_benchmark(symbol="JPM", dates=trades_ms.index, sv=sv)

    cr_ms, adr_ms, sddr_ms, _ = access(portvals_ms)
    cr_sl, adr_sl, sddr_sl, _ = access(portvals_sl)
    cr, adr, sddr, _ = access(benchmark)

    f.write('\n')
    f.write('Out of sample:\n')
    f.write('Manual strategy:\n')
    f.write('Cumulative return: {}\n'.format(cr_ms))
    f.write('Mean of daily returns: {}\n'.format(adr_ms))
    f.write('Stdev of daily returns: {}\n'.format(sddr_ms))
    f.write('Strategy learner:\n')
    f.write('Cumulative return: {}\n'.format(cr_sl))
    f.write('Mean of daily returns: {}\n'.format(adr_sl))
    f.write('Stdev of daily returns: {}\n'.format(sddr_sl))
    f.write('Benchmark:\n')
    f.write('Cumulative return: {}\n'.format(cr))
    f.write('Mean of daily returns: {}\n'.format(adr))
    f.write('Stdev of daily returns: {}\n'.format(sddr))
    f.close()

    # Normalization
    portvals_ms = portvals_ms / portvals_ms[0]
    portvals_sl = portvals_sl / portvals_sl[0]
    benchmark = benchmark / benchmark[0]

    plt.figure(figsize=(14, 8))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(portvals_ms, label="Manual strategy", color="red")
    plt.plot(portvals_sl, label="Strategy learner")
    plt.title("Strategy learner vs. Manual strategy - out of sample")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig("images/out.png")
    plt.clf()


