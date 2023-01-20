import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from util import get_data


def author():
    """
    :return: The GT username of the student  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """
    return "jyu497"


def stochastic(prices, window_k, window_d):
    high = prices.rolling(window=window_k, center=False).max()
    low = prices.rolling(window=window_k, center=False).min()

    K = 100 * (prices - low) / (high - low)
    D = K.rolling(window=window_d, center=False).mean()

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    plt.plot(prices, label="JPM price")
    plt.plot(high, label="Rolling max")
    plt.plot(low, label="Rolling min")
    plt.legend()
    plt.title("Stochastic oscillator({},{})".format(window_k, window_d))
    plt.ylabel('Price')
    plt.grid(linestyle='dotted')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(K, label="K")
    plt.plot(D, label="D")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel('Stochastic oscillator value')
    plt.grid(linestyle='dotted')
    fig.autofmt_xdate()
    plt.savefig("images/stochastic.png")
    plt.clf()

    return D


def rsi(prices, window):
    delta = prices.diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    up_gain = up.ewm(span=window, adjust=False).mean()  # TODO: no historical?
    down_loss = down.abs().ewm(span=window, adjust=False).mean()

    RS = up_gain / down_loss
    rsi_value = 100 - (100 / (1 + RS))

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    plt.plot(prices, label="JPM price")
    plt.legend()
    plt.title("RSI({})".format(window))
    plt.ylabel('Price')
    plt.grid(linestyle='dotted')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(rsi_value)
    plt.xlabel("Date")
    plt.ylabel('RSI value')
    plt.grid(linestyle='dotted')
    fig.autofmt_xdate()
    plt.savefig("images/rsi.png")
    plt.clf()

    return rsi_value


def ema(prices, window):
    ema = prices.ewm(span=window, adjust=False).mean()  # TODO: no historical?
    ema_value = prices / ema

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    plt.plot(prices, label="JPM price")
    plt.plot(ema, label="EMA")
    plt.legend()
    plt.title("EMA({})".format(window))
    plt.ylabel('Price')
    plt.grid(linestyle='dotted')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(ema_value)
    plt.xlabel("Date")
    plt.ylabel('Price / EMA')
    plt.grid(linestyle='dotted')
    fig.autofmt_xdate()
    plt.savefig("images/ema.png")
    plt.clf()

    return ema_value


def macd(prices, window1, window2, window3):
    ema_1 = prices.ewm(span=window1, adjust=False).mean()  # TODO: no historical?
    ema_2 = prices.ewm(span=window2, adjust=False).mean()
    macd = ema_1 - ema_2
    macd_signal = macd.ewm(span=window3, adjust=False).mean()

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    plt.plot(prices, label="JPM price")
    plt.plot(ema_1, label="{} days EMA".format(window1))
    plt.plot(ema_2, label="{} days EMA".format(window2))
    plt.legend()
    plt.title("MACD({},{},{})".format(window1, window2, window3))
    plt.ylabel('Price')
    plt.grid(linestyle='dotted')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(macd, label="MACD")
    plt.plot(macd_signal, label="Signal")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel('MACD value')
    plt.grid(linestyle='dotted')
    fig.autofmt_xdate()
    plt.savefig("images/macd.png")
    plt.clf()

    return macd_signal


def bollinger(prices, window, width):
    sma = prices.rolling(window=window, center=False).mean()
    std = prices.rolling(window=window, center=False).std()

    upper_band = sma + (width * std)
    lower_band = sma - (width * std)

    # BB value
    bb_value = (prices - sma) / (2 * std)
    bbp = (prices - lower_band) / (upper_band - lower_band)

    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(211)
    plt.plot(prices, label="JPM price")
    plt.plot(sma, label="SMA")
    plt.plot(upper_band, label="Upper band", color="purple")
    plt.plot(lower_band, label="Lower band", color="purple")
    plt.legend()
    plt.title("Bollinger bands({}, {})".format(window, width))
    plt.ylabel('Price')
    plt.grid(linestyle='dotted')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(bb_value)
    plt.xlabel("Date")
    plt.ylabel('Bollinger bands value')
    plt.grid(linestyle='dotted')
    fig.autofmt_xdate()
    plt.savefig("images/bollinger.png")
    plt.clf()

    return bb_value


def run():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000

    df_prices = get_data(["JPM"], pd.date_range(sd, ed))
    df_prices = df_prices["JPM"]
    df_prices = df_prices.ffill().bfill()

    # Stochastic oscillator
    stochastic(df_prices, 14, 3)

    # RSI
    rsi(df_prices, 14)

    # EMA
    ema(df_prices, 20)

    # MACD
    macd(df_prices, 12, 26, 9)

    # Bollinger bands
    bollinger(df_prices, 20, 2)






