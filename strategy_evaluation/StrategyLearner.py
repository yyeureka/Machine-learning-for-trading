""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Atlanta, Georgia 30332  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
All Rights Reserved  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Template code for CS 4646/7646  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
and other users of this template code are advised not to share it with others  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or to make it available on publicly viewable websites including repositories  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
or edited.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
We do grant permission to share solutions privately with non-students such  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
as potential employers. However, sharing with other current or future  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT honor code violation.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
-----do not edit anything above this line---  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

import datetime as dt
import pandas as pd

import sys
sys.path.append("..")
import util as ut
import indicators
import RTLearner as rt
import BagLearner as bl


class StrategyLearner(object):
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type verbose: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type impact: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type commission: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Constructor method  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """
        self.commission = commission
        self.impact = impact
        self.verbose = verbose
        self.transaction_num = 0

        # Hyper-parameters
        leaf_size = 8
        bags = 15
        self.look_back = 20
        self.N = 5
        target_cr = 0.03
        self.YBUY = target_cr + self.impact
        self.YSELL = -target_cr - self.impact

        self.learner = bl.BagLearner(rt.RTLearner, {"leaf_size": leaf_size}, bags)

    # this method should create a QLearner, and train it for trading  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def add_evidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """

        prices = self.get_prices(symbol, sd, ed)

        train_x = self.get_features(prices)
        train_x = train_x[:-self.N]

        train_y = prices.copy() * 0
        train_y = train_y[:-self.N]

        for i in range(len(prices) - self.N):
            cr = prices.iloc[i + self.N, :] / prices.iloc[i, :] - 1

            if cr[0] > self.YBUY:  # long
                train_y.iloc[i, :] = 1
            elif cr[0] < self.YSELL:  # short
                train_y.iloc[i, :] = -1

        self.learner.add_evidence(train_x.values, train_y.values)

    # this method should use the existing policy and test it against new data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """

        prices = self.get_prices(symbol, sd, ed)
        test_x = self.get_features(prices)[:]
        test_y = self.learner.query(test_x.values)

        dates = prices.index
        trades = pd.DataFrame(0, index=dates, columns=[symbol])
        shares = 0

        for i in range(len(trades)):
            if test_y[i] > 0:
                trades.loc[dates[i]] = 1000 - shares  # long
                shares = 1000
                self.transaction_num += 1
            elif test_y[i] < 0:
                trades.loc[dates[i]] = -1000 - shares  # short
                shares = -1000
                self.transaction_num += 1
            else:  # cash
                trades.loc[dates[i]] = -shares
                shares = 0

        return trades

    def get_prices(self, symbol, sd, ed):
        syms = [symbol]
        prices = ut.get_data(syms, pd.date_range(sd, ed))
        prices = prices.ffill().bfill()

        return prices[syms]

    def get_features(self, prices):
        rsi = indicators.rsi(prices, self.look_back)
        rsi.columns = ['RSI']
        ema = indicators.ema(prices, self.look_back)
        ema.columns = ['EMA']
        bbp = indicators.bollinger(prices, self.look_back, 2)
        bbp.columns = ['BBP']

        features = pd.concat((rsi, ema, bbp), axis=1)
        features.fillna(0, inplace=True)

        return features

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jyu497"


if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print("One does not simply think up a strategy")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
