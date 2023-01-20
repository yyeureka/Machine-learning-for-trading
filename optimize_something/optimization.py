""""""
"""MC1-P2: Optimize a portfolio.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Student Name: Jing Yu (replace with your name)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT User ID: jyu497 (replace with your User ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
GT ID: 902852040 (replace with your GT ID)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from util import get_data, plot_data


# This is the function that will be tested by the autograder  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
# The student must update this code to properly implement the functionality  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    statistics.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type sd: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type ed: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        symbol in the data directory)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type syms: list  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        code with gen_plot = False.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type gen_plot: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        standard deviation of daily returns, and Sharpe ratio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: tuple  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

    # Fill forward and backward
    prices = prices.ffill().bfill()

    # Find the allocations for the optimal portfolio
    allocs = optimizer(prices, len(syms))

    # Assess optimal portfolio
    cr, adr, sddr, sr, port_val = assess_portfolio(allocs, prices)

    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if gen_plot:
        # add code to plot here  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        df_temp = df_temp / df_temp.iloc[0, :]

        ax = df_temp.plot(title='Daily Portfolio Value and SPY')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        plt.grid(linestyle='dotted')
        plt.savefig('images/Figure1.png')

    return allocs, cr, adr, sddr, sr


def assess_portfolio(allocs, prices):
    # Daily portfolio value
    port_prices = prices / prices.iloc[0, :] * allocs
    port_val = port_prices.sum(axis=1)

    # Daily returns
    dr = port_val / port_val.shift(1) - 1
    dr[0] = 0

    # Cumulative returns
    cr = port_val[-1] / port_val[0] - 1

    # Average daily returns
    adr = dr[1:].mean()

    # Standard deviation of daily returns
    sddr = dr[1:].std()

    # Sharpe Ratio
    sr = np.sqrt(252) * (adr - 0) / sddr

    return cr, adr, sddr, sr, port_val


def loss_function(allocs, prices):
    return -1.0 * assess_portfolio(allocs, prices)[3]


def optimizer(prices, asset_number):
    x0 = [1.0 / asset_number for _ in range(asset_number)]  # uniform initialization
    bounds = [(0.0, 1.0) for _ in range(asset_number)]
    constraints = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    allocs = minimize(loss_function, x0, args=(prices,), method='SLSQP', bounds=bounds, constraints=constraints)  # TODO

    return allocs.x


def test_code():
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    This function WILL NOT be called by the auto grader.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]  # ["GOOG", "AAPL", "GLD", "XOM", "IBM"]

    # Assess the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Do not assume that it will be called  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()
