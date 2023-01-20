""""""
"""Assess a betting strategy.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def author():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The GT username of the student  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return "jyu497"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def gtid():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The GT ID of the student  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return 902852040  # replace with your GT ID number
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def get_spin_result(win_prob):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param win_prob: The probability of winning  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type win_prob: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :return: The result of the spin.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :rtype: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    result = False  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if np.random.random() <= win_prob:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        result = True  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return result  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def test_code():  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    Method to test your code  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    win_prob = 18 / 38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # add your code here to implement the experiments

    # 10 episodes with infinite bankroll
    winnings = simulator(win_prob, 10)
    plt.plot(winnings.T)

    plt.axis([0, 300, -256, 100])
    plt.title("10 episodes w/ infinite bankroll")
    plt.xlabel("Spin round")
    plt.ylabel("Winning")
    plt.legend(["Episode {}".format(i) for i in range(1, 11)])
    plt.savefig('images/figure1.png')
    plt.clf()

    # 1000 episodes with infinite bankroll
    winnings = simulator(win_prob, 1000)
    means = np.mean(winnings, axis=0)
    stds = np.std(winnings, axis=0)
    medians = np.median(winnings, axis=0)

    plt.plot(means, label="mean")
    plt.plot(means + stds, label="mean+std")
    plt.plot(means - stds, label="mean-std")
    plt.axis([0, 300, -256, 100])
    plt.title("Means of 1000 episodes w/ infinite bankroll")
    plt.xlabel("Spin round")
    plt.ylabel("Winning")
    plt.legend()
    plt.savefig('images/figure2.png')
    plt.clf()

    plt.plot(medians, label="median")
    plt.plot(medians + stds, label="median+std")
    plt.plot(medians - stds, label="median-std")
    plt.axis([0, 300, -256, 100])
    plt.title("Medians of 1000 episodes w/ infinite bankroll")
    plt.xlabel("Spin round")
    plt.ylabel("Winning")
    plt.legend()
    plt.savefig('images/figure3.png')
    plt.clf()

    # 1000 episodes with $256 bankroll
    winnings = simulator(win_prob, 1000, realistic=True)
    means = np.mean(winnings, axis=0)
    stds = np.std(winnings, axis=0)
    medians = np.median(winnings, axis=0)

    print("Number of win episode: ", np.sum(winnings[:, 1000] == 80))

    plt.plot(means, label="mean")
    plt.plot(means + stds, label="mean+std")
    plt.plot(means - stds, label="mean-std")
    plt.axis([0, 300, -256, 100])
    plt.title("Means of 1000 episodes w/ $256 bankroll")
    plt.xlabel("Spin round")
    plt.ylabel("Winning")
    plt.legend()
    plt.savefig('images/figure4.png')
    plt.clf()

    plt.plot(medians, label="median")
    plt.plot(medians + stds, label="median+std")
    plt.plot(medians - stds, label="median-std")
    plt.axis([0, 300, -256, 100])
    plt.title("Medians of 1000 episodes w/ $256 bankroll")
    plt.xlabel("Spin round")
    plt.ylabel("Winning")
    plt.legend()
    plt.savefig('images/figure5.png')
    plt.clf()
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

def simulator(win_prob, max_episode, max_spin=1000, target_winning=80, realistic=False, bankroll=256):
    winnings = np.empty([max_episode, max_spin + 1])

    for i in range(max_episode):
        winning = 0
        spin_round = 1
        bet = 1
        episode_winnings = np.full(max_spin + 1, target_winning)
        episode_winnings[0] = 0

        while winning < target_winning and spin_round <= max_spin:
            won = get_spin_result(win_prob)

            if won:
                winning += bet
                bet = 1
            else:
                winning -= bet

                if realistic:
                    if winning <= -bankroll:
                        episode_winnings[spin_round:] = -bankroll
                        break

                    bet = min(bet * 2, winning + bankroll)
                else:
                    bet = bet * 2

            episode_winnings[spin_round] = winning
            spin_round += 1

        winnings[i] = episode_winnings

    return winnings


if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
