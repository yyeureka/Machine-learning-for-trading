""""""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
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
"""  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import math  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it


def assess(y, pred_y, metric):
    if "rmse" == metric:
        return math.sqrt(((y - pred_y) ** 2).sum() / y.shape[0])
    if "corr" == metric:
        return np.corrcoef(pred_y, y=y)
    if "mae" == metric:
        return np.sum(np.abs(y - pred_y)) / y.shape[0]


def experiment1(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 100)
    in_rmses = []
    out_rmses = []

    for leaf_size in leaf_sizes:
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        in_rmses.append(assess(train_y, pred_y, "rmse"))

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        out_rmses.append(assess(test_y, pred_y, "rmse"))

    plt.plot(leaf_sizes, in_rmses, label="In sample")
    plt.plot(leaf_sizes, out_rmses, label="Out of sample")
    plt.title("Decision Tree RMSE vs Leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.savefig('images/E1.png')
    plt.clf()


def experiment2(train_x, train_y, test_x, test_y):
    leaf_sizes = np.arange(1, 100)
    bags = np.array([1, 10, 20, 50, 100])

    for i, bag in enumerate(bags):
        in_rmses = []
        out_rmses = []

        for leaf_size in leaf_sizes:
            learner = bl.BagLearner(dt.DTLearner, {"leaf_size": leaf_size}, bag)
            learner.add_evidence(train_x, train_y)  # train it

            # evaluate in sample
            pred_y = learner.query(train_x)  # get the predictions
            in_rmses.append(assess(train_y, pred_y, "rmse"))

            # evaluate out of sample
            pred_y = learner.query(test_x)  # get the predictions
            out_rmses.append(assess(test_y, pred_y, "rmse"))

        plt.plot(leaf_sizes, in_rmses, label="In sample")
        plt.plot(leaf_sizes, out_rmses, label="Out of sample")
        plt.title("Bag Learner RMSE vs Leaf size, bags={}".format(bag))
        plt.xlabel("Leaf size")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid()
        plt.savefig('images/E2-{}.png'.format(i + 1))
        plt.clf()


def experiment3(train_x, train_y, test_x, test_y):
    dt_learners = []
    rt_learners = []
    f = open('p3_results.txt', 'w')

    # training time
    dt_height = 0
    start = time.time()
    for i in range(100):
        learner = dt.DTLearner(leaf_size=1)
        dt_learners.append(learner)
        learner.add_evidence(train_x, train_y)  # train it
        dt_height += learner.height
    end = time.time()
    f.write('100-loop DT training time: {}\n'.format(end - start))

    rt_height = 0
    start = time.time()
    for i in range(100):
        learner = rt.RTLearner(leaf_size=1)
        rt_learners.append(learner)
        learner.add_evidence(train_x, train_y)  # train it
        rt_height += learner.height
    end = time.time()
    f.write('100-loop RT training time: {}\n'.format(end - start))

    # height
    f.write('100-loop DT average height: {}\n'.format(dt_height / 100))
    f.write('100-loop RT average height: {}\n'.format(rt_height / 100))

    # query time
    start = time.time()
    for learner in dt_learners:
        learner.query(train_x)
    end = time.time()
    f.write('100-loop DT query time: {}\n'.format(end - start))

    start = time.time()
    for learner in rt_learners:
        learner.query(train_x)
    end = time.time()
    f.write('100-loop RT query time: {}\n'.format(end - start))

    f.close()

    leaf_sizes = np.arange(1, 100)
    dt_in_mae = []
    dt_out_mae = []
    rt_in_mae = []
    rt_out_mae = []

    for leaf_size in leaf_sizes:
        # Decision tree
        learner = dt.DTLearner(leaf_size=leaf_size)
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        dt_in_mae.append(assess(train_y, pred_y, "mae"))

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        dt_out_mae.append(assess(test_y, pred_y, "mae"))

        # Random tree
        learner = rt.RTLearner(leaf_size=leaf_size)
        learner.add_evidence(train_x, train_y)  # train it

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rt_in_mae.append(assess(train_y, pred_y, "mae"))

        # evaluate out of sample
        pred_y = learner.query(test_x)  # get the predictions
        rt_out_mae.append(assess(test_y, pred_y, "mae"))

    plt.plot(leaf_sizes, dt_in_mae, label="Decision Tree in sample")
    plt.plot(leaf_sizes, dt_out_mae, label="Decision Tree out of sample")
    plt.plot(leaf_sizes, rt_in_mae, label="Random Tree in sample")
    plt.plot(leaf_sizes, rt_out_mae, label="Random Tree out of sample")
    plt.title("Decision tree learner & Random tree learner MAE vs Leaf size")
    plt.xlabel("Leaf size")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    plt.savefig('images/E3-1.png')
    plt.clf()


if __name__ == "__main__":  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if len(sys.argv) != 2:  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        print("Usage: python testlearner.py <filename>")  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        sys.exit(1)  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    inf = open(sys.argv[1])

    if "Istanbul" in sys.argv[1]:
        data = np.array([list(map(str, s.strip().split(','))) for s in inf.readlines()])
        data = data[1:, 1:]
        data = data.astype(float)
    else:
        data = np.array([list(map(float, s.strip().split(","))) for s in inf.readlines()])

    np.random.seed(902852040)
    np.random.shuffle(data)

    # compute how much of the data is training and testing  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_rows = int(0.6 * data.shape[0])  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_x = data[:train_rows, 0:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    train_y = data[:train_rows, -1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_x = data[train_rows:, 0:-1]  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_y = data[train_rows:, -1]

    experiment1(train_x, train_y, test_x, test_y)
    experiment2(train_x, train_y, test_x, test_y)
    experiment3(train_x, train_y, test_x, test_y)

    # # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # learner = it.InsaneLearner()
    # learner = dt.DTLearner()
    # learner.add_evidence(train_x, train_y)  # train it

    # # evaluate in sample
    # pred_y = learner.query(train_x)  # get the predictions
    # rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0,1]}")
    #
    # # evaluate out of sample
    # pred_y = learner.query(test_x)  # get the predictions
    # rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    # c = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0,1]}")
