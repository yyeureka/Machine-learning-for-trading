import numpy as np


class BagLearner(object):
    """
    This is a Bootstrap Aggregation Learner.

    :param leaf_size: Defines the maximum number of samples to be aggregated at a leaf
    :rtype: int

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :rtype: str
    """

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        """
        Constructor method
        """
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))

        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "jyu497"

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        for learner in self.learners:
            sample_i = np.random.choice(data_x.shape[0], data_x.shape[0], replace=True)
            learner.add_evidence(data_x[sample_i], data_y[sample_i])

        if self.boost:  # TODO
            pass

        if self.verbose:
            pass

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        preds = np.zeros([self.bags, len(points)])

        for i, learner in enumerate(self.learners):
            preds[i] = learner.query(points)

        return np.mean(preds, axis=0)

