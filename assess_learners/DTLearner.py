import numpy as np


class DTLearner(object):
    """
    This is a Decision Tree Learner.

    :param leaf_size: Defines the maximum number of samples to be aggregated at a leaf
    :rtype: int

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :rtype: str
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.learner = None
        self.height = 0

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

        self.learner = self.build_tree(data_x, data_y, 1)

        if self.verbose:
            print("Decision tree shape: ", self.learner.shape)
            print("Decision tree details:")
            print(self.learner)

    def build_tree(self, data_x, data_y, height):
        self.height = max(self.height, height)

        leaf = np.array([[-1, np.mean(data_y), np.nan, np.nan]])  # TODO: mean? most common? random?

        if data_x.shape[0] <= self.leaf_size:
            return leaf
        if np.all(data_y == data_y[0]):
            return leaf

        # Get best feature i and split value
        # corrs = np.zeros(data_x.shape[1])
        # for i in range(data_x.shape[1]):
        #     corrs[i] = np.absolute(np.corrcoef(data_x[:, i], data_y)[0, 1])
        corrs = np.abs(np.corrcoef(data_x, y=data_y, rowvar=False))[:-1, -1]
        best_i = np.nanargmax(corrs)
        split_val = np.median(data_x[:, best_i])

        left_mask = data_x[:, best_i] <= split_val
        if np.all(left_mask) or np.all(~left_mask):  # Avoid infinite loop
            return leaf

        left_tree = self.build_tree(data_x[left_mask], data_y[left_mask], height + 1)
        right_tree = self.build_tree(data_x[~left_mask], data_y[~left_mask], height + 1)
        root = np.array([[best_i, split_val, 1, left_tree.shape[0] + 1]])

        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        preds = np.zeros(len(points))

        for i, point in enumerate(points):
            preds[i] = self.predict(point, 0)

        return preds

    def predict(self, point, node):
        best_i = int(self.learner[node, 0])
        split_val = self.learner[node, 1]

        if -1 == best_i:  # leaf node
            return split_val
        if point[best_i] <= split_val:
            return self.predict(point, node + int(self.learner[node, 2]))
        else:
            return self.predict(point, node + int(self.learner[node, 3]))

