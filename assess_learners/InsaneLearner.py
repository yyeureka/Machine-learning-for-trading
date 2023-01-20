import numpy as np
import LinRegLearner as lrl
import BagLearner as bl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(lrl.LinRegLearner, {}, 20, False, False) for _ in range(20)]
        self.verbose = verbose
    def author(self):
        return "jyu497"
    def add_evidence(self, data_x, data_y):
        for learner in self.learners: learner.add_evidence(data_x, data_y)
    def query(self, points):
        return np.mean([learner.query(points) for learner in self.learners], axis=0)