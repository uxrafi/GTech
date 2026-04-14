import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = []
        for _ in range(20):
            bag_learner = bl.BagLearner(lrl.LinRegLearner, {}, 20, False, verbose)
            self.learners.append(bag_learner)

    def author(self):
        return "urafi3"

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        predictions = []
        for learner in self.learners:
            pred = learner.query(points)
            predictions.append(pred)
        return np.mean(predictions, axis=0)
