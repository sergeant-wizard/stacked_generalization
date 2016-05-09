from os import path
import pickle
import numpy
from stacked_generalization import StackedGeneralization

class Generalizer:
    def guess_partial(self, sg):
        raise NotImplementedError

    def guess_whole(self, sg):
        raise NotImplementedError

class PerpetualGeneralizer(Generalizer):
    def __init__(self, filename):
        self.partial = None # FIXME
        self.whole = None

    def guess_partial(self, _):
        return(self.partial)

    def guess_whole(self, _):
        return(self.whole)

class EphemeralGeneralizer(Generalizer):
    def __init__(self, from_file = False):
        if from_file:
            self.load_model()
        else:
            self.model = None

    def guess_partial(self, sg):
        assert(isinstance(sg, StackedGeneralization))
        generalizer_prediction = numpy.empty((0, sg.n_classes))
        for train_index, test_index in sg.skf:
            self.train(sg.train_data[train_index],
                       sg.train_target[train_index])
            generalizer_prediction = numpy.vstack((
                generalizer_prediction,
                self.predict(sg.train_data[test_index])))

        reorder_index = [test_index for _, test_indices in sg.skf for test_index in test_indices]
        return(generalizer_prediction[reorder_index, :])

    def guess_whole(self, sg):
        assert(isinstance(sg, StackedGeneralization))
        return(self.guess(sg.train_data, sg.train_target, sg.test_data))

    def guess(self, input_data, input_target, test_data):
        self.train(input_data, input_target)
        return(self.predict(test_data))

    def train(self, data, label):
        if self.model:
            print("overwriting model")
        self._train(data, label)

    def _train(self, data, label):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save_model(self):
        pickle.dump(self.model, open(self.path(), "wb"))

    def load_model(self):
        self.model = pickle.load(open(self.path(), "rb"))

    def path(self):
        return(path.join("data", self.name() + ".p"))

    def assert_model(self):
        assert(self.model != None)

