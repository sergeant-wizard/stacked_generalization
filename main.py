from os import path
import pickle

class Generalizer:
    def __init__(self, from_file = False):
        if from_file:
            self.load_model()
        else:
            self.model = None

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

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
class RandomForest(Generalizer):
    def name(self):
        return("random_forest")

    def train(self, data, label):
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
        self.model = rfc.fit(data, label)

    def predict(self, data):
        return(self.model.predict(data))

class ExtraTrees(Generalizer):
    def name(self):
        return("extra_trees")

    def train(self, data, label):
        et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
        self.model= et.fit(data, label)

    def predict(self, data):
        return(self.model.predict(data))

from sklearn import datasets
import numpy
from sklearn.cross_validation import StratifiedKFold

def guess_layer0_with_partition(generalizers, n_folds, train_data, train_target):
    skf = StratifiedKFold(y=train_target, n_folds=n_folds)
    layer1_input = numpy.empty((0, len(train_data)))
    for generalizer in generalizers:
        generalizer_prediction = numpy.array([])
        for train_index, test_index in skf:
            generalizer.train(train_data[train_index], train_target[train_index])
            generalizer_prediction = numpy.append(
                generalizer_prediction,
                generalizer.predict(train_data[test_index]))

        layer1_input = numpy.vstack((
            layer1_input,
            generalizer_prediction))

    reorder_index = [test_index for _, test_indices in skf for test_index in test_indices]
    return(layer1_input[:, reorder_index].T)

def guess_layer0_with_whole(generalizers, n_folds, train_data, train_target, test_data):
    layer1_input = numpy.empty((0, len(train_data)))
    for generalizer in generalizers:
        generalizer.train(train_data, train_target)
        generalizer_prediction = generalizer.predict(test_data)
        layer1_input = numpy.vstack((
            layer1_input,
            generalizer_prediction))

    return(layer1_input.T)

def main():
    n_folds = 3
    iris = datasets.load_iris()
    train_data = iris.data
    test_data = iris.data # FIXME: for simplification
    train_target = numpy.array(iris.target)
    generalizers = [RandomForest(), ExtraTrees()]

    layer1_train_data = guess_layer0_with_partition(
        generalizers, n_folds, train_data, train_target)
    layer1_generalizer = RandomForest()
    layer1_generalizer.train(layer1_train_data, train_target)

    layer1_test_input = guess_layer0_with_whole(generalizers, n_folds, train_data, train_target, test_data)
    result = layer1_generalizer.predict(layer1_test_input)
    print(result)

if __name__ == "__main__":
    main()
