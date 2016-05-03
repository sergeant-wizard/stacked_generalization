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

# class StackGeneralization:
#     def __init__(self, layer0_generalizers):
#         self.layer0_generalizers = layer0_generalizers
# 
#     def train(self, data, target):

from sklearn import datasets
import numpy
from sklearn.cross_validation import StratifiedKFold
def main():
    n_folds = 3
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    generalizers = [RandomForest(), ExtraTrees()]

    skf = StratifiedKFold(y=iris.target, n_folds=n_folds)
    layer1_input = numpy.empty((0, len(data)))
    for generalizer in generalizers:
        generalizer_prediction = numpy.array([])
        for train_index, test_index in skf:
            generalizer.train(data[train_index], target[train_index])
            generalizer_prediction = numpy.append(
                generalizer_prediction,
                generalizer.predict(data[test_index]))

        layer1_input = numpy.vstack((
            layer1_input,
            generalizer_prediction))

    layer1_input = layer1_input.T




    for generalizer in generalizers:
        generalizer.train(iris.data[0:124], iris.target[0:124])
    predicted = numpy.array(
        [generalizer.predict(iris.data[125:149]) for generalizer in generalizers])
    predicted

if __name__ == "__main__":
    main()
