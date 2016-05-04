from generalizer import Generalizer
from sklearn.ensemble import ExtraTreesClassifier

class ExtraTrees(Generalizer):
    def name(self):
        return("extra_trees")

    def train(self, data, label):
        et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
        self.model= et.fit(data, label)

    def predict(self, data):
        return(self.model.predict(data))
