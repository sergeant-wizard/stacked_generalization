from sklearn.ensemble import RandomForestClassifier
from generalizer import Generalizer

class RandomForest(Generalizer):
    def name(self):
        return("random_forest")

    def train(self, data, label):
        rfc = RandomForestClassifier(n_estimators=32, n_jobs=-1, criterion='gini')
        self.model = rfc.fit(data, label)

    def predict(self, data):
        return(self.model.predict_proba(data))
