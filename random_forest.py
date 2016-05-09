from sklearn.ensemble import RandomForestClassifier
from generalizer import EphemeralGeneralizer

class RandomForest(EphemeralGeneralizer):
    def name(self):
        return("random_forest")

    def train(self, data, label):
        rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
        self.model = rfc.fit(data, label)

    def predict(self, data):
        return(self.model.predict_proba(data))
