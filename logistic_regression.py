from sklearn.linear_model import LogisticRegression as LR
from generalizer import EphemeralGeneralizer

class LogisticRegression(EphemeralGeneralizer):
    def name(self):
        return("logistic_regression")

    def train(self, data, label):
        rfc = LR()
        self.model = rfc.fit(data, label)

    def predict(self, data):
        return(self.model.predict_proba(data))
