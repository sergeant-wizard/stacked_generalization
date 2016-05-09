from sklearn.linear_model import LinearRegression as LR
from generalizer import Generalizer

class LinearRegression(Generalizer):
    def name(self):
        return("linear_regression")

    def train(self, data, label):
        rfc = LR()
        self.model = rfc.fit(data, label)

    def predict(self, data):
        return(self.model.predict(data))
