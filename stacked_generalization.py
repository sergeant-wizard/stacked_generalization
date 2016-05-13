from sklearn.cross_validation import StratifiedKFold
import numpy

class StackedGeneralization:
    def __init__(self, n_folds, train_data, train_target, test_data, n_classes):
        self.n_folds = n_folds
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.n_classes = n_classes

        self.skf = StratifiedKFold(y=train_target, n_folds=n_folds)

