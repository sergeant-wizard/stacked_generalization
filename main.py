from sklearn.metrics import log_loss
import numpy

from random_forest import RandomForest
from extra_trees import ExtraTrees
from logistic_regression import LogisticRegression
from stacked_generalization import StackedGeneralization
from sklearn import datasets # for debugging with iris

def load_bio_data():
    raw_train = numpy.loadtxt('train.csv', delimiter=',', skiprows=1)
    train_target = raw_train[:, 0]
    train_data = raw_train[:, 1:]
    test_data = numpy.loadtxt('test.csv', delimiter=',', skiprows=1)
    return(train_data, train_target, test_data)

def load_iris_data():
    iris = datasets.load_iris()
    train_data = iris.data
    train_target = iris.target
    test_data = iris.data # for simplicity
    return(train_data, train_target, test_data)

def main():
    n_folds = 3
    (train_data, train_target, test_data) = load_bio_data()
    # (train_data, train_target, test_data) = load_iris_data()
    generalizers = [RandomForest(), ExtraTrees()]
    id_column = numpy.array(range(len(test_data))) + 1

    sg = StackedGeneralization(n_folds, train_data, train_target, test_data)
    layer0_partition_guess = numpy.array([generalizer.guess_partial(sg) for generalizer in generalizers])

    # not necessary, but nice to have for tuning each layer0 classifiers
    for generalizer_index, generalizer in enumerate(generalizers):
        print("log loss for {} : {}".format(
            generalizer.name(),
            log_loss(train_target, layer0_partition_guess[generalizer_index, :, :])
        ))
        numpy.savetxt(
            '{}.csv'.format(generalizer.name()),
            numpy.array([id_column, generalizer.predict(test_data)[:, 1]]).T,
            fmt='%d,%1.6f',
            header='id, activation')

    layer0_whole_guess = numpy.array([generalizer.guess_whole(sg) for generalizer in generalizers])

    result = LogisticRegression().guess(
        numpy.hstack(layer0_partition_guess),
        train_target,
        numpy.hstack(layer0_whole_guess))

    numpy.savetxt(
        'predicted.csv',
        numpy.array([id_column, result[:, 1]]).T,
        fmt='%d,%1.6f',
        header='id, activation')

if __name__ == "__main__":
    main()
