from sklearn.metrics import log_loss
import numpy

from random_forest import RandomForest
from extra_trees import ExtraTrees
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

    sg = StackedGeneralization(n_folds, train_data, train_target, test_data)
    layer0_partition_guess = numpy.array([sg.guess_layer0_with_partition(generalizer) for
                              generalizer in generalizers])

    # not necessary, but nice to have for tuning each layer0 classifiers
    for generalizer_index, generalizer in enumerate(generalizers):
        print("log loss for {} : {}".format(
            generalizer.name(),
            log_loss(train_target, layer0_partition_guess[generalizer_index, :, :])
        ))

    layer0_whole_guess = numpy.array([sg.guess_layer0_with_whole(generalizer) for
                          generalizer in generalizers])

    result = StackedGeneralization.guess_layer1(
        RandomForest(),
        numpy.hstack(layer0_partition_guess),
        train_target,
        numpy.hstack(layer0_whole_guess))

    id_column = numpy.array(range(len(result))) + 1
    numpy.savetxt(
        'predicted.csv',
        numpy.array([id_column, result[:, 1]]).T,
        fmt='%d,%1.6f',
        header='id, activation')

if __name__ == "__main__":
    main()
