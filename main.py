from random_forest import RandomForest
from extra_trees import ExtraTrees
from stacked_generalization import StackedGeneralization
import numpy

# from sklearn import datasets

def load_bio_data():
    raw_train = numpy.loadtxt('train.csv', delimiter=',', skiprows=1)
    train_target = raw_train[:, 0]
    train_data = raw_train[:, 1:]
    test_data = numpy.loadtxt('test.csv', delimiter=',', skiprows=1,
                              dtype=numpy.int16)
    return(train_data, train_target, test_data)

def main():
    n_folds = 3
    (train_data, train_target, test_data) = load_bio_data()
    generalizers = [RandomForest(), ExtraTrees()]

    sg = StackedGeneralization(n_folds, train_data, train_target, test_data)
    layer0_partition_guess = StackedGeneralization.merge([sg.guess_layer0_with_partition(generalizer) for
                              generalizer in generalizers])
    layer0_whole_guess = StackedGeneralization.merge([sg.guess_layer0_with_whole(generalizer) for
                          generalizer in generalizers])

    result = StackedGeneralization.guess_layer1(
        RandomForest(),
        layer0_partition_guess,
        train_target,
        layer0_whole_guess)

    print(result)

if __name__ == "__main__":
    main()
