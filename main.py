from random_forest import RandomForest
from extra_trees import ExtraTrees
from stacked_generalization import StackedGeneralization

from sklearn import datasets

def main():
    n_folds = 3
    iris = datasets.load_iris()
    train_data = iris.data
    test_data = iris.data # FIXME: for simplification
    train_target = iris.target
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
