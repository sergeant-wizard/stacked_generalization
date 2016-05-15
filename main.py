from sklearn.metrics import log_loss
import numpy

from random_forest import RandomForest
from extra_trees import ExtraTrees
from gen_xgboost import Xgboost
from logistic_regression import LogisticRegression
from stacked_generalization import StackedGeneralization
from generalizer import Generalizer
from sklearn import datasets # for debugging with iris

n_classes = 39

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

def load_sfcc_data(class_index):
    year = 0 # deprecated
    train_data = numpy.load("../sfcc/train_{}.npy".format(year))
    test_data = numpy.load("../sfcc/test_{}.npy".format(year))
    train_target = (train_data[:, 0] == class_index).astype(int)
    train_data = train_data[:, 1:]
    return(train_data, train_target, test_data)

def print_logloss(generalizer_name, answer, prediction):
    loss = log_loss(answer, prediction, eps = 1.0E-15)
    print("log loss for {} : {}".format(generalizer_name, loss))

def train_partial(sg, generalizers, save_predictions = True, suffix = ''):
    layer0_partial_guess = numpy.array([generalizer.guess_partial(sg) for generalizer in generalizers])

    for generalizer_index, generalizer in enumerate(generalizers):
        if save_predictions:
            Generalizer.save_partial(generalizer.name() + suffix,
                                              layer0_partial_guess[generalizer_index].astype(numpy.float16))
        # guess = layer0_partial_guess[generalizer_index, :, numpy.unique(sg.train_target)].T
        # print_logloss(generalizer.name(), sg.train_target, guess)
    return(layer0_partial_guess)

def train_whole(sg, generalizers, save_predictions = True, suffix = ''):
    layer0_whole_guess = numpy.array([generalizer.guess_whole(sg) for generalizer in generalizers])
    for generalizer_index, generalizer in enumerate(generalizers):
        if save_predictions:
            Generalizer.save_whole(generalizer.name() + suffix,
                                   layer0_whole_guess[generalizer_index].astype(numpy.float16))
    return(layer0_whole_guess)

def load_layer0(filenames):
    layer0_partial_guess = numpy.array([Generalizer.load_partial(filename) for
                                        filename in filenames])
    layer0_whole_guess = numpy.array([Generalizer.load_whole(filename) for
                                        filename in filenames])
    return(layer0_partial_guess, layer0_whole_guess)

def load_with_suffix(filenames, class_index):
    return(load_layer0([filename + "_{}".format(class_index) for filename in filenames]))

def initialize_sg(class_index):
    # local classes
    n_classes = 2
    n_folds = 3
    # (train_data, train_target, test_data) = load_bio_data()
    # (train_data, train_target, test_data) = load_iris_data()
    (train_data, train_target, test_data) = load_sfcc_data(class_index)
    return(StackedGeneralization(
        n_folds,
        train_data,
        train_target,
        test_data,
        n_classes))

def layer0():
    for class_index in range(n_classes):
        sg = initialize_sg(class_index)
        generalizers = [ExtraTrees()]
        suffix = "_{}".format(class_index)
        layer0_partial_guess = train_partial(sg, generalizers, True, suffix)
        del layer0_partial_guess
        layer0_whole_guess = train_whole(sg, generalizers, True, suffix)

def layer1():
    # loading predictions
    for class_index in range(n_classes):
        print("processing class index{}".format(class_index))
        sg = initialize_sg(class_index)
        layer0_partial_guess, layer0_whole_guess = load_with_suffix(['extra_trees', 'xg_boost'], class_index)

        prediction = LogisticRegression().guess(
            numpy.hstack(layer0_partial_guess),
            sg.train_target,
            numpy.hstack(layer0_whole_guess))
        expanded = Generalizer.expand_all_classes(prediction,
                                                  numpy.unique(sg.train_target),
                                                  sg.n_classes)
        numpy.save("layer1_result_{}".format(class_index), expanded)

def concatenate():
    header = 'Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'
    fmt = '%d,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f'

    n_rows = 884262
    result = numpy.empty((n_rows, n_classes), dtype=numpy.float16)
    for class_index in range(n_classes):
        print("class_index: {}".format(class_index))
        class_result = numpy.load("layer1_result_{}.npy".format(class_index))
        result[:, class_index] = class_result[:, 1]

    id_column = numpy.array(range(result.shape[0]))

    numpy.savetxt(
        'predicted.csv',
        numpy.hstack((numpy.array([id_column]).T, result)),
        fmt=fmt,
        header=header,
        comments='')

if __name__ == "__main__":
    concatenate()
