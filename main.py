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

def load_sfcc_data():
    year = 0 # deprecated
    train_data = numpy.load("../sfcc/train_{}.npy".format(year))
    test_data = numpy.load("../sfcc/test_{}.npy".format(year))
    train_target = train_data[:, 0].astype(int)
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

def initialize_sg():
    # local classes
    n_folds = 3
    # (train_data, train_target, test_data) = load_bio_data()
    # (train_data, train_target, test_data) = load_iris_data()
    (train_data, train_target, test_data) = load_sfcc_data()
    return(StackedGeneralization(
        n_folds,
        train_data,
        train_target,
        test_data,
        n_classes))

def layer0():
    sg = initialize_sg()
    generalizers = [RandomForest()]
    layer0_partial_guess = train_partial(sg, generalizers, True)
    del layer0_partial_guess
    layer0_whole_guess = train_whole(sg, generalizers, True)

def layer1():
    # loading predictions
    sg = initialize_sg()
    layer0_partial_guess, layer0_whole_guess = load_layer0(['xg_boost'])

    prediction = LogisticRegression().guess(
        numpy.hstack(layer0_partial_guess),
        sg.train_target,
        numpy.hstack(layer0_whole_guess))
    expanded = Generalizer.expand_all_classes(prediction,
                                              numpy.unique(sg.train_target),
                                              sg.n_classes)
    numpy.save("layer1_result", expanded)

def format_for_kaggle():
    header = 'Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'
    fmt = '%d,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f'

    n_rows = 884262
    result = numpy.load("layer1_result.npy")

    id_column = numpy.array(range(result.shape[0]))

    numpy.savetxt(
        'predicted.csv',
        numpy.hstack((numpy.array([id_column]).T, result)),
        fmt=fmt,
        header=header,
        comments='')

if __name__ == "__main__":
    layer1()
