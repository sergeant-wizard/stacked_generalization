from sklearn.metrics import log_loss
import numpy

from random_forest import RandomForest
from extra_trees import ExtraTrees
from logistic_regression import LogisticRegression
from stacked_generalization import StackedGeneralization
from generalizer import Generalizer
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

def load_sfcc_data():
    train_data = numpy.load('../scff/train.npy')
    test_data = numpy.load('../scff/test.npy')
    train_target = train_data[:, 0]
    train_data = train_data[:, 1:].astype(int)
    return(train_data, train_target, test_data)

def train_partial(sg, generalizers, save_predictions = True):
    layer0_partial_guess = numpy.array([generalizer.guess_partial(sg) for generalizer in generalizers])

    for generalizer_index, generalizer in enumerate(generalizers):
        if save_predictions:
            Generalizer.save_partial(generalizer.name(),
                                              layer0_partial_guess[generalizer_index])
        print("log loss for {} : {}".format(
            generalizer.name(),
            log_loss(sg.train_target, layer0_partial_guess[generalizer_index, :, :])
        ))
    return(layer0_partial_guess)

def train_whole(sg, generalizers, save_predictions = True):
    layer0_whole_guess = numpy.array([generalizer.guess_whole(sg) for generalizer in generalizers])
    for generalizer_index, generalizer in enumerate(generalizers):
        if save_predictions:
            Generalizer.save_whole(generalizer.name(),
                                   layer0_whole_guess[generalizer_index])
    return(layer0_whole_guess)

def load_layer0(filenames):
    layer0_partial_guess = numpy.array([Generalizer.load_partial(filename) for
                                        filename in filenames])
    layer0_whole_guess = numpy.array([Generalizer.load_whole(filename) for
                                        filename in filenames])
    return(layer0_partial_guess, layer0_whole_guess)

def initialize_sg():
    n_folds = 3
    # (train_data, train_target, test_data) = load_bio_data()
    # (train_data, train_target, test_data) = load_iris_data()
    (train_data, train_target, test_data) = load_sfcc_data()
    return(StackedGeneralization(n_folds, train_data, train_target, test_data))

def main():
    sg = initialize_sg()
    # for ad-hoc training
    # generalizers = [ExtraTrees()]
    # layer0_partial_guess = train_partial(sg, generalizers)
    # del layer0_partial_guess
    # layer0_whole_guess = train_whole(sg, generalizers)
    # return

    # loading predictions
    layer0_partial_guess, layer0_whole_guess = load_layer0(['random_forest',
                                                            'extra_trees'])

    result = LogisticRegression().guess(
        numpy.hstack(layer0_partial_guess),
        sg.train_target,
        numpy.hstack(layer0_whole_guess))

    id_column = numpy.array(range(len(sg.test_data))) + 1
    print(1)
    header = 'Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS'
    print(2)

    numpy.savetxt(
        'predicted.csv',
        numpy.array([id_column, result[:, 1]]).T,
        fmt='%d,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f',
        header=header,
        comments='')

if __name__ == "__main__":
    main()
