from os import path
import numpy
from stacked_generalization import StackedGeneralization

class Generalizer:
    def __init__(self):
        self.model = None

    def name(self):
        raise NotImplementedError

    def guess_partial(self, sg):
        assert(isinstance(sg, StackedGeneralization))
        generalizer_prediction = numpy.zeros((sg.train_data.shape[0], sg.n_classes))
        for train_index, test_index in sg.skf:
            self.train(sg.train_data[train_index],
                       sg.train_target[train_index])
            # fold_prediction may have less # of classes than sg.n_classes
            limited_classes = numpy.unique(sg.train_target[train_index])
            fold_prediction = self.predict(sg.train_data[test_index])
            expanded = Generalizer.expand_all_classes(fold_prediction, limited_classes, sg.n_classes)
            generalizer_prediction[test_index, :] = expanded

        return(generalizer_prediction)

    def guess_whole(self, sg):
        assert(isinstance(sg, StackedGeneralization))
        prediction = self.guess(sg.train_data, sg.train_target, sg.test_data)
        limited_classes = numpy.unique(sg.train_target)
        return(Generalizer.expand_all_classes(prediction, limited_classes, sg.n_classes))

    def guess(self, input_data, input_target, test_data):
        self.train(input_data, input_target)
        return(self.predict(test_data))

    def train(self, data, label):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    @staticmethod
    def expand_all_classes(thin_matrix, limited_classes, num_classes):
        assert(thin_matrix.shape[1] == len(limited_classes))
        ret = numpy.zeros((thin_matrix.shape[0], num_classes))
        ret[:, limited_classes] = thin_matrix
        return(ret)

    @staticmethod
    def load_partial(name):
        return(numpy.load(Generalizer._partial_path(name)))

    @staticmethod
    def load_whole(name):
        return(numpy.load(Generalizer._whole_path(name)))

    @staticmethod
    def save_partial(name, prediction):
        numpy.save(Generalizer._partial_path(name), prediction)

    @staticmethod
    def save_whole(name, prediction):
        numpy.save(Generalizer._whole_path(name), prediction)

    @staticmethod
    def _partial_path(name, has_ext = True):
        return(path.join("data", "partial", Generalizer._add_ext(name, has_ext)))

    @staticmethod
    def _whole_path(name, has_ext = True):
        return(path.join("data", "whole", Generalizer._add_ext(name, has_ext)))

    @staticmethod
    def _add_ext(name, has_ext):
        if has_ext:
            return(name + '.npy')
        else:
            return(name)

