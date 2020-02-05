#todo after algorithm works replace numpy array to csr matrix for performance
from scipy.sparse import csr_matrix

import numpy as np


class HistogramDataSet:

    def __init__(self, xhi_size, data_dim, data_set=None, create_random=False, tabular_data_set=False, histogram_data_set=False):
        if create_random:
            self.histogram = self.__create_uniform_distribution_histogram(xhi_size, data_dim)
        elif tabular_data_set:
            self.histogram = self.__tabular_data_set_to_histogram_data_set(data_set, xhi_size)
        elif histogram_data_set:
            self.histogram = data_set


    def __tabular_data_set_to_histogram_data_set(self, tabular_data_set, xhi_size):
        hist_data_set = np.zeros(xhi_size)
        for row in tabular_data_set:
            num = 0
            for j, e in reversed(list(enumerate(row))):
                num += int(e) * (2 ** j)
            hist_data_set[num] += 1.0
        # normalized_hist_data = self.__normalize(hist_data_set) todo figure out if the algorithm assumes normalized data
        return hist_data_set


    def __create_uniform_distribution_histogram(self, xhi_size, data_dim):
        x_0_hist_data_set = np.array([1 / float(xhi_size)] * (2 ** data_dim))
        return x_0_hist_data_set


    def evaluate(self, query):
        return np.dot(self.histogram, query)


    def __normalize(self, histogram):
        return histogram * (1 / sum(histogram))
