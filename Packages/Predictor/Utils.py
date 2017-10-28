import numpy as np
import collections
import math as m


def softmax(vector):
    """
    Softmax function that favours the highest score
    :param vector: input vector of scores
    :return: output of the softmax
    """
    if sum(vector) == 0.0:
        return np.array([1.0 / len(vector)]*len(vector))
    vector = np.array(vector) / (0.3 * sum(vector))
    e_scaled = []
    for value in vector:
        e_scaled.append(m.exp(value))
    sum_e = sum(e_scaled)

    return np.array(e_scaled) / sum_e


class Scaler(object):
    def __init__(self, map_range=(0.0, 1.0)):
        self.min = np.zeros(1)
        self.max = np.zeros(1)
        self.mapped_mean = np.zeros(1)
        self.mapped_std = np.zeros(1)
        self._map_range = map_range
        self.ncols = 1

    @property
    def map_size(self):
        return self._map_range[1] - self._map_range[0]

    def scale(self):
        return (self.max - self.min) / np.array([self.map_size]*self.ncols)

    @property
    def map_range(self):
        return self._map_range

    @map_range.setter
    def map_range(self, value):
        assert len(value) == 2
        assert value[1] > value[0]
        self._map_range = value

    def fit(self, array):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        self.ncols = array.shape[1]
        self.min = np.array([col.min() for col in array.transpose()])
        self.max = np.array([col.max() for col in array.transpose()])

        temp = self.apply(array, standarize=False)
        self.mapped_mean = np.array([col.mean() for col in temp.transpose()])
        self.mapped_std = np.array([col.std() for col in temp.transpose()])
        return self

    def apply(self, array, standarize=True):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        return_1d = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            return_1d = True
        assert array.shape[1] == self.ncols

        array = ((array - self.min) / self.scale().transpose()) + self.map_range[0]

        if standarize:
            array = (array - self.mapped_mean)

        if return_1d:
            return array[:, 0]
        else:
            return array

    def revert(self, array, standarize=True):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        return_1d = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            return_1d = True
        assert array.shape[1] == self.ncols

        if standarize:
            array = (array) + self.mapped_mean

        scale = 1.0 / self.scale()
        array = ((array - self.map_range[0]) / scale.transpose()) + self.min

        if return_1d:
            return array[:, 0]
        else:
            return array


if __name__ == "__main__":
    a = [1, 3, 2]
    scaler = Scaler(map_range=(-1.0, 1.0))
    scaler.fit(a)
    b = scaler.apply(a)
    c = scaler.revert(b)
    print(a)
    print(b)
    print(c)







