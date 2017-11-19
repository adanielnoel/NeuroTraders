import numpy as np
import collections
import math as m
import pandas as pd


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

    @property
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

    def fit(self, data_array):
        assert isinstance(data_array, collections.Iterable)
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)

        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)

        # Make array where only the last dims are left as columns
        dims_arr = data_array.reshape((np.prod(np.array(data_array.shape[:-1])), data_array.shape[-1]))

        self.ncols = data_array.shape[-1]
        self.min = np.array([col.min() for col in dims_arr.transpose()])
        self.max = np.array([col.max() for col in dims_arr.transpose()])

        return self

    def apply(self, data_array):
        assert isinstance(data_array, collections.Iterable)
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)

        return_1d = False
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
            return_1d = True

        assert data_array.shape[-1] == self.ncols

        # Make array where only the last dims are left as columns
        dims_arr = data_array.reshape((np.prod(np.array(data_array.shape[:-1])), data_array.shape[-1]))
        dims_arr = ((dims_arr - self.min) / self.scale.transpose()) + self.map_range[0]

        if return_1d:
            dims_arr.reshape(len(data_array))
        else:
            return dims_arr.reshape(data_array.shape)

    def revert(self, data_array):
        assert isinstance(data_array, collections.Iterable)
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)

        return_1d = False
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
            return_1d = True

        assert data_array.shape[-1] == self.ncols

        dims_arr = data_array.reshape((np.prod(np.array(data_array.shape[:-1])), data_array.shape[-1]))
        scale = 1.0 / self.scale
        dims_arr = ((dims_arr - self.map_range[0]) / scale.transpose()) + self.min

        if return_1d:
            return dims_arr.reshape(len(data_array))
        else:
            return dims_arr.reshape(data_array.shape)


def format_time_series(dataframe, input_time_steps, batch_size, input_cols, target_cols=None):
    """
    Formats data for LSTM.
    :param dataframe: dataframe with columns for each feature and a row for each point in time
    :param input_time_steps: number of time steps in the resulting LSTM input data
    :param batch_size: number of samples per batch.
    :param input_cols: List of names of the columns to use as inputs
    :param target_cols: List of names of the columns to use as targets
    :param target_time_steps: Number of time steps to predict
    :return: a tuple with an array of inputs, an array of outputs and a list of lists
             of dates that the network will be predicting (shape=(number_of_batches, batch_size, target_time_steps)).
             Note that if target_cols is None (default) then the array of outputs will be None.
    """
    assert isinstance(dataframe, pd.DataFrame)
    formatted_input_df = pd.DataFrame(columns=input_cols)
    formatted_target_df = pd.DataFrame(columns=target_cols)
    output_dates = []

    batch_size = batch_size if batch_size else len(dataframe)
    final_shift = 2 if target_cols else 1
    batch_size = min(batch_size, (len(dataframe) - 1) // input_time_steps - final_shift)
    n_batches = len(dataframe) - input_time_steps * (2 + batch_size)

    # Format inputs
    for i in range(n_batches):  # Make input batches
        batch_dates = []
        for j in range(batch_size):  # Make each sample in the batch
            batch_shift = j * input_time_steps
            # Note that LSTM state is kept within a batch, so the samples must not overlap
            # Instead, samples from different batches can overlap
            formatted_input_df = formatted_input_df.append(dataframe[input_cols].iloc
                                                           [i + batch_shift:
                                                            i + batch_shift + input_time_steps])

            if target_cols is not None:
                formatted_target_df = formatted_target_df.append(dataframe[target_cols].iloc
                                                                 [i + batch_shift + input_time_steps])

            batch_dates.append(dataframe.index[i + batch_shift + input_time_steps])
        output_dates.append(batch_dates)

    # if target_time_steps == 1:
    #     output_dates = list(map(list, zip(*output_dates)))

    input_array = formatted_input_df.values.reshape(n_batches, batch_size, input_time_steps, len(input_cols))
    if target_cols:
        target_array = formatted_target_df.values.reshape(n_batches, batch_size, len(target_cols))
    else:
        target_array = None

    return input_array, target_array, output_dates


def split_data(data, *ratios):
    assert sum(ratios) == 1.0, "Ratios must sum up to 1"
    splits = []
    for i in range(len(ratios)):
        start = int(sum(ratios[:i]) * len(data))
        end = int(start + ratios[i] * len(data))
        splits.append(data.iloc[start:end])
    return splits


if __name__ == "__main__":
    a = [1, 3, 2]
    scaler = Scaler(map_range=(-2.0, 2.0))
    scaler.fit(a)
    b = scaler.apply(a)
    c = scaler.revert(b)
    print(a)
    print(b)
    print(c)







