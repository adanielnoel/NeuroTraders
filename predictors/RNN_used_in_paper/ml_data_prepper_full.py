"""
The MIT License (MIT)
Copyright (c) 2017 Paolo Rizzo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
import numpy as np
from numpy import concatenate

'''
Paolo Rizzo         10/16/2017

This is a series of methods to help convert a time series into a form that can be fed into a neural network using Keras API.

'''


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    From Jason Brownlee - https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning. In the following form:
        For instance, with 3 features and N timesteps:

       var1(t-n_in) var2(t-n_in) var3(t-n_in) var1(t-n_in+1) var2(t-n_in+1) var3 (t-n_in+1) ... var1(t) var2(t) var3(t)
     1
     2
     ...
     N

    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = DataFrame(concat(cols, axis=1))
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_lstm_data(dataframe, predict, del_inputs = None, timestep = 1, use_present_vars = False,n_in = 1,
                      n_out = 1, scaler = 'MinMax', feature_range = (-1,1),
                      train_interval_1 = (0.0,0.7), test_interval_1 = (0.7,1.0), train_interval_2 = (0.7,0.9),
                      test_interval_2 = (0.9,1.0)):

    if type(predict) is not str:
        raise TypeError('predict parameter must be of type string - please enter the column name of the dataframe '
                        'that you wish to predict')
    if del_inputs is not None:
        if type(del_inputs) is not list:
            raise TypeError('del_inputs must be a list of strings, containing the names of the columns of the dataframe'
                            'that you wish not to use as predictors')
        else:
            for i in range(len(del_inputs)):
                if type(del_inputs[i]) is not str:
                    raise TypeError('del_inputs must be a list of strings, containing the names of the columns of the '
                                    'dataframe that you wish not to use as predictors')
                else:
                    del dataframe[del_inputs[i]]

    # place the predict column last for easier indexing down the line
    cols = list(dataframe)
    cols[cols.index(predict)], cols[-1] = cols[-1], cols[cols.index(predict)]
    dataframe = dataframe.ix[:, cols]

    # Prep time series for machine learning
    values = dataframe.values.astype('float32')
    n_vars = values.shape[1]
    if scaler == 'Custom':
        warnings.warn('You are using a custom scaling method. This must be inverted manually later!')
        # Implement a custom scaling method below -- must be inverted manually later
        for i in range(len(values)):
            # DO SOMETHING TO SCALE
            # values[i] = values[i] - np.mean(values[i])
            # values[i] = values[i] / np.std(values[i])
            pass
        scaler = None
        scaled = values
    else:
        if scaler != 'MinMax' and scaler != 'StandardScaler':
            warnings.warn('The scaler '+str(scaler)+' is not supported, using MinMax instead ...\n', stacklevel=3)
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler == 'MinMax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler == 'StandardScaler':
            scaler = StandardScaler()
        scaled = scaler.fit_transform(values)

    processed = series_to_supervised(scaled, n_in=n_in, n_out=n_out)
    values_processed = processed.values

    # Partition the dataset into train and test intervals, defined by the user
    train_1 = values_processed[int(train_interval_1[0] * len(values_processed)):
                             int(train_interval_1[1] * len(values_processed))]
    test_1 = values_processed[int(test_interval_1[0] * len(values_processed)):
                            int(test_interval_1[1] * len(values_processed))]
    train_2 = values_processed[int(train_interval_2[0] * len(values_processed)):
                            int(train_interval_2[1] * len(values_processed))]
    test_2 = values_processed[int(test_interval_2[0] * len(values_processed)):
                            int(test_interval_2[1] * len(values_processed))]

    # Divide the train and test partitions into inputs and outputs
    if use_present_vars is False:
        # If the user wishes to only use variables before time t as features
        train_X_1, train_Y_1 = train_1[:, :-n_vars*n_out], train_1[:, -n_out]
        test_X_1, test_Y_1 = test_1[:, :-n_vars*n_out], test_1[:, -n_out]
        train_X_2 = train_2[:, :-n_vars*n_out]
        test_X_2 = test_2[:, :-n_vars*n_out]
    elif use_present_vars is True:
        # If the user wishes to use all of the variables (before and at time t) as features
        train_X_1, train_Y_1 = train_1[:, :-n_out], train_1[:, -n_out]
        test_X_1, test_Y_1 = test_1[:, :-n_out], test_1[:, -n_out]
        train_X_2 = train_2[:, :-n_out]
        test_X_2 = test_2[:, :-n_out]
    else:
        raise TypeError('use_present_vars parameter must be of type bool')

    # Reshape inputs into 3d format as required by LSTM: [samples, timesteps, features]
    train_X_1 = train_X_1.reshape((train_X_1.shape[0], timestep, train_X_1.shape[1]))
    test_X_1 = test_X_1.reshape((test_X_1.shape[0], timestep, test_X_1.shape[1]))
    train_X_2 = train_X_2.reshape((train_X_2.shape[0], timestep, train_X_2.shape[1]))
    test_X_2 = test_X_2.reshape((test_X_2.shape[0], timestep, test_X_2.shape[1]))

    return train_X_1, train_Y_1, test_X_1, test_Y_1, values, scaler, train_X_2, test_X_2


def descale_output(scaled_output, scaler, input_X):
    if input_X is not None:
        input_X = input_X.reshape(input_X.shape[0], input_X.shape[2])
        rest = input_X[:, :-1]
        descaled_output = concatenate((rest, scaled_output), axis=1)
    else:
        descaled_output = scaled_output
    descaled_output = scaler.inverse_transform(descaled_output)
    descaled_output = descaled_output[:, -1]
    return descaled_output