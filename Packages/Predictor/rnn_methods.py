import numpy as np
import json
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
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

# each row is a different roll, built up like:
# [[ var1(t-10) var2(t-10) var1(t-9)...var2(t)]
# In the next row var(t-10) <-- var(t-9) as we
# roll in time


# invert scaling for actual
#test_Y = test_Y.reshape((len(test_Y), 1))
#test_Y = concatenate((test_Y,dummy), axis=1)
#inv_y = scaler.inverse_transform(test_Y)
#inv_y = inv_y[:, 0]

'''
Paolo Rizzo         NeuroTraders        14/10/2017

Here we create an RNN with LSTM cells capable of predicting the stock price of the next day based on historical data
and sentiment analysis conducted on related news issues.

There are actually two RNN networks that together make up the entire stock prediction model. The first RNN predicts the
stock price, the second neural net is trained on the output of the first net to predict the confidence level (risk) on
the first prediction

Structure of the RNN (to be updated):

- try: stacked LSTM layers
- IT IS IMPERATIVE THAT THE TOTAL NUMBER OF TUNABLE PARAMETERS MUST BE LESS THAN THE NUMBER OF TRAINING POINTS - OTHERWISE
OVERFITTING AND THE MODEL WILL PERFORM VERY WELL ON TRAINING DATA BUT VERY POORLY ON THE TEST DATA AS IT WILL NOT BE ABLE TO
GENERALIZE, BUT ONLY MEMORIZE
- On the other hand, if too few tunable parameters - underfitting - bad training performance and also bad test performance

'''

'''
###################### ------------ THE SECOND NEURAL NETWORK ------------------------------------ #####################
#---------------------------------------------------------------------------------------------------------------------#


# FOR NET 2
look_back2 = 50              # very important for RNN applications -- consider tuning with Bayesian Optimizer
epochs2 = 100
batch_size2 = 32
learning_rate2 = 0.001       # 0.001 is default for Adam optimizer -- usually an ok learning rate

##### ------ THE SECOND RNN - PREDICTS THE RISK

# design network -- 3 hidden layers (stacked LSTM layers)
model2 = Sequential()

# First Hidden layer (LSTM)
model2.add(LSTM(50, return_sequences = True, input_shape=(train_X.shape[1], train_X.shape[2])))  # for stateful, must specify batch size

# Second Hidden layer (LSTM)
model2.add(LSTM(50, return_sequences = True))    # essentially, if you want to stack another LSTM layer below, you must
                                                # set return_sequences = True

# Third Hidden layer (LSTM)
model2.add(LSTM(50))

# Output Layer
model2.add(Dense(1))

adam2 = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)    # this allows you to set the learning rate yourself
                                                                                                 # instead of using the default one
model2.compile(loss='mae', optimizer=adam2)

# fit network
history2 = model1.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
'''
