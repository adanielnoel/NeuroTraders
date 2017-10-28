from math import sqrt
from numpy import concatenate
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
import numpy as np
from Packages.Predictor.rnn_methods import *

### Tentative data for Apple stock
dataset = read_csv('./../../Database/AAPL/record.csv', header=0, index_col=0)

# Delete all undesired features
del dataset['price_m_ave_50']
del dataset['price_m_ave_100']

#del dataset['uncertainty_sentiment']
#del dataset['news_volume']

# Which feature is to be predicted? - place this feature as the last column for easier indexing
predict = 'relative_change'
cols = list(dataset)
cols[cols.index(predict)] , cols[-1] = cols[-1] , cols[cols.index(predict)]
dataset = dataset.ix[:,cols]
print("Building a deep RNN with LSTM cells to predict:", predict)
# print(dataset)
offset = 0              # to get rid of NaN instances
np.random.seed(19)      # Fix random seed so results can be reproduced

##### ------ HYPERPARAMETERS -- to be tuned

# FOR NET 1
look_back1 = 1               # very important for RNN applications -- consider tuning with Bayesian Optimizer
epochs1 = 100
batch_size1 = 32
learning_rate1 = 0.001       # 0.001 is default for Adam optimizer -- usually an ok learning rate
train_percentage1 = 0.6      # Dataset must be split into three parts
train_percentage2 = 0.3
test_percentage2 = 0.1
n_features = dataset.shape[1]     # n. of features flowing into first layer of the net


##### ------ PREP DATASET FOR MACHINE LEARNING
values = dataset.values.astype('float32')[offset:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)                                     # normalize features
processed = series_to_supervised(scaled, n_in = look_back1, n_out = 1)

#print("Here is the data as outputted by the supervised function:")
#print(processed.head())

# split into train and test datasets
values_proc = processed.values
# train1 is the first partition of the dataset
train1 = values_proc[:int(train_percentage1*len(values_proc)), :]
# test1 is the union of the second and third partitions of the dataset
test1 = values_proc[int(train_percentage1*len(values_proc)):,:]


# separate into inputs and outputs
train_X, train_Y = train1[:, :-n_features] , train1[:, -1]                # train_X excludes all variables at time t b/c
test_X, test_Y = test1[:, :-n_features], test1[:, -1]                     # these are not used as predictors
                                                                          # train_Y is the last variable (column) at time t
print("Input shape", train_X.shape, "Output shape", train_Y.shape)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

##### ------ THE FIRST RNN - PREDICTS STOCK PRICE

# network design -- 3 hidden layers (stacked LSTM layers)
model1 = Sequential()

# First Hidden layer (LSTM)
model1.add(LSTM(40, return_sequences = True,
                input_shape=(train_X.shape[1], train_X.shape[2])))  # for stateful, must specify batch size
# Second Hidden layer (LSTM)
#model1.add(LSTM(30, return_sequences=True))  # essentially, if you want to stack another LSTM layer below, you must
                                              # set return_sequences = True
# Second Hidden layer (LSTM)
model1.add(LSTM(40))
# Output Layer
model1.add(Dense(1))
adam1 = optimizers.Adam(lr=learning_rate1, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                        decay=0.0)  # this allows you to set the learning rate yourself
# instead of using the default one
model1.compile(loss='mae', optimizer=adam1)
# fit network
print('\n Training Sequence for RNN 1 (Stock Price Predictor) Initiated!\n')
history1 = model1.fit(train_X, train_Y, epochs=epochs1, batch_size=batch_size1, validation_data=(test_X, test_Y),
                      verbose=1, shuffle=False)
print('\n RNN 1 has been trained and tested! \n')

##### ------ PREDICT USING THE MODEL OUTPUT

# make a prediction
yhat = model1.predict(test_X)
dummy = np.ones((len(yhat),n_features-1))

# invert scaling for prediction
inv_yhat = concatenate((yhat, dummy), axis=1)       # Verify that this step is actually legit - if so: me = genius
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]


# calculate RMSE by comparing with actual closing prices
target_outputs1 = values[:,-1][len(train1):len(train1)+len(test1)]
rmse = sqrt(mean_squared_error(target_outputs1, inv_yhat))
print('Test RMSE: %.3f' % rmse)
rel_errors = abs((inv_yhat-target_outputs1)/target_outputs1)
model1.summary()


##### ------ THE SECOND RNN - PREDICTS CONFIDENCE

# train2 is the second partition of the dataset
train2 = values_proc[int(train_percentage1*len(values_proc)):int((train_percentage1+train_percentage2)*len(values_proc)),:]
train2_X = train2[:, :-n_features]
train2_Y = rel_errors[:-int(test_percentage2*len(values_proc))-1]       # rough patch: if this line fails, delete the -1
#print(any(n < 0 for n in train2_Y))                                    # False -- no negative numbers in range
print(train2_X.shape,train2_Y.shape)

# test2 is the third partition of the dataset
test2 = values_proc[int((train_percentage1 + train_percentage2)*len(values_proc)):,:]      # the third and final partition
test2_X = test2[:, :-n_features]
test2_Y = rel_errors[int(train_percentage2*len(values_proc)):]
print(test2_X.shape,test2_Y.shape)

# reshape input to be 3D [samples, timesteps, features]
train2_X = train2_X.reshape((train2_X.shape[0], 1, train2_X.shape[1]))
test2_X = test2_X.reshape((test2_X.shape[0], 1, test2_X.shape[1]))

# Hyper-parameters 2
epochs2 = 100
learning_rate2 = 0.00001
batch_size2 = 32

# design network
model2 = Sequential()
model2.add(LSTM(40, return_sequences=False,input_shape=(train2_X.shape[1], train2_X.shape[2])))
model2.add(Dense(1))
adam2 = optimizers.Adam(lr=learning_rate2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model2.compile(loss='mae', optimizer=adam2)
print('\n Training Sequence for RNN 2 (Relative Error Predictor) Initiated!\n')
# fit network
history2 = model2.fit(train2_X, train2_Y, epochs=epochs2, batch_size=batch_size2, validation_data=(test2_X, test2_Y),
                      verbose=1, shuffle=False)
print('\n RNN 2 has been trained and tested! \n')

# make a prediction
yhat2 = model2.predict(test2_X)
#dummy = np.ones((len(yhat2),n_features-1))

# invert scaling for prediction
#inv_yhat2 = concatenate((yhat2, dummy), axis=1)       # Verify that this step is actually legit - if so: me = genius
#inv_yhat2 = scaler.inverse_transform(inv_yhat2)
#inv_yhat2 = inv_yhat2[:, 0]

# calculate RMSE by comparing with actual closing prices
target_outputs2 = test2_Y
rmse2 = sqrt(mean_squared_error(target_outputs2, abs(yhat2)))
print('Test RMSE: %.3f' % rmse2)
model2.summary()

##### Performance Assessment

# Plots to assess RNN 1 performance
plt.subplot(2,2,1)
plt.title("RNN 1 (Stock Price Predictor) Losses vs Epochs")
plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='test')
plt.yscale('log')
plt.legend()

plt.subplot(2,2,3)
plt.title("RNN 1 (Stock Price Predictor) Prediction vs Truth")
plt.plot(range(len(test1)),inv_yhat, label = 'Prediction')
plt.plot(range(len(test1)),target_outputs1, label = 'Truth')
plt.legend()

plt.subplot(2,2,2)
plt.title("RNN 2 (Relative Error Predictor) Losses vs Epochs")
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.yscale('log')
plt.legend()

plt.subplot(2,2,4)
plt.title("RNN 2 (Relative Error Predictor) Prediction vs Truth")
plt.plot(range(len(test2)),yhat2, label = 'Prediction')
plt.plot(range(len(test2)),target_outputs2, label = 'Truth')
plt.legend()

plt.show()

# Final MMM (Magic Money-Multiplying) Plot

inv_yhat = inv_yhat.reshape((len(inv_yhat),1))

plt.title("Price Prediction and Associated Confidence")
plt.plot(range(len(test2)),inv_yhat[-len(test2):] + yhat2*inv_yhat[-len(test2):], label = 'Upper Bound for Price')
plt.plot(range(len(test2)),inv_yhat[-len(test2):] - yhat2*inv_yhat[-len(test2):], label = 'Lower Bound for Price')
plt.plot(range(len(test2)), target_outputs1[-len(test2):], label = 'Actual Price')
plt.legend()
plt.show()
