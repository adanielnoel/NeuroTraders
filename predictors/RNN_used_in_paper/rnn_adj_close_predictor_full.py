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

from math import sqrt
import numpy as np
np.random.seed(19)
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras import regularizers
import csv
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from predictors.RNN_used_in_paper.ml_data_prepper_full import prepare_lstm_data, descale_output

ticker = "AAPL"

# Shift adjusted open one day back so that it is used on the day
dataframe = read_csv('./../../new_database/aapl/time_data.csv', header=0, index_col=0)
dataframe = dataframe[:-1]

# Choosing which inputs to delete - set 1 to use, 0 to delete
input_dictionary = {'adj_vol': 0,
                    'adj_low': 1,
                    'adj_close': 0,
                    'adj_high': 0,
                    'adj_open': 0,
                    'relative_intraday': 0,
                    'vol_m_ave_10': 0,
                    'vol_rel_m_ave_10': 0,
                    'adj_close_tomorrow': 1,
                    'sentiment_p': 1,
                    'sentiment_n': 1,
                    'sentiment_u': 0,
                    'relative_intraday_tomorrow': 0,
                    'relative_overnight': 0,
                    'adj_open_tomorrow': 1,
                    'relative_overnight_tomorrow': 0,
                    }

predict = 'adj_close_tomorrow'
del_inputs = [key for key in input_dictionary if not input_dictionary[key]]
used_inputs = [key for key in input_dictionary if input_dictionary[key]]

# To save model results
directory = "./Results/" + \
            ticker + '_' + predict + '_full_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(directory):
    os.makedirs(directory)

# A few options for Neural Net 2 - Error predictor
scale_targets_net2 = False          # True --> targets for neural net 2. MinMax scaler will be used
which_scaler_2 = "MinMax"           # Options: "MinMax" or "StandardScaler"
which_errors = "Relative Unsigned"  # Options:  "Relative Unsigned","Relative Signed","Actual Unsigned","Actual Signed"
                                    # "Actual squared"
feature_range_2 = (0, 1)            # Feature range to map target errors into

# Regularizers
which_regularizer_1 = None          # Choose between "l1", "l2"
which_regularizer_2 = None          # Choose between "l1", "l2"

# Define partitions for dataset - partition 1: train1 - partition 2: test1 and train 2 - partition 3: test 2
train_interval_1 = (0.0, 0.6)
test_interval_1 = (0.6, 1.0)
train_interval_2 = (0.6, 0.9)
test_interval_2 = (0.9, 1.0)

# TUNABLE NEURAL NET HYPERPARAMETERS
# Net 1
epochs1 = 106
batch_size1 = 64
learning_rate1 = 0.0001  # 0.001 is default for Adam optimizer -- usually an ok learning rate
dropout1 = 0
# Net 2
epochs2 = 150
batch_size2 = 64
learning_rate2 = 0.0001  # 0.001 is default for Adam optimizer -- usually an ok learning rate
dropout2 = 0

# Pose dataset as a supervised learning problem - see ml_data_prepper methods
prepared_data = prepare_lstm_data(dataframe=dataframe, predict=predict, del_inputs=del_inputs,
                                  timestep=1, use_present_vars=False, n_in=1, n_out=1, scaler='MinMax',
                                  feature_range=(0, 1), train_interval_1=train_interval_1,
                                  test_interval_1=test_interval_1, train_interval_2=train_interval_2,
                                  test_interval_2=test_interval_2)

train_X_1 = prepared_data[0]
train_Y_1 = prepared_data[1]
test_X_1 = prepared_data[2]
test_Y_1 = prepared_data[3]
values = prepared_data[4]
scaler = prepared_data[5]
train_X_2 = prepared_data[6]
test_X_2 = prepared_data[7]

# Create and compile first Neural Network Model - stock price predictor
model1 = Sequential()
if which_regularizer_1 == "l1":
    model1.add(LSTM(120, return_sequences=True, input_shape=(train_X_1.shape[1], train_X_1.shape[2]),
                    kernel_regularizer=regularizers.l1(0.01)))
    model1.add(Dropout(dropout1))
    model1.add(LSTM(120, return_sequences=False, kernel_regularizer=regularizers.l1(0.01)))
    model1.add(Dropout(dropout1))
elif which_regularizer_1 == "l2":
    model1.add(LSTM(120, return_sequences=True, input_shape=(train_X_1.shape[1], train_X_1.shape[2]),
                    kernel_regularizer=regularizers.l2(0.01)))
    model1.add(Dropout(dropout1))
    model1.add(LSTM(120, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model1.add(Dropout(dropout1))
else:
    model1.add(LSTM(120, return_sequences=True, input_shape=(train_X_1.shape[1], train_X_1.shape[2])))
    model1.add(Dropout(dropout1))
    model1.add(LSTM(120, return_sequences=False))
    model1.add(Dropout(dropout1))
model1.add(Dense(1))
adam1 = optimizers.Adam(lr=learning_rate1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model1.compile(loss='mae', optimizer=adam1)

# Create second Neural Network Model - ABSOLUTE ERROR PREDICTOR
model2 = Sequential()
if which_regularizer_2 == "l1":
    model2.add(LSTM(100, return_sequences=True, input_shape=(train_X_2.shape[1], train_X_2.shape[2]),
                    kernel_regularizer=regularizers.l1(0.01)))
    model2.add(Dropout(dropout2))
    model2.add(LSTM(100, return_sequences=False, kernel_regularizer=regularizers.l1(0.01)))
    model2.add(Dropout(dropout2))
elif which_regularizer_2 == "l2":
    model2.add(LSTM(100, return_sequences=True, input_shape=(train_X_2.shape[1], train_X_2.shape[2]),
                    kernel_regularizer=regularizers.l2(0.01)))
    model2.add(Dropout(dropout2))
    model2.add(LSTM(100, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
    model2.add(Dropout(dropout2))
else:
    model2.add(LSTM(20, return_sequences=True, input_shape=(train_X_2.shape[1], train_X_2.shape[2])))
    model2.add(Dropout(dropout2))
    model2.add(LSTM(20, return_sequences=False))
    model2.add(Dropout(dropout2))

model2.add(Dense(1))
adam2 = optimizers.Adam(lr=learning_rate2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model2.compile(loss='mae', optimizer=adam2)

# Fit first Neural Network
print('\n Training Sequence for RNN 1 (Stock Price Predictor) Initiated!\n')
history1 = model1.fit(train_X_1, train_Y_1, epochs=epochs1, batch_size=batch_size1,
                      validation_data=(test_X_1, test_Y_1),
                      verbose=2, shuffle=False)
print('\n RNN 1 has been trained and tested! \n')

# Predict the test data and descale the prediction
yhat_test = model1.predict(test_X_1)
inv_yhat_test_1 = descale_output(yhat_test, scaler=scaler, input_X=test_X_1)

# Predict the training data and descale the prediction
yhat_train = model1.predict(train_X_1)
inv_yhat_train_1 = descale_output(yhat_train, scaler=scaler, input_X=train_X_1)

# Compute error to train the second neural network
target_outputs_test_1 = dataframe[predict][len(train_X_1):(len(train_X_1) + len(test_X_1))]
errors = []
try:
    if which_errors == "Relative Unsigned":
        errors = abs(inv_yhat_test_1 - target_outputs_test_1) / target_outputs_test_1
    elif which_errors == "Relative Signed":
        errors = (inv_yhat_test_1 - target_outputs_test_1) / target_outputs_test_1
    elif which_errors == "Actual Unsigned":
        errors = abs(inv_yhat_test_1 - target_outputs_test_1)
    elif which_errors == "Actual Signed":
        errors = inv_yhat_test_1 - target_outputs_test_1
    elif which_errors == "Actual Squared":
        errors = (inv_yhat_test_1 - target_outputs_test_1)**2
except:
    raise TypeError('Illegal input- which_errors must be a string chosen from: "Relative Unsigned", "Relative Signed", '
                    '"Actual Unsigned", "Actual Signed"')


# Fit second Neural Network and predict errors
if scale_targets_net2:
    scaler_2 = None
    if which_scaler_2 == "MinMax":
        scaler_2 = MinMaxScaler(feature_range=feature_range_2)
    elif which_scaler_2 == "StandardScaler":
        scaler_2 = StandardScaler()
    aggregate_Y_2 = errors
    scaled_2 = scaler_2.fit_transform(aggregate_Y_2.reshape(-1,1))
    train_Y_2 = scaled_2[:len(train_X_2)]
    test_Y_2 = scaled_2[len(train_X_2):]
    print('\n Training Sequence for RNN 2 (Error Predictor) Initiated!\n')
    history2 = model2.fit(train_X_2, train_Y_2, epochs=epochs2, batch_size=batch_size2,
                          validation_data=(test_X_2, test_Y_2), verbose=2, shuffle=False)
    print('\n RNN 2 has been trained and tested! \n')
    yhat_test_2 = model2.predict(test_X_2)
    inv_yhat_test_2 = descale_output(yhat_test_2, scaler_2, None)
    yhat_train_2 = model2.predict(train_X_2)
    inv_yhat_train_2 = descale_output(yhat_train_2, scaler_2, None)

else:  # --> elif not scale_target_net2
    train_Y_2 = errors[:len(train_X_2)]
    test_Y_2 = errors[len(train_X_2):]
    print('\n Training Sequence for RNN 2 (Error Predictor) Initiated!\n')
    history2 = model2.fit(train_X_2, train_Y_2, epochs=epochs2, batch_size=batch_size2,
                          validation_data=(test_X_2, test_Y_2),
                          verbose=2, shuffle=False)
    print('\n RNN 2 has been trained and tested! \n')
    yhat_test_2 = model2.predict(test_X_2)
    inv_yhat_test_2 = yhat_test_2
    yhat_train_2 = model2.predict(train_X_2)
    inv_yhat_train_2 = yhat_train_2

# Compute test RMSE, train RMSE and baseline RMSE for Neural Net 1
target_outputs_train_1 = dataframe[predict][:len(train_X_1)]
test_rmse_1 = sqrt(mean_squared_error(target_outputs_test_1, inv_yhat_test_1))
train_rmse_1 = sqrt(mean_squared_error(target_outputs_train_1, inv_yhat_train_1))
baseline_rmse_1 = sqrt(mean_squared_error(target_outputs_test_1,
                                          dataframe["adj_open_tomorrow"][len(train_X_1):(len(train_X_1) + len(test_X_1))
                                          ]))
print('Test RMSE 1 : ', test_rmse_1)
print('Train RMSE 1 : ', train_rmse_1)
print('Baseline RMSE 1 : ', baseline_rmse_1)

# Compute test RMSE, train RMSE for Neural Net 2
target_outputs_test_2 = errors[len(train_X_2):]
target_outputs_train_2 = errors[:len(train_X_2)]
test_rmse_2 = sqrt(mean_squared_error(target_outputs_test_2, inv_yhat_test_2))
train_rmse_2 = sqrt(mean_squared_error(target_outputs_train_2, inv_yhat_train_2))
print('Test RMSE 2 : ', test_rmse_2)
print('Train RMSE 2 : ', train_rmse_2)

# Plots to assess RNN  1 performance
plt.subplot(311)
plt.title("RNN 1 (Stock Price Predictor) Losses vs Epochs")
plt.plot(history1.history['loss'], label='train')
plt.plot(history1.history['val_loss'], label='test')
plt.yscale('log')
plt.legend()

plt.subplot(312)
plt.title("RNN 1 (Stock Price Predictor) Prediction vs Truth - TEST")
plt.plot(range(len(test_X_1)), inv_yhat_test_1, label='Prediction')
plt.plot(range(len(test_X_1)), target_outputs_test_1, label='Truth')
plt.legend()
plt.xlabel(str('\n RMSE:   ' + str(test_rmse_1)), fontsize=16)

plt.subplot(313)
plt.title("RNN 1 (Stock Price Predictor) Prediction vs Truth - TRAIN")
plt.plot(range(len(train_X_1)), inv_yhat_train_1, label='Prediction')
plt.plot(range(len(train_X_1)), target_outputs_train_1, label='Truth')
plt.legend()

plt.savefig(directory+"/RNN_1_"+predict+'.png', dpi=400, bbox_inches='tight')
plt.show()

# Plots to assess RNN 2 performance
plt.subplot(311)
plt.title("RNN 2 (Error Predictor) Losses vs Epochs")
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.yscale('log')
plt.legend()

plt.subplot(312)
plt.title("RNN 2 (Error Predictor) Prediction vs Truth - TEST")
plt.plot(range(len(test_X_2)), inv_yhat_test_2, label='Prediction')
plt.plot(range(len(test_X_2)), target_outputs_test_2, label='Truth')
plt.legend()

plt.subplot(313)
plt.title("RNN 2 (Error Predictor) Prediction vs Truth - TRAIN")
plt.plot(range(len(train_X_2)), inv_yhat_train_2, label='Prediction')
plt.plot(range(len(train_X_2)), target_outputs_train_2, label='Truth')
plt.legend()

plt.savefig(directory+'/RNN_2_'+which_errors+' Error.png',dpi=400, bbox_inches='tight')
plt.show()

#Other plots to show off
plt.title("RNN 1 (Stock Price Predictor) Prediction vs Truth - TEST")
plt.plot(range(len(test_X_1)), inv_yhat_test_1, label='Prediction')
plt.plot(range(len(test_X_1)), target_outputs_test_1, label='Truth')
plt.legend()
plt.savefig(directory+'/RNN_1_'+predict+'_test_performance.png',dpi=800, bbox_inches='tight')
plt.show()

# Write useful results to CSV file

predict_for_csv = ["Predict: ", predict]
used_inputs_for_csv = ["Using: "] + used_inputs
rmse_1_for_csv = [["Train RMSE 1: ", train_rmse_1], ["Test RMSE 1: ", test_rmse_1],
                  ["Baseline RMSE 1: ", baseline_rmse_1]]
rmse_2_for_csv = [["Train RMSE 2: ", train_rmse_2], ["Test RMSE 2: ", test_rmse_2]]
headers_for_csv = ["Predicted " + predict, "True " + predict, "Relative Error on " + predict]
headers_2_for_csv = ["Predicted" + which_errors, "True" + which_errors, "Relative Error on" + which_errors]
results_for_csv = [[inv_yhat_test_1[i], target_outputs_test_1[i],
                    abs(inv_yhat_test_1[i] - target_outputs_test_1[i]) / target_outputs_test_1[i]]
                   for i in range(len(inv_yhat_test_1))]
results_2_for_csv = [[inv_yhat_test_2[i], target_outputs_test_2[i],
                      abs(inv_yhat_test_2[i] - target_outputs_test_2[i]) / target_outputs_test_2[i]]
                     for i in range(len(inv_yhat_test_2))]



# Write Summary of RNN 1
with open(directory + "/" + 'RNN_1_summary.txt', 'w') as fh:
    fh.write('RNN 1 - Stock Price Predictor - Summary\n')
    model1.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.write('Epochs: ' + str(epochs1) + '\n')
    fh.write('Learning Rate: ' + str(learning_rate1) + '\n')
    fh.write('Batch Size: ' + str(batch_size1) + '\n')
    fh.write('Dropout: ' + str(dropout1) + '\n')

# Write Summary of RNN 2
with open(directory + "/" + 'RNN_2_summary.txt', 'w') as fh:
    fh.write('RNN 2 - Error Predictor - Summary\n')
    model2.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.write('Epochs: ' + str(epochs2) + '\n')
    fh.write('Learning Rate: ' + str(learning_rate2) + '\n')
    fh.write('Batch Size: ' + str(batch_size2) + '\n')
    fh.write('Dropout: ' + str(dropout2) + '\n')

# Write Useful Results
with open(directory + "/" + "RNN_1_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(used_inputs_for_csv)
    writer.writerow(predict_for_csv)
    writer.writerows(rmse_1_for_csv)
    writer.writerow(headers_for_csv)
    writer.writerows(results_for_csv)

with open(directory + "/" + "RNN_2_results.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(used_inputs_for_csv)
    writer.writerow(predict_for_csv)
    writer.writerows(rmse_2_for_csv)
    writer.writerow(headers_2_for_csv)
    writer.writerows(results_2_for_csv)