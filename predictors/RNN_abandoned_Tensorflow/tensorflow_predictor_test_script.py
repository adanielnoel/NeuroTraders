import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from predictors.RNN_abandoned_Tensorflow.RNN_model_tensorflow import RNNPredictor
from sklearn.preprocessing import RobustScaler

from predictors.RNN_under_development.Utils import format_time_series, split_data

input_cols = ["vol_rel_m_ave_10",
              "relative_intraday",
              "relative_overnight_tomorrow",
              "sentiment_p",
              "sentiment_n",
              "sentiment_u"]
target_cols = ["relative_intraday_tomorrow"]
data_ratios = (0.8, 0.2)

input_time_steps = 20
output_time_steps = 1
batch_size = 5

regenerate_data = True
retrain_model = True

aapl_data = pd.read_csv('./../../new_database/aapl/time_data.csv', header=0, index_col=0)
aapl_data.fillna(0, inplace=True)

# Scale the columns to be within -0.7 and 0.7
scalers = {}
for col in input_cols + target_cols:
    scalers[col] = RobustScaler()
    aapl_data[[col]] = scalers[col].fit_transform(aapl_data[[col]])

train_data, test_data = split_data(aapl_data, *data_ratios)

if regenerate_data:
    # Split data into training and testing

    # Format data for LSTM
    print("Formatting train data")
    trainX, trainY, train_prediction_dates = format_time_series(train_data,
                                                                input_time_steps=input_time_steps,
                                                                batch_size=batch_size,
                                                                input_cols=input_cols,
                                                                target_cols=target_cols)

    print("Formatting test data")
    testX, testY, test_prediction_dates = format_time_series(test_data,
                                                             input_time_steps=input_time_steps,
                                                             batch_size=None,
                                                             input_cols=input_cols,
                                                             target_cols=target_cols)

    pickle.dump((trainX, trainY, testX, testY, train_prediction_dates, test_prediction_dates), open("rnn_data", "wb"))
trainX, trainY, testX, testY, train_prediction_dates, test_prediction_dates = pickle.load(open("rnn_data", "rb"))

if retrain_model:
    price_change_predictor = RNNPredictor(name="Price predictor",
                                          input_time_steps=input_time_steps,
                                          input_length=len(input_cols),
                                          output_length=len(target_cols),
                                          lstm_size=100,
                                          num_layers=2)

    # Remove dimension for output time steps
    price_change_predictor.train(trainX, trainY, testX, testY, max_epochs=5)

    testX, _, test_dates = format_time_series(test_data,
                                              input_time_steps=input_time_steps,
                                              batch_size=1,
                                              input_cols=input_cols,
                                              target_cols=target_cols)
    predictions_test = []
    for i in range(testX.shape[0]):
        predictions_test.append(price_change_predictor.predict(testX[i]))
    predictions_test = np.array(predictions_test).reshape(1, -1)
    pickle.dump((predictions_test, test_dates), open("predictions_temp.pickle", "wb"))

predictions_test, test_dates = pickle.load(open("predictions_temp.pickle", "rb"))

predictions_test = scalers[target_cols[0]].inverse_transform(predictions_test)
test_data[target_cols] = scalers[target_cols[0]].inverse_transform(test_data[target_cols])

test_data = test_data[target_cols].loc[[date[0] for date in test_dates]]
test_data["predictions"] = predictions_test.reshape(-1)
test_data.plot()
plt.show()
