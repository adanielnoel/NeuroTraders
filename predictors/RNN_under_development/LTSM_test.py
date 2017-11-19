# Code adapted from http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
# Author: Jakob Aungiers

import pickle
import time
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import predictors.RNN_under_development.LSTM1 as lstm
from keras.models import load_model

from predictors.RNN_under_development.Utils import Scaler, split_data, format_time_series


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + list(data), label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':
    global_start_time = time.time()

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
    prediction_steps = 3

    print('> Loading data... ')
    regenerate_data = False
    retrain_model = False

    if regenerate_data:
        # X_train, y_train, X_test, y_test = lstm.load_data('./../../new_database/aapl/time_data.csv', seq_len, True)
        aapl_data = pd.read_csv('./../new_database/aapl/time_data.csv', header=0, index_col=0)
        aapl_data.fillna(0, inplace=True)
        train_data, test_data = split_data(aapl_data, *data_ratios)
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
    trainX, trainY, testX, testY, train_prediction_dates, test_prediction_dates = pickle.load(open("rnn_data.pickle", "rb"))

    trainX = trainX.reshape(trainX.shape[0] * trainX.shape[1], *trainX.shape[2:])
    trainY = trainY.reshape(trainY.shape[0] * trainY.shape[1], *trainY.shape[2:])
    testX = testX.reshape(testX.shape[0] * testX.shape[1], *testX.shape[2:])
    testY = testY.reshape(testY.shape[0] * testY.shape[1], *testY.shape[2:])

    trainX_scaler = Scaler(map_range=(-0.6, 0.6))
    trainY_scaler = Scaler(map_range=(-0.6, 0.6))
    testX_scaler = Scaler(map_range=(-0.6, 0.6))
    testY_scaler = Scaler(map_range=(-0.6, 0.6))
    trainX = trainX_scaler.fit(trainX).apply(trainX)
    trainY = trainY_scaler.fit(trainY).apply(trainY)
    testX = testX_scaler.fit(testX).apply(testX)
    testY = testY_scaler.fit(testY).apply(testY)

    if retrain_model:
        print('> Data Loaded. Compiling...')

        model = lstm.build_model([6, 50, 100, 1])
        model.fit(
            trainX,
            trainY,
            batch_size=batch_size,
            nb_epoch=2,
            validation_split=0.05)

        model.save("LSTM_model_trained.h5")
    model = load_model("LSTM_model_trained.h5")

    print("> Predicting...")
    predictions = np.array(lstm.predict_sequences_multiple(model, testX, input_time_steps, prediction_steps))
    # predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    # predicted = lstm.predict_point_by_point(model, X_test)
    predictions = np.array([testY_scaler.revert(predictions[:, i]) for i in range(predictions.shape[-1])])
    predictions = predictions.swapaxes(0, 1)
    targets = testY_scaler.revert(testY)

    print('Training duration (s) : ', time.time() - global_start_time)
    plot_results_multiple(predictions, targets, prediction_steps)

