import pickle
import os
import pandas
from pandas import Series, DataFrame
import numpy as np

results = pickle.load(open(os.path.dirname(os.path.realpath(__file__)) + "/predictions8.pickle", "rb"))
error_pred, price_pred = results

db = pandas.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/../../Database/AAPL/record.csv', header=0, index_col=0)

price_pred = np.array(price_pred.reshape(1, 272)[0], dtype='float64')
error_pred = np.array(error_pred.reshape(1, 272)[0])
