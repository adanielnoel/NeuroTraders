import pandas
import numpy as np
from Packages.Predictor.load_results import price_pred, error_pred

# dates,adj_open,adj_close,volume,price_m_ave_50,price_m_ave_100,volume_m_ave_50,relative_change,positive_sentiment,negative_sentiment,uncertainty_sentiment,news_volume


run_true = False
if run_true:
    tables = pandas.read_csv("record.csv", header=0, index_col=0).iloc[-272:]
    prices = tables["adj_close"]
    relative_change = [0.0]
    for i in range(1, prices.shape[0]):
        close_yesterday = price_pred[i-1]
        close_today = price_pred[i]
        relative_change.append(close_today / close_yesterday - 1.0)

    tables["predicted_changes"] = relative_change
    tables["predicted_err"] = error_pred
    risk_free_rates = pandas.read_csv("risk-free-rates.csv", header=0, index_col=0)["risk free multiplier"]

else:
    tables = pandas.read_csv("record.csv", header=0, index_col=0).iloc[:]
    prices = tables["adj_close"]
    relative_change = [0.0]
    for i in range(1, prices.shape[0]):
        close_yesterday = prices.iloc[[i - 1]].values[0]
        close_today = prices.iloc[[i]].values[0]
        relative_change.append(close_today / close_yesterday - 1.0)

    np.random.seed(20)
    predicted_changes = np.random.normal(relative_change, 0.01, len(prices)) + prices - prices
    predicted_err = abs(np.random.normal(0.0, 0.01, len(prices))) + prices - prices
    tables["predicted_changes"] = predicted_changes
    tables["predicted_err"] = predicted_err
    risk_free_rates = pandas.read_csv("risk-free-rates.csv", header=0, index_col=0)["risk free multiplier"]
