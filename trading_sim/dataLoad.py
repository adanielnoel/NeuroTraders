import pandas
import numpy as np

run_on_true_data = True
if run_on_true_data:
    csv = pandas.read_csv("./Best_Modelzz/AMZN/RNN_2_results.csv", header=4)
    errors = csv.iloc[:, 0]

    csv = pandas.read_csv("./Best_Modelzz/AMZN/RNN_1_results.csv", header=5)
    prices = csv.iloc[:, 0]
    relative_change = prices.divide(prices.shift(1)) - 1.0
    relative_change = relative_change[-errors.shape[0]:]

    tables = pandas.read_csv("../new_database/amzn/time_data.csv", header=0, index_col=0).iloc[-relative_change.shape[0]:]

    errors.index = tables.index
    predicted_err = errors.multiply(tables["adj_close"]).divide(tables["adj_close"].shift(1))

    tables["predicted_changes"] = relative_change.values
    tables["predicted_err"] = predicted_err
    tables.fillna(0, inplace=True)  # Replace NAN entries with 0

    risk_free_rates = pandas.read_csv("risk-free-rates.csv", header=0, index_col=0)["risk free multiplier"]

else:
    # Generate simulated predictions for price change and prediction error
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
