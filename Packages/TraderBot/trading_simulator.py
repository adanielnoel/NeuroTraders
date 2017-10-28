import matplotlib.pyplot as plt
import pandas as pd

from Packages.TraderBot.dataLoad import tables
from Packages.TraderBot.trader import PredictiveTrader, ClassicalTrader, HybridTrader


class TradingSimulator:
    def __init__(self, traders):
        self.traders = {}
        for trader in traders:
            self.traders[trader] = {"Wallet history": []}

    def run(self):
        for date in tables.index.values:
            for trader in self.traders.keys():
                # Let a few days of observation to non-predictive traders
                # Traders that do not require observation will ignore the setting
                trader.trade(date=date, operate=(date in tables.index.values[3:]))
                self.traders[trader]["Wallet history"].append(trader.wallet.get_total(date=date))

    def plot_wallet_history(self):
        for trader in self.traders.keys():
            rel_gains = pd.Series(self.traders[trader]["Wallet history"],
                                  index=tables.index.values) / self.traders[trader]["Wallet history"][0]
            rel_gains.plot(label=trader.name)
        (tables["adj_close"] / tables["adj_close"].iloc[[0]].values[0]).plot(label='Stock original price')
        plt.title("Wallet history and Stock price movement vs Date")
        plt.legend()
        plt.ylabel("Relative gain")
        plt.show()

    def print_traders_statistics(self):
        for trader in self.traders.keys():
            print(trader.name, "\tBuy: ", trader.broker.buy_count, "\tSell: ", trader.broker.sell_count)


if __name__ == "__main__":
    td = TradingSimulator((ClassicalTrader(), PredictiveTrader(), HybridTrader()))
    td.run()
    td.print_traders_statistics()
    td.plot_wallet_history()
