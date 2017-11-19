from trading_sim.dataLoad import tables, risk_free_rates
from datetime import datetime
from trading_sim.market import market
from trading_sim.broker import Broker
import scipy.stats


class Wallet:
    def __init__(self, money, stock_count):
        self.money = money
        self.stock_count = stock_count

    def get_total(self, date=datetime.today().strftime("%Y-%m-%d")):
        return self.money + self.stock_count * market.open_prices.ix[date]

    # TODO: use this: # self.wallet.money *= risk_free_rates.ix[date]


class Trader:
    name = "Generic Trader"

    def __init__(self):
        self.wallet = Wallet(money=10000.0, stock_count=0)
        # self.risk_allowed = 0.2
        # self.buy_threshold = 0.9
        # self.sell_all_threshold = 0.9
        # self.risk_ratio = 0.5
        # self.raise_ratio_threshold = 1.001
        self.buy_threshold = 0.9
        self.sell_all_threshold = 0.9
        self.risk_ratio = 0.3
        self.raise_ratio_threshold = 1.001

        self.broker = Broker()
        self.last_minima = 0.0
        self.last_buy_price = 0.0
        self.stop_loss = 0.0

    def trade(self, date=datetime.today().strftime("%Y-%m-%d"), operate=True):
        raise NotImplemented


class ClassicalTrader(Trader):
    name = "Classical Trader"

    # Trade purely based on current price (today's open)
    def trade(self, date=datetime.today().strftime("%Y-%m-%d"), operate=True):
        today_price = market.open_prices.ix[date]
        if today_price > self.last_minima*self.raise_ratio_threshold:
            if operate:
                stock_count, cost = self.broker.buy(self.wallet.money, date=date)
                self.wallet.money -= cost
                self.wallet.stock_count += stock_count
            self.last_minima = today_price
            self.stop_loss = today_price * (1 - self.risk_ratio)
        elif today_price < self.stop_loss:
            if operate:
                self.wallet.money += self.broker.sell(self.wallet.stock_count, date=date)
                self.wallet.stock_count = 0.0
            self.stop_loss = 0.0
            if today_price < self.last_minima:
                self.last_minima = today_price


class PredictiveTrader(Trader):
    name = "Predictive Trader"

    # Trade purely based on prediction
    def trade(self, date=datetime.today().strftime("%Y-%m-%d"), operate=True):
        yesterday_price = market.close_prices.ix[market.close_prices.index.get_loc(date)-1]
        predicted_change = tables["predicted_changes"].ix[date]
        predicted_err = tables["predicted_err"].ix[date]
        loose_prob = scipy.stats.norm(predicted_change, predicted_err).cdf(0.0)
        gain_prob = 1.0 - loose_prob
        if gain_prob > self.buy_threshold and yesterday_price > self.last_minima*self.raise_ratio_threshold:
            stock_count, cost = self.broker.buy(self.wallet.money, date=date)
            self.wallet.money -= cost
            self.wallet.stock_count += stock_count
            self.last_minima = yesterday_price
            self.stop_loss = yesterday_price * (1 - self.risk_ratio)
        elif yesterday_price*(1+predicted_change) < self.stop_loss or loose_prob > self.sell_all_threshold:
            self.wallet.money += self.broker.sell(self.wallet.stock_count, date=date)
            self.wallet.stock_count -= self.wallet.stock_count
            self.stop_loss = 0.0
            if yesterday_price < self.last_minima:
                self.last_minima = yesterday_price


class PredictiveTraderWithoutErrorPrediction(Trader):
    name = "Predictive Trader"
    up_threshold = 0.00001
    down_threshold = 0.01

    # Trade purely based on prediction
    def trade(self, date=datetime.today().strftime("%Y-%m-%d"), operate=True):
        yesterday_price = market.close_prices.ix[market.close_prices.index.get_loc(date)-1]
        today_price = market.open_prices.ix[date]
        predicted_change = tables["predicted_changes"].ix[date]
        if predicted_change > self.up_threshold:
            stock_count, cost = self.broker.buy(self.wallet.money, date=date)
            if operate:
                self.wallet.money -= cost
                self.wallet.stock_count += stock_count
            self.last_minima = yesterday_price
            self.stop_loss = yesterday_price * (1 - self.risk_ratio)
        elif today_price < self.stop_loss or predicted_change < -self.down_threshold:
            if operate:
                self.wallet.money += self.broker.sell(self.wallet.stock_count, date=date)
                self.wallet.stock_count -= self.wallet.stock_count
            self.stop_loss = 0.0
            if today_price < self.last_minima:
                self.last_minima = yesterday_price


class HybridTrader(Trader):
    name = "Hybrid Trader"

    # Trade based on prediction and current price
    def trade(self, date=datetime.today().strftime("%Y-%m-%d"), operate=True):
        today_price = market.open_prices.ix[date]
        yesterday_price = market.close_prices.ix[market.close_prices.index.get_loc(date) - 1]
        predicted_change = tables["predicted_changes"].ix[date]
        predicted_err = tables["predicted_err"].ix[date]
        loose_prob = scipy.stats.norm(predicted_change, predicted_err).cdf(0.0)
        gain_prob = 1.0 - loose_prob
        if gain_prob > self.buy_threshold and today_price > self.last_minima * self.raise_ratio_threshold:
            stock_count, cost = self.broker.buy(self.wallet.money, date=date)
            self.wallet.money -= cost
            self.wallet.stock_count += stock_count
            self.last_minima = yesterday_price
            self.stop_loss = yesterday_price * (1 - self.risk_ratio)
        elif today_price < self.stop_loss or loose_prob > self.sell_all_threshold:
            self.wallet.money += self.broker.sell(self.wallet.stock_count, date=date)
            self.wallet.stock_count -= self.wallet.stock_count
            self.stop_loss = 0.0
            if yesterday_price < self.last_minima:
                self.last_minima = yesterday_price


if __name__ == "__main__":
    trader = Trader()
    trader.trade()




