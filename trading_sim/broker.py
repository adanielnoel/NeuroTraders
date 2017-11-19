from trading_sim.market import market
from pandas import datetime


class Broker:
    def __init__(self):
        self.fixed_commission = 0.5
        self.scale_commission = 0.0002
        self.buy_count = 0
        self.sell_count = 0

    def commission(self, stock_count, date):
        if stock_count == 0:
            return 0.0
        total_capital = market.open_prices.ix[date] * stock_count
        commission = (total_capital * self.scale_commission + self.fixed_commission) / total_capital
        return commission

    def buy(self, cash, date=datetime.today().strftime("%Y-%m-%d")):
        stock_count = cash // market.open_prices.ix[date]
        commission = self.commission(stock_count, date)
        total_cost = stock_count * market.open_prices.ix[date] * (1 + commission)
        if total_cost > cash:
            stock_count -= (total_cost - cash) // market.open_prices.ix[date] + 1
            stock_count = max(0, stock_count)
            total_cost = stock_count * market.open_prices.ix[date] * (1 + commission)
        if stock_count > 0:
            self.buy_count += 1
        return stock_count, total_cost

    def sell(self, stock_count, date=datetime.today().strftime("%Y-%m-%d")):
        commission = self.commission(stock_count, date)
        self.sell_count += 1
        return stock_count * market.open_prices.ix[date] * (1 - commission)


if __name__ == "__main__":
    broker = Broker()
    print(broker.buy(100, datetime(year=2017, month=10, day=9).strftime("%Y-%m-%d")))

