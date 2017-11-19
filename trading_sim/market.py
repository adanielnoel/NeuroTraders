from trading_sim.dataLoad import tables


class Market:
    def __init__(self):
        self.close_prices = tables["adj_close"]
        self.open_prices = tables["adj_open"]


market = Market()
if __name__ == "__main__":
    a = 2


