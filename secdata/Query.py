
class Query:
    def __init__(self, init_date=None, end_date=None, **kwargs):
        self.init_date = init_date
        self.end_date = end_date
        for key, value in kwargs.items():
            self.add_attribute(key, value)

    def add_attribute(self, key, value):
        setattr(self, key, value)


class StockQuery(Query):
    def __init__(self, ticker, company_name="", keywords=(), market="Nasdaq", quandl_database="WIKI", **kwargs):
        Query.__init__(self, **kwargs)
        self.ticker = ticker
        self.company_name = company_name
        self.keywords = keywords
        self.market = market
        self.quandl_database = quandl_database


