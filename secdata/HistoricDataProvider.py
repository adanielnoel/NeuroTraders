
class HistoricDataProvider:
    def __init__(self, init_date, end_date):
        self.init_date = init_date
        self.end_date = end_date

    def get_cols(self, cols):
        raise NotImplementedError

