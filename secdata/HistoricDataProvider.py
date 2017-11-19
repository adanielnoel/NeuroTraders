
class HistoricDataProvider:
    def __init__(self, query):
        self.init_date = query.init_date
        self.end_date = query.end_date

    def get_cols(self, cols):
        raise NotImplementedError

