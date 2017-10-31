from secdata.HistoricDataProvider import HistoricDataProvider
from secdata.settings import settings
from secdata import utils
from datetime import datetime, timedelta
import logging
import quandl
from quandl.errors.quandl_error import ForbiddenError, NotFoundError

quandl.ApiConfig.api_key = settings["quandl_api_key"]
logger = logging.getLogger(__name__)


class StockValues(HistoricDataProvider):
    def __init__(self, ticker, init_date, end_date):
        HistoricDataProvider.__init__(self, init_date, end_date)
        # Set start date earlier than requested so as to calculate moving averages correctly
        start_date = utils.to_datetime(init_date) - timedelta(days=20)
        end_date = utils.to_datetime(end_date)
        try:
            self.quandl_data = quandl.get("WIKI/{}".format(ticker), start_date=start_date, end_date=end_date)
        except ForbiddenError:
            logger.error("Access to dataset is forbidden")
            self.quandl_data = None
        except NotFoundError:
            logger.error("Ticker could not be found")
            logger.error("  Reason: The default quandl dataset WIKI does not include the stock")
            logger.error("          A subscription dataset may be required, for instance the dataset EOD")
            self.quandl_data = None

    def get_cols(self, cols):
        if self.quandl_data is None:
            return None
        cols_directly_from_quandl = [col for col in cols if "quandl_name" in settings["stock_cols"][col]]
        data_frame = self.quandl_data[[settings["stock_cols"][col]["quandl_name"] for col in cols_directly_from_quandl]]
        cols_to_process = list(set(cols).difference(cols_directly_from_quandl))
        for col in cols_to_process:
            try:
                data_frame = data_frame.assign(**{col: eval("self.{}".format(col))})
            except Exception as e:
                logger.error("Column {} cannot be computed or appended, check that property exists".format(col))
                logger.error("  Reason: %s" % e)

        # Return the columns while eliminating the earlier dates that are used for moving averages
        return data_frame[utils.to_datetime(self.init_date):]

    @property
    def vol_m_ave_10(self):
        return self.quandl_data['Adj. Volume'].rolling(window=10).mean()

    @property
    def relative_close(self):
        return self.quandl_data['Adj. Close'].divide(self.quandl_data['Adj. Close'].shift(1), axis='index') - 1.0


# Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio',
#        'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'],
#       dtype='object')

