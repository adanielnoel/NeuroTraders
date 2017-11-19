"""
The MIT License (MIT)
Copyright (c) 2017 Alejandro Daniel Noel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

from secdata.HistoricDataProvider import HistoricDataProvider
from secdata.settings import settings
from secdata import utils
from datetime import timedelta
import logging
import quandl
from quandl.errors.quandl_error import ForbiddenError, NotFoundError

quandl.ApiConfig.api_key = settings["quandl_api_key"]
logger = logging.getLogger(__name__)


class StockValues(HistoricDataProvider):
    extra_rows = 20  # Rows added to allow for computing moving averages and others.

    def __init__(self, query):
        HistoricDataProvider.__init__(self, query)
        # Set start date earlier than requested so as to calculate moving averages correctly
        start_date = utils.text_to_datetime(query.init_date) - timedelta(days=self.extra_rows)
        end_date = utils.text_to_datetime(query.end_date)
        try:
            self.quandl_data = quandl.get("{}/{}".format(query.quandl_database, query.ticker),
                                          start_date=start_date,
                                          end_date=end_date)
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
        # TODO: auto determine notation for columns depending on database
        # TODO: not that relevant unless you have access to paid databases
        try:
            data_frame = self.quandl_data[[settings["stock_cols"][col]["quandl_name"]
                                           for col in cols_directly_from_quandl]]
            # Rename columns to project's standard names
            data_frame.columns = [col for col in cols_directly_from_quandl]
        except KeyError:
            logger.error("Some columns where not recognised by quandl")
            logger.error("  Reason: Either a mapping is misspelled in the settings file")
            logger.error("          or you are using the mapping of another database")
            return None
        cols_to_process = list(set(cols).difference(cols_directly_from_quandl))
        for col in cols_to_process:
            try:
                data_frame = data_frame.assign(**{col: eval("self.{}".format(col))})
            except Exception as e:
                logger.error("Column {} cannot be computed or appended, check that property exists".format(col))
                logger.error("  Reason: %s" % e)

        # Return the columns while eliminating the earlier dates that are used for moving averages
        return data_frame[utils.text_to_datetime(self.init_date):]

    @property
    def adj_close_tomorrow(self):
        return self.quandl_data['Adj. Close'].shift(-1)

    @property
    def adj_open_tomorrow(self):
        return self.quandl_data['Adj. Open'].shift(-1)

    @property
    def vol_m_ave_10(self):
        return self.quandl_data['Adj. Volume'].rolling(window=10).mean()

    @property
    def vol_rel_m_ave_10(self):
        return (self.quandl_data['Adj. Volume'] / self.vol_m_ave_10) - 1

    @property
    def relative_intraday(self):
        return self.quandl_data['Adj. Close'].divide(self.quandl_data['Adj. Open'], axis='index') - 1.0

    @property
    def relative_intraday_tomorrow(self):
        return self.relative_intraday.shift(-1)

    @property
    def relative_overnight_tomorrow(self):
        return self.relative_overnight.shift(-1)

    @property
    def relative_overnight(self):
        return self.quandl_data['Adj. Open'].divide(self.quandl_data['Adj. Close'].shift(1), axis='index') - 1.0

    @property
    def dates_filtered_by_price_range_std(self, std_threshold=3):
        price_range_col = self.quandl_data['Adj. High'] - self.quandl_data['Adj. Low']
        std = price_range_col.std()
        price_range_col = price_range_col.abs()
        return self.quandl_data.ix[price_range_col/std >= std_threshold][self.init_date:].index

    @property
    def all_dates(self):
        return self.quandl_data[self.init_date:].index


# Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Ex-Dividend', 'Split Ratio',
#        'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume'],
#       dtype='object')

