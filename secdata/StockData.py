from secdata.SecurityData import SecurityData
import os
from datetime import datetime, timedelta
import logging
from secdata.settings import settings
from secdata.StockValues import StockValues
from secdata.StockSentiment import StockSentiment
from secdata.StockNews import StockNews
from secdata import utils
logger = logging.getLogger(__name__)


class StockData(SecurityData):

    def __init__(self, ticker, database_dir, auto_load=True):
        database_dir = database_dir if ticker in database_dir else os.path.join(database_dir, ticker)
        SecurityData.__init__(self, database_dir, auto_load=auto_load)
        assert isinstance(ticker, str)
        assert isinstance(database_dir, str)
        self.ticker = ticker.upper()
        if auto_load:
            self.load()

    def update(self, cols=None, exclude=None, init_date=None):
        if init_date is None:
            if self.latest_date == datetime.today().strftime(settings["time_format_str"]):
                logger.info("Database for stock {} is already up to date".format(self.ticker))
                return
            else:
                # Get the date of the day after self.last_date
                init_date = utils.to_datetime(self.latest_date) + timedelta(days=1)
                init_date = init_date.strftime(settings["time_format_str"])

        # ----------------------------------
        # Prepare list of columns to update
        if cols is None:
            cols = self.cols
        else:
            # Warn user if specified columns to include are invalid
            if not set(cols).issubset(self.cols):
                wrong_cols = set(exclude).difference(cols)
                logger.warning("The following columns requested are unknown:")
                for wrong_col in wrong_cols:
                    logger.warning("\t{}".format(wrong_col))

        # ---------------------------------------
        # Remove exclude columns from the update
        if exclude is not None:
            cols = list(set(cols).difference(exclude))
            logger.info("Columns where removed from the update, values will be set to their defaults")
            # Warn user if specified columns to exclude are invalid
            if not set(exclude).issubset(cols):
                wrong_cols = set(exclude).difference(cols)
                logger.warning("The following columns to exclude are unknown:")
                for wrong_col in wrong_cols:
                    logger.warning("\t{}".format(wrong_col))

        logger.info("Gathering data for stock {}. From {} to {}".format(self.ticker, init_date, utils.today()))
        logger.info("Data will be fetched for the following columns:")
        for col in cols:
            logger.info("\t{}".format(col))

        # ----------------------------------------------------------------------------------
        # Get info on how to retrieve columns
        col_info = settings["stock_cols"]

        # ----------------------------------
        # Retrieve columns from StockValues
        provider = StockValues(self.ticker, init_date=init_date, end_date=utils.today())
        new_data = provider.get_cols([col for col in cols if col_info[col]["provider"] == "StockValues"])
        if new_data is None:
            logger.info("Update cancelled")
            return
        print(new_data)
        # ------------------------------------
        # Issue a news search to StockNews
        # TODO

        # ------------------------------------
        # Retrieve columns from StockSentiment
        # TODO: get data from sentiment analysis

        # ------------------------------------
        # Append new rows to self.time_data
        # TODO

    def initialise(self, init_date):
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        utils.update_info("latest_update", utils.today(), self.info_file_dir)
        self.update(init_date=init_date)

    @property
    def cols(self):
        return settings["stock_cols"].keys()














