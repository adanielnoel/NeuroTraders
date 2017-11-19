import json
import logging
import os
import pandas
from datetime import datetime, timedelta

from secdata import utils
from secdata.StockValues import StockValues
from secdata.SecurityData import SecurityData
from secdata.news_scraper.stockNewsCrawler import StockNewsCrawler
from secdata.sentiment_analyser.NewsAnalyser import NewsAnalyser
from secdata.sentiment_analyser.utils import collapse_daily_news
from secdata.settings import settings

logger = logging.getLogger(__name__)


class StockData(SecurityData):
    def __init__(self, query, database_dir, auto_load=True):
        database_dir = database_dir if query.ticker in database_dir else os.path.join(database_dir, query.ticker)
        SecurityData.__init__(self, query, database_dir, auto_load=auto_load)
        if auto_load:
            self.load()
        self.news_analyser = NewsAnalyser()

    def update(self, cols=None, exclude=None, init_date=None, end_date=utils.today()):
        if init_date is None:
            if self.latest_date == datetime.today().strftime(settings["time_format_str"]):
                logger.info("Database for stock {} is already up to date".format(self.query["ticker"]))
                return
            else:
                # Get the date of the day after self.last_date
                init_date = utils.text_to_datetime(self.latest_date) + timedelta(days=1)
                init_date = init_date.strftime(settings["time_format_str"])
        elif utils.text_to_datetime(init_date) > utils.text_to_datetime(end_date):
            logger.error("Init date is later than end date ({} > {})".format(init_date, end_date))
            return

        self.query.init_date = init_date
        self.query.end_date = end_date

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

        logger.info("Gathering data for stock {}. From {} to {}".format(self.query.ticker, init_date, end_date))
        logger.info("Data will be fetched for the following columns:")
        for col in cols:
            logger.info("\t{}".format(col))

        # ----------------------------------------------------------------------------------
        # Get info on how to retrieve columns
        col_info = settings["stock_cols"]

        # ----------------------------------
        # Retrieve columns from StockValues
        provider = StockValues(self.query)
        new_data = provider.get_cols([col for col in cols if col_info[col]["provider"] == "StockValues"])
        if new_data is None:
            logger.info("Failed to download data. Update cancelled")
            return

        all_trading_days = list(map(utils.datetime_to_text, provider.all_dates))
        filtered_dates = list(map(utils.datetime_to_text, provider.dates_filtered_by_price_range_std))
        filtered_dates_prev = [utils.days_from_date(date, -1) for date in filtered_dates]
        filtered_dates_plus_prev = list(set(filtered_dates + filtered_dates_prev))

        # ------------------------------------
        # Issue a news search to StockNews
        news_crawler = StockNewsCrawler(self.query)
        news_crawler.read_headlines(filter_dates=filtered_dates_plus_prev)
        # news_crawler.snap_to_closest_date(all_trading_days)
        # news_crawler.filter_dates(filtered_dates)

        # Download news headings and corresponding link, date and time
        news_crawler.save_headlines(self.db_dir)

        # Download bodies
        news_crawler.read_articles(save_continuously=True, save_dir=self.db_dir)

        # ------------------------------------
        # Retrieve columns from StockSentiment

        # 1) Set by default same probability for each of the 3 categories
        new_data["sentiment_p"] = pandas.DataFrame(1./3., index=new_data.index, columns=["sentiment_p"])
        new_data["sentiment_n"] = pandas.DataFrame(1./3., index=new_data.index, columns=["sentiment_n"])
        new_data["sentiment_u"] = pandas.DataFrame(1./3., index=new_data.index, columns=["sentiment_u"])

        # 2) Update those cells for which sentiment can be computed
        news = self.get_all_news()
        for date, content in news:
            sentiment_score = utils.normalise(self.news_analyser.analyse(collapse_daily_news(content['news'])))
            new_data.at[date, "sentiment_p"] = sentiment_score[0]
            new_data.at[date, "sentiment_n"] = sentiment_score[1]
            new_data.at[date, "sentiment_u"] = sentiment_score[2]

        # ------------------------------------
        # Update Database
        if os.path.exists(self.time_data_file_path):
            previous_data = pandas.read_csv(self.time_data_file_path, header=0, index_col=0)
            all_data = previous_data.append(new_data)
            if set(all_data.columns) == set(new_data.columns):
                all_data.to_csv(self.time_data_file_path)
            else:
                new_data_path = self.time_data_file_path[:-4] + "_new.csv"
                new_data.to_csv(new_data_path)
                logger.warning("New query had different columns than database")
                logger.warning("New time data has been saved to {}".format(new_data_path))
        else:
            new_data.to_csv(self.time_data_file_path)

    def initialise(self, init_date, end_date=utils.today()):
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        utils.update_info("latest_update", utils.today(), self.info_file_dir)
        self.update(init_date=init_date, end_date=end_date)

    @property
    def cols(self):
        return settings["stock_cols"].keys()

    def get_all_news(self, **kwargs):
        for file in utils.absolute_file_paths(os.path.join(self.db_dir, "news")):
            date = os.path.basename(file).replace('.json', '')
            return_obj = self.get_news_of_day(date, **kwargs)
            if return_obj is None:
                continue
            else:
                yield return_obj

    def get_news_of_day(self, date, pair_with_cols=(), snap_to_closest_date=True):
        news_file = os.path.join(self.db_dir, "news", date + '.json')
        if not os.path.exists(news_file):
            logger.warning("news file not found at ()".format(news_file))
        else:
            if snap_to_closest_date:
                date = utils.closest_date(date, list_of_dates=self.time_data.index)
                if date is None:
                    return None
            return date, {'news': json.load(open(news_file, 'r')),
                          **{col: self.time_data.at[date, col] for col in pair_with_cols}}
