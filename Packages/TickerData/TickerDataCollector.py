# from Packages.WebScrapper.stockQueryProcess import StockQueryProcess
import json
import os
from datetime import datetime, timedelta

import pandas
import quandl

quandl.ApiConfig.api_key = "bYNHAsGguxFWJsjg3ccN"

from Packages.WebScrapper.stockNewsCrawler import StockNewsCrawler
import logging

ARTICLES = "ARTICLES"
INDICATORS = "INDICATORS"


class TickerDataCollector:
    """
    Collects data of a certain ticker. Data includes stock price, financial indicators and sentiment analysis
    results.
    """
    record_file_name = "record.csv"
    info_file_name = "info.json"
    news_file_name = "news_articles.json"

    def __init__(self):
        self.data = {}
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.record_path = None
        self.row_count = 0

        self.keys_in_order = ["dates",
                              "adj_open",
                              "adj_close",
                              "volume",
                              "price_m_ave_50",
                              "price_m_ave_100",
                              "volume_m_ave_50",
                              "relative_change",
                              "positive_sentiment",
                              "negative_sentiment",
                              "uncertainty_sentiment",
                              "news_volume"]

    def reset(self):
        self.data = {}
        self.ticker = None
        self.start_date = None
        self.end_date = None
        self.record_path = None
        self.row_count = 0

    def create_record(self):
        self.record_path = "../../Database/{}".format(self.ticker)
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
            with open(self.record_path + '/' + self.record_file_name, "w") as f:
                f.writelines([','.join(self.keys_in_order) + '\n'])
            return True
        else:
            return False

    def get_latest_date(self):
        return

    def collect(self, ticker, company, start_date, end_date):
        self.reset()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

        if not self.create_record():
            new_start_date = datetime.strptime(self.get_info("latest date"), "%Y-%m-%d")
            logging.info("Part of the queried dates are already in the record")
            logging.info("\tStarting from {} instead of {}".format(new_start_date.strftime("%Y-%m-%d"),
                                                                   start_date.strftime("%Y-%m-%d")))
            # start_date = new_start_date
        self.update_info("latest update", datetime.today().strftime("%Y-%m-%d"))
        self.update_info("company_name", company)

        logging.info("Gathering historical stock price data")
        quandl_data = quandl.get("WIKI/{}".format(ticker),
                                 start_date=start_date,
                                 end_date=end_date)
        self.row_count = quandl_data.shape[0]

        logging.info("Computing generic stock indicators")
        self.data["dates"] = [pandas.to_datetime(date).strftime("%Y-%m-%d") for date in quandl_data.index.values]
        self.update_info("latest date", self.data["dates"][-1])
        self.data["adj_open"] = list(quandl_data['Adj. Open'].values)
        self.data["adj_close"] = list(quandl_data['Adj. Close'].values)
        self.data["volume"] = list(quandl_data['Adj. Volume'].values)
        self.data["price_m_ave_50"] = list(quandl_data['Adj. Close'].rolling(window=50).mean().values)
        self.data["price_m_ave_100"] = list(quandl_data['Adj. Close'].rolling(window=100).mean().values)
        self.data["volume_m_ave_50"] = list(quandl_data['Adj. Volume'].rolling(window=100).mean().values)

        relative_change = [0.0]
        for i in range(1, self.row_count):
            close_yesterday = quandl_data['Adj. Close'].iloc[[i - 1]].values[0]
            close_today = quandl_data['Adj. Close'].iloc[[i]].values[0]
            relative_change.append(close_today / close_yesterday - 1.0)
        self.data["relative_change"] = relative_change

        logging.info("Downloading stock news")
        crawler = StockNewsCrawler(ticker, start_date, end_date)
        # new_articles = crawler.get_news()
        new_articles = self.get_record(ARTICLES)

        # all_articles = self.get_record(ARTICLES) + new_articles
        all_articles = new_articles
        # self.update_record(all_articles, ARTICLES)

        logging.info("Computing sentiment scores")
        sentiment_scores = crawler.analyse_articles(new_articles)
        self.data["positive_sentiment"] = [0] * self.row_count
        self.data["negative_sentiment"] = [0] * self.row_count
        self.data["uncertainty_sentiment"] = [0] * self.row_count
        self.data["news_volume"] = [0] * self.row_count
        for date, info in sentiment_scores.items():
            score = info["score"]
            article_count = info["article_count"]
            index = -1
            for i in range(6):  # Look for the first open market date in the next 3 days
                try:
                    index = self.data["dates"].index(date)
                    self.data["positive_sentiment"][index] += score[0]
                    self.data["negative_sentiment"][index] += score[1]
                    self.data["uncertainty_sentiment"][index] += score[2]
                    self.data["news_volume"][index] += article_count
                    break
                except (ValueError, IndexError):
                    # Add 1 day to the date
                    logging.warning("No trading data for day {}, trying next day".format(date))
                    date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        logging.info("Updating persistent record")
        self.update_record(self.data, INDICATORS)

    def declare_new_query(self):
        identifier = self.ticker + "_" + self.start_date.strftime("%Y-%m-%d") + "-" + self.end_date.strftime("%Y-%m-%d")
        description = {"Identifier": identifier,
                       "Issue date": datetime.today().strftime("%Y-%m-%d_%H:%M:%S"),
                       "Ticker": self.ticker,
                       "init_date": self.start_date.strftime("%Y-%m-%d"),
                       "end_date": self.end_date.strftime("%Y-%m-%d")}
        # updated_query_list = StockQuery.get_query_history().append(description)
        # StockQuery.save_query_history(updated_query_list)
        return description

    def get_record(self, record_type):
        if record_type == "INDICATORS":
            with open(self.record_path + '/' + self.record_file_name, "r") as f:
                pass
        elif record_type == "ARTICLES":
            news_file = self.record_path + "/" + self.news_file_name
            if not os.path.exists(news_file):
                return []
            with open(self.record_path + "/" + self.news_file_name, 'r') as f:
                return json.load(f)

    def update_record(self, data, record_type):
        if record_type == "INDICATORS":
            with open(self.record_path + '/' + self.record_file_name, "a") as f:
                lines = []
                for i in range(self.row_count):
                    lines.append(','.join(map(str, [data[key][i] for key in self.keys_in_order])) + '\n')
                f.writelines(lines)
        elif record_type == "ARTICLES":
            with open(self.record_path + "/" + self.news_file_name, 'w') as f:
                json.dump(data, f, indent=4)

    def update_info(self, key, value):
        info_file_path = self.record_path + '/' + self.info_file_name
        if os.path.exists(info_file_path):
            info = json.load(open(info_file_path, "r"))
        else:
            info = {}
        info[key] = value
        json.dump(info, open(self.record_path + '/' + self.info_file_name, "w"))

    def get_info(self, key):
        info = json.load(open(self.record_path + '/' + self.info_file_name, "r"))
        return info[key]


if __name__ == "__main__":
    data_collector = TickerDataCollector()
    data_collector.collect("AAPL", "Apple",
                           datetime(year=2007, month=1, day=3),
                           datetime(year=2017, month=10, day=13))
