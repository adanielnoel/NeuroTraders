from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings
from scrapy import signals
import secdata.news_scraper.spiders as spiders
from datetime import datetime, timedelta
import nltk
nltk.download('punkt')
# Todo: check if package is downloaded
import numpy as np
import os
from goose3 import Goose
import unicodedata

import logging
logging.getLogger('scrapy').setLevel(logging.FATAL)


class Headline:
    def __init__(self):
        title = ""
        link = ""
        date = ""


class StockNewsCrawler:

    def __init__(self,
                 ticker,
                 init_date=datetime.now()-timedelta(days=10),
                 end_date=datetime.today(),
                 exchange="NASDAQ"):
        self.ticker = ticker
        self.exchange = exchange
        self.start_date = init_date
        self.end_date = end_date
        self.links = {}  # key: spider name, value: list of links to process
        self.pages = []  # List of downloaded pages (each element is a dictionary)
        self.scores = {}  # Dictionary of dates with article count and (cumulative) sentiment score for each

        with open(os.path.dirname(os.path.realpath(__file__)) + "/resources/positive.csv", "r") as f:
            self.positive_tks = [l.strip() for l in f.readlines()]
        with open(os.path.dirname(os.path.realpath(__file__)) + "/resources/negative.csv", "r") as f:
            self.negative_tks = [l.strip() for l in f.readlines()]
        with open(os.path.dirname(os.path.realpath(__file__)) + "/resources/uncertainty.csv", "r") as f:
            self.uncertainty_tks = [l.strip() for l in f.readlines()]

    def get_headlines(self):
        """

        :return: a list of Headline objects
        """
        # STEP 1: Search for the links of news pages
        settings = Settings()
        settings.set("USER_AGENT",
                     "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Safari/604.1.38")
        # settings.set("ITEM_PIPELINES", {'pipelines.NewsLinkPipeline': 100})
        crawler = CrawlerProcess(settings)
        # crawler.settings.
        for Spider in spiders.search_spiders:
            crawler.crawl(Spider,
                          ticker=self.ticker,
                          exchange=self.exchange,
                          start_date=self.start_date,
                          end_date=self.end_date,
                          manager_crawler=self)
        # for individual_crawler in crawler.crawlers:
        #     individual_crawler.signals.connect(self.add_headline, signal=signals.engine_started)
            # TODO: Check what's wrong with the signals (not getting triggered)
        crawler.start()
        crawler.stop()

        for page in self.pages:
            page["body"] = ""
            if "www.reuters.com" in page["link"]:
                continue
            try:
                article = Goose().extract(url=page["link"])
                body = article.cleaned_text
                body = unicodedata.normalize('NFKD', body).encode('ascii', 'ignore').decode("utf-8").replace('\u00a0', ' ')
                page["body"] = body
            except:
                continue

        return self.pages

    def analyse_articles(self, articles=None):
        if articles is None:
            articles = self.pages
        self.scores = {}
        for article in articles:
            print('Analysing "%s"' % article["title"])
            scores = self.process_sentiment(article["title"], article["body"])
            if article["date"] in self.scores.keys():
                self.scores[article["date"]]["score"] += np.array(scores)
                self.scores[article["date"]]["article_count"] += 1
            else:
                self.scores[article["date"]] = {}
                self.scores[article["date"]]["score"] = np.array(scores)
                self.scores[article["date"]]["article_count"] = 1
        # Snap article count into levels 0% = 0, 33% < 10, 66% < 100, 100% > 100
        for key in self.scores.keys():
            if self.scores[key]["article_count"] == 0:
                continue
            elif self.scores[key]["article_count"] < 10:
                self.scores[key]["article_count"] = 0.33
            elif self.scores[key]["article_count"] < 100:
                self.scores[key]["article_count"] = 0.66
            else:
                self.scores[key]["article_count"] = 1.0
        return self.scores

    def process_sentiment(self, title, body):
        # TODO: Check stemming as a better method
        # TODO: https://machinelearnings.co/text-classification-using-neural-networks-f5cd7b8765c6
        tokens = nltk.word_tokenize(title + " " + body)
        positive_pts = 0
        negative_pts = 0
        uncertainty_pts = 0
        for token in tokens:
            token = token.upper()
            for positive_tkn in self.positive_tks:
                if token == positive_tkn:
                    positive_pts += 1
            for negative_tkn in self.negative_tks:
                if token == negative_tkn:
                    negative_pts += 1
            for uncertainty_tkn in self.uncertainty_tks:
                if token == uncertainty_tkn:
                    uncertainty_pts += 1
        return self.softmax((positive_pts, negative_pts, uncertainty_pts))

    @staticmethod
    def softmax(vector):
        """
        Softmax function that favours the highest score
        :param vector: input vector of scores
        :return: output of the softmax
        """
        import math as m
        vector = np.array(vector) / (0.5 * sum(vector))
        e_scaled = []
        for value in vector:
            e_scaled.append(m.exp(value))
        sum_e = sum(e_scaled)
        return np.array(e_scaled) / sum_e

    def add_headline(self, item):
        self.pages.append(item)


if __name__ == "__main__":
    # pipelines.results = []
    crawler = StockNewsCrawler(ticker="AAPL",
                               init_date=datetime(year=2015, month=1, day=1),
                               end_date=datetime.today(),
                               exchange="NASDAQ")
    import json
    articles = crawler.get_headlines()
    articles = json.load(open("news_data.json", "r"))["news"]

    for article in articles[:20]:
        try:
            art = Goose().extract(url=article["link"])
            article["body"] = art.cleaned_text
            print(article)
        except:
            continue

    with open("news_data.json", "w") as f:
        json.dump({"news": articles}, f, indent=4)

    crawler.analyse_articles(articles[:6000])
    for date, score in crawler.scores.items():
        print(date, score["score"])
