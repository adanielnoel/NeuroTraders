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

import difflib
import logging
import os
from random import shuffle
import json

import scrapy.settings
from goose3 import Goose
from goose3.network import NetworkError
from requests.exceptions import MissingSchema
from scrapy.crawler import CrawlerProcess

import secdata.news_scraper.spiders as spiders
from secdata import utils
from secdata.Query import StockQuery
from secdata.news_scraper.UserAgents import UserAgent
from secdata.news_scraper.items import NewsLink
from secdata.settings import settings

# logging.getLogger('scrapy').setLevel(logging.FATAL)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockNewsCrawler:
    heads_file_name = "headlines.json"
    news_folder = "news"
    news_file_name = "news.json"

    def __init__(self, query):
        self.query = query
        self.headlines = {}
        self.news = {}

    def read_headlines(self, filter_dates=None):
        if filter_dates is not None:
            self.query.add_attribute("filter_dates", filter_dates)
        process_settings = scrapy.settings.Settings()
        process_settings.set("USER_AGENT", UserAgent().random())
        process_settings.set("LOG_ENABLED", "False")
        process = CrawlerProcess(process_settings)
        for Spider in spiders.search_spiders:
            process.crawl(Spider, **{"query": self.query, "callback": self.add_headline})
        logger.info("Starting headline scraping")
        process.start()
        process.stop()
        logger.info("Finished headline scraping")
        self.remove_duplicates()
        return self.headlines

    def add_headline(self, item):
        if isinstance(item, NewsLink):
            new_entry = {"header": item["header"], "link": item["link"], "body": "", "time": item["time"]}
            if item["date"] not in self.headlines.keys():
                self.headlines[item["date"]] = [new_entry]
            else:
                self.headlines[item["date"]].append(new_entry)
        else:
            logger.warning("Received unknown item")

    def remove_duplicates(self):
        def are_duplicates(str1, str2):
            return difflib.SequenceMatcher(a=str1, b=str2).ratio() > 0.9

        logger.info("Looking for duplicates")

        duplicate_count = 0
        filtered_dict = {}
        for date, headlines in self.headlines.items():
            filtered_dict[date] = self.headlines[date].copy()
            for headline1 in headlines:
                found_duplicate = True
                while found_duplicate:
                    temp = filtered_dict[date].copy()
                    found_duplicate = False
                    found_once = False
                    for headline2 in temp:
                        if are_duplicates(headline1["header"], headline2["header"]):
                            if found_once:
                                filtered_dict[date].remove(headline2)
                                found_duplicate = True
                                duplicate_count += 1
                                break
                            else:
                                found_once = True
        self.headlines = filtered_dict
        logger.info("Removed {} duplicate headlines".format(duplicate_count))

    def snap_to_closest_date(self, dates, time_of_day_threshold="9:30am EST"):
        logger.info("Snapping headlines to closest date")
        # TODO: account for time of publication (move news after market opens to next day)
        #       kind of important, nonetheless companies usually publish their official
        #       announcements before market opens, so statistically it's sounds acceptable to not do it.
        len_before = len(self.headlines)
        readjusted_heads = {}
        for headline_date in self.headlines.keys():
            for i in range(7):
                test_day = utils.days_from_date(headline_date, i)
                if test_day in dates:
                    if test_day not in readjusted_heads.keys():
                        readjusted_heads[test_day] = self.headlines[headline_date]
                    else:
                        readjusted_heads[test_day] += self.headlines[headline_date]
                    break
                logger.warning("News on {} ignored, no trading day found close".format(headline_date))
        self.headlines = readjusted_heads
        logger.info("Snapped to closest date forward, {} news days grouped".format(len_before - len(self.headlines)))
        return self.headlines

    def filter_dates(self, dates):
        len_before = len(self.headlines)
        self.headlines = {date: self.headlines[date] for date in dates if date in self.headlines.keys()}
        logger.info("Filter applied, {} news days removed".format(len_before - len(self.headlines)))
        return self.headlines

    def read_articles(self, headlines=None, save_continuously=False, save_dir=""):
        if headlines is None:
            headlines = self.headlines
        extractor = Goose()
        for date, daily_news in headlines.items():
            # Shuffle since if there are too many some will be ignored
            # and we want the ignored ones to be randomly deselected
            shuffle(daily_news)

            news_read = []
            for new in daily_news:
                try:
                    body = extractor.extract(url=new["link"]).cleaned_text
                    news_read.append({**new, "body": body})
                    if len(self.news) == settings["max_news_per_day"]:
                        break
                except NetworkError:
                    logger.error("Page not found in {}".format(new["link"]))
                except MissingSchema:
                    logger.warning("Couldn't read link {}".format(new["link"]))
                    logger.warning("  Reason: string 'http://' might be missing")
                except Exception as e:
                    logger.warning("Unknown exception while trying to read {}".format(new["link"]))
                    logger.warning("   {}".format(e))
            if len(news_read) > 0:
                self.news[date] = news_read
                if save_continuously:
                    if save_dir == "":
                        logger.warning("Please provide a save directory")
                    else:
                        self.save_news(save_dir, {date: news_read})
        logger.info("From {} headlines, {} of their articles where correctly downloaded".format(
            sum([len(headers) for headers in self.headlines.values()]),
            sum([len(day_news) for day_news in self.news.values()])))
        return self.news

    def save_headlines(self, save_dir):
        logger.info("Saving news headlines to {}".format(save_dir))
        if not os.path.exists(save_dir):
            logger.error("Directory not found, headlines will not be saved")
        else:
            headlines_file_path = os.path.join(save_dir, self.heads_file_name)
            if os.path.exists(headlines_file_path):
                prev_headlines = json.load(open(headlines_file_path, "r"))
            else:
                prev_headlines = {}
            with open(os.path.join(save_dir, self.heads_file_name), "w") as f:
                json.dump({**prev_headlines, **self.headlines}, f, indent=4)

    def save_news(self, save_dir, news=None):
        if news is None:
            news = self.news
        save_dir = os.path.join(save_dir, self.news_folder)
        logger.info("Saving news articles to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for date, daily_news in news.items():
            json.dump(daily_news, open(os.path.join(save_dir, date + ".json"), "w"), indent=4)
            logger.info("Saved {} news for day {}".format(len(news[date]), date))

    def save_all(self, save_dir):
        self.save_headlines(save_dir)
        self.save_news(save_dir)


if __name__ == "__main__":
    crawler = StockNewsCrawler(StockQuery(ticker="AAPL",
                                          company_name="Apple",
                                          init_date="2009-11-03",
                                          end_date="2009-12-10",
                                          keywords=["Apple", "Iphone"],
                                          exchange="NASDAQ"))

    heads = crawler.read_headlines()
    json.dump(heads, open("./heads.json", "w"), indent=4)
    arts = crawler.read_articles()
    json.dump(arts, open("./bodies.json", "w"), indent=4)
