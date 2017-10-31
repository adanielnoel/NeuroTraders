import scrapy
from scrapy.spiders import Spider
from scrapy.selector import Selector
from scrapy import Request
from dateutil import parser
from secdata.news_scraper.items import NewsLink
from datetime import datetime
import nltk
from Database.utils import get_company_name

import logging
logging.getLogger('scrapy').setLevel(logging.FATAL)


class ReutersSearch(Spider):
    name = "google_finance"
    allowed_domains = ["reuters.com"]

    def __init__(self, ticker, start_date, end_date, exchange, manager_crawler):
        scrapy.Spider.__init__(self)
        self.company_name = get_company_name(ticker).lower()
        self.ticker = ticker
        self.start_urls = ["http://www.reuters.com/resources/archive/us/2007.html",
                           "http://www.reuters.com/resources/archive/us/2008.html",
                           "http://www.reuters.com/resources/archive/us/2009.html",
                           "http://www.reuters.com/resources/archive/us/2010.html",
                           "http://www.reuters.com/resources/archive/us/2011.html",
                           "http://www.reuters.com/resources/archive/us/2012.html",
                           "http://www.reuters.com/resources/archive/us/2013.html",
                           "http://www.reuters.com/resources/archive/us/2014.html",
                           "http://www.reuters.com/resources/archive/us/2015.html",
                           "http://www.reuters.com/resources/archive/us/2016.html",
                           "http://www.reuters.com/resources/archive/us/2017.html"]
        self.manager_crawler = manager_crawler
        # TODO: Fix this shitty way of passing around the crawler

    def parse(self, response):
        print(response.url)
        sel = Selector(response)
        # news = sel.xpath('//div[@class="news"]').extract()
        # news = sel.css('.news') #.css('.name').xpath('a/text()').extract()
        days = sel.xpath('//div[@class="moduleBody"]/p/a/@href').extract()
        for day_link in days:
            yield Request("http://www.reuters.com" + day_link, callback=self.parse_news_list)

    def parse_news_list(self, response):
        sel = Selector(response)
        news= sel.xpath('//div[@class="module"]//a')
        date = datetime.strptime(response.url.split('/')[-1].split('.')[0], "%Y%m%d").strftime("%Y-%m-%d")

        for new in news:
            link = new.xpath('@href').extract()[0]
            title = new.xpath('text()').extract()[0].lower()
            if self.company_name in title or self.ticker in title:
                yield NewsLink(date=date, title=title, link=link)

