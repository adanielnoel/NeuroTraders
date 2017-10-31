import scrapy
from scrapy.spiders import Spider
from scrapy.selector import Selector
from dateutil import parser
from Packages.WebScrapper.items import NewsLink
from datetime import datetime

import logging
logging.getLogger('scrapy').setLevel(logging.FATAL)


class GoogleFinanceSearchSpider(Spider):
    name = "google_finance"
    allowed_domains = ["finance.google.com/finance/company_news"]

    def __init__(self, ticker, start_date, end_date, exchange, manager_crawler):
        scrapy.Spider.__init__(self)
        self.start_urls = [
            "https://finance.google.com/finance/company_news?q={}%3A{}&startdate={}&enddate={}&start=0&num=1000".
            format(exchange, ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))]
        print(self.start_urls)
        self.manager_crawler = manager_crawler
        # TODO: Fix this shitty way of passing around the crawler

    def parse(self, response):
        print(response.url)
        sel = Selector(response)
        # news = sel.xpath('//div[@class="news"]').extract()
        # news = sel.css('.news') #.css('.name').xpath('a/text()').extract()
        news = sel.xpath('//div[contains(concat(" ", normalize-space(@class), " "), "news")'
                         'and not(contains(concat(" ",normalize-space(@id)," "),"articles-published"))]')
        for new in news:
            title = new.css('.name').xpath('a/text()').extract()[0].replace('\u00a0', ' ').replace('\"', "'")
            link = new.css('.name').xpath('a/@href').extract()[0]
            date = new.css('.date::text').extract()[0]
            # Convert date to be consistent with the rest of the project
            print(date)
            if "ago" in date:
                date = datetime.today().strftime("%Y-%m-%d")
            else:
                date = parser.parse(date).strftime("%Y-%m-%d")
            yield NewsLink(date=date, title=title, link=link)
