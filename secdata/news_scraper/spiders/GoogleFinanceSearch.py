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

import scrapy
from scrapy.spiders import Spider
from scrapy.selector import Selector
from dateutil import parser
from secdata.news_scraper.items import NewsLink
from datetime import datetime
from secdata import utils
from secdata.settings import settings

import logging
logging.getLogger('scrapy').setLevel(logging.FATAL)


class GoogleFinanceSearchSpider(Spider):
    name = "google_finance"
    allowed_domains = ["finance.google.com/finance/company_news"]

    def __init__(self, query, callback):
        scrapy.Spider.__init__(self)
        self.start_urls = [
            "https://finance.google.com/finance/company_news?q={}%3A{}&startdate={}&enddate={}&start=0&num=1000".
            format(query.exchange, query.ticker, query.start_date, query.end_date)]
        print(self.start_urls)
        self.callback = callback

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
                date = utils.today()
            else:
                date = parser.parse(date).strftime(settings["time_format_str"])
            yield NewsLink(date=date, title=title, link=link, time="")
