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
from datetime import datetime
from dateutil import rrule

from secdata.news_scraper.items import NewsLink
from secdata import utils

import logging
# logging.getLogger('scrapy').setLevel(logging.INFO)
# logging.getLogger('scrapy.core.scraper').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


class ReutersSearch(Spider):
    name = "google_finance"
    allowed_domains = ["reuters.com"]

    def __init__(self, query, callback):
        scrapy.Spider.__init__(self)
        if hasattr(query, "filter_dates"):
            self.start_urls = ["http://www.reuters.com/resources/archive/us/{}.html".format(
                               utils.text_to_datetime(date).strftime('%Y%m%d')) for date in query.filter_dates]
        else:
            self.start_urls = ["http://www.reuters.com/resources/archive/us/{}.html".format(date.strftime('%Y%m%d'))
                               for date in rrule.rrule(rrule.DAILY,
                                                       dtstart=utils.text_to_datetime(query.init_date),
                                                       until=utils.text_to_datetime(query.end_date))]
        print(self.start_urls)
        self.query = query
        self.callback = callback

    def parse(self, response):
        sel = Selector(response)
        news = sel.xpath('//div[@class="module"]/div[@class="headlineMed"]')
        date = datetime.strptime(response.url.split('/')[-1].split('.')[0], "%Y%m%d").strftime("%Y-%m-%d")

        logger.info("Scraping news for date {}".format(date))
        for new in news:
            try:
                link = new.xpath('a/@href').extract()[0]
                title = new.xpath('a/text()').extract()[0]
            except Exception:
                logger.debug("   New discarded due to empty field")
                continue
            try:
                time = new.xpath('text()').extract()[0].strip()
            except Exception:
                time = ""
            for keyword in self.query.keywords:
                if keyword.lower() in title.lower():
                    item = NewsLink(date=date, header=title, link=link, time=time)
                    self.callback(item)
                    logger.info("   Headline: {}".format(title))
                    # yield item
                    break

