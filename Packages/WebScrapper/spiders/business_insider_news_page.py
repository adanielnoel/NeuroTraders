# -*- coding: utf-8 -*-
import scrapy


class BusinessInsiderNewsPageSpider(scrapy.Spider):
    name = "businessInsiderNewsPage"
    allowed_domains = ["https://www.bloomberg.com"]
    start_urls = ['http://https://www.bloomberg.com/']

    def __init__(self):
        super.__init__(self)

    def parse(self, response):
        pass
