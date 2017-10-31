# -*- coding: utf-8 -*-
import scrapy
import requests
import httplib2


class FinancialTimesSearchSpider(scrapy.Spider):
    name = "financial_times"
    allowed_domains = ["ft.com"]
    start_urls = []

    company_names = {"appl": "Apple_Inc",
                     "googl": "Google_Inc"}

    def __init__(self, ticker, start_date, end_date):
        scrapy.Spider.__init__(self)
        assert ticker in FinancialTimesSearchSpider.company_names.keys()
        company_name = FinancialTimesSearchSpider.company_names[ticker]
        page_num = 1
        h = httplib2.Http()
        while True:
            page = "https://www.ft.com/topics/organisations/{}?page={}".format(company_name, page_num)
            request = h.request(page, 'HEAD')
            if int(request[0]['status']) == 200:
                self.start_urls.append(page)
                page_num += 1
            else:
                break


    def parse(self, response):
        pass
