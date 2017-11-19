# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

from secdata.news_scraper.spiders.GoogleFinanceSearch import GoogleFinanceSearchSpider
from secdata.news_scraper.spiders.ReutersSearch import ReutersSearch

search_spiders = [GoogleFinanceSearchSpider, ReutersSearch]

