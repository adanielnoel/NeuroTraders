# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.

from Packages.WebScrapper.spiders.FinancialTimesSearchSpider import FinancialTimesSearchSpider
from Packages.WebScrapper.spiders.GoogleFinanceSearch import GoogleFinanceSearchSpider
from Packages.WebScrapper.spiders.ReutersSearch import ReutersSearch

# search_spiders = [GoogleFinanceSearchSpider, ReutersSearch
search_spiders = [ReutersSearch]
scraping_spiders = []

