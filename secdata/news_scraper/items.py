import scrapy


class NewsLink(scrapy.Item):
    header = scrapy.Field()
    link = scrapy.Field()
    date = scrapy.Field()
    time = scrapy.Field()
