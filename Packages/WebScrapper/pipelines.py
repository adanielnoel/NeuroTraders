# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from Packages.WebScrapper.items import NewsLink


class NewsLinkPipeline(object):
    @staticmethod
    def process_item(self, item, spider):
        if isinstance(item, NewsLink):
            spider.manager_crawler.add_headline(dict(item))
            # TODO: Insert into database instead


