from secdata.StockData import StockData
import os
import logging
logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s]%(levelname)s: %(message)s", level=logging.DEBUG)


stock_data = StockData("pulm", "./new_database")
stock_data.initialise("2011-10-13")

