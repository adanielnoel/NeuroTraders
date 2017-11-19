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

import logging

from secdata.Query import StockQuery
from secdata.StockData import StockData

logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s]%(levelname)s: %(message)s", level=logging.INFO)

# query = StockQuery(ticker="aapl",
#                    company_name="Apple",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" aapl", " apple", "iphone", "macbook", " osx", "ipad", " ios"]
#                    )
# query = StockQuery(ticker="amzn",
#                    company_name="Amazon",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=["Amazon", " amzn", "kindle", " aws", "Jeff Bezos"],
#                    )
# query = StockQuery(ticker="wmt",
#                    company_name="Walmart",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=["Walmart", " wmt", "Wal-Mart", "wal mart", " Asda ", "Seiyu Group", "Walton family",
#                              "Walton enterprises",
#                              "Doug McMillon"],
#                    )
# query = StockQuery(ticker="jpm",
#                    company_name="JPMorgan",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" jpm ", "JPMorgan", "Chase & Co", "J.P. Morgan",
#                             "J.P.Morgan", "JP Morgan", "J P Morgan"],
#                    )
# query = StockQuery(ticker="msft",
#                    company_name="Microsoft",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" msft", "Microsoft", "Visual Studio", "Windows", "XBox",
#                              "X Box", "direct3d", "kinect", " havok", "Silverlight",
#                              "Skype", "OneDrive", "MS-DOS", " Azure",
#                              " Cortana", "DirectX"],
#                    )
# query = StockQuery(ticker="mmm",
#                    company_name="3M",
#                    market="nyse",
#                    quandl_database="WIKI",
#                    keywords=[" mmm ", " 3M "],
#                    )
# query = StockQuery(ticker="amd",
#                    company_name="AMD",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" amd ", "Radeon", "Ryzen", "Opteron", "Phenom ", " Athlon",
#                              "Sempron"],
#                    )
# query = StockQuery(ticker="vrx",
#                    company_name="Valiant Pharmaceuticals",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" vrx ", "Valeant"],
#                    )
# query = StockQuery(ticker="tsla",
#                    company_name="Tesla",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=["tsla", "Tesla", "model 3", "model3", "model s", "Gigafactory"],
#                    )
# query = StockQuery(ticker="nvda",
#                    company_name="Nvidia",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" nvda ", "Nvidia"],
#                    )
# query = StockQuery(ticker="googl",
#                    company_name="Google",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" googl ", "Google", "Alphabet"],
#                    )
# query = StockQuery(ticker="intc",
#                    company_name="Intel",
#                    market="nasdaq",
#                    quandl_database="WIKI",
#                    keywords=[" intc ", "Intel "],
#                    )
query = StockQuery(ticker="nflx",
                   company_name="Netflix",
                   market="nasdaq",
                   quandl_database="WIKI",
                   keywords=[" nflx ", "Netflix"],
                   )
# queries = [StockQuery(ticker="aapl"),
#            StockQuery(ticker="amd"),
#            StockQuery(ticker="amzn"),
#            StockQuery(ticker="googl"),
#            StockQuery(ticker="intc"),
#            StockQuery(ticker="jpm"),
#            StockQuery(ticker="mmm"),
#            StockQuery(ticker="msft"),
#            StockQuery(ticker="nflx"),
#            StockQuery(ticker="nvda"),
#            StockQuery(ticker="tsla"),
#            StockQuery(ticker="vrx"),
#            StockQuery(ticker="wmt")]

stock_data = StockData(query, "./new_database")

# for query in queries:
#     stock_data = StockData(query, "./new_database")
#     stock_data.initialise(init_date="2007-01-01", end_date="2007-02-02")
# stock_data.update(end_date="2011-10-25")
