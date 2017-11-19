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

class Query:
    def __init__(self, init_date=None, end_date=None, **kwargs):
        self.init_date = init_date
        self.end_date = end_date
        for key, value in kwargs.items():
            self.add_attribute(key, value)

    def add_attribute(self, key, value):
        setattr(self, key, value)


class StockQuery(Query):
    def __init__(self, ticker, company_name="", keywords=(), market="Nasdaq", quandl_database="WIKI", **kwargs):
        Query.__init__(self, **kwargs)
        self.ticker = ticker
        self.company_name = company_name
        self.keywords = keywords
        self.market = market
        self.quandl_database = quandl_database


