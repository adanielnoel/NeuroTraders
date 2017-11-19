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

import pandas
import os
import json
from datetime import datetime
from secdata import utils
from secdata.settings import settings
import logging
logger = logging.getLogger(__name__)


class SecurityData:
    time_data_file_name = "time_data.csv"
    statistics_file_name = "fixed_data.json"
    info_file_name = "info.json"

    def __init__(self, query, database_dir, auto_load=True):
        self.time_data = pandas.DataFrame()
        self.statistics = dict()
        self.query = query
        self.db_dir = database_dir
        self.info_file_dir = os.path.join(self.db_dir, self.info_file_name)
        self.time_data_file_path = os.path.join(self.db_dir, self.time_data_file_name)
        self.statistics_file_path = os.path.join(self.db_dir, self.statistics_file_name)

    def __getitem__(self, attr):
        if attr in self.time_data.columns:
            return self.time_data[attr]
        elif attr in self.statistics:
            return self.statistics[attr]
        else:
            raise AttributeError

    def load(self):
        if not os.path.exists(self.db_dir):
            return
        # TODO: also save statistics
        # try:
        #     with open(self.statistics_file_dir, "r") as f:
        #         self.statistics = json.load(f)
        # except Exception as e:
        #     logger.warning("Statistics could not be loaded from {}".format(self.statistics_file_dir))
        #     logger.warning("  Reason: %s" % e)
        try:
            self.time_data = pandas.read_csv(self.time_data_file_path, header=0, index_col=0)
        except Exception as e:
            logger.warning("Time data could not be loaded from {}".format(self.time_data_file_path))
            logger.warning("  Reason: %s" % e)

    def save(self):
        self.time_data.to_csv(self.time_data_file_path)
        with open(self.statistics_file_path, "w") as f:
            json.dump(self.statistics, f)

    @property
    def cols(self):
        return NotImplementedError

    @property
    def latest_date(self):
        try:
            utils.get_info("latest_date", self.info_file_dir)
        except (AttributeError, FileNotFoundError, FileExistsError):
            return None

    @latest_date.setter
    def latest_date(self, date):
        if isinstance(date, datetime):
            date = date.strftime(settings["time_format_str"])
        utils.update_info("latest_date", date, self.info_file_dir)








