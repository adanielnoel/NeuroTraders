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

    def __init__(self, database_dir, auto_load=True):
        self.time_data = pandas.DataFrame()
        self.statistics = dict()
        self.db_dir = database_dir
        self.info_file_dir = os.path.join(self.db_dir, self.info_file_name)
        self.time_data_file_dir = os.path.join(self.db_dir, self.time_data_file_name)
        self.statistics_file_dir = os.path.join(self.db_dir, self.statistics_file_name)

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
        try:
            with open(self.statistics_file_dir, "r") as f:
                self.statistics = json.load(f)
        except Exception as e:
            logger.warning("Statistics could not be loaded from {}".format(self.statistics_file_dir))
            logger.warning("  Reason: %s" % e)
        try:
            self.time_data_file_name = pandas.read_csv(self.time_data_file_dir, header=0, index_col=0)
        except Exception as e:
            logger.warning("Time data could not be loaded from {}".format(self.time_data_file_dir))
            logger.warning("  Reason: %s" % e)

    def save(self):
        self.time_data.to_csv(self.time_data_file_dir)
        with open(self.statistics_file_dir, "w") as f:
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








