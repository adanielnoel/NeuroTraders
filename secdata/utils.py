import os
import json
from datetime import datetime
from secdata.settings import settings
import logging
logger = logging.getLogger(__name__)


def update_info(key, value, info_file_path):
    if os.path.exists(info_file_path):
        info = json.load(open(info_file_path, "r"))
    else:
        info = {}
    info[key] = value
    json.dump(info, open(info_file_path, "w"))


def get_info(key, info_file_path):
    if not os.path.exists(info_file_path):
        logger.error("Info file \"{}\" does not exist".format(info_file_path))
        return None
    info = json.load(open(info_file_path, "r"))
    return info[key]


def today():
    return datetime.today().strftime(settings["time_format_str"])


def to_datetime(date_str):
    return datetime.strptime(date_str, settings["time_format_str"])

