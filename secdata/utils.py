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

import json
import logging
import os
from datetime import datetime, timedelta

import numpy as np

from secdata.settings import settings

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


def day_before(text_date):
    return datetime_to_text(text_to_datetime(text_date) - timedelta(days=1))


def day_after(text_date):
    return datetime_to_text(text_to_datetime(text_date) + timedelta(days=1))


def days_from_date(text_date, days):
    return datetime_to_text(text_to_datetime(text_date) + timedelta(days=days))


def text_to_datetime(date_str):
    assert isinstance(date_str, str)
    return datetime.strptime(date_str, settings["time_format_str"])


def datetime_to_text(date):
    assert isinstance(date, datetime)
    return date.strftime(settings["time_format_str"])


def softmax(vector, penalty=1.0):
    """
    Softmax function that favours the highest score
    :param vector: input vector of scores
    :param penalty: penalty for misclassification. Values in (0, 1.0) make highest element stand out
    :return: output of the softmax
    """
    import math as m
    if sum(vector) == 0:
        vector = np.array([1 for v in vector])
    vector = np.array(vector) / (penalty * sum(vector))
    e_scaled = []
    for value in vector:
        e_scaled.append(m.exp(value))
    sum_e = sum(e_scaled)
    return np.array(e_scaled) / sum_e


def absolute_file_paths(directory):
    for folder, _, file_names in os.walk(directory):
        for f in file_names:
            yield os.path.abspath(os.path.join(folder, f))


def closest_date(date, list_of_dates, direction="forward", max_days_forward=5):
    direction = 1 if direction == "forward" else -1
    days_forward = 10
    while date not in list_of_dates:
        date = datetime_to_text(text_to_datetime(date) + timedelta(days=direction))
        days_forward += 1
        if days_forward >= max_days_forward:
            date = None
            break
    return date


def normalise(vector):
    if sum(vector) == 0:
        return np.array([1./len(vector)]*len(vector))
    return np.array(vector) / sum(vector)

