import logging
from collections import defaultdict
import numpy as np
import collections
from itertools import chain

import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree
from nltk.chunk import ne_chunk
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
logger = logging.getLogger(__name__)


def collapse_daily_news(news_list):
    assert isinstance(news_list, list)
    total_day_text = ""
    for news_article in news_list:
        new_text = news_article["header"] + " " + news_article["body"] + " "
        if is_english(new_text):
            total_day_text += new_text
    return total_day_text


def sum_dicts(dicts):
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


def is_english(text):
    languages_ratios = {}

    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    # Compute per language included in nltk number of unique stopwords appearing in analyzed text
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)

        languages_ratios[language] = len(common_elements)  # language "score"

    most_rated_language = max(languages_ratios, key=languages_ratios.get)

    return most_rated_language == 'english'


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_names(sentence):
    assert isinstance(sentence, list), "Sentence must be tokenized first"
    tagged_sent = nltk.tag.pos_tag(sentence)
    names = [i[0] for i in list(chain(*[chunk.leaves() for chunk in ne_chunk(tagged_sent) if isinstance(chunk, Tree)]))]
    possessives = [word for word in sentence if word.endswith("s'")]
    return names + possessives

class Scaler(object):
    def __init__(self, map_range=(0.0, 1.0)):
        self.min = np.zeros(1)
        self.max = np.zeros(1)
        self.mapped_mean = np.zeros(1)
        self.mapped_std = np.zeros(1)
        self._map_range = map_range
        self.ncols = 1

    @property
    def map_size(self):
        return self._map_range[1] - self._map_range[0]

    def scale(self):
        return (self.max - self.min) / np.array([self.map_size]*self.ncols)

    @property
    def map_range(self):
        return self._map_range

    @map_range.setter
    def map_range(self, value):
        assert len(value) == 2
        assert value[1] > value[0]
        self._map_range = value

    def fit(self, array):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        if array.ndim == 1:
            array = array.reshape(-1, 1)

        self.ncols = array.shape[1]
        self.min = np.array([col.min() for col in array.transpose()])
        self.max = np.array([col.max() for col in array.transpose()])

        temp = self.apply(array, standarize=False)
        self.mapped_mean = np.array([col.mean() for col in temp.transpose()])
        self.mapped_std = np.array([col.std() for col in temp.transpose()])
        return self

    def apply(self, array, standarize=True):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        return_1d = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            return_1d = True
        assert array.shape[1] == self.ncols

        array = ((array - self.min) / self.scale().transpose()) + self.map_range[0]

        if standarize:
            array = (array - self.mapped_mean)

        if return_1d:
            return array[:, 0]
        else:
            return array

    def revert(self, array, standarize=True):
        assert isinstance(array, collections.Iterable)
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        assert array.ndim == 1 or array.ndim == 2
        return_1d = False
        if array.ndim == 1:
            array = array.reshape(-1, 1)
            return_1d = True
        assert array.shape[1] == self.ncols

        if standarize:
            array = (array) + self.mapped_mean
            # TODO: Bring back standardization

        scale = 1.0 / self.scale()
        array = ((array - self.map_range[0]) / scale.transpose()) + self.min

        if return_1d:
            return array[:, 0]
        else:
            return array


