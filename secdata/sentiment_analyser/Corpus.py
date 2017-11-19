import nltk
import os
import logging
import json
from secdata.sentiment_analyser.utils import sum_dicts, get_names
from secdata.sentiment_analyser import text_processing as tp
logger = logging.getLogger(__name__)


class Corpus:
    word_freq_file_name = "word_frequencies.json"

    def __init__(self, corpus_dir=""):
        self.word_counts = {}
        self.corpus_dir = corpus_dir
        self.word_freq_file_path = os.path.join(self.corpus_dir, self.word_freq_file_name)
        if corpus_dir != "":
            self.load_stats()

    @property
    def words(self):
        return self.word_counts.keys()

    @property
    def stems(self):
        return list(set(tp.stem(self.words)))

    def new_text(self, text):
        tokens = tp.tokenize(text, remove_punctuation=True)
        self.word_counts = sum_dicts((self.word_counts, nltk.FreqDist(tokens)))

    def save_stats(self, corpus_dir=None):
        corpus_dir = self.corpus_dir if corpus_dir is None else corpus_dir
        if not os.path.exists(corpus_dir):
            logger.warning("Path '{}' not found".format(corpus_dir))
        else:
            json.dump(self.word_counts, open(self.word_freq_file_path, "w"))

    def load_stats(self, corpus_dir=None):
        corpus_dir = self.corpus_dir if corpus_dir is None else corpus_dir
        word_freq_file_path = os.path.join(self.corpus_dir, self.word_freq_file_name)
        if not os.path.exists(word_freq_file_path):
            logger.warning("File '{}' not found".format(corpus_dir))
        else:
            self.word_counts = json.load(open(word_freq_file_path, "r"))
