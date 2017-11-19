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
