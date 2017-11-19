import logging
import os
import numpy as np

from secdata.sentiment_analyser import text_processing as tp
from secdata.sentiment_analyser.Corpus import Corpus
from secdata.sentiment_analyser.utils import get_names
from secdata.sentiment_analyser.sentiment_models.WordGraph import WordGraph

logger = logging.getLogger(__name__)


class NewsAnalyser:

    def __init__(self, resources_dir=""):
        if resources_dir == "":
            resources_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
        if not os.path.exists(resources_dir):
            logger.info("Resources path not found, creating it")
            os.makedirs(resources_dir)
        self.resources_dir = resources_dir
        self.word_graph = WordGraph(model_data_dir=self.resources_dir)
        self.corpus = Corpus(corpus_dir=self.resources_dir)

    def clear_resources(self):
        if os.path.exists(self.word_graph.graph_save_path):
            os.remove(self.word_graph.graph_save_path)
        if os.path.exists(self.corpus.word_freq_file_path):
            os.remove(self.corpus.word_freq_file_path)

    def analyse(self, text):
        # Remove infrequent words
        sentences = self.prepare_text(text,
                                      remove_non_frequent_words=False,
                                      keep_names=True,
                                      tokenize=True,
                                      do_stemming=False,
                                      convert_to_sentences=True,
                                      remove_numbers=True,
                                      remove_punctuation=True,
                                      to_lowercase=True)
        total_score = np.array((0, 0, 0), dtype=np.float64)   # Just 0s
        for sentence in sentences:
            # stemmed_tokens = tp.tokenize(sentence)
            total_score += np.array(self.word_graph.find_score(sentence))
        return total_score

    def train_iter(self, text, label):
        self.corpus.new_text(text)
        for sentence in tp.to_sentences(text):
            self.word_graph.train_batch(tp.tokenize(sentence,
                                                    remove_punctuation=True,
                                                    remove_numbers=True,
                                                    to_lowercase=True,
                                                    with_stemming=False), label)

    def train(self, pairs):
        for text, score in pairs:
            self.train_iter(text, score)

    def save(self):
        self.corpus.save_stats()
        self.word_graph.save()

    def prepare_text(self, text,
                     tokenize=False,
                     do_stemming=False,
                     remove_non_frequent_words=False,
                     convert_to_sentences=False,
                     remove_numbers=True,
                     remove_punctuation=True,
                     keep_names=True,
                     to_lowercase=False):

        if not remove_non_frequent_words:
            # Names will be kept anyway, but this disables a costly check
            keep_names = False

        sentences = tp.to_sentences(text)
        filtered_sentences = []
        for sentence in sentences:
            filtered_sentence = []
            if keep_names:
                names = get_names(sentence)
            else:
                names = []
            for word in tp.tokenize(sentence,
                                    remove_punctuation=remove_punctuation,
                                    remove_numbers=remove_numbers,
                                    with_stemming=do_stemming):
                if word in names:
                    filtered_sentence.append(word)
                elif remove_non_frequent_words:
                    if word in self.corpus.words:
                        filtered_sentence.append(word)
                else:
                    filtered_sentence.append(word)
            if to_lowercase:
                filtered_sentence = [_t.lower() for _t in filtered_sentence]
            filtered_sentences.append(filtered_sentence)

        if convert_to_sentences:
            if tokenize:
                return np.array(filtered_sentences)
            else:
                return [' '.join(sentence) for sentence in sentences]
        else:
            filtered_sentences = [token for sentence in sentences for token in sentence]
            if tokenize:
                return filtered_sentences
            else:
                return ' '.join(filtered_sentences)
