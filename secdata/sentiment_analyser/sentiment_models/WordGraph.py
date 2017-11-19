import os
import networkx as nx
import numpy as np
import logging

from secdata.sentiment_analyser.sentiment_models.prototype import SentimentWeightingModel

logger = logging.getLogger(__name__)


class WordGraph(SentimentWeightingModel):
    graph_filename = "word_graph.pickle"

    def __init__(self, model_data_dir=""):
        self.graph = nx.Graph()
        self.model_data_dir = model_data_dir
        self.graph_save_path = os.path.join(model_data_dir, self.graph_filename)
        self.graph.add_node("data_points_accumulator", value=np.array((0, 0, 0), dtype=np.float64))
        if model_data_dir != "":
            self.load()

    def train_batch(self, nodes, labels):
        assert len(labels) == 3
        self.graph.node["data_points_accumulator"]["value"] += labels
        labels = np.array(labels)
        for word, next_word in zip(nodes[:-1], nodes[1:]):
            self.reinforce_connection(word, next_word, labels)

    def reinforce_connection(self, a, b, deltas):
        assert len(deltas) == 3
        deltas = np.array(deltas)
        if a not in self.graph.nodes:
            self.graph.add_node(a)
        if b not in self.graph.nodes:
            self.graph.add_node(b)

        if self.graph.has_edge(a, b):
            self.graph[a][b]['scores'] += deltas
        else:
            self.graph.add_edge(a, b, scores=deltas)

    @property
    def data_point_count(self):
        return sum(self.graph.node["data_points_accumulator"]["value"])

    @property
    def normalization_factor(self):
        return self.data_point_count / self.graph.node["data_points_accumulator"]["value"]

    @property
    def min_sum(self):
        return int(0.001 * self.data_point_count)

    def find_score(self, nodes):
        accumulated_score = np.array((0, 0, 0), dtype=np.float64)
        for word, next_word in zip(nodes[:-1], nodes[1:]):
            if (word not in self.graph.nodes) or (next_word not in self.graph.nodes):
                # Filter the unknown
                continue
            elif not self.graph.has_edge(word, next_word):
                # Filter the unknown
                continue
            elif sum(self.graph[word][next_word]['scores']) < self.min_sum:
                # Filter the very uncommon
                continue
            elif max(self.graph[word][next_word]['scores']) < 1.4 * np.mean(self.graph[word][next_word]['scores']):
                # Filter the ones that appear too much in all contexts
                continue
            else:
                accumulated_score += self.graph[word][next_word]['scores']
                # print(word, next_word, self.graph[word][next_word]['scores'])
        return accumulated_score * self.normalization_factor

    def save(self, model_data_dir=None):
        model_data_dir = self.model_data_dir if model_data_dir is None else model_data_dir
        graph_save_path = os.path.join(model_data_dir, self.graph_filename)
        if not os.path.exists(model_data_dir):
            logger.warning("Path '{}' not found".format(model_data_dir))
        if os.path.exists(self.model_data_dir):
            nx.write_gpickle(self.graph, graph_save_path)

    def load(self, model_data_dir=None):
        model_data_dir = self.model_data_dir if model_data_dir is None else model_data_dir
        if not os.path.exists(self.graph_save_path):
            logger.warning("File '{}' not found".format(model_data_dir))
        else:
            self.graph = nx.read_gpickle(self.graph_save_path)


