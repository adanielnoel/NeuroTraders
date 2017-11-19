import os

import numpy as np

from secdata.sentiment_analyser.sentiment_models.prototype import SentimentWeightingModel


class GeneralInquirer(SentimentWeightingModel):
    def __init__(self):
        rd = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)),
                          "resources")
        with open(os.path.join(rd, "positive.csv")) as f:
            self.positive_tks = [l.strip().lower() for l in f.readlines()]
        with open(os.path.join(rd, "negative.csv")) as f:
            self.negative_tks = [l.strip().lower() for l in f.readlines()]
        with open(os.path.join(rd, "uncertainty.csv")) as f:
            self.uncertainty_tks = [l.strip().lower() for l in f.readlines()]

    def train_batch(self, nodes, labels):
        pass

    def find_score(self, tokens):
        score = np.array((0, 0, 0), dtype=np.float64)
        for token in tokens:
            if token.lower() in self.positive_tks:
                score += self.POSITIVE
            elif token.lower() in self.negative_tks:
                score += self.NEGATIVE
            elif token.lower() in self.uncertainty_tks:
                score += self.UNCERTAIN
        return score
