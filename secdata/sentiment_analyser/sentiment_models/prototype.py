import numpy as np


class SentimentWeightingModel:
    POSITIVE = np.array((1, 0, 0), dtype=np.float64)
    NEGATIVE = np.array((0, 1, 0), dtype=np.float64)
    UNCERTAIN = np.array((0, 0, 1), dtype=np.float64)
    NEUTRAL = np.array((0, 0, 0), dtype=np.float64)
    labels = (POSITIVE, NEGATIVE, UNCERTAIN, NEUTRAL)

    def train_batch(self, nodes, labels):
        raise NotImplementedError

    def find_score(self, tokens):
        raise NotImplementedError


