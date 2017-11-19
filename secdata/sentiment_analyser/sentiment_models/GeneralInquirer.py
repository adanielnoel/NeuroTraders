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
