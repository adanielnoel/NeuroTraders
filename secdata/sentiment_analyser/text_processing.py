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
from secdata.sentiment_analyser import utils
from nltk.tokenize import RegexpTokenizer


def tokenize(text, remove_punctuation=True, remove_numbers=True, to_lowercase=False, with_stemming=False):
    if isinstance(text, list):
        return text
    if to_lowercase:
        text = text.lower()
    if remove_punctuation:
        tokens = RegexpTokenizer(r'\w+').tokenize(text)
    else:
        tokens = nltk.word_tokenize(text, language='english')
    if remove_numbers:
        tokens = [token for token in tokens if not utils.is_number(token)]
    if with_stemming:
        tokens = [nltk.stem.SnowballStemmer(language='english').stem(token) for token in tokens]
    return tokens


def to_bigrams(text_or_tokens):
    return nltk.bigrams(tokenize(text_or_tokens))


def to_sentences(text):
    return nltk.sent_tokenize(text)
