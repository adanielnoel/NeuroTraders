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
