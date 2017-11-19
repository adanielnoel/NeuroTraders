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

import logging
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from secdata.Query import StockQuery
from secdata.StockData import StockData
from secdata.sentiment_analyser import text_processing as tp
from secdata.sentiment_analyser import utils
from secdata.sentiment_analyser.NewsAnalyser import NewsAnalyser
from secdata.sentiment_analyser.sentiment_models.GeneralInquirer import GeneralInquirer
from secdata.sentiment_analyser.sentiment_models.WordGraph import WordGraph

logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s]%(levelname)s: %(message)s", level=logging.INFO)

# Word-Graph test settings
retrain_feature_extractor = False  # If true retrains the graph model for sentiment (about 30 to 60 seconds)
retrain_classifier = False
regenerate_train_data = False
regenerate_test_data = False

# General inquirer test settings
test_general_inquirer = False


def custom_norm(array):
    std = np.std(array)
    mean = np.mean(array)
    for i in range(len(array)):
        devs_away = (np.floor(np.abs(array[i] - mean) / std) + 1.0)
        array[i] = (array[i] - mean) / (1.4 * devs_away) + mean
    return array


queries_train = [StockQuery(ticker="amd"),
                 StockQuery(ticker="amzn"),
                 StockQuery(ticker="intc"),
                 StockQuery(ticker="jpm"),
                 StockQuery(ticker="mmm"),
                 StockQuery(ticker="googl"),
                 StockQuery(ticker="nflx"),
                 StockQuery(ticker="nvda"),
                 StockQuery(ticker="tsla"),
                 StockQuery(ticker="vrx"),
                 StockQuery(ticker="wmt")]
queries_test = [StockQuery(ticker="aapl"),
                StockQuery(ticker="msft")]

change_threshold_p = 0.011  # % increase threshold for day to be positive
change_threshold_n = 0.011  # % increase threshold for day to be negative

news_analyser = NewsAnalyser()

print("-----------------------------------------------------------------")
print("     Test classification performance of the graph-based model")
print("-----------------------------------------------------------------")

if retrain_feature_extractor:
    print("Training feature extractor")
    news_analyser.clear_resources()  # Empty the resources dir, else words may be counted twice
    for query in queries_train:
        print("\tTraining with {}".format(query.ticker))
        stock_data = StockData(query, "./new_database")
        news = stock_data.get_all_news(pair_with_cols=["relative_intraday"])
        for date, contents in news:
            collapsed_news = utils.collapse_daily_news(contents['news'])
            if contents['relative_intraday'] >= change_threshold_p:
                label = WordGraph.POSITIVE
            elif contents['relative_intraday'] <= -change_threshold_n:
                label = WordGraph.NEGATIVE
            else:
                label = WordGraph.UNCERTAIN
            news_analyser.train_iter(collapsed_news, label)
        news_analyser.save()

if regenerate_train_data:
    trainX = []
    trainY = []
    counter_1 = 0
    print("Generating training data")
    for query in queries_train:
        print("\tGenerating for {}".format(query.ticker))
        stock_data = StockData(query, "./new_database")
        news = stock_data.get_all_news(pair_with_cols=["relative_intraday"])
        for _, contents in news:
            collapsed_news = utils.collapse_daily_news(contents['news'])
            counter_1 += 1
            if contents['relative_intraday'] >= change_threshold_p:
                label = WordGraph.POSITIVE
            elif contents['relative_intraday'] <= -change_threshold_n:
                label = WordGraph.NEGATIVE
            else:
                label = WordGraph.UNCERTAIN
            score = news_analyser.analyse(collapsed_news)
            trainX.append(score)
            trainY.append(label)
    pickle.dump((trainX, trainY), open("train_data.pickle", "wb"))

trainX, trainY = pickle.load(open("train_data.pickle", "rb"))
trainX = custom_norm(trainX)

if retrain_classifier:
    print("Training classifier")
    classifier = KNeighborsClassifier(n_neighbors=1)
    # classifier = SVC(gamma=2, C=1)
    # classifier = RandomForestClassifier(max_depth=5)
    classifier.fit(trainX, trainY)
    pickle.dump(classifier, open("classifier.pickle", "wb"))
classifier = pickle.load(open("classifier.pickle", "rb"))

if regenerate_test_data:
    testX = []
    testY = []

    print("Generating test data")
    for query in queries_test:
        print("\tGenerating for {}".format(query.ticker))
        stock_data = StockData(query, "./new_database")
        news = stock_data.get_all_news(pair_with_cols=["relative_intraday"])
        for _, contents in news:
            collapsed_news = utils.collapse_daily_news(contents['news'])
            score = news_analyser.analyse(collapsed_news)
            testX.append(score)
            if contents['relative_intraday'] >= change_threshold_p:
                label = WordGraph.POSITIVE
            elif contents['relative_intraday'] <= -change_threshold_n:
                label = WordGraph.NEGATIVE
            else:
                label = WordGraph.UNCERTAIN
            testY.append(label)
    pickle.dump((testX, testY), open("test_data.pickle", "wb"))

testX, testY = pickle.load(open("test_data.pickle", "rb"))
testX = custom_norm(testX)

classifications = classifier.predict(testX)
bad_classifications = (np.array(testY) - classifications)
bad_classifications[bad_classifications < 0] = 0  # Remove those -1 where classification did not coincide with testY
accuracies = 1.0 - (np.sum(bad_classifications, axis=0) / np.sum(testY, axis=0))
print("\nPrecision of graph model")
print(" {:3.2f}% of positive classified correctly".format(100 * accuracies[0]))
print(" {:3.2f}% of negative classified correctly".format(100 * accuracies[1]))
print(" {:3.2f}% of uncertain classified correctly".format(100 * accuracies[2]))
print(" {:3.2f}% overall accuracy".format(100 * np.average(accuracies)))


if test_general_inquirer:
    print("\n------------------------------------------------------------------")
    print("   Test classification performance of the General Inquirer model")
    print("------------------------------------------------------------------")
    queries_test = [StockQuery(ticker="aapl"),
                    StockQuery(ticker="msft")]
    trainX = []
    trainY = []
    testX = []
    testY = []

    print("Generating training data")
    gi_model = GeneralInquirer()
    for query in queries_train:
        print("\tGenerating for {}".format(query.ticker))
        stock_data = StockData(query, "./new_database")
        news = stock_data.get_all_news(pair_with_cols=["relative_intraday"])
        for _, contents in news:
            collapsed_news = utils.collapse_daily_news(contents['news'])
            score = gi_model.find_score(tp.tokenize(collapsed_news))
            trainX.append(score)
            if contents['relative_intraday'] >= change_threshold_p:
                label = WordGraph.POSITIVE
            elif contents['relative_intraday'] <= -change_threshold_n:
                label = WordGraph.NEGATIVE
            else:
                label = WordGraph.UNCERTAIN
            trainY.append(label)

    print("Generating test data")
    gi_model = GeneralInquirer()
    for query in queries_test:
        print("\tGenerating for {}".format(query.ticker))
        stock_data = StockData(query, "./new_database")
        news = stock_data.get_all_news(pair_with_cols=["relative_intraday"])
        for _, contents in news:
            collapsed_news = utils.collapse_daily_news(contents['news'])
            score = gi_model.find_score(tp.tokenize(collapsed_news))
            testX.append(score)
            if contents['relative_intraday'] >= change_threshold_p:
                label = WordGraph.POSITIVE
            elif contents['relative_intraday'] <= -change_threshold_n:
                label = WordGraph.NEGATIVE
            else:
                label = WordGraph.UNCERTAIN
            testY.append(label)
    # pickle.dump((testX, testY), open("test_data.pickle", "wb"))
    trainX = custom_norm(trainX)
    testX = custom_norm(testX)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(trainX, trainY)

    classifications = classifier.predict(testX)
    bad_classifications = (np.array(testY) - classifications)
    bad_classifications[bad_classifications < 0] = 0  # Remove those -1 where classification did not coincide with testY
    accuracies = 1.0 - (np.sum(bad_classifications, axis=0) / np.sum(testY, axis=0))
    print("\nPrecision of general inquirer model")
    print(" {:3.2f}% of positive classified correctly".format(100 * accuracies[0]))
    print(" {:3.2f}% of negative classified correctly".format(100 * accuracies[1]))
    print(" {:3.2f}% of uncertain classified correctly".format(100 * accuracies[2]))
    print(" {:3.2f}% overall accuracy".format(100 * np.average(accuracies)))
