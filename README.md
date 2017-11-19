With the advent of recurrent neural networks becoming increasingly popular in solving complex time-series predictions, we decided to give it a try in predicting stock prices. Moreover, we combine stock market data with online news from different sources.
The paper can be downloaded [HERE].
 
##Project structure
The overall structure of the code is as follows:

`secdata/` : This directory contains all the code relating to building and managing a database for a certain security (so far only stocks)

`secdata/news_scraper/`: this is the sub-package responsible of fetching all the online news that can be found on a certain stock. So far there are spiders for Reuters and Google Finance.

`secdata/sentiment_analyser`: this subpackage handles the extraction of sentiment from news.

`secdata/sentiment_analyser/models`: the different models available. So far the popular General Inquirer and the custom Word-Graph method.

`predictors/`: So far two different LSTM-based predictors, one in **Tensorflow** and one in **Keras**. Note: this part of the project is a bit messy.

`trading_sim/`: A simple trading simulator for testing the performance of different trading strategies. Note: this part of the project is also quite messy.

`new_database/`: The database we compiled for the project and that we decided to include so that it is easier to start experimenting with the code.

##How to get started
The code has been developed in Python 3.5, we do not know if it will work properly in other versions. Moreover, the following packages must be present in the system:

- **Scikit learn:** Required for predictors and for `sentiment_benchmark_script.py`
- **Tensorflow:** Required for predictors
- **Keras:** Required for predictors
- **Pandas:** Required in all the project to handle collections of data
- **Quandl:** Required for downloading stock market data
- **Networkx:** Required for Word-Graph model in sentiment analysis 
- **NLTK:** Required in sentiment analysis
- **Scrapy:** Required in news scraping
- **Goose3:** Required in news scraping
- **Requests:** Required in news scraping

The best way to see how the code works is to run the scripts we have made demonstrating the different parts of the project. These are:

`new_stock_script.py`: A script that shows how to add a new stock to the database, also fetching market data and news articles.

`sentiment_benchmark_script`: A script that benchmarks the performance of the Word-Graph and General Inquirer methods, as documented in the paper. You will have to set all the settings to true the first time so that it generates the training and test data and trains the models. After that the data is saved into pickle files, so you may deactivate some settings to make the script run faster.

Other runnable scripts, used for "dirty testing" in the project are (scroll to the end of each for the test script):

`stockNewsCrawler.py`: test the crawling process.

`RNN_model_tensorflow`: test the model.

`trading_simulator.py`: test the simulator. 