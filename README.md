# NeuroTraders
Motivated by the notable performance of modern machine learning methods in internalizing time series,
we decided to build a deep learning pipeline for the task of stock price forecasting. 

We propose a novel sentiment analysis technique, the Word-Graph, and use it to process thousands of online news articles.
Sentiment data is used together with stock price data to train a system of two deep recurrent neural networks with LSTM
architecture to predict closing prices at market open, as well as the associated absolute errors. 

We have co-authored a scientific paper about our work, titled *A Deep Learning Approach to Stock Price Forecasting*, 
in which a more detailed explanation of the learning algorithms employed and their performance can be found. 
The paper is included in the main folder of the repository.

## The team
This project was developed in a team of 4 engineering students/hackathon hackers with a passion for AI and finance:
- Alejandro Daniel Noel (adanielnoel@gmail.com)
- Paolo Rizzo (paolo.rizzo@outlook.com)
- Anja Meunier
- Daniel Martini Jimenez


## Awards
- Best AI Hack - *iNTUition 2017 Hackathon* in Singapore
- Best Project in Statistical Learning & Data Mining - *Nanyang Technological University (NTU)*, Singapore

## Project structure

The overall structure of the code is as follows:

- `secdata/` : This directory contains all the code relating to building and managing a database for a certain security (so far only stocks)

- `secdata/news_scraper/`: this is the sub-package responsible of fetching all the online news that can be found on a certain stock. So far there are spiders for Reuters and Google Finance.

- `secdata/sentiment_analyser`: this subpackage handles the extraction of sentiment from news.

- `secdata/sentiment_analyser/models`: the different models available. So far the popular General Inquirer and the custom Word-Graph method.

- `predictors/`: So far two different LSTM-based predictors, one in **Tensorflow** and one in **Keras**. 

    - **NOTE**: this part of the project is a bit messy,
      it was developed at a Hackathon under serious time constraints and sleep deprivation and has not been cleaned up much since.

- `trading_sim/`: A simple trading simulator for testing the performance of different trading strategies. 
   
    - **NOTE**: this part of the project is also quite messy, see previous note.


- `new_database/`: The database we compiled for the project and that we decided to include so that it is easier to start experimenting with the code.

## How to get started

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

- `new_stock_script.py`: A script that shows how to add a new stock to the database, also fetching market data and news articles.

- `sentiment_benchmark_script.py`: A script that benchmarks the performance of the Word-Graph and General Inquirer methods, as documented in the paper. You will have to set all the settings to true the first time so that it generates the training and test data and trains the models. After that the data is saved into pickle files, so you may deactivate some settings to make the script run faster.

- `rnn_adj_close_predictor_full.py`: This can be found in `predictors/RNN_used_in_paper`. No object oriented programming was used in this part of the project, the entire neural network training and testing is a script that can directly run.

Other runnable scripts, used for "dirty testing" in the project are (scroll to the end of each for the test script):

- `stockNewsCrawler.py`: test the crawling process.

- `RNN_model_tensorflow.py`: test the model.

- `trading_simulator.py`: test the simulator.

- `LSTM_test.py`: This can be found in `predictors/RNN_under_development` and might not work since it's under development. Still interesting to take a look since it's cleaner that the scripts in `predictors/RNN_used_in_paper` and has better data formatting methods.

## Further work
The project is still under development, but at a sporadic pace, as we are very busy with our studies.

For inquiries, ideas or collaboration opportunities, please feel free to contact *Paolo* (paolo.rizzo@outlook.com) or *Alejandro* (adanielnoel@gmail.com).