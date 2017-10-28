import quandl
from datetime import datetime
import pickle

quandl.ApiConfig.api_key = "bYNHAsGguxFWJsjg3ccN"
mydata = quandl.get("WIKI/AAPL", start_date="2005-12-31", end_date=datetime.today())
pickle.dump(mydata, open("aapl_data.pickle", "wb"))
