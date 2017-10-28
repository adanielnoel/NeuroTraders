import os
import json


def get_company_name(ticker):
    path = os.path.dirname(os.path.realpath(__file__)) + "/" + ticker + "/info.json"
    try:
        company = json.load(open(path, 'r'))["company_name"]
        return company
    except:
        return ""


