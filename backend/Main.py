import requests
import json
import nltk
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# API endpoint from the newly deployed service
API_URL = "https://zfgp45ih7i.execute-api.eu-west-1.amazonaws.com/sandbox/api/search"
API_KEY = secrets.API_KEY

def QueryAmplyFi():

    headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

    payload = {
    "query_text": "What are the current reviews on the release of the Nintendo switch?",
    "result_size": 100,
    "include_highlights":True,
    "ai_answer": "basic"
    }