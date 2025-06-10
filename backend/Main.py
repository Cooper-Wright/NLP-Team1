import requests
import json
import nltk
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# API endpoint from the newly deployed service
API_URL = "https://zfgp45ih7i.execute-api.eu-west-1.amazonaws.com/sandbox/api/search"
API_KEY = "ZR38746G38B7RB46GBER"

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

    return headers, payload 

def GetResponse(API_URL, headers, payload):
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    json_response = response.json()

    return json_response

def WriteExcel(json_response, payload):
    # Create a better Excel file with multiple sheets
    with pd.ExcelWriter("nintendo_switch_reviews.xlsx") as writer:
        # Extract the results list into a DataFrame
        if 'results' in json_response:
            results_df = pd.json_normalize(json_response['results'])
            results_df.to_excel(writer, sheet_name="Reviews", index=False)
            
        # Save query details to another sheet
        if 'query_details' in json_response:
            query_df = pd.json_normalize(json_response['query_details'])
            query_df.to_excel(writer, sheet_name="Query Details", index=False)
        
        # Save AI answer if available
        if 'ai_answer' in json_response:
            ai_answer_df = pd.json_normalize(json_response['ai_answer'])
            ai_answer_df.to_excel(writer, sheet_name="AI Answer", index=False)
        
        # Save metadata
        metadata = {
            "count": json_response.get('count', 0),
            "total_results": len(json_response.get('results', [])),
            "query": payload["query_text"]
        }
        pd.DataFrame([metadata]).to_excel(writer, sheet_name="Metadata", index=False)

    print(f"JSON response saved to nintendo_switch_reviews.xlsx with multiple sheets")

def main():
    headers, payload = QueryAmplyFi()
    
    attempt_num = 1

    print("Attempt", attempt_num)
    json_response = GetResponse(API_URL, headers, payload)

    while json_response.get("message") == "Endpoint request timed out":
        attempt_num += 1
        print("Attempt", attempt_num)
        json_response = GetResponse(API_URL, headers, payload)

    WriteExcel(json_response)

main()