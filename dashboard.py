from shiny import App, render, ui, reactive
import requests
import json
import nltk
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# API Configuration
API_URL = "https://zfgp45ih7i.execute-api.eu-west-1.amazonaws.com/sandbox/api/search"
API_KEY = "ZR38746G38B7RB46GBER"

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

def clean_text(text):
    """Clean text for analysis"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_data(query, result_size):
    """Fetch data from Amplyfi API"""
    payload = {
        "query_text": query,
        "result_size": result_size,
        "include_highlights": True
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        return response.json()
    except Exception as e:
        return {"results": [], "error": str(e)}



# UI Definition
app_ui = ui.page_fluid(
    # Search Input
    ui.include_css("style.css"),
    ui.div(
        class_="navbar",
    ),
    ui.row(
            ui.input_text("search_query", label=None, placeholder="What are you looking for today?"),
            ui.input_action_button("search_btn", "Search",)
    ),
    
    # Main Dashboard Layout
    ui.row(
        ui.column(6,
            ui.br(),
            # Article List
            ui.card(
                ui.card_header("Top Articles"),
                ui.div(
                    ui.output_ui("articles_list")
                )
            ),
            # WordCloud
            ui.card(
                ui.card_header("Word Cloud"),
                ui.output_ui("wordcloud_output"),
                height="350px"
            )
        ),
        
        # Right Column - Sentiment Graph
        ui.column(6,
            ui.card(
                ui.card_header("Sentiment Analysis"),
                ui.output_ui("sentiment_plot"),
                height="780px"
            )
        )
    )
)


# Server Logic
def server(input, output, session):
    
    # Reactive data storage
    current_data = reactive.Value(pd.DataFrame())
    
    @reactive.Effect
    @reactive.event(input.search_btn, ignore_none=False)
    def fetch_and_process_data():
        """Fetch and process data when search button is clicked"""
        query = input.search_query()
        if not query:
            return
        
        # Show loading state
        ui.notification_show(f"Searching for '{query}'...", type="message")
        
        # Fetch data
        json_response = fetch_data(query, 5)
        
        if "error" in json_response:
            ui.notification_show(f"Error: {json_response['error']}", type="error")
            return
        
        if not json_response.get('results'):
            ui.notification_show("No results found", type="warning")
            return
        
        # Process data
        df = pd.json_normalize(json_response['results'])
        
        # Clean and analyze text
        df['clean_summary'] = df['summary'].apply(clean_text)
        df['sentiment'] = df['clean_summary'].apply(lambda x: sia.polarity_scores(x))
        df['sent_compound'] = df['sentiment'].apply(lambda d: d['compound'])
        df['sent_pos'] = df['sentiment'].apply(lambda d: d['pos'])
        df['sent_neg'] = df['sentiment'].apply(lambda d: d['neg'])
        
        # Store processed data
        current_data.set(df)
        
        ui.notification_show(f"Found {len(df)} articles", type="success")
    
    @output
    @render.ui
    def articles_list():
        """Render top articles list"""
        df = current_data()
        if df.empty:
            return ui.div("Search for articles to see results", class_="text-center mt-3")
        
        # Get top 3 articles by sentiment
        top_articles = df.nlargest(3, 'sent_compound')
        
        articles_html = []
        for idx, article in top_articles.iterrows():
            sentiment_color = "success" if article['sent_compound'] > 0.1 else "danger" if article['sent_compound'] < -0.1 else "warning"
            
            article_card = ui.div(
                ui.div(
                    ui.h6(article.get('title', 'No Title')[:100] + "..." if len(str(article.get('title', ''))) > 100 else article.get('title', 'No Title')),
                    ui.p(article.get('summary', 'No Summary')[:200] + "..." if len(str(article.get('summary', ''))) > 200 else article.get('summary', 'No Summary')),
                    ui.span(f"Sentiment: {article['sent_compound']:.3f}", class_=f"badge bg-{sentiment_color}"),
                    class_="card-body"
                ),
                class_="card mb-3"
            )
            articles_html.append(article_card)
        
        return ui.div(*articles_html)

# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()