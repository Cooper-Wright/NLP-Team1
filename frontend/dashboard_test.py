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
API_KEY = "ZR38746G38B7RB46GBER"  # Use your actual API key from the hackathon

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

def fetch_data(query, result_size=20):
    """Fetch data from Amplyfi API"""
    payload = {
        "query_text": query,
        "result_size": result_size,
        "include_highlights": True,
        "ai_answer": "basic"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        return response.json()
    except Exception as e:
        return {"results": [], "error": str(e)}

def generate_wordcloud(text_data):
    """Generate word cloud from text data"""
    if not text_data:
        return None
    
    # Combine all text
    combined_text = " ".join(text_data)
    
    if not combined_text.strip():
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=400, 
        height=300, 
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate(combined_text)
    
    # Convert to base64 for display
    img = io.BytesIO()
    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    
    img.seek(0)
    img_b64 = base64.b64encode(img.read()).decode()
    return f"data:image/png;base64,{img_b64}"

def create_sentiment_plot(df):
    """Create sentiment over time plot"""
    if df.empty:
        return go.Figure()
    
    # Sort by date if available, otherwise use index
    if 'published_date' in df.columns:
        df_sorted = df.sort_values('published_date')
        x_axis = df_sorted['published_date']
        x_title = "Publication Date"
    else:
        df_sorted = df.reset_index()
        x_axis = df_sorted.index
        x_title = "Article Index"
    
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_sorted['sent_compound'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6)
    ))
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sentiment Analysis Over Time",
        xaxis_title=x_title,
        yaxis_title="Sentiment Score",
        height=400,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

# UI Definition
app_ui = ui.page_fluid(
    # Custom CSS for styling
    ui.include_css("style.css"),

    
    # Navigation Bar
    ui.div(
        ui.div(
            ui.img(src="star.png", class_="star-icon"),
            "Seren",
            class_="navbar-brand"
        ),
        class_="navbar-custom"
    ),
    
    # Search Section
    ui.div(
        ui.div(
            ui.div(
                ui.input_text(
                    "search_query", 
                    None,
                    value="artificial intelligence",
                    placeholder="What are you looking at today?",
                    width="100%"
                ),
                ui.input_action_button("search_btn", "🔍", class_="search-btn"),
                class_="search-wrapper"
            ),
            class_="container"
        ),
        class_="search-container"
    ),
    
    # Main Dashboard Layout
    ui.row(
        # Left Column - WordCloud and Article List
        ui.column(6,
            # WordCloud
            ui.card(
                ui.card_header("Word Cloud"),
                ui.output_ui("wordcloud_output"),
                height="350px"
            ),
            ui.br(),
            # Article List
            ui.card(
                ui.card_header("Top Articles"),
                ui.div(
                    ui.output_ui("articles_list"),
                    style="height: 400px; overflow-y: auto;"
                )
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
        json_response = fetch_data(query, result_size=15)
        
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
    def wordcloud_output():
        """Render word cloud"""
        df = current_data()
        if df.empty:
            return ui.div("Search for articles to generate word cloud", class_="text-center mt-5")
        
        # Generate word cloud from clean summaries
        clean_texts = df['clean_summary'].tolist()
        wordcloud_img = generate_wordcloud(clean_texts)
        
        if wordcloud_img:
            return ui.img(src=wordcloud_img, style="width: 100%; height: auto;")
        else:
            return ui.div("No text data available for word cloud", class_="text-center mt-5")
    
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
    
    @output
    @render.ui
    def sentiment_plot():
        """Render sentiment plot"""
        df = current_data()
        if df.empty:
            return ui.div("Search for articles to see sentiment analysis", class_="text-center mt-5")
        
        fig = create_sentiment_plot(df)
        
        # Convert plotly figure to HTML
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id="sentiment-plot")
        return ui.HTML(plot_html)

# Create the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()