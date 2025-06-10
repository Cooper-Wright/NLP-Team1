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
        "query_text": "Articles with the words 'reviews' or 'rating' or 'experiences' " + query + " in the Title", #What are the latest and most unbiased reviews or ratings or experiences on the " + query + ". Please ensure the articles are solely on the " + query + ".
        "result_size": result_size,
        "include_highlights": True,
        "ai_answer": "basic"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        json_response = response.json()

        while json_response.get("message") == "Endpoint request timed out":
            json_response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

        print(json_response)
        write_to_excel(json_response, payload)
        return json_response
    except Exception as e:
        return {"results": [], "error": str(e)}
    
def write_to_excel(json_response, payload):
    df = find_sentimental_value(json_response)

    # Create a better Excel file with multiple sheets
    with pd.ExcelWriter("data.xlsx") as writer:
        # Extract the results list into a DataFrame
        if 'results' in json_response:
            results_df = pd.json_normalize(json_response['results'])

            # Add sentiment columns from the df dataframe
            results_df['sent_compound'] = df['sent_compound']
            results_df['sent_pos'] = df['sent_pos']
            results_df['sent_neg'] = df['sent_neg']
            
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

    print(f"JSON response saved to data.xlsx with multiple sheets")

def find_sentimental_value(json_response):
    # Process data
    df = pd.json_normalize(json_response['results'])
    
    # Clean and analyze text
    df['clean_summary'] = df['summary'].apply(clean_text)
    df['sentiment'] = df['clean_summary'].apply(lambda x: sia.polarity_scores(x))
    df['sent_compound'] = df['sentiment'].apply(lambda d: d['compound'])
    df['sent_pos'] = df['sentiment'].apply(lambda d: d['pos'])
    df['sent_neg'] = df['sentiment'].apply(lambda d: d['neg'])

    return df

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
            ui.br(),
            # Article List
            ui.card(
                ui.card_header("Top Articles"),
                # Add sorting dropdown here
                ui.input_select(
                    "sort_by", 
                    "Sort Articles By:", 
                    {
                        "sent_compound_desc": "Most Positive First",
                        "sent_compound_asc": "Most Negative First", 
                        "title": "Title (A-Z)",
                        "timestamp": "Date (if available)",
                        "score": "Relevance Score"
                    },
                    selected="sent_compound_desc",
                    width="100%"
                ),
                ui.div(
                    ui.output_ui("articles_list"),
                    style="height: 400px; overflow-y: auto;"
                ),

                ui.download_button(
                "download_basic", 
                "Full Excel Download", 
                class_="btn btn-success"
                )
            )
        ),
        
        # Right Column - Sentiment Graph
        ui.column(6,
            ui.card(
                ui.card_header("Sentiment Analysis"),
                ui.output_ui("sentiment_plot"),
                height="780px"
            ),

            # WordCloud
            ui.card(
                ui.card_header("Word Cloud"),
                ui.output_ui("wordcloud_output"),
                height="500px"
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
        
        df = find_sentimental_value(json_response)
        
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
        
        # Get sort option
        sort_option = input.sort_by()
        
        # Apply sorting based on selection
        if sort_option == "sent_compound_desc":
            sorted_df = df.nlargest(100, 'sent_compound')
        elif sort_option == "sent_compound_asc":
            sorted_df = df.nsmallest(100, 'sent_compound')
        elif sort_option == "title":
            sorted_df = df.sort_values('title', key=lambda x: x.str.lower() if pd.api.types.is_string_dtype(x) else x)
        elif sort_option == "timestamp" and "timestamp" in df.columns:
            sorted_df = df.sort_values('timestamp', ascending=False)
        elif sort_option == "score" and "score" in df.columns:
            sorted_df = df.nlargest(100, 'score')
        else:
            # Default fallback
            sorted_df = df.nlargest(100, 'sent_compound')
        
        articles_html = []
        for idx, article in sorted_df.iterrows():
            sentiment_color = "success" if article['sent_compound'] > 0.1 else "danger" if article['sent_compound'] < -0.1 else "warning"
            
            # Get the title and truncate if needed
            title_text = article.get('title', 'No Title')
            if len(str(title_text)) > 100:
                title_display = title_text[:100] + "..."
            else:
                title_display = title_text
            
            # Check for URL in different possible field names
            article_url = None
            for url_field in ['url', 'link', 'source_url', 'web_url']:
                if url_field in article and isinstance(article[url_field], str):
                    article_url = article[url_field]
                    break
            
            # Create title element based on whether URL is available
            if article_url:
                title_element = ui.tags.a(
                    title_display, 
                    href=article_url,
                    target="_blank",  # Open in new tab
                    style="color: #2c3e50; text-decoration: none; font-weight: 600; font-size: 1.1rem;"
                )
            else:
                title_element = ui.h6(title_display)
                
            article_card = ui.div(
                ui.div(
                    title_element,  # Use the dynamic title element
                    ui.p(article.get('summary', 'No Summary')[:200] + "..." if len(str(article.get('summary', ''))) > 200 else article.get('summary', 'No Summary')),
                    ui.span(f"Sentiment: {article['sent_compound']:.3f}", class_=f"badge bg-{sentiment_color}"),
                    class_="card-body"
                ),
                class_="card mb-3"
            )
            articles_html.append(article_card)
        
        return ui.div(*articles_html)

    @session.download(filename="data.xlsx")
    def download_basic():
        file_path = "data.xlsx"
        
        # Read the file and return its contents
        with open(file_path, "rb") as f:
            return f.read()

    
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