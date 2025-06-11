from pathlib import Path
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
from collections import Counter

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
    text = re.sub(r"[^a-z\s.,!?;:'\"\(\)\[\]\{\}-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def fetch_data(query, result_size):
    """Fetch data from Amplyfi API"""
    payload = {
        "query_text": "Articles with the words 'reviews' or 'rating' or 'experiences' and must have a product by the name '" + query + "' in the Title", #What are the latest and most unbiased reviews or ratings or experiences on the " + query + ". Please ensure the articles are solely on the " + query + ".
        "result_size": result_size,
        "include_highlights": True
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

def generate_wordcloud(text_data, exclude_words=None):
    """Generate word cloud from text data"""
    if not text_data:
        return None
    
    # Combine all text
    combined_text = " ".join(text_data)
    
    if not combined_text.strip():
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=650,
        height=150, 
        background_color='white',
        colormap='plasma',
        max_words=50,
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
    return [f"data:image/png;base64,{img_b64}", wordcloud.words_.keys()]

def create_sentiment_plot(df):
    """Create sentiment scatter plot over time"""
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
    
    # Add sentiment scatter plot (no lines, only markers)
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=df_sorted['sent_compound'],
        mode='markers',  # Only markers for pure scatter plot
        name='Sentiment Score',
        marker=dict(
            size=12,
            color=df_sorted['sent_compound'],  # Color points by sentiment value
            colorscale='RdYlGn',  # Red (negative) to Yellow (neutral) to Green (positive)
            colorbar=dict(title="Sentiment"),
            showscale=True,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        text=df_sorted['title'] if 'title' in df_sorted.columns else None,
        hovertemplate='<b>%{text}</b><br>Sentiment: %{y:.2f}<extra></extra>' if 'title' in df_sorted.columns 
                    else 'Sentiment: %{y:.2f}<extra></extra>'
    ))
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sentiment Analysis Over Time",
        xaxis_title=x_title,
        yaxis_title="Sentiment Score",
        height=380,  # Reduced height to fit better
        margin=dict(l=50, r=50, t=50, b=30),  # Tighter margins
        template="plotly_white",
        autosize=True
    )
    
    return fig

def create_sentiment_pie(df):
    """Create pie chart showing distribution of sentiment categories"""
    if df.empty:
        return go.Figure()
    
    # Categorize sentiments
    sentiment_categories = {
        "Positive": (df['sent_compound'] > 0.4).sum(),
        "Neutral": ((df['sent_compound'] >= -0.4) & (df['sent_compound'] <= 0.4)).sum(),
        "Negative": (df['sent_compound'] < -0.4).sum()
    }
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(sentiment_categories.keys()),
        values=list(sentiment_categories.values()),
        hole=.3,  # Donut chart style
        marker_colors=['#28a745', '#6c757d', '#dc3545']  # Green, Gray, Red
    )])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400,
        template="plotly_white"
    )
    
    return fig

# UI Definition
app_ui = ui.page_fixed(
    # Custom CSS for styling
    ui.tags.link(href="style.css", rel="stylesheet"),

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
                    value="",
                    placeholder="Enter the name of a product...",
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

        ui.column(6,
            # Article List
                ui.card(
                    ui.row(
                        ui.column(6,
                            ui.h1("Product Articles"),
                        ),
                        ui.column(6,
                            # Add sorting dropdown here
                            ui.input_select(
                            "sort_by", 
                            label=None, 
                            choices={
                                "score": "Relevance Score",
                                "sent_compound_desc": "Most Positive First",
                                "sent_compound_asc": "Most Negative First", 
                                "title": "Title (A-Z)",
                                "timestamp": "Date (if available)"
                            },
                            selected="score",
                            width="100%"
                        ),
                        ),
                        class_="section-header"
                    ),
                    ui.div(
                        ui.output_ui("articles_list"),
                        style="height: 98.5vh; overflow-y: auto;"
                    ),

                    ui.download_button(
                    "download_basic", 
                    "Full Excel Download", 
                    class_="btn btn-success"
                    )
                ),
                class_="articles-list"
            ),
            
        ui.column(6,
            ui.card(
                ui.h1("Analytics Dashboard"),
                ui.card(
                    ui.row(
                        ui.column(8, ui.card_header("Sentiment Analysis")),
                        ui.column(4, 
                            ui.input_action_button(
                                "toggle_sentiment", 
                                "Scatter View", 
                                class_="btn btn-outline-primary btn-sm float-end",
                                style="margin-top: -5px;"
                            )
                        ),
                        class_="d-flex align-items-center"
                    ),
                    ui.output_ui("sentiment_plot"),
                    class_="sentiment-analysis",
                    height="61vh"
                ),

                ui.card(
                    ui.card_header(
                        ui.row(
                            ui.column(8, "Word Cloud"),
                            ui.column(4, 
                                ui.input_action_button(
                                    "toggle_wordcloud", 
                                    "List View", 
                                    class_="btn btn-outline-primary btn-sm float-end",
                                    style="margin-top: -5px;"
                                )
                            )
                        )
                    ),
                    ui.output_ui("wordcloud_output"),
                    class_="word_cloud", 
                     height="49vh"
                )
            ),
            class_="analytics"
        ),
    ),

)

# Server Logic
def server(input, output, session):
    
    # Reactive data storage
    current_data = reactive.Value(pd.DataFrame())
    # Toggle state for word cloud view
    show_wordcloud = reactive.Value(True)
    # Toggle state for sentiment view
    show_pie_chart = reactive.Value(True)

    @reactive.Effect
    @reactive.event(input.toggle_wordcloud)
    def toggle_view():
        """Toggle between word cloud and list view"""
        current_state = show_wordcloud()
        show_wordcloud.set(not current_state)
        
        # Update button text based on current view
        if show_wordcloud():
            ui.update_action_button("toggle_wordcloud", label="List View")
        else:
            ui.update_action_button("toggle_wordcloud", label="Cloud View")
    
    @reactive.Effect
    @reactive.event(input.toggle_sentiment)
    def toggle_sentiment_view():
        """Toggle between pie chart and scatter plot view"""
        current_state = show_pie_chart()
        show_pie_chart.set(not current_state)
        
        # Update button text based on current view
        if show_pie_chart():
            ui.update_action_button("toggle_sentiment", label="Scatter View")
        else:
            ui.update_action_button("toggle_sentiment", label="Pie View")

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
        json_response = fetch_data(query, result_size=100)
        
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
        """Render word cloud or word list based on toggle state"""
        df = current_data()
        if df.empty:
            return ui.div("Search for articles to generate word cloud", class_="text-center mt-5")
        
        # Extract words to exclude from the search query
        query = input.search_query()
        exclude_words = []
        if query:
            # Split query into individual words and clean them
            query_words = re.split(r'[^\w]+', query.lower())
            exclude_words = [word.strip() for word in query_words if word.strip()]
        
        # Generate word cloud from clean summaries
        clean_texts = df['clean_summary'].tolist()
        wordcloud_data = generate_wordcloud(clean_texts, exclude_words=exclude_words)
        
        if not wordcloud_data or not wordcloud_data[0]:
            return ui.div("No text data available for word cloud", class_="text-center mt-5")
        
        if show_wordcloud():
            # Show word cloud
            return ui.img(src=wordcloud_data[0], style="height: 100%;")
        else:
            # Show top words list using the already sorted words from wordcloud
            wordcloud_words = list(wordcloud_data[1])[:15]  # Take top 15 words
            
            if not wordcloud_words:
                return ui.div("No words available", class_="text-center mt-5")
            
            # Create a styled list of top words
            word_items = []
            for i, word in enumerate(wordcloud_words, 1):
                word_item = ui.div(
                    ui.div(
                        ui.span(f"{i}.", class_="text-muted me-2"),
                        ui.span(word.title(), class_="fw-bold"),
                        class_="d-flex align-items-center"
                    ),
                    class_="list-group-item border-0 py-1"
                )
                word_items.append(word_item)
            
            return ui.div(
                ui.div(
                    *word_items,
                    class_="list-group list-group-flush",
                    style="height: 100%; overflow-y: auto; padding: 0 10px;"
                ),
                style="height: 100%;"
            )
    
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
            sentiment_color = "success" if article['sent_compound'] > 0.4 else "danger" if article['sent_compound'] < -0.4 else "secondary"
            
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
                
            review = ""

            if article['sent_compound'] > 0.4:
                review = "Good"
            elif article['sent_compound'] < -0.4:
                review = "Bad"
            else:
                review = "Neutral"
            
            article_card = ui.div(
                ui.div(
                    title_element,  # Use the dynamic title element
                    ui.p(article.get('summary', 'No Summary')[:200] + "..." if len(str(article.get('summary', ''))) > 200 else article.get('summary', 'No Summary')),    
                    ui.span(f"Opinion: {review}", class_=f"badge bg-{sentiment_color}"),
                    class_="card-body"
                ),
                class_="card mb-3"
            )
            articles_html.append(article_card)
        
        return ui.div(*articles_html)
    
    @render.download(filename="data.xlsx")
    def download_basic():
        file_path = "data.xlsx"
        
        # Read the file and return its contents
        with open(file_path, "rb") as f:
            return f.read()

    
    @output
    @render.ui
    # @reactive.event(current_data)
    def sentiment_plot():
        """Render sentiment plot based on selected view type"""
        df = current_data()
        if df.empty:
            return ui.div("Search for articles to see sentiment analysis", class_="text-center mt-5")
        
        # Use the reactive value instead of the input switch
        if show_pie_chart():
            fig = create_sentiment_pie(df)
        else:
            fig = create_sentiment_plot(df)
        
        # Convert plotly figure to HTML
        plot_html = fig.to_html(include_plotlyjs='cdn', div_id="sentiment-plot")
        return ui.HTML(plot_html)

# Create the app
app_dir = Path(__file__).parent
app = App(app_ui, server, static_assets=app_dir / "www")

if __name__ == "__main__":
    app.run()