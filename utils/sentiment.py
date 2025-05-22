import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import tweepy
import praw
import os
from datetime import datetime, timedelta
import time

class SentimentAnalyzer:
    """
    Handles sentiment analysis from various sources including:
    - Social media (Twitter/X, Reddit)
    - Direct citizen feedback
    - Chat interactions
    """
    
    def __init__(self):
        # Initialize NLTK sentiment analyzer
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize social media APIs if keys are available
        self._init_social_apis()
        
    def _init_social_apis(self):
        """Initialize social media API clients if credentials are available"""
        self.twitter_api = None
        self.reddit_api = None
        
        # Twitter/X API setup
        twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if twitter_bearer_token:
            try:
                self.twitter_api = tweepy.Client(bearer_token=twitter_bearer_token)
            except Exception as e:
                print(f"Twitter API initialization error: {e}")
        
        # Reddit API setup
        reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'CitizenAI Sentiment Analyzer v1.0')
        
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit_api = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
            except Exception as e:
                print(f"Reddit API initialization error: {e}")
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        
        Args:
            text: String to analyze
            
        Returns:
            Dict with sentiment scores and category
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Determine sentiment category
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            category = 'positive'
        elif compound_score <= -0.05:
            category = 'negative'
        else:
            category = 'neutral'
            
        return {
            'scores': sentiment_scores,
            'category': category
        }
    
    def analyze_conversation(self, conversation_history):
        """
        Analyze sentiment trends in a conversation
        
        Args:
            conversation_history: List of conversation messages
            
        Returns:
            Dict with overall sentiment and trend
        """
        # Extract only user messages
        user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        
        if not user_messages:
            return {"overall": "neutral", "trend": "stable"}
        
        # Calculate sentiment for each message
        sentiments = [self.analyze_text(msg)["scores"]["compound"] for msg in user_messages]
        
        # Calculate overall sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Determine category
        if avg_sentiment >= 0.05:
            category = "positive"
        elif avg_sentiment <= -0.05:
            category = "negative"
        else:
            category = "neutral"
            
        # Calculate trend (improving/declining/stable)
        if len(sentiments) >= 3:
            # Simple trend: compare last message to average of previous messages
            previous_avg = sum(sentiments[:-1]) / (len(sentiments) - 1)
            last_sentiment = sentiments[-1]
            
            if last_sentiment > previous_avg + 0.1:
                trend = "improving"
            elif last_sentiment < previous_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
            
        return {
            "overall": category,
            "trend": trend
        }
    
    def get_social_media_sentiment(self, query, source="all", limit=100):
        """
        Fetch and analyze social media posts related to the query
        
        Args:
            query: Search term
            source: 'twitter', 'reddit', or 'all'
            limit: Maximum number of posts to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        # Query Twitter/X if available
        if (source == "twitter" or source == "all") and self.twitter_api:
            try:
                tweets = self.twitter_api.search_recent_tweets(
                    query=query, 
                    max_results=min(100, limit),
                    tweet_fields=['created_at', 'lang']
                )
                
                if tweets and hasattr(tweets, 'data') and tweets.data:
                    for tweet in tweets.data:
                        if tweet.lang == 'en':  # Filter for English tweets
                            sentiment = self.analyze_text(tweet.text)
                            results.append({
                                'source': 'Twitter',
                                'text': tweet.text,
                                'date': tweet.created_at,
                                'sentiment': sentiment['category'],
                                'score': sentiment['scores']['compound']
                            })
            except Exception as e:
                print(f"Twitter API error: {e}")
        
        # Query Reddit if available
        if (source == "reddit" or source == "all") and self.reddit_api:
            try:
                # Search in relevant subreddits
                subreddit = self.reddit_api.subreddit("all")
                posts = subreddit.search(query, limit=min(100, limit))
                
                for post in posts:
                    sentiment = self.analyze_text(post.title + " " + post.selftext)
                    results.append({
                        'source': 'Reddit',
                        'text': post.title,
                        'date': datetime.fromtimestamp(post.created_utc),
                        'sentiment': sentiment['category'],
                        'score': sentiment['scores']['compound']
                    })
            except Exception as e:
                print(f"Reddit API error: {e}")
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['source', 'text', 'date', 'sentiment', 'score'])
    
    def generate_sentiment_report(self, query=None, timeframe='week'):
        """
        Generate a comprehensive sentiment report for a specific query or general civic topics
        
        Args:
            query: Specific search term (or None for general civic topics)
            timeframe: 'day', 'week', or 'month'
            
        Returns:
            DataFrame with aggregated sentiment data
        """
        # In a real implementation, this would fetch data from social media
        # APIs and other sources based on the query and timeframe
        
        # For the hackathon demo, we'll generate synthetic data
        today = datetime.now()
        
        if timeframe == 'day':
            date_range = pd.date_range(end=today, periods=24, freq='H')
            date_column = 'hour'
        elif timeframe == 'week':
            date_range = pd.date_range(end=today, periods=7, freq='D')
            date_column = 'date'
        else:  # month
            date_range = pd.date_range(end=today, periods=30, freq='D')
            date_column = 'date'
        
        # Generate synthetic data with a slight positive trend
        import numpy as np
        
        # Base values
        positive_base = 60
        neutral_base = 25
        negative_base = 15
        
        # Add some randomness and a slight trend
        positive_values = [min(75, max(50, positive_base + i*0.5 + np.random.randint(-3, 4))) for i in range(len(date_range))]
        neutral_values = [min(35, max(15, neutral_base - i*0.2 + np.random.randint(-2, 3))) for i in range(len(date_range))]
        
        # Ensure percentages add up to 100
        negative_values = [100 - p - n for p, n in zip(positive_values, neutral_values)]
        
        # Create DataFrame
        df = pd.DataFrame({
            date_column: date_range,
            'positive': positive_values,
            'neutral': neutral_values,
            'negative': negative_values
        })
        
        return df