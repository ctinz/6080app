#!/usr/bin/env python3
"""
Tariff Change Probability Dashboard - Live Data Version
A web scraping and NLP-based system to monitor and predict tariff changes
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import nltk
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import hashlib
import feedparser
from newspaper import Article as NewsArticle
import tldextract

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

@dataclass
class Article:
    """Data class for storing article information"""
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    sentiment: float = 0.0
    relevance: float = 0.0
    
class TariffMonitor:
    """Main class for monitoring tariff-related news and calculating probabilities"""
    
    def __init__(self):
        # Source credibility weights (0-1 scale)
        self.source_weights = {
            'reuters.com': 0.95,
            'bloomberg.com': 0.93,
            'wsj.com': 0.91,
            'ft.com': 0.90,
            'politico.com': 0.85,
            'cnbc.com': 0.80,
            'cnn.com': 0.78,
            'foxnews.com': 0.75,
            'nytimes.com': 0.88,
            'washingtonpost.com': 0.87,
            'generic': 0.70
        }
        
        # Keywords for tariff-related content
        self.tariff_keywords = [
            'tariff', 'trade war', 'import duty', 'export tax', 
            'customs', 'trade policy', 'trade agreement', 'USMCA',
            'trade deal', 'sanctions', 'quota', 'WTO', 'dumping',
            'trade barrier', 'protectionism', 'free trade', 'trade deficit',
            'trade surplus', 'antidumping', 'countervailing'
        ]
        
        # Keywords specific to food/agriculture
        self.food_keywords = [
            'agricultural', 'food', 'produce', 'fruit', 'berry',
            'frozen', 'import', 'export', 'FDA', 'USDA', 'farm',
            'crop', 'harvest', 'agriculture', 'agribusiness'
        ]
        
        self.articles = []
        self.probability_history = []
        
    def get_source_weight(self, url: str) -> float:
        """Get credibility weight for a source"""
        url_lower = url.lower()
        for domain, weight in self.source_weights.items():
            if domain in url_lower:
                return weight
        return self.source_weights['generic']
    
    def calculate_relevance(self, text: str) -> float:
        """Calculate relevance score based on keyword presence"""
        text_lower = text.lower()
        
        # Count tariff keywords
        tariff_score = sum(1 for keyword in self.tariff_keywords 
                          if keyword in text_lower)
        
        # Count food keywords (bonus points for Wyman's-relevant content)
        food_score = sum(0.5 for keyword in self.food_keywords 
                        if keyword in text_lower)
        
        # Normalize to 0-1 scale
        total_score = (tariff_score + food_score) / 10
        return min(1.0, total_score)
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text[:1000])  # Limit text length for performance
            
            # Look for specific phrases that indicate policy changes
            change_indicators = [
                'will impose', 'considering tariffs', 'announced new',
                'plans to increase', 'threatening to', 'may implement',
                'proposed tariff', 'tariff increase', 'raising duties'
            ]
            
            # Boost sentiment if change indicators are present
            change_boost = 0.3 if any(ind in text.lower() for ind in change_indicators) else 0
            
            # Convert sentiment to probability scale
            # Negative sentiment = higher probability of tariff changes
            sentiment_score = (-blob.sentiment.polarity + 1) / 2
            
            return min(1.0, sentiment_score + change_boost)
        except:
            return 0.5  # Neutral if analysis fails
    
    def get_article_text(self, url: str) -> str:
        """Fetch and parse full article text using newspaper3k"""
        try:
            news_article = NewsArticle(url)
            news_article.download()
            news_article.parse()
            
            # Combine title and text for better context
            full_text = f"{news_article.title or ''} {news_article.text or ''}"
            
            # Limit text length to prevent processing issues
            return full_text[:5000]
        except Exception as e:
            st.warning(f"Could not fetch full text from {url}: {str(e)}")
            return ""
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            ext = tldextract.extract(url)
            return f"{ext.domain}.{ext.suffix}"
        except:
            return "unknown"
    
    def scrape_google_news(self, query="tariff OR trade OR import duty OR export tax OR customs OR trade policy", max_articles=15) -> List[Article]:
        """Scrape news from Google News RSS feed"""
        # Add food-related terms to the query
        enhanced_query = f"{query} OR agricultural OR food OR frozen fruit"
        
        feed_url = f"https://news.google.com/rss/search?q={requests.utils.quote(enhanced_query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            with st.spinner(f"Fetching {len(feed.entries[:max_articles])} articles from Google News..."):
                progress_bar = st.progress(0)
                
                for idx, entry in enumerate(feed.entries[:max_articles]):
                    try:
                        # Update progress
                        progress_bar.progress((idx + 1) / min(len(feed.entries), max_articles))
                        
                        # Get full article text
                        article_text = self.get_article_text(entry.link)
                        
                        # Skip if no content retrieved
                        if not article_text:
                            article_text = entry.get('summary', '')
                        
                        # Parse publication date
                        try:
                            pub_date = datetime(*entry.published_parsed[:6])
                        except:
                            pub_date = datetime.now()
                        
                        article = Article(
                            title=entry.title,
                            content=article_text,
                            source=self.extract_domain(entry.link),
                            url=entry.link,
                            timestamp=pub_date
                        )
                        articles.append(article)
                        
                    except Exception as e:
                        st.warning(f"Error parsing article: {e}")
                        continue
                
                progress_bar.empty()
            
            return articles
            
        except Exception as e:
            st.error(f"Error fetching Google News feed: {e}")
            return []
    
    def collect_articles(self) -> None:
        """Collect articles from all sources"""
        self.articles = []
        
        # Collect from Google News with different queries for comprehensive coverage
        queries = [
            "tariff trade policy United States",
            "agricultural import duty food",
            "trade war sanctions customs",
            "WTO trade barriers protectionism"
        ]
        
        for query in queries:
            self.articles.extend(self.scrape_google_news(query=query, max_articles=5))
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        for article in self.articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        self.articles = unique_articles
        
        # Analyze each article
        st.info(f"Analyzing {len(self.articles)} unique articles...")
        for article in self.articles:
            article.sentiment = self.analyze_sentiment(
                article.title + " " + article.content
            )
            article.relevance = self.calculate_relevance(
                article.title + " " + article.content
            )
    
    def calculate_probability(self) -> float:
        """Calculate overall probability of tariff changes"""
        if not self.articles:
            return 0.0
        
        # Filter for relevant articles only
        relevant_articles = [a for a in self.articles if a.relevance > 0.1]
        
        if not relevant_articles:
            return 0.0
        
        # Weight each article by source credibility, sentiment, relevance, and recency
        weighted_scores = []
        
        for article in relevant_articles:
            # Recency weight (decay over 7 days)
            age_hours = (datetime.now() - article.timestamp).total_seconds() / 3600
            recency_weight = np.exp(-age_hours / 168)  # 168 hours = 7 days
            
            # Source credibility
            source_weight = self.get_source_weight(article.url)
            
            # Combined score
            article_score = (
                article.sentiment * 
                article.relevance * 
                source_weight * 
                recency_weight
            )
            
            weighted_scores.append(article_score)
        
        # Calculate weighted average
        if weighted_scores:
            probability = np.mean(weighted_scores)
        else:
            probability = 0.0
        
        # Apply Bayesian adjustment based on historical base rate
        base_rate = 0.15  # Historical rate of tariff changes
        adjusted_probability = (probability * 0.7) + (base_rate * 0.3)
        
        return min(1.0, adjusted_probability)
    
    def get_confidence_level(self) -> Tuple[float, str]:
        """Calculate confidence level based on source agreement"""
        relevant_articles = [a for a in self.articles if a.relevance > 0.1]
        
        if not relevant_articles:
            return 0.0, "No Data"
        
        if len(relevant_articles) < 3:
            return 0.4, "Limited Data"
        
        # Calculate standard deviation of sentiments
        sentiments = [a.sentiment for a in relevant_articles]
        std_dev = np.std(sentiments)
        
        # Also consider number of high-quality sources
        quality_sources = sum(1 for a in relevant_articles 
                            if self.get_source_weight(a.url) > 0.85)
        
        # Lower std_dev = higher confidence
        if std_dev < 0.15 and quality_sources >= 3:
            return 0.9, "High"
        elif std_dev < 0.25 and quality_sources >= 2:
            return 0.7, "Medium"
        else:
            return 0.5, "Low"

def create_dashboard():
    """Create Streamlit dashboard"""
    st.set_page_config(
        page_title="Tariff Change Probability Dashboard - Live",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .stAlert {
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎯 Tariff Change Probability Dashboard - Live Data")
    st.markdown("**Real-time monitoring of trade policy changes affecting Wyman's operations**")
    st.caption("Powered by Google News RSS feeds and NLP analysis")
    
    # Initialize monitor
    if 'monitor' not in st.session_state:
        st.session_state.monitor = TariffMonitor()
        st.session_state.last_update = None
        st.session_state.probability_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Custom search query
        st.subheader("🔍 Custom Search")
        custom_query = st.text_input(
            "Add search terms:",
            placeholder="e.g., frozen fruit tariff",
            help="Add specific terms to search for"
        )
        
        max_articles = st.slider(
            "Articles per query",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of articles to fetch per search query"
        )
        
        auto_refresh = st.checkbox("Auto-refresh (every 5 min)", value=False)
        
        if st.button("🔄 Refresh Data", type="primary") or auto_refresh:
            with st.spinner("Collecting latest articles from Google News..."):
                # Add custom query if provided
                if custom_query:
                    st.session_state.monitor.articles = []
                    articles = st.session_state.monitor.scrape_google_news(
                        query=custom_query,
                        max_articles=max_articles
                    )
                    st.session_state.monitor.articles.extend(articles)
                    
                    # Analyze custom query articles
                    for article in articles:
                        article.sentiment = st.session_state.monitor.analyze_sentiment(
                            article.title + " " + article.content
                        )
                        article.relevance = st.session_state.monitor.calculate_relevance(
                            article.title + " " + article.content
                        )
                else:
                    st.session_state.monitor.collect_articles()
                
                st.session_state.last_update = datetime.now()
                
                # Store probability history
                current_prob = st.session_state.monitor.calculate_probability()
                st.session_state.probability_history.append({
                    'timestamp': datetime.now(),
                    'probability': current_prob
                })
        
        st.markdown("---")
        st.markdown("### 📊 Source Weights")
        
        # Show top sources
        top_sources = sorted(
            [(k, v) for k, v in st.session_state.monitor.source_weights.items() 
             if k != 'generic'],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for source, weight in top_sources:
            st.markdown(f"**{source}**: {weight:.2f}")
        
        st.caption("Higher weights = more trusted sources")
    
    # Main dashboard
    if st.session_state.monitor.articles:
        # Calculate current probability
        current_probability = st.session_state.monitor.calculate_probability()
        confidence_score, confidence_label = st.session_state.monitor.get_confidence_level()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_probability * 100,
                title={'text': "Tariff Change Probability %"},
                delta={'reference': st.session_state.probability_history[-2]['probability'] * 100 
                       if len(st.session_state.probability_history) > 1 else current_probability * 100},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.markdown('<p class="medium-font">Confidence Level</p>', unsafe_allow_html=True)
            confidence_color = {
                "High": "green",
                "Medium": "orange", 
                "Low": "red",
                "Limited Data": "gray",
                "No Data": "gray"
            }.get(confidence_label, "gray")
            
            st.markdown(f'<p class="big-font" style="color: {confidence_color};">{confidence_label}</p>', 
                       unsafe_allow_html=True)
            st.metric("Confidence Score", f"{confidence_score:.2f}")
            
            # Quality sources count
            quality_count = sum(1 for a in st.session_state.monitor.articles 
                              if st.session_state.monitor.get_source_weight(a.url) > 0.85)
            st.caption(f"High-quality sources: {quality_count}")
        
        with col3:
            st.markdown('<p class="medium-font">Articles Analyzed</p>', unsafe_allow_html=True)
            total_articles = len(st.session_state.monitor.articles)
            relevant_articles = len([a for a in st.session_state.monitor.articles if a.relevance > 0.1])
            
            st.markdown(f'<p class="big-font">{relevant_articles}/{total_articles}</p>', 
                       unsafe_allow_html=True)
            st.caption("Relevant / Total")
            
            if st.session_state.last_update:
                time_diff = datetime.now() - st.session_state.last_update
                if time_diff.seconds < 60:
                    st.caption(f"Updated {time_diff.seconds} seconds ago")
                else:
                    st.caption(f"Updated {time_diff.seconds // 60} minutes ago")
        
        with col4:
            st.markdown('<p class="medium-font">Risk Level</p>', unsafe_allow_html=True)
            if current_probability < 0.25:
                risk_level = "Low"
                risk_color = "green"
                risk_icon = "✅"
            elif current_probability < 0.5:
                risk_level = "Medium"
                risk_color = "orange"
                risk_icon = "⚠️"
            else:
                risk_level = "High"
                risk_color = "red"
                risk_icon = "🚨"
            
            st.markdown(f'<p class="big-font" style="color: {risk_color};">{risk_icon} {risk_level}</p>', 
                       unsafe_allow_html=True)
            
            # Trend indicator
            if len(st.session_state.probability_history) > 1:
                prev_prob = st.session_state.probability_history[-2]['probability']
                change = current_probability - prev_prob
                if change > 0.05:
                    st.caption("📈 Increasing risk")
                elif change < -0.05:
                    st.caption("📉 Decreasing risk")
                else:
                    st.caption("➡️ Stable")
        
        st.markdown("---")
        
        # Article analysis section
        st.header("📰 Live News Analysis")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            min_relevance = st.slider("Min. Relevance", 0.0, 1.0, 0.1)
        with col2:
            sort_by = st.selectbox("Sort by", ["Time", "Relevance", "Sentiment", "Source Weight"])
        with col3:
            show_content = st.checkbox("Show article preview", value=False)
        
        # Filter and sort articles
        filtered_articles = [a for a in st.session_state.monitor.articles 
                           if a.relevance >= min_relevance]
        
        if sort_by == "Time":
            filtered_articles.sort(key=lambda x: x.timestamp, reverse=True)
        elif sort_by == "Relevance":
            filtered_articles.sort(key=lambda x: x.relevance, reverse=True)
        elif sort_by == "Sentiment":
            filtered_articles.sort(key=lambda x: x.sentiment, reverse=True)
        else:  # Source Weight
            filtered_articles.sort(
                key=lambda x: st.session_state.monitor.get_source_weight(x.url), 
                reverse=True
            )
        
        # Display articles
        if filtered_articles:
            for article in filtered_articles[:20]:  # Limit display
                with st.expander(
                    f"📄 {article.title[:100]}... | "
                    f"🔗 {article.source} | "
                    f"📅 {article.timestamp.strftime('%Y-%m-%d %H:%M')}"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sentiment", f"{article.sentiment:.2f}")
                    with col2:
                        st.metric("Relevance", f"{article.relevance:.2f}")
                    with col3:
                        weight = st.session_state.monitor.get_source_weight(article.url)
                        st.metric("Source Weight", f"{weight:.2f}")
                    with col4:
                        impact = article.sentiment * article.relevance * weight
                        st.metric("Impact Score", f"{impact:.2f}")
                    
                    if show_content and article.content:
                        st.markdown("**Article Preview:**")
                        st.text(article.content[:500] + "...")
                    
                    st.markdown(f"[Read full article]({article.url})")
        else:
            st.warning("No articles match the current filter criteria")
        
        # Source analysis
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Distribution")
            
            relevant_sentiments = [a.sentiment for a in st.session_state.monitor.articles 
                                 if a.relevance > 0.1]
            
            if relevant_sentiments:
                fig_hist = px.histogram(
                    relevant_sentiments, 
                    nbins=20,
                    title="Article Sentiment Distribution (Relevant Articles)",
                    labels={'value': 'Sentiment Score', 'count': 'Number of Articles'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.add_vline(x=np.mean(relevant_sentiments), 
                                 line_dash="dash", 
                                 annotation_text="Mean")
                fig_hist.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Top Sources")
            
            source_data = defaultdict(lambda: {'count': 0, 'avg_relevance': []})
            for article in st.session_state.monitor.articles:
                source_data[article.source]['count'] += 1
                source_data[article.source]['avg_relevance'].append(article.relevance)
            
            # Calculate averages and sort
            source_summary = []
            for source, data in source_data.items():
                avg_rel = np.mean(data['avg_relevance'])
                weight = st.session_state.monitor.get_source_weight(source)
                source_summary.append({
                    'Source': source,
                    'Articles': data['count'],
                    'Avg Relevance': avg_rel,
                    'Trust Score': weight
                })
            
            df_sources = pd.DataFrame(source_summary)
            df_sources = df_sources.sort_values('Articles', ascending=False).head(10)
            
            fig_sources = px.bar(
                df_sources, 
                x='Articles', 
                y='Source',
                color='Trust Score',
                color_continuous_scale='blues',
                title="Article Count by Source",
                orientation='h'
            )
            fig_sources.update_layout(height=300)
            st.plotly_chart(fig_sources, use_container_width=True)
        
        # Alerts section
        st.markdown("---")
        st.header("⚠️ Alerts & Recommendations")
        
        # Dynamic alerts based on current data
        alert_container = st.container()
        
        with alert_container:
            if current_probability > 0.7:
                st.error(f"""
                **🚨 HIGH RISK ALERT**: {current_probability*100:.1f}% probability of tariff changes
                
                **Key Indicators:**
                - {sum(1 for a in filtered_articles if a.sentiment > 0.7)} articles with high change sentiment
                - {confidence_label} confidence based on {len(filtered_articles)} relevant sources
                
                **Immediate Actions Required:**
                - Review current inventory levels for imported ingredients
                - Contact suppliers for expedited shipment options
                - Prepare alternative sourcing strategies
                - Schedule emergency supply chain meeting
                - Model financial impact of 10-25% tariff increase
                """)
            elif current_probability > 0.5:
                st.warning(f"""
                **⚠️ ELEVATED RISK**: {current_probability*100:.1f}% probability of tariff changes
                
                **Monitoring Required:**
                - Increased mentions of trade policy in {len(filtered_articles)} articles
                - Continue monitoring over next 48-72 hours
                
                **Recommended Actions:**
                - Review contingency plans
                - Assess current inventory buffer levels
                - Identify alternative suppliers
                - Prepare communication for stakeholders
                """)
            else:
                st.success(f"""
                **✅ NORMAL CONDITIONS**: {current_probability*100:.1f}% probability of tariff changes
                
                **Status:**
                - No significant policy changes indicated
                - Normal market conditions prevailing
                
                **Recommended Actions:**
                - Maintain standard monitoring schedule
                - Continue regular supply chain operations
                - Update risk assessments as scheduled
                """)
        
        # Historical trend
        if len(st.session_state.probability_history) > 1:
            st.markdown("---")
            st.header("📈 Probability Trend")
            
            # Convert history to DataFrame
            df_history = pd.DataFrame(st.session_state.probability_history)
            df_history['probability'] = df_history['probability'] * 100
            
            fig_trend = px.line(
                df_history,
                x='timestamp',
                y='probability',
                title="Real-time Tariff Change Probability Trend",
                labels={'probability': 'Probability (%)', 'timestamp': 'Time'}
            )
            
            # Add threshold lines
            fig_trend.add_hline(y=70, line_dash="dash", line_color="red", 
                              annotation_text="High Risk", annotation_position="right")
            fig_trend.add_hline(y=50, line_dash="dash", line_color="orange", 
                              annotation_text="Medium Risk", annotation_position="right")
            
            # Add markers for updates
            fig_trend.update_traces(mode='lines+markers')
            fig_trend.update_layout(height=400)
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current", f"{df_history['probability'].iloc[-1]:.1f}%")
            with col2:
                st.metric("Average", f"{df_history['probability'].mean():.1f}%")
            with col3:
                st.metric("Max (Session)", f"{df_history['probability'].max():.1f}%")
        
    else:
        # No data state
        st.info("""
        👆 Click **'Refresh Data'** in the sidebar to start monitoring live news
        
        This dashboard will:
        - Fetch recent articles from Google News
        - Analyze sentiment and relevance
        - Calculate probability of tariff changes
        - Provide actionable recommendations
        
        The system monitors multiple queries related to trade policy, tariffs, and agricultural imports.
        """)
    
    # Auto-refresh logic
    if auto_refresh and st.session_state.last_update:
        time_since_update = (datetime.now() - st.session_state.last_update).seconds
        if time_since_update > 300:  # 5 minutes
            st.rerun()

if __name__ == "__main__":
    create_dashboard()