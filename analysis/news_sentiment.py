# analysis/news_sentiment.py

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
from collections import defaultdict
import openai
import feedparser
import yfinance as yf
from functools import lru_cache
import re

from config.settings import Config

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment for trading signals
    Uses OpenAI for advanced sentiment analysis
    """

    def __init__(self, openai_api_key: str = None):
        """Initialize news analyzer"""
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        openai.api_key = self.openai_api_key

        # Cache for API efficiency
        self.sentiment_cache = {}
        self.cache_duration = 3600  # 1 hour

        # Free news sources
        self.rss_feeds = {
            'reuters_market': 'https://feeds.reuters.com/reuters/businessNews',
            'yahoo_finance': 'https://finance.yahoo.com/rss/topfinstories',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }

        # Sentiment keywords for quick filtering
        self.positive_keywords = [
            'surge', 'soar', 'jump', 'rally', 'gain', 'rise', 'boost', 'upgrade',
            'beat', 'exceed', 'strong', 'robust', 'growth', 'profit', 'record'
        ]

        self.negative_keywords = [
            'plunge', 'crash', 'fall', 'drop', 'decline', 'loss', 'weak', 'concern',
            'downgrade', 'miss', 'cut', 'warning', 'recession', 'fear', 'risk'
        ]

    def fetch_news(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """
        Fetch recent news articles

        Args:
            symbol: Stock symbol (None for general market news)
            hours: How many hours back to fetch

        Returns:
            List of news articles
        """
        articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Fetch from RSS feeds
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:20]:  # Limit per source
                    # Parse publish date
                    pub_date = self._parse_date(entry.get('published', ''))
                    if pub_date and pub_date < cutoff_time:
                        continue

                    article = {
                        'title': entry.get('title', ''),
                        'summary': self._clean_text(entry.get('summary', '')[:500]),
                        'url': entry.get('link', ''),
                        'published': pub_date or datetime.now(),
                        'source': source
                    }

                    # Filter by symbol if provided
                    if symbol:
                        content = f"{article['title']} {article['summary']}".lower()
                        # Check for symbol or company name
                        if self._contains_symbol(content, symbol):
                            articles.append(article)
                    else:
                        articles.append(article)

            except Exception as e:
                logger.error(f"Error fetching from {source}: {e}")

        # If symbol provided, also get Yahoo Finance news
        if symbol:
            yahoo_news = self._fetch_yahoo_news(symbol)
            articles.extend(yahoo_news)

        # Sort by date
        articles.sort(key=lambda x: x['published'], reverse=True)

        # Remove duplicates
        seen_titles = set()
        unique_articles = []
        for article in articles:
            title_key = article['title'][:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        return unique_articles[:50]  # Limit total articles

    def _fetch_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance for specific symbol"""
        articles = []

        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            for item in news[:10]:
                article = {
                    'title': item.get('title', ''),
                    'summary': '',  # Yahoo doesn't provide summary in API
                    'url': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'source': 'yahoo_finance_api'
                }
                articles.append(article)

        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {symbol}: {e}")

        return articles

    def analyze_sentiment(self, text: str, symbol: str = None) -> Dict:
        """
        Analyze sentiment using OpenAI

        Args:
            text: Text to analyze
            symbol: Stock symbol for context

        Returns:
            Sentiment analysis results
        """
        # Check cache
        cache_key = f"{hash(text[:200])}_{symbol}"
        if cache_key in self.sentiment_cache:
            cached_time, cached_result = self.sentiment_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_result

        try:
            # Prepare prompt
            prompt = self._create_sentiment_prompt(text, symbol)

            # Call OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst specializing in news sentiment analysis for swing trading (1-5 day holds). Provide concise, actionable insights."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )

            # Parse response
            content = response['choices'][0]['message']['content']
            result = self._parse_sentiment_response(content)

            # Add quick sentiment check
            result['quick_sentiment'] = self._quick_sentiment_check(text)

            # Cache result
            self.sentiment_cache[cache_key] = (time.time(), result)

            return result

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Fallback to simple sentiment
            return self._fallback_sentiment(text)

    def _create_sentiment_prompt(self, text: str, symbol: str = None) -> str:
        """Create prompt for sentiment analysis"""
        symbol_context = f" for {symbol}" if symbol else ""

        prompt = f"""Analyze the sentiment and trading implications of this financial news{symbol_context}:

"{text}"

Provide a JSON response with ONLY these fields:
{{
    "sentiment": "bullish" or "bearish" or "neutral",
    "confidence": 0.0 to 1.0,
    "impact": "high" or "medium" or "low",
    "time_horizon": "immediate" or "short_term" or "medium_term",
    "key_points": ["point1", "point2", "point3"] (max 3),
    "affected_sectors": ["sector1", "sector2"] (if applicable)
}}

Focus on implications for swing trading (1-5 day holding period)."""

        return prompt

    def _parse_sentiment_response(self, response: str) -> Dict:
        """Parse OpenAI response"""
        # Try to extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                # Validate and clean
                return {
                    'sentiment': result.get('sentiment', 'neutral'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'impact': result.get('impact', 'medium'),
                    'time_horizon': result.get('time_horizon', 'short_term'),
                    'key_points': result.get('key_points', [])[:3],
                    'affected_sectors': result.get('affected_sectors', [])
                }
        except:
            pass

        # Fallback parsing
        sentiment = 'neutral'
        if 'bullish' in response.lower():
            sentiment = 'bullish'
        elif 'bearish' in response.lower():
            sentiment = 'bearish'

        return {
            'sentiment': sentiment,
            'confidence': 0.5,
            'impact': 'medium',
            'time_horizon': 'short_term',
            'key_points': [],
            'affected_sectors': []
        }

    def _quick_sentiment_check(self, text: str) -> str:
        """Quick keyword-based sentiment check"""
        text_lower = text.lower()

        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)

        if positive_count > negative_count * 1.5:
            return 'positive'
        elif negative_count > positive_count * 1.5:
            return 'negative'
        else:
            return 'neutral'

    def _fallback_sentiment(self, text: str) -> Dict:
        """Fallback sentiment analysis without OpenAI"""
        quick_sentiment = self._quick_sentiment_check(text)

        sentiment_map = {
            'positive': 'bullish',
            'negative': 'bearish',
            'neutral': 'neutral'
        }

        return {
            'sentiment': sentiment_map[quick_sentiment],
            'confidence': 0.3,  # Low confidence for fallback
            'impact': 'medium',
            'time_horizon': 'short_term',
            'key_points': [],
            'affected_sectors': [],
            'method': 'fallback'
        }

    def analyze_symbol_news(self, symbol: str, hours: int = 48) -> Dict:
        """
        Analyze all recent news for a symbol

        Args:
            symbol: Stock symbol
            hours: Hours of news to analyze

        Returns:
            Aggregated sentiment analysis
        """
        # Fetch news
        articles = self.fetch_news(symbol, hours)

        if not articles:
            return {
                'symbol': symbol,
                'sentiment': 'neutral',
                'confidence': 0,
                'article_count': 0,
                'key_events': [],
                'recommendation': 'no_signal'
            }

        # Analyze each article
        sentiments = []
        all_key_points = []
        high_impact_events = []

        for article in articles[:10]:  # Limit to avoid too many API calls
            # Combine title and summary
            text = f"{article['title']}. {article['summary']}"

            if len(text) > 50:  # Skip very short items
                analysis = self.analyze_sentiment(text, symbol)

                sentiments.append({
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'impact': analysis['impact'],
                    'article': article
                })

                all_key_points.extend(analysis['key_points'])

                # Track high impact events
                if analysis['impact'] == 'high' and analysis['confidence'] > 0.7:
                    high_impact_events.append({
                        'title': article['title'],
                        'sentiment': analysis['sentiment'],
                        'published': article['published']
                    })

        # Aggregate sentiments
        overall_sentiment, confidence = self._aggregate_sentiments(sentiments)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_sentiment, confidence, high_impact_events
        )

        return {
            'symbol': symbol,
            'sentiment': overall_sentiment,
            'confidence': confidence,
            'article_count': len(articles),
            'analyzed_count': len(sentiments),
            'key_events': list(set(all_key_points))[:5],
            'high_impact_events': high_impact_events,
            'latest_articles': articles[:3],
            'recommendation': recommendation,
            'timestamp': datetime.now()
        }

    def _aggregate_sentiments(self, sentiments: List[Dict]) -> Tuple[str, float]:
        """Aggregate multiple sentiment scores"""
        if not sentiments:
            return 'neutral', 0

        # Weight by confidence and impact
        impact_weights = {'high': 3, 'medium': 2, 'low': 1}

        bullish_score = 0
        bearish_score = 0
        neutral_score = 0
        total_weight = 0

        for sent in sentiments:
            weight = sent['confidence'] * impact_weights.get(sent.get('impact', 'medium'), 2)

            if sent['sentiment'] == 'bullish':
                bullish_score += weight
            elif sent['sentiment'] == 'bearish':
                bearish_score += weight
            else:
                neutral_score += weight

            total_weight += weight

        if total_weight == 0:
            return 'neutral', 0

        # Determine overall sentiment
        if bullish_score > bearish_score * 1.5:
            sentiment = 'bullish'
            confidence = bullish_score / total_weight
        elif bearish_score > bullish_score * 1.5:
            sentiment = 'bearish'
            confidence = bearish_score / total_weight
        else:
            sentiment = 'neutral'
            confidence = neutral_score / total_weight

        return sentiment, min(confidence, 1.0)

    def _generate_recommendation(self, sentiment: str, confidence: float,
                                 high_impact_events: List[Dict]) -> str:
        """Generate trading recommendation based on sentiment"""
        # Check for recent high impact events
        recent_high_impact = any(
            event['published'] > datetime.now() - timedelta(hours=6)
            for event in high_impact_events
        )

        if sentiment == 'bullish' and confidence > 0.7:
            if recent_high_impact:
                return 'strong_buy'
            else:
                return 'buy'
        elif sentiment == 'bullish' and confidence > 0.5:
            return 'weak_buy'
        elif sentiment == 'bearish' and confidence > 0.7:
            if recent_high_impact:
                return 'strong_sell'
            else:
                return 'sell'
        elif sentiment == 'bearish' and confidence > 0.5:
            return 'weak_sell'
        else:
            return 'no_signal'

    def get_market_sentiment(self, hours: int = 12) -> Dict:
        """Get overall market sentiment"""
        # Fetch general market news
        articles = self.fetch_news(hours=hours)

        if not articles:
            return {
                'sentiment': 'neutral',
                'confidence': 0,
                'major_themes': [],
                'market_moving_events': []
            }

        # Analyze top articles
        sentiments = []
        themes = defaultdict(int)
        market_events = []

        for article in articles[:15]:
            text = f"{article['title']}. {article['summary']}"

            if len(text) > 50:
                analysis = self.analyze_sentiment(text)

                sentiments.append({
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'impact': analysis.get('impact', 'medium')
                })

                # Extract themes
                for sector in analysis.get('affected_sectors', []):
                    themes[sector] += 1

                # Track market moving events
                if analysis.get('impact') == 'high':
                    market_events.append({
                        'title': article['title'],
                        'sentiment': analysis['sentiment'],
                        'time': article['published']
                    })

        # Aggregate
        overall_sentiment, confidence = self._aggregate_sentiments(sentiments)

        # Top themes
        major_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'sentiment': overall_sentiment,
            'confidence': confidence,
            'major_themes': [theme[0] for theme in major_themes],
            'market_moving_events': market_events[:5],
            'article_count': len(articles),
            'timestamp': datetime.now()
        }

    def get_earnings_calendar(self, symbols: List[str], days_ahead: int = 7) -> Dict[str, Dict]:
        """Get earnings dates for symbols"""
        earnings_dates = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get earnings dates
                if hasattr(ticker, 'calendar') and ticker.calendar is not None:
                    calendar = ticker.calendar

                    # Extract next earnings date
                    if 'Earnings Date' in calendar.index:
                        earnings_date = calendar.loc['Earnings Date']

                        if isinstance(earnings_date, pd.Series):
                            earnings_date = earnings_date.iloc[0]

                        if pd.notna(earnings_date):
                            earnings_dates[symbol] = {
                                'date': earnings_date,
                                'days_until': (earnings_date - datetime.now()).days
                            }

            except Exception as e:
                logger.debug(f"No earnings data for {symbol}: {e}")

        return earnings_dates

    def analyze_pre_earnings_sentiment(self, symbol: str) -> Dict:
        """Special analysis for pre-earnings period"""
        # Check if earnings are upcoming
        earnings_info = self.get_earnings_calendar([symbol])

        if symbol not in earnings_info:
            return {'has_earnings': False}

        days_until = earnings_info[symbol]['days_until']

        if days_until > 7 or days_until < 0:
            return {'has_earnings': False}

        # Get recent news sentiment
        news_analysis = self.analyze_symbol_news(symbol, hours=72)

        # Adjust recommendation for earnings
        base_recommendation = news_analysis['recommendation']

        # Be more cautious before earnings
        if base_recommendation == 'strong_buy':
            adjusted_recommendation = 'buy'  # Downgrade due to earnings risk
        elif base_recommendation == 'buy':
            adjusted_recommendation = 'weak_buy'
        else:
            adjusted_recommendation = base_recommendation

        return {
            'has_earnings': True,
            'days_until_earnings': days_until,
            'pre_earnings_sentiment': news_analysis['sentiment'],
            'confidence': news_analysis['confidence'] * 0.8,  # Reduce confidence
            'recommendation': adjusted_recommendation,
            'warning': 'Earnings announcement increases volatility risk'
        }

    def get_sector_sentiment(self, sector_symbols: Dict[str, List[str]],
                             hours: int = 24) -> Dict[str, Dict]:
        """Analyze sentiment by sector"""
        sector_sentiments = {}

        for sector, symbols in sector_symbols.items():
            sector_articles = []

            # Collect news for sector symbols
            for symbol in symbols[:5]:  # Sample of sector
                articles = self.fetch_news(symbol, hours)
                sector_articles.extend(articles[:3])  # Limit per symbol

            if not sector_articles:
                sector_sentiments[sector] = {
                    'sentiment': 'neutral',
                    'confidence': 0
                }
                continue

            # Analyze sector sentiment
            sentiments = []
            for article in sector_articles[:10]:
                text = f"{article['title']}. {article['summary']}"
                if len(text) > 50:
                    analysis = self.analyze_sentiment(text)
                    sentiments.append({
                        'sentiment': analysis['sentiment'],
                        'confidence': analysis['confidence']
                    })

            # Aggregate
            overall_sentiment, confidence = self._aggregate_sentiments(sentiments)

            sector_sentiments[sector] = {
                'sentiment': overall_sentiment,
                'confidence': confidence,
                'article_count': len(sector_articles),
                'top_symbols': symbols[:5]
            }

        return sector_sentiments

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_string:
            return None

        # Common RSS date formats
        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%d %H:%M:%S'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_string.replace('GMT', '+0000'), fmt)
            except:
                continue

        return None

    def _clean_text(self, text: str) -> str:
        """Clean HTML and special characters from text"""
        # Remove HTML tags
        text = re.sub('<.*?>', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _contains_symbol(self, text: str, symbol: str) -> bool:
        """Check if text contains symbol or company reference"""
        symbol_lower = symbol.lower()

        # Direct symbol match
        if f" {symbol_lower} " in f" {text} ":
            return True
        if f"${symbol_lower}" in text:
            return True

        # Common company name patterns
        # This could be enhanced with a symbol->company name mapping
        company_patterns = {
            'aapl': ['apple'],
            'msft': ['microsoft'],
            'googl': ['google', 'alphabet'],
            'amzn': ['amazon'],
            'meta': ['meta', 'facebook'],
            'tsla': ['tesla'],
            'nvda': ['nvidia']
        }

        if symbol_lower in company_patterns:
            for company_name in company_patterns[symbol_lower]:
                if company_name in text:
                    return True

        return False

    def generate_sentiment_report(self, watchlist: List[str]) -> str:
        """Generate comprehensive sentiment report"""
        report_lines = ["=== Market Sentiment Report ==="]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Market sentiment
        market = self.get_market_sentiment()
        report_lines.append(f"Market Sentiment: {market['sentiment'].upper()}")
        report_lines.append(f"Confidence: {market['confidence']:.1%}")
        report_lines.append(f"Major Themes: {', '.join(market['major_themes'])}\n")

        # High impact symbols
        report_lines.append("High Impact News:")

        high_impact_symbols = []
        for symbol in watchlist[:20]:  # Check top 20 symbols
            analysis = self.analyze_symbol_news(symbol, hours=24)

            if analysis['confidence'] > 0.7 and analysis['sentiment'] != 'neutral':
                high_impact_symbols.append({
                    'symbol': symbol,
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'recommendation': analysis['recommendation']
                })

        # Sort by confidence
        high_impact_symbols.sort(key=lambda x: x['confidence'], reverse=True)

        for item in high_impact_symbols[:10]:
            report_lines.append(
                f"  {item['symbol']}: {item['sentiment']} "
                f"({item['confidence']:.1%}) - {item['recommendation']}"
            )

        return "\n".join(report_lines)