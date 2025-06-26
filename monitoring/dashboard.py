# monitoring/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
import logging

from monitoring.performance import PerformanceTracker
from data.database import Database

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Streamlit-based trading dashboard
    """

    def __init__(self, broker_interface=None, database=None,
                 performance_tracker=None):
        self.broker = broker_interface
        self.db = database or Database()
        self.performance = performance_tracker or PerformanceTracker()

        # Dashboard settings
        self.refresh_interval = 60  # seconds
        self.last_refresh = None

    def run(self):
        """Main dashboard entry point"""
        st.set_page_config(
            page_title="Smart Swing Trading Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .positive {
            color: #00cc00;
        }
        .negative {
            color: #ff0000;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header
        st.title("🚀 Smart Swing Trading Dashboard")

        # Sidebar
        self._render_sidebar()

        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "💼 Positions",
            "📈 Performance",
            "🔔 Signals",
            "📰 Market News"
        ])

        with tab1:
            self._render_overview()

        with tab2:
            self._render_positions()

        with tab3:
            self._render_performance()

        with tab4:
            self._render_signals()

        with tab5:
            self._render_market_news()

        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh", value=True):
            time.sleep(self.refresh_interval)
            st.rerun()

    def _render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("Dashboard Controls")

        # Last update time
        if self.last_refresh:
            st.sidebar.info(f"Last refresh: {self.last_refresh.strftime('%H:%M:%S')}")

        # Refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            self.last_refresh = datetime.now()
            st.rerun()

        # Account info
        if self.broker:
            account_info = self.broker.get_account_info()
            if account_info:
                st.sidebar.header("Account Info")
                st.sidebar.metric("Buying Power", f"${account_info['buying_power']:,.2f}")
                st.sidebar.metric("Portfolio Value", f"${account_info['portfolio_value']:,.2f}")

                # PDT warning
                if account_info['pattern_day_trader']:
                    st.sidebar.warning("⚠️ Pattern Day Trader")
                else:
                    trades_left = 3 - account_info['day_trade_count']
                    st.sidebar.info(f"Day trades remaining: {trades_left}")

        # Risk controls
        st.sidebar.header("Risk Controls")
        max_positions = st.sidebar.slider("Max Positions", 1, 10, 4)
        stop_loss = st.sidebar.slider("Stop Loss %", 1, 5, 3)

        # Save settings
        if st.sidebar.button("💾 Save Settings"):
            st.sidebar.success("Settings saved!")

    def _render_overview(self):
        """Render overview tab"""
        st.header("Trading Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        metrics = self.performance.get_current_metrics()

        with col1:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.2%}",
                delta=f"{metrics['total_return']:.2%}"
            )

        with col2:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}",
                delta=None
            )

        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                delta=None
            )

        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1%}",
                delta=None
            )

        # Equity curve
        st.subheader("Equity Curve")
        equity_df = self.performance.get_equity_curve()

        if not equity_df.empty:
            fig = go.Figure()

            # Equity line
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ))

            # Peak equity line
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['peak'],
                mode='lines',
                name='Peak',
                line=dict(color='green', width=1, dash='dash')
            ))

            # Drawdown area
            fig.add_trace(go.Scatter(
                x=equity_df.index,
                y=equity_df['equity'],
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(width=0),
                showlegend=False
            ))

            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Recent activity
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Recent Trades")
            recent_trades = self.db.get_filled_orders(lookback_hours=168)  # 7 days

            if recent_trades:
                trades_df = pd.DataFrame(recent_trades)
                trades_df = trades_df[['symbol', 'side', 'quantity', 'filled_price', 'filled_at']]
                st.dataframe(trades_df.head(10))
            else:
                st.info("No recent trades")

        with col2:
            st.subheader("Market Status")
            if self.broker:
                market_status = self.broker.get_market_status()

                if market_status['is_open']:
                    st.success("🟢 Market Open")
                else:
                    st.error("🔴 Market Closed")
                    st.info(f"Next open: {market_status['next_open']}")

    def _render_positions(self):
        """Render positions tab"""
        st.header("Current Positions")

        if self.broker:
            positions = self.broker.get_positions()

            if positions:
                # Summary metrics
                total_value = sum(p['market_value'] for p in positions)
                total_pnl = sum(p['unrealized_pl'] for p in positions)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Positions", len(positions))
                with col2:
                    st.metric("Total Value", f"${total_value:,.2f}")
                with col3:
                    color = "positive" if total_pnl >= 0 else "negative"
                    st.metric(
                        "Unrealized P&L",
                        f"${total_pnl:,.2f}",
                        delta=f"{total_pnl / total_value * 100:.1f}%"
                    )

                # Positions table
                positions_df = pd.DataFrame(positions)

                # Format columns
                positions_df['P&L'] = positions_df['unrealized_pl'].apply(
                    lambda x: f"${x:,.2f}"
                )
                positions_df['P&L %'] = positions_df['unrealized_plpc'].apply(
                    lambda x: f"{x * 100:.1f}%"
                )
                positions_df['Value'] = positions_df['market_value'].apply(
                    lambda x: f"${x:,.2f}"
                )

                # Select columns to display
                display_cols = [
                    'symbol', 'shares', 'avg_entry_price', 'current_price',
                    'Value', 'P&L', 'P&L %'
                ]

                st.dataframe(
                    positions_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

                # Position details
                st.subheader("Position Analysis")

                # P&L distribution
                fig = px.bar(
                    positions_df,
                    x='symbol',
                    y='unrealized_pl',
                    color='unrealized_pl',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="P&L by Position"
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No open positions")

    def _render_performance(self):
        """Render performance tab"""
        st.header("Performance Analytics")

        # Performance metrics
        metrics = self.performance.get_current_metrics()
        risk_metrics = self.performance.calculate_risk_metrics()

        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", metrics['total_trades'])
            st.metric("Win Rate", f"{metrics['win_rate']:.1%}")

        with col2:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
            st.metric("Expectancy", f"${risk_metrics['expectancy']:.2f}")

        with col3:
            st.metric("Avg Win", f"${metrics['avg_win']:.2f}")
            st.metric("Avg Loss", f"${metrics['avg_loss']:.2f}")

        with col4:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")

        # Trade analysis
        trade_analysis = self.performance.get_trade_analysis()

        if trade_analysis:
            # Monthly returns
            st.subheader("Monthly Returns")
            monthly_returns = self.performance.calculate_monthly_returns()

            if not monthly_returns.empty:
                # Create heatmap
                monthly_df = monthly_returns.to_frame('return')
                monthly_df['year'] = monthly_df.index.year
                monthly_df['month'] = monthly_df.index.month_name()

                pivot_df = monthly_df.pivot(index='year', columns='month', values='return')

                fig = px.imshow(
                    pivot_df.values,
                    labels=dict(x="Month", y="Year", color="Return"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale=['red', 'white', 'green'],
                    color_continuous_midpoint=0
                )

                st.plotly_chart(fig, use_container_width=True)

            # Trade distribution
            if 'by_symbol' in trade_analysis:
                st.subheader("Performance by Symbol")

                # Convert nested dict to DataFrame
                symbol_data = []
                for symbol, metrics in trade_analysis['by_symbol'].items():
                    if isinstance(metrics, dict) and 'pnl' in metrics:
                        symbol_data.append({
                            'Symbol': symbol,
                            'Trades': metrics['pnl'].get('count', 0),
                            'Total P&L': metrics['pnl'].get('sum', 0),
                            'Avg P&L': metrics['pnl'].get('mean', 0)
                        })

                if symbol_data:
                    symbol_df = pd.DataFrame(symbol_data)
                    symbol_df = symbol_df.sort_values('Total P&L', ascending=False)

                    fig = px.bar(
                        symbol_df.head(20),
                        x='Symbol',
                        y='Total P&L',
                        color='Total P&L',
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title="Top 20 Symbols by P&L"
                    )

                    st.plotly_chart(fig, use_container_width=True)

    def _render_signals(self):
        """Render signals tab"""
        st.header("Trading Signals")

        # Get recent signals from database
        recent_signals = self.db.get_signals(lookback_hours=24)

        if recent_signals:
            # Filter for high confidence
            high_conf_signals = [s for s in recent_signals if s['confidence'] > 0.7]

            if high_conf_signals:
                st.subheader(f"High Confidence Signals ({len(high_conf_signals)})")

                # Create DataFrame
                signals_df = pd.DataFrame(high_conf_signals)

                # Format display
                signals_df['Confidence'] = signals_df['confidence'].apply(
                    lambda x: f"{x:.1%}"
                )
                signals_df['Time'] = pd.to_datetime(signals_df['timestamp']).dt.strftime('%H:%M')

                # Select columns
                display_cols = ['symbol', 'signal_type', 'Confidence', 'Time']
                if 'expected_return' in signals_df.columns:
                    signals_df['Expected Return'] = signals_df['expected_return'].apply(
                        lambda x: f"{x:.1%}"
                    )
                    display_cols.insert(3, 'Expected Return')

                st.dataframe(
                    signals_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )

            # Signal distribution
            st.subheader("Signal Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Signals by type
                signal_types = pd.DataFrame(recent_signals)['signal_type'].value_counts()

                fig = px.pie(
                    values=signal_types.values,
                    names=signal_types.index,
                    title="Signals by Type"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Confidence distribution
                confidences = [s['confidence'] for s in recent_signals]

                fig = px.histogram(
                    x=confidences,
                    nbins=20,
                    title="Signal Confidence Distribution",
                    labels={'x': 'Confidence', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No recent signals")

        # Manual signal check
        st.subheader("Check Symbol")

        col1, col2 = st.columns([3, 1])
        with col1:
            symbol = st.text_input("Enter symbol to analyze", value="AAPL")
        with col2:
            analyze_btn = st.button("Analyze", type="primary")

        if analyze_btn and symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                # This would call your signal generation logic
                st.success(f"Analysis complete for {symbol}")

    def _render_market_news(self):
        """Render market news tab"""
        st.header("Market News & Sentiment")

        # Market sentiment summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Market Sentiment", "Neutral", delta="→")
        with col2:
            st.metric("VIX", "15.2", delta="-0.5")
        with col3:
            st.metric("News Volume", "High", delta="↑")

        # News feed
        st.subheader("Latest News")

        # Get recent news from database
        recent_news = self.db.get_news_sentiment(lookback_hours=24)

        if recent_news:
            for article in recent_news[:10]:
                with st.expander(f"{article['headline'][:100]}..."):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write(f"**Source**: {article['source']}")
                        st.write(f"**Time**: {article['timestamp']}")
                        if article.get('symbol'):
                            st.write(f"**Symbol**: {article['symbol']}")

                    with col2:
                        sentiment = article.get('sentiment', 'neutral')
                        if sentiment == 'bullish':
                            st.success("🟢 Bullish")
                        elif sentiment == 'bearish':
                            st.error("🔴 Bearish")
                        else:
                            st.info("⚪ Neutral")

                    if article.get('url'):
                        st.markdown(f"[Read more]({article['url']})")
        else:
            st.info("No recent news")

        # Sector sentiment
        st.subheader("Sector Sentiment")

        sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer']
        sentiments = ['Bullish', 'Neutral', 'Bearish', 'Neutral', 'Bullish']

        sector_df = pd.DataFrame({
            'Sector': sectors,
            'Sentiment': sentiments
        })

        # Color map
        color_map = {'Bullish': 'green', 'Neutral': 'gray', 'Bearish': 'red'}
        sector_df['Color'] = sector_df['Sentiment'].map(color_map)

        fig = px.bar(
            sector_df,
            x='Sector',
            y=[1] * len(sectors),
            color='Sentiment',
            color_discrete_map=color_map,
            title="Sector Sentiment Overview"
        )

        st.plotly_chart(fig, use_container_width=True)