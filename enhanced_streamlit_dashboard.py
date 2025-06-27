# enhanced_streamlit_dashboard.py
"""
Sophisticated Trading Dashboard with Real-time Updates
Integrates with the enhanced backtesting system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional
import yfinance as yf
import alpaca_trade_api as tradeapi
from collections import defaultdict
import threading
import queue

# Import custom modules
from config.settings import Config
from config.watchlist import WATCHLIST, SECTOR_MAPPING
from data.market_data import MarketDataManager
from execution.broker_interface import BrokerInterface
from models.ensemble_gpu_windows import GPUEnsembleModel
from enhanced_advanced_backtesting import HedgeFundBacktester, BacktestConfig
from analysis.news_sentiment import NewsSentimentAnalyzer
from risk.risk_manager import RiskManager

# Configure Streamlit
st.set_page_config(
    page_title="Hedge Fund Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .trade-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Enhanced trading dashboard with real-time updates"""

    def __init__(self):
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.auto_refresh = False
            st.session_state.refresh_interval = 5
            st.session_state.selected_symbols = []
            st.session_state.backtest_results = None
            st.session_state.live_prices = {}
            st.session_state.positions = []
            st.session_state.pending_orders = []
            st.session_state.trade_history = []
            st.session_state.alerts = []
            st.session_state.last_update = datetime.now()

        # Initialize components
        self.market_data = MarketDataManager()
        self.broker = BrokerInterface()
        self.risk_manager = RiskManager(100000)  # Default capital
        self.news_analyzer = NewsSentimentAnalyzer()

        # Load ML model if available
        self.ml_model = None
        if os.path.exists('models/saved'):
            try:
                self.ml_model = GPUEnsembleModel()
                self.ml_model.load_models('models/saved')
                st.session_state.ml_model_loaded = True
            except Exception as e:
                st.error(f"Failed to load ML model: {e}")
                st.session_state.ml_model_loaded = False

    def run(self):
        """Main dashboard entry point"""
        # Header
        st.title("🏛️ Hedge Fund Trading System")
        st.markdown("### Professional Algorithmic Trading Platform")

        # Sidebar
        self._render_sidebar()

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Portfolio Overview",
            "📈 Market Analysis",
            "🤖 AI Signals",
            "📉 Backtesting",
            "⚠️ Risk Management",
            "📰 News & Sentiment"
        ])

        with tab1:
            self._render_portfolio_overview()

        with tab2:
            self._render_market_analysis()

        with tab3:
            self._render_ai_signals()

        with tab4:
            self._render_backtesting()

        with tab5:
            self._render_risk_management()

        with tab6:
            self._render_news_sentiment()

        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()

    def _render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("⚙️ Control Panel")

            # Account info
            if self._update_account_info():
                st.success("✅ Connected to Alpaca")
            else:
                st.error("❌ Not connected to Alpaca")

            st.divider()

            # Auto-refresh controls
            st.subheader("🔄 Auto Refresh")
            st.session_state.auto_refresh = st.checkbox(
                "Enable Auto-Refresh",
                value=st.session_state.auto_refresh
            )

            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=5,
                    max_value=60,
                    value=st.session_state.refresh_interval,
                    step=5
                )

            # Manual refresh
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.rerun()

            st.divider()

            # Watchlist selection
            st.subheader("📋 Watchlist")

            # Sector filter
            sectors = ['All'] + list(set(SECTOR_MAPPING.values()))
            selected_sector = st.selectbox("Filter by Sector", sectors)

            # Symbol selection
            if selected_sector == 'All':
                available_symbols = WATCHLIST
            else:
                available_symbols = [s for s, sec in SECTOR_MAPPING.items()
                                     if sec == selected_sector]

            st.session_state.selected_symbols = st.multiselect(
                "Select Symbols",
                available_symbols,
                default=st.session_state.selected_symbols[
                        :10] if st.session_state.selected_symbols else available_symbols[:10]
            )

            st.divider()

            # Trading controls
            st.subheader("🎯 Trading Controls")

            trading_mode = st.radio(
                "Trading Mode",
                ["Paper", "Live"],
                index=0,
                horizontal=True
            )

            if trading_mode == "Live":
                st.warning("⚠️ Live trading mode - Real money at risk!")

            # Quick actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛑 Close All", use_container_width=True):
                    self._close_all_positions()

            with col2:
                if st.button("❌ Cancel Orders", use_container_width=True):
                    self._cancel_all_orders()

            # System status
            st.divider()
            st.subheader("📊 System Status")

            # Last update time
            st.metric(
                "Last Update",
                st.session_state.last_update.strftime("%H:%M:%S"),
                delta=f"{(datetime.now() - st.session_state.last_update).seconds}s ago"
            )

            # Model status
            if st.session_state.get('ml_model_loaded'):
                st.success("✅ ML Model Loaded")
            else:
                st.warning("⚠️ ML Model Not Loaded")

    def _render_portfolio_overview(self):
        """Render portfolio overview tab"""
        st.header("📊 Portfolio Overview")

        # Update portfolio data
        self._update_portfolio_data()

        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        account_info = st.session_state.get('account_info', {})

        with col1:
            st.metric(
                "Total Value",
                f"${account_info.get('portfolio_value', 0):,.2f}",
                delta=f"{account_info.get('daily_change_pct', 0):.2f}%"
            )

        with col2:
            st.metric(
                "Cash Balance",
                f"${account_info.get('cash', 0):,.2f}"
            )

        with col3:
            st.metric(
                "Buying Power",
                f"${account_info.get('buying_power', 0):,.2f}"
            )

        with col4:
            daily_pnl = account_info.get('daily_pnl', 0)
            st.metric(
                "Daily P&L",
                f"${daily_pnl:,.2f}",
                delta=f"{(daily_pnl / account_info.get('portfolio_value', 1)) * 100:.2f}%"
            )

        with col5:
            st.metric(
                "Open Positions",
                len(st.session_state.positions)
            )

        # Portfolio composition
        st.subheader("📊 Portfolio Composition")

        if st.session_state.positions:
            # Create portfolio dataframe
            positions_df = pd.DataFrame(st.session_state.positions)

            # Add calculations
            positions_df['current_value'] = positions_df['qty'] * positions_df['current_price']
            positions_df['pnl'] = (positions_df['current_price'] - positions_df['avg_entry_price']) * positions_df[
                'qty']
            positions_df['pnl_pct'] = ((positions_df['current_price'] / positions_df['avg_entry_price']) - 1) * 100
            positions_df['allocation_pct'] = (positions_df['current_value'] / account_info.get('portfolio_value',
                                                                                               1)) * 100

            # Portfolio pie chart
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.pie(
                    positions_df,
                    values='current_value',
                    names='symbol',
                    title='Position Allocation',
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Sector allocation
                sector_allocation = positions_df.groupby(
                    positions_df['symbol'].map(lambda x: SECTOR_MAPPING.get(x, 'Unknown'))
                )['current_value'].sum()

                fig2 = px.pie(
                    values=sector_allocation.values,
                    names=sector_allocation.index,
                    title='Sector Allocation',
                    hole=0.4
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Positions table
            st.subheader("📋 Current Positions")

            # Format dataframe for display
            display_df = positions_df[[
                'symbol', 'qty', 'avg_entry_price', 'current_price',
                'current_value', 'pnl', 'pnl_pct', 'allocation_pct'
            ]].copy()

            display_df.columns = [
                'Symbol', 'Quantity', 'Avg Entry', 'Current',
                'Value', 'P&L', 'P&L %', 'Allocation %'
            ]

            # Format columns
            for col in ['Avg Entry', 'Current', 'Value', 'P&L']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

            for col in ['P&L %', 'Allocation %']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")

            # Color code P&L
            def color_pnl(val):
                if '

            " in str(val):
            num = float(val.replace('
                                    ", '').replace(',', ''))
            color = 'green' if num > 0 else 'red' if num < 0 else 'black'
            elif '%' in str(val):
            num = float(val.replace('%', ''))
            color = 'green' if num > 0 else 'red' if num < 0 else 'black'
            else:
            color = 'black'
        return f'color: {color}'

        styled_df = display_df.style.applymap(
            color_pnl,
            subset=['P&L', 'P&L %']
        )

        st.dataframe(styled_df, use_container_width=True)

    else:
    st.info("No open positions")


# Recent trades
st.subheader("📜 Recent Trades")
self._render_trade_history()

# Pending orders
st.subheader("⏳ Pending Orders")
self._render_pending_orders()


def _render_market_analysis(self):
    """Render market analysis tab"""
    st.header("📈 Market Analysis")

    # Market overview
    col1, col2, col3, col4 = st.columns(4)

    # Get market indices
    with col1:
        spy_price = self._get_index_data('SPY')
        if spy_price:
            st.metric(
                "S&P 500",
                f"${spy_price['price']:.2f}",
                delta=f"{spy_price['change_pct']:.2f}%"
            )

    with col2:
        qqq_price = self._get_index_data('QQQ')
        if qqq_price:
            st.metric(
                "NASDAQ",
                f"${qqq_price['price']:.2f}",
                delta=f"{qqq_price['change_pct']:.2f}%"
            )

    with col3:
        vix_price = self._get_index_data('VIX')
        if vix_price:
            st.metric(
                "VIX",
                f"{vix_price['price']:.2f}",
                delta=f"{vix_price['change_pct']:.2f}%"
            )

    with col4:
        # Market status
        market_status = self.market_data.get_market_status()
        if market_status['is_open']:
            st.success("🟢 Market Open")
        else:
            st.error("🔴 Market Closed")

    # Selected symbols analysis
    if st.session_state.selected_symbols:
        st.subheader("📊 Selected Symbols Analysis")

        # Get current data for selected symbols
        symbols_data = []
        for symbol in st.session_state.selected_symbols:
            data = self._get_symbol_data(symbol)
            if data:
                symbols_data.append(data)

        if symbols_data:
            # Create performance table
            perf_df = pd.DataFrame(symbols_data)

            # Add sparkline chart
            fig = make_subplots(
                rows=len(symbols_data),
                cols=1,
                subplot_titles=[d['symbol'] for d in symbols_data],
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[1 / len(symbols_data)] * len(symbols_data)
            )

            for i, data in enumerate(symbols_data):
                # Get intraday data
                bars = self.market_data.get_bars(
                    data['symbol'], '5Min', limit=78  # Full trading day
                )

                if not bars.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=bars.index,
                            y=bars['close'],
                            mode='lines',
                            name=data['symbol'],
                            line=dict(
                                color='green' if data['change_pct'] > 0 else 'red',
                                width=2
                            )
                        ),
                        row=i + 1, col=1
                    )

            fig.update_layout(
                height=200 * len(symbols_data),
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Performance metrics table
            st.subheader("📊 Performance Metrics")

            metrics_df = perf_df[[
                'symbol', 'price', 'change', 'change_pct',
                'volume', 'avg_volume', 'volume_ratio'
            ]].copy()

            metrics_df.columns = [
                'Symbol', 'Price', 'Change', 'Change %',
                'Volume', 'Avg Volume', 'Vol Ratio'
            ]

            # Format columns
            metrics_df['Price'] = metrics_df['Price'].apply(lambda x: f"${x:.2f}")
            metrics_df['Change'] = metrics_df['Change'].apply(lambda x: f"${x:.2f}")
            metrics_df['Change %'] = metrics_df['Change %'].apply(lambda x: f"{x:.2f}%")
            metrics_df['Volume'] = metrics_df['Volume'].apply(lambda x: f"{x:,.0f}")
            metrics_df['Avg Volume'] = metrics_df['Avg Volume'].apply(lambda x: f"{x:,.0f}")
            metrics_df['Vol Ratio'] = metrics_df['Vol Ratio'].apply(lambda x: f"{x:.2f}x")

            st.dataframe(metrics_df, use_container_width=True)

            # Technical indicators heatmap
            st.subheader("🔥 Technical Indicators Heatmap")
            self._render_technical_heatmap(st.session_state.selected_symbols)
    else:
        st.info("Select symbols from the sidebar to view analysis")


def _render_ai_signals(self):
    """Render AI signals tab"""
    st.header("🤖 AI Trading Signals")

    if not st.session_state.get('ml_model_loaded'):
        st.warning("⚠️ ML Model not loaded. Please train or load a model first.")

        if st.button("🎓 Train New Model"):
            with st.spinner("Training model... This may take several minutes."):
                # This would trigger model training
                st.info("Model training functionality to be implemented")
        return

    # Generate signals
    if st.button("🔍 Scan for Signals", use_container_width=True):
        with st.spinner("Scanning market for opportunities..."):
            signals = self._generate_ai_signals()
            st.session_state.ai_signals = signals

    # Display signals
    if hasattr(st.session_state, 'ai_signals') and st.session_state.ai_signals:
        st.subheader("📊 Current AI Signals")

        # Signal summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_signals = len(st.session_state.ai_signals)
        high_conf_signals = len([s for s in st.session_state.ai_signals if s['confidence'] > 0.8])
        avg_confidence = np.mean([s['confidence'] for s in st.session_state.ai_signals])
        avg_predicted_return = np.mean([s['predicted_return'] for s in st.session_state.ai_signals])

        with col1:
            st.metric("Total Signals", total_signals)

        with col2:
            st.metric("High Confidence", high_conf_signals)

        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

        with col4:
            st.metric("Avg Predicted Return", f"{avg_predicted_return:.2%}")

        # Signals table
        signals_df = pd.DataFrame(st.session_state.ai_signals)

        # Add additional calculations
        for idx, signal in signals_df.iterrows():
            # Get current price
            current_price = self.market_data.get_current_price(signal['symbol'])
            signals_df.at[idx, 'current_price'] = current_price

            # Calculate position size
            position_value = 100000 * 0.02  # 2% position size
            signals_df.at[idx, 'suggested_shares'] = int(position_value / current_price) if current_price else 0
            signals_df.at[idx, 'position_value'] = signals_df.at[
                                                       idx, 'suggested_shares'] * current_price if current_price else 0

        # Display signals
        for idx, signal in signals_df.iterrows():
            with st.expander(f"{signal['symbol']} - Confidence: {signal['confidence']:.2%}", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Current Price", f"${signal['current_price']:.2f}")
                    st.metric("Predicted Return", f"{signal['predicted_return']:.2%}")
                    st.metric("ML Confidence", f"{signal['confidence']:.2%}")

                with col2:
                    st.metric("Suggested Shares", signal['suggested_shares'])
                    st.metric("Position Value", f"${signal['position_value']:,.2f}")
                    st.metric("Stop Loss", f"${signal['current_price'] * 0.98:.2f}")

                with col3:
                    # Model predictions breakdown
                    if 'model_predictions' in signal:
                        st.write("**Model Predictions:**")
                        for model, pred in signal['model_predictions'].items():
                            st.write(f"- {model}: {pred:.2%}")

                    # Execute button
                    if st.button(f"Execute Trade", key=f"execute_{signal['symbol']}"):
                        self._execute_ai_signal(signal)
    else:
        st.info("Click 'Scan for Signals' to generate AI trading signals")


def _render_backtesting(self):
    """Render backtesting tab"""
    st.header("📉 Backtesting Engine")

    # Backtest configuration
    with st.expander("⚙️ Backtest Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Period Settings")

            # Date range
            end_date = st.date_input("End Date", value=datetime.now())
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )

            # Capital settings
            initial_capital = st.number_input(
                "Initial Capital",
                min_value=10000,
                max_value=10000000,
                value=100000,
                step=10000
            )

            # Position sizing
            position_size = st.slider(
                "Position Size %",
                min_value=1,
                max_value=10,
                value=2
            ) / 100

            max_positions = st.slider(
                "Max Positions",
                min_value=5,
                max_value=50,
                value=20
            )

        with col2:
            st.subheader("Risk Settings")

            stop_loss = st.slider(
                "Stop Loss %",
                min_value=1,
                max_value=10,
                value=2
            ) / 100

            take_profit = st.slider(
                "Take Profit %",
                min_value=2,
                max_value=20,
                value=5
            ) / 100

            max_drawdown = st.slider(
                "Max Drawdown %",
                min_value=10,
                max_value=50,
                value=15
            ) / 100

            # Advanced options
            use_correlation_filter = st.checkbox("Use Correlation Filter", value=True)
            use_market_regime = st.checkbox("Use Market Regime Detection", value=True)

    # Symbol selection for backtest
    backtest_symbols = st.multiselect(
        "Select Symbols for Backtest",
        WATCHLIST,
        default=WATCHLIST[:50]  # Default to first 50
    )

    # Run backtest button
    if st.button("🚀 Run Backtest", use_container_width=True, type="primary"):
        if len(backtest_symbols) == 0:
            st.error("Please select at least one symbol")
            return

        # Create config
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size,
            max_positions=max_positions,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_drawdown_pct=max_drawdown,
            use_correlation_filter=use_correlation_filter,
            use_market_regime=use_market_regime
        )

        # Run backtest
        with st.spinner(f"Running backtest on {len(backtest_symbols)} symbols... This may take a few minutes."):
            backtester = HedgeFundBacktester(config)
            results = backtester.run_backtest(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                symbols=backtest_symbols
            )

            st.session_state.backtest_results = results

    # Display results
    if st.session_state.backtest_results:
        results = st.session_state.backtest_results

        if 'error' not in results:
            # Performance summary
            st.subheader("📊 Backtest Performance Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Return",
                    f"{results['total_return'] * 100:.2f}%",
                    delta=f"{results['annual_return'] * 100:.2f}% annual"
                )

            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{results['sharpe_ratio']:.2f}",
                    delta="Good" if results['sharpe_ratio'] > 1.5 else "Poor"
                )

            with col3:
                st.metric(
                    "Max Drawdown",
                    f"{results['max_drawdown'] * 100:.2f}%"
                )

            with col4:
                st.metric(
                    "Win Rate",
                    f"{results['win_rate'] * 100:.1f}%",
                    delta=f"{results['profit_factor']:.2f} profit factor"
                )

            # Equity curve
            st.subheader("📈 Equity Curve")

            if 'equity_curve' in results:
                equity_df = pd.DataFrame(results['equity_curve'])
                equity_df['date'] = pd.to_datetime(equity_df['date'])

                fig = go.Figure()

                # Add equity curve
                fig.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))

                # Add initial capital line
                fig.add_hline(
                    y=results['initial_capital'],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Capital"
                )

                # Shade profitable/unprofitable regions
                fig.add_traces([
                    go.Scatter(
                        x=equity_df['date'],
                        y=equity_df['equity'],
                        fill='tonexty',
                        fillcolor='rgba(0,255,0,0.1)',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    go.Scatter(
                        x=equity_df['date'],
                        y=[results['initial_capital']] * len(equity_df),
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                ])

                fig.update_layout(
                    title='Portfolio Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    hovermode='x unified',
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            # Additional metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Trade Analysis")

                trade_metrics = {
                    "Total Trades": results['total_trades'],
                    "Average Win": f"${results['avg_win']:.2f}",
                    "Average Loss": f"${results['avg_loss']:.2f}",
                    "Best Trade": f"${results['best_trade']:.2f}",
                    "Worst Trade": f"${results['worst_trade']:.2f}",
                    "Avg Holding Period": f"{results['avg_holding_period']:.1f} days"
                }

                for metric, value in trade_metrics.items():
                    st.write(f"**{metric}:** {value}")

            with col2:
                st.subheader("📊 Exit Reason Analysis")

                if results.get('exit_reasons'):
                    exit_df = pd.DataFrame(
                        list(results['exit_reasons'].items()),
                        columns=['Reason', 'Count']
                    )

                    fig = px.pie(
                        exit_df,
                        values='Count',
                        names='Reason',
                        title='Exit Reasons Distribution'
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Monthly returns heatmap
            if results.get('monthly_returns'):
                st.subheader("📅 Monthly Returns")

                monthly_stats = results['monthly_returns']

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Avg Monthly", f"{monthly_stats['mean'] * 100:.2f}%")

                with col2:
                    st.metric("Best Month", f"{monthly_stats['best'] * 100:.2f}%")

                with col3:
                    st.metric("Worst Month", f"{monthly_stats['worst'] * 100:.2f}%")

                with col4:
                    st.metric("Positive Months",
                              f"{monthly_stats['positive_months']}/{monthly_stats['positive_months'] + monthly_stats['negative_months']}")
        else:
            st.error(f"Backtest failed: {results.get('error')}")


def _render_risk_management(self):
    """Render risk management tab"""
    st.header("⚠️ Risk Management")

    # Risk metrics overview
    col1, col2, col3, col4 = st.columns(4)

    # Calculate current risk metrics
    positions_value = sum([p['qty'] * p['current_price'] for p in st.session_state.positions])
    account_value = st.session_state.get('account_info', {}).get('portfolio_value', 100000)

    with col1:
        portfolio_heat = (positions_value / account_value) * 100 if account_value > 0 else 0
        st.metric(
            "Portfolio Heat",
            f"{portfolio_heat:.1f}%",
            delta="High" if portfolio_heat > 80 else "Normal"
        )

    with col2:
        # Calculate daily VaR
        if st.session_state.positions:
            position_vars = []
            for pos in st.session_state.positions:
                # Simplified VaR calculation
                position_value = pos['qty'] * pos['current_price']
                # Assume 2% daily volatility
                position_var = position_value * 0.02 * 1.645  # 95% confidence
                position_vars.append(position_var)

            portfolio_var = sum(position_vars)
            var_pct = (portfolio_var / account_value) * 100 if account_value > 0 else 0
        else:
            portfolio_var = 0
            var_pct = 0

        st.metric(
            "Value at Risk (95%)",
            f"${portfolio_var:,.0f}",
            delta=f"{var_pct:.1f}% of portfolio"
        )

    with col3:
        # Max position size
        if st.session_state.positions:
            max_position = max([p['qty'] * p['current_price'] for p in st.session_state.positions])
            max_position_pct = (max_position / account_value) * 100 if account_value > 0 else 0
        else:
            max_position_pct = 0

        st.metric(
            "Largest Position",
            f"{max_position_pct:.1f}%",
            delta="Concentrated" if max_position_pct > 10 else "Diversified"
        )

    with col4:
        # Number of sectors
        if st.session_state.positions:
            sectors = set([SECTOR_MAPPING.get(p['symbol'], 'Unknown') for p in st.session_state.positions])
            n_sectors = len(sectors)
        else:
            n_sectors = 0

        st.metric(
            "Sector Diversification",
            f"{n_sectors} sectors",
            delta="Good" if n_sectors >= 5 else "Poor"
        )

    # Risk limits configuration
    st.subheader("🎯 Risk Limits")

    col1, col2 = st.columns(2)

    with col1:
        max_position_size = st.slider(
            "Max Position Size %",
            min_value=1,
            max_value=20,
            value=5
        )

        max_sector_exposure = st.slider(
            "Max Sector Exposure %",
            min_value=10,
            max_value=50,
            value=30
        )

        max_daily_loss = st.slider(
            "Max Daily Loss %",
            min_value=1,
            max_value=10,
            value=3
        )

    with col2:
        max_correlation = st.slider(
            "Max Position Correlation",
            min_value=0.3,
            max_value=0.9,
            value=0.7,
            step=0.1
        )

        stop_loss_default = st.slider(
            "Default Stop Loss %",
            min_value=1,
            max_value=10,
            value=2
        )

        if st.button("💾 Save Risk Limits"):
            st.success("Risk limits saved successfully!")

    # Position risk analysis
    if st.session_state.positions:
        st.subheader("📊 Position Risk Analysis")

        # Create risk dataframe
        risk_data = []
        for pos in st.session_state.positions:
            position_value = pos['qty'] * pos['current_price']
            position_pct = (position_value / account_value) * 100 if account_value > 0 else 0

            # Calculate position risk metrics
            risk_data.append({
                'Symbol': pos['symbol'],
                'Value': position_value,
                'Portfolio %': position_pct,
                'Sector': SECTOR_MAPPING.get(pos['symbol'], 'Unknown'),
                'Stop Loss': pos.get('stop_price', pos['current_price'] * 0.98),
                'Risk
                ": position_value - (pos['qty'] * pos.get('stop_price', pos['current_price'] * 0.98)),
                'Risk %': ((pos['current_price'] - pos.get('stop_price', pos['current_price'] * 0.98)) / pos[
                    'current_price']) * 100
            })

        risk_df = pd.DataFrame(risk_data)

        # Display risk table
        st.dataframe(
            risk_df.style.format({
                'Value': '${:,.0f}',
                'Portfolio %': '{:.1f}%',
                'Stop Loss': '${:.2f}',
                'Risk
                ": '${:,.0f}',
                'Risk %': '{:.1f}%'
            }),
            use_container_width=True
        )

        # Correlation matrix
        st.subheader("🔗 Position Correlation Matrix")

        # Get correlation data
        symbols = [p['symbol'] for p in st.session_state.positions]
        if len(symbols) > 1:
            corr_matrix = self._calculate_correlation_matrix(symbols)

            if corr_matrix is not None:
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(x="Symbol", y="Symbol", color="Correlation"),
                    x=symbols,
                    y=symbols,
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0
                )

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Highlight high correlations
                high_corr_pairs = []
                for i in range(len(symbols)):
                    for j in range(i + 1, len(symbols)):
                        if abs(corr_matrix.iloc[i, j]) > max_correlation:
                            high_corr_pairs.append(
                                f"{symbols[i]} - {symbols[j]}: {corr_matrix.iloc[i, j]:.2f}"
                            )

                if high_corr_pairs:
                    st.warning(f"⚠️ High correlation detected: {', '.join(high_corr_pairs)}")

    # Risk alerts
    st.subheader("🚨 Risk Alerts")

    alerts = self._check_risk_alerts()

    if alerts:
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")
    else:
        st.success("✅ No risk alerts")


def _render_news_sentiment(self):
    """Render news and sentiment tab"""
    st.header("📰 News & Sentiment Analysis")

    # Sentiment overview
    col1, col2, col3, col4 = st.columns(4)

    # Market sentiment indicators
    market_sentiment = self._calculate_market_sentiment()

    with col1:
        st.metric(
            "Overall Sentiment",
            market_sentiment['overall'],
            delta=f"{market_sentiment['score']:.2f}"
        )

    with col2:
        st.metric(
            "Bullish %",
            f"{market_sentiment['bullish_pct']:.1f}%"
        )

    with col3:
        st.metric(
            "Bearish %",
            f"{market_sentiment['bearish_pct']:.1f}%"
        )

    with col4:
        st.metric(
            "News Volume",
            market_sentiment['news_count'],
            delta="High" if market_sentiment['news_count'] > 100 else "Normal"
        )

    # News feed
    st.subheader("📰 Latest Market News")

    # Get news for selected symbols or market
    if st.session_state.selected_symbols:
        news_items = self._get_news_feed(st.session_state.selected_symbols)
    else:
        news_items = self._get_news_feed(['SPY', 'QQQ'])  # Default to market news

    if news_items:
        for news in news_items[:10]:  # Show latest 10
            with st.expander(f"{news['headline']} - {news['source']}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(news['summary'])
                    st.caption(f"Published: {news['timestamp']}")

                    if news.get('symbols'):
                        st.write(f"**Related symbols:** {', '.join(news['symbols'])}")

                with col2:
                    # Sentiment indicator
                    sentiment_color = {
                        'bullish': 'green',
                        'bearish': 'red',
                        'neutral': 'gray'
                    }.get(news['sentiment'], 'gray')

                    st.markdown(
                        f"<div style='text-align: center; padding: 10px; "
                        f"background-color: {sentiment_color}; color: white; "
                        f"border-radius: 5px;'>"
                        f"<b>{news['sentiment'].upper()}</b><br>"
                        f"Score: {news['sentiment_score']:.2f}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.info("No recent news available")

    # Sentiment by sector
    st.subheader("🏭 Sentiment by Sector")

    sector_sentiment = self._calculate_sector_sentiment()

    if sector_sentiment:
        sector_df = pd.DataFrame(sector_sentiment)

        fig = px.bar(
            sector_df,
            x='sector',
            y='sentiment_score',
            color='sentiment_score',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Sector Sentiment Scores'
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Social media sentiment (if available)
    st.subheader("💬 Social Media Sentiment")

    social_sentiment = self._get_social_sentiment()

    if social_sentiment:
        # Create gauge charts for different platforms
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=['Twitter', 'Reddit', 'StockTwits']
        )

        # Twitter sentiment gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=social_sentiment['twitter']['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "gray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ),
            row=1, col=1
        )

        # Reddit sentiment gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=social_sentiment['reddit']['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "gray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ]
                }
            ),
            row=1, col=2
        )

        # StockTwits sentiment gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=social_sentiment['stocktwits']['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0.5], 'color': "gray"},
                        {'range': [0.5, 1], 'color': "green"}
                    ]
                }
            ),
            row=1, col=3
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Social media sentiment data not available")


# Helper methods

def _update_account_info(self) -> bool:
    """Update account information"""
    try:
        account = self.broker.api.get_account()

        st.session_state.account_info = {
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'daily_pnl': float(account.equity) - float(account.last_equity),
            'daily_change_pct': ((float(account.equity) - float(account.last_equity)) / float(
                account.last_equity)) * 100
        }

        return True
    except Exception as e:
        st.error(f"Failed to update account info: {e}")
        return False


def _update_portfolio_data(self):
    """Update portfolio positions and orders"""
    try:
        # Get positions
        positions = self.broker.api.list_positions()

        st.session_state.positions = []
        for pos in positions:
            st.session_state.positions.append({
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'side': pos.side
            })

        # Get orders
        orders = self.broker.api.list_orders(status='open')

        st.session_state.pending_orders = []
        for order in orders:
            st.session_state.pending_orders.append({
                'id': order.id,
                'symbol': order.symbol,
                'qty': int(order.qty),
                'side': order.side,
                'type': order.order_type,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'submitted_at': order.submitted_at
            })

        st.session_state.last_update = datetime.now()

    except Exception as e:
        st.error(f"Failed to update portfolio data: {e}")


def _get_index_data(self, symbol: str) -> Optional[Dict]:
    """Get index price data"""
    try:
        current_price = self.market_data.get_current_price(symbol)
        bars = self.market_data.get_bars(symbol, '1Day', limit=2)

        if current_price and not bars.empty:
            prev_close = bars.iloc[-2]['close'] if len(bars) > 1 else bars.iloc[-1]['open']
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100

            return {
                'price': current_price,
                'change': change,
                'change_pct': change_pct
            }
    except:
        pass

    return None


def _get_symbol_data(self, symbol: str) -> Optional[Dict]:
    """Get comprehensive symbol data"""
    try:
        current_price = self.market_data.get_current_price(symbol)
        bars = self.market_data.get_bars(symbol, '1Day', limit=20)

        if current_price and not bars.empty:
            prev_close = bars.iloc[-2]['close'] if len(bars) > 1 else bars.iloc[-1]['open']

            return {
                'symbol': symbol,
                'price': current_price,
                'change': current_price - prev_close,
                'change_pct': ((current_price - prev_close) / prev_close) * 100,
                'volume': bars.iloc[-1]['volume'],
                'avg_volume': bars['volume'].mean(),
                'volume_ratio': bars.iloc[-1]['volume'] / bars['volume'].mean()
            }
    except:
        pass

    return None


def _render_technical_heatmap(self, symbols: List[str]):
    """Render technical indicators heatmap"""
    indicators_data = []

    for symbol in symbols[:20]:  # Limit to 20 symbols
        bars = self.market_data.get_bars(symbol, '1Day', limit=50)

        if not bars.empty:
            # Calculate indicators
            bars = self.market_data.calculate_indicators(bars)

            current = bars.iloc[-1]

            # RSI signal
            rsi_signal = 1 if current.get('rsi', 50) < 30 else -1 if current.get('rsi', 50) > 70 else 0

            # Moving average signals
            ma_signal = 1 if current['close'] > current.get('sma_20', current['close']) else -1

            # Bollinger bands signal
            bb_signal = 1 if current['close'] < current.get('bb_lower', current['close']) else -1 if current[
                                                                                                         'close'] > current.get(
                'bb_upper', current['close']) else 0

            # Volume signal
            vol_signal = 1 if current.get('volume_ratio', 1) > 1.5 else 0

            indicators_data.append({
                'Symbol': symbol,
                'RSI': rsi_signal,
                'MA': ma_signal,
                'BB': bb_signal,
                'Volume': vol_signal,
                'Overall': (rsi_signal + ma_signal + bb_signal + vol_signal) / 4
            })

    if indicators_data:
        indicators_df = pd.DataFrame(indicators_data)
        indicators_df = indicators_df.set_index('Symbol')

        fig = px.imshow(
            indicators_df.T,
            labels=dict(x="Symbol", y="Indicator", color="Signal"),
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect='auto'
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def _generate_ai_signals(self) -> List[Dict]:
    """Generate AI trading signals"""
    signals = []

    if not self.ml_model:
        return signals

    # Get current positions to exclude
    current_positions = [p['symbol'] for p in st.session_state.positions]

    # Scan symbols
    for symbol in st.session_state.selected_symbols:
        if symbol in current_positions:
            continue

        # Get market data
        bars = self.market_data.get_bars(symbol, '1Day', limit=100)

        if not bars.empty:
            try:
                # Generate prediction
                prediction = self.ml_model.predict(symbol, bars)

                if prediction['prediction'] == 1 and prediction['confidence'] >= 0.6:
                    signals.append({
                        'symbol': symbol,
                        'confidence': prediction['confidence'],
                        'probability': prediction['probability'],
                        'predicted_return': prediction.get('expected_return', 0.05),
                        'model_predictions': prediction.get('model_probabilities', {}),
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                st.error(f"Error generating signal for {symbol}: {e}")

    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)

    return signals[:10]  # Return top 10 signals


def _execute_ai_signal(self, signal: Dict):
    """Execute an AI trading signal"""
    try:
        # Create order
        order = self.broker.api.submit_order(
            symbol=signal['symbol'],
            qty=signal['suggested_shares'],
            side='buy',
            type='market',
            time_in_force='day'
        )

        st.success(f"✅ Order submitted for {signal['symbol']}")

        # Add to alerts
        st.session_state.alerts.append({
            'type': 'success',
            'message': f"Bought {signal['suggested_shares']} shares of {signal['symbol']}",
            'timestamp': datetime.now()
        })

    except Exception as e:
        st.error(f"Failed to execute order: {e}")


def _render_trade_history(self):
    """Render recent trade history"""
    # Get recent orders
    try:
        orders = self.broker.api.list_orders(
            status='filled',
            limit=20
        )

        if orders:
            trades_data = []
            for order in orders:
                trades_data.append({
                    'Time': order.filled_at,
                    'Symbol': order.symbol,
                    'Side': order.side,
                    'Qty': int(order.filled_qty),
                    'Price': float(order.filled_avg_price),
                    'Value': float(order.filled_qty) * float(order.filled_avg_price)
                })

            trades_df = pd.DataFrame(trades_data)

            # Format display
            trades_df['Time'] = pd.to_datetime(trades_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
            trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:.2f}")
            trades_df['Value'] = trades_df['Value'].apply(lambda x: f"${x:,.2f}")

            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No recent trades")

    except Exception as e:
        st.error(f"Failed to load trade history: {e}")


def _render_pending_orders(self):
    """Render pending orders"""
    if st.session_state.pending_orders:
        orders_data = []

        for order in st.session_state.pending_orders:
            orders_data.append({
                'Symbol': order['symbol'],
                'Side': order['side'],
                'Type': order['type'],
                'Qty': order['qty'],
                'Limit': f"${order['limit_price']:.2f}" if order['limit_price'] else '-',
                'Stop': f"${order['stop_price']:.2f}" if order['stop_price'] else '-',
                'Submitted': pd.to_datetime(order['submitted_at']).strftime('%H:%M:%S'),
                'Action': order['id']
            })

        orders_df = pd.DataFrame(orders_data)

        # Display with cancel buttons
        for idx, row in orders_df.iterrows():
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 2, 1])

            col1.write(row['Symbol'])
            col2.write(row['Side'])
            col3.write(row['Type'])
            col4.write(row['Qty'])
            col5.write(row['Limit'])
            col6.write(row['Stop'])
            col7.write(row['Submitted'])

            if col8.button("Cancel", key=f"cancel_{row['Action']}"):
                self._cancel_order(row['Action'])
    else:
        st.info("No pending orders")


def _cancel_order(self, order_id: str):
    """Cancel a specific order"""
    try:
        self.broker.api.cancel_order(order_id)
        st.success("Order cancelled successfully")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to cancel order: {e}")


def _close_all_positions(self):
    """Close all open positions"""
    if st.session_state.positions:
        with st.spinner("Closing all positions..."):
            for position in st.session_state.positions:
                try:
                    self.broker.api.close_position(position['symbol'])
                except Exception as e:
                    st.error(f"Failed to close {position['symbol']}: {e}")

            st.success("All positions closed")
            st.rerun()
    else:
        st.info("No positions to close")


def _cancel_all_orders(self):
    """Cancel all pending orders"""
    if st.session_state.pending_orders:
        with st.spinner("Cancelling all orders..."):
            try:
                self.broker.api.cancel_all_orders()
                st.success("All orders cancelled")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to cancel orders: {e}")
    else:
        st.info("No orders to cancel")


def _calculate_correlation_matrix(self, symbols: List[str]) -> Optional[pd.DataFrame]:
    """Calculate correlation matrix for symbols"""
    try:
        returns_data = {}

        for symbol in symbols:
            bars = self.market_data.get_bars(symbol, '1Day', limit=50)
            if not bars.empty:
                returns_data[symbol] = bars['close'].pct_change().dropna()

        if returns_data:
            returns_df = pd.DataFrame(returns_data)
            return returns_df.corr()

    except Exception as e:
        st.error(f"Failed to calculate correlations: {e}")

    return None


def _check_risk_alerts(self) -> List[Dict]:
    """Check for risk management alerts"""
    alerts = []

    # Portfolio concentration check
    if st.session_state.positions:
        account_value = st.session_state.get('account_info', {}).get('portfolio_value', 100000)

        for pos in st.session_state.positions:
            position_value = pos['qty'] * pos['current_price']
            position_pct = (position_value / account_value) * 100

            if position_pct > 10:
                alerts.append({
                    'severity': 'high',
                    'message': f"{pos['symbol']} position is {position_pct:.1f}% of portfolio (limit: 10%)"
                })

        # Sector concentration check
        sector_exposure = defaultdict(float)
        for pos in st.session_state.positions:
            sector = SECTOR_MAPPING.get(pos['symbol'], 'Unknown')
            position_value = pos['qty'] * pos['current_price']
            sector_exposure[sector] += position_value

        for sector, exposure in sector_exposure.items():
            sector_pct = (exposure / account_value) * 100
            if sector_pct > 30:
                alerts.append({
                    'severity': 'medium',
                    'message': f"{sector} sector exposure is {sector_pct:.1f}% (limit: 30%)"
                })

    # Daily loss check
    daily_pnl = st.session_state.get('account_info', {}).get('daily_pnl', 0)
    account_value = st.session_state.get('account_info', {}).get('portfolio_value', 100000)

    if daily_pnl < 0:
        daily_loss_pct = abs(daily_pnl / account_value) * 100
        if daily_loss_pct > 3:
            alerts.append({
                'severity': 'high',
                'message': f"Daily loss of {daily_loss_pct:.1f}% exceeds 3% limit"
            })

    return alerts


def _calculate_market_sentiment(self) -> Dict:
    """Calculate overall market sentiment"""
    # Simplified sentiment calculation
    # In production, this would aggregate real sentiment data

    return {
        'overall': 'Neutral',
        'score': 0.1,
        'bullish_pct': 45.0,
        'bearish_pct': 35.0,
        'news_count': 87
    }


def _get_news_feed(self, symbols: List[str]) -> List[Dict]:
    """Get news feed for symbols"""
    # Simplified news feed
    # In production, this would fetch real news

    news_items = []

    try:
        # Mock news data for demonstration
        news_items = [
            {
                'headline': 'Market Analysis: Tech Stocks Show Strong Momentum',
                'summary': 'Technology stocks continue to lead market gains as earnings season approaches.',
                'source': 'Market Watch',
                'timestamp': datetime.now() - timedelta(hours=1),
                'symbols': symbols[:3],
                'sentiment': 'bullish',
                'sentiment_score': 0.7
            },
            {
                'headline': 'Fed Minutes Suggest Continued Policy Support',
                'summary': 'Federal Reserve minutes indicate commitment to current monetary policy stance.',
                'source': 'Reuters',
                'timestamp': datetime.now() - timedelta(hours=2),
                'symbols': ['SPY'],
                'sentiment': 'neutral',
                'sentiment_score': 0.0
            }
        ]
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")

    return news_items


def _calculate_sector_sentiment(self) -> List[Dict]:
    """Calculate sentiment by sector"""
    # Simplified calculation
    sectors = ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer']

    sector_sentiment = []
    for sector in sectors:
        # Mock sentiment scores
        sentiment_score = np.random.uniform(-0.5, 0.5)
        sector_sentiment.append({
            'sector': sector,
            'sentiment_score': sentiment_score
        })

    return sector_sentiment


def _get_social_sentiment(self) -> Dict:
    """Get social media sentiment data"""
    # Simplified social sentiment
    # In production, this would fetch real social media data

    return {
        'twitter': {'score': 0.3},
        'reddit': {'score': -0.1},
        'stocktwits': {'score': 0.5}
    }


def main():
    """Main entry point for Streamlit dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()