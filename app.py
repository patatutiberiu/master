
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the prediction overlay and styling
st.markdown("""
<style>
    .main-container {
        position: relative;
        min-height: 80vh;
    }

    .prediction-overlay {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 8vw;
        font-weight: bold;
        color: rgba(128, 128, 128, 0.15);
        z-index: 1;
        pointer-events: none;
        text-transform: uppercase;
        user-select: none;
    }

    .bullish-overlay {
        color: rgba(0, 255, 0, 0.15) !important;
    }

    .bearish-overlay {
        color: rgba(255, 0, 0, 0.15) !important;
    }

    .analysis-panel {
        position: relative;
        z-index: 2;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }

    .focus-mode {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(0, 0, 0, 0.95);
        z-index: 1000;
        padding: 20px;
        overflow-y: auto;
    }

    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }

    .prediction-confidence {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .high-confidence {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }

    .medium-confidence {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
    }

    .low-confidence {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictionSystem:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # Popular stocks for demo
        self.stock_list = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'TSLA': 'Tesla Inc.',
            'AMZN': 'Amazon.com Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc.',
            'NFLX': 'Netflix Inc.',
            'AMD': 'Advanced Micro Devices',
            'UBER': 'Uber Technologies'
        }

    def generate_stock_data(self, symbol, days=90):
        """Generate realistic stock data for demo"""
        np.random.seed(hash(symbol) % 1000)  # Consistent data per symbol

        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')

        # Base price varies by stock
        base_prices = {
            'AAPL': 180, 'GOOGL': 140, 'MSFT': 340, 'TSLA': 250,
            'AMZN': 140, 'NVDA': 450, 'META': 320, 'NFLX': 450,
            'AMD': 110, 'UBER': 45
        }

        base_price = base_prices.get(symbol, 100)
        prices = [base_price]

        for i in range(1, len(dates)):
            change = np.random.normal(0, base_price * 0.02)
            new_price = max(prices[-1] + change, base_price * 0.5)
            prices.append(new_price)

        stock_data = []
        for i, date in enumerate(dates):
            open_price = prices[i]
            close_price = prices[i] + np.random.normal(0, base_price * 0.01)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, base_price * 0.015))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, base_price * 0.015))
            volume = np.random.randint(1000000, 50000000)

            stock_data.append({
                'Date': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })

        return pd.DataFrame(stock_data)

    def generate_sentiment_data(self, symbol, days=30):
        """Generate social media sentiment data"""
        np.random.seed(hash(symbol + 'sentiment') % 1000)

        sources = ['News', 'Twitter', 'Reddit', 'TikTok', 'Instagram', 'Facebook']
        sentiment_data = []

        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')

        for date in dates:
            for source in sources:
                # Different sentiment patterns per source
                if source == 'News':
                    sentiment = np.random.normal(0.1, 0.3)
                elif source == 'Twitter':
                    sentiment = np.random.normal(0, 0.4)
                elif source == 'Reddit':
                    sentiment = np.random.normal(-0.05, 0.35)
                elif source == 'TikTok':
                    sentiment = np.random.normal(0.15, 0.4)
                elif source == 'Instagram':
                    sentiment = np.random.normal(0.2, 0.25)
                else:  # Facebook
                    sentiment = np.random.normal(0.05, 0.3)

                sentiment = max(-1, min(1, sentiment))
                mentions = np.random.poisson(100)
                engagement = mentions * np.random.randint(10, 100)

                sentiment_data.append({
                    'Date': date,
                    'Source': source,
                    'Sentiment': sentiment,
                    'Mentions': mentions,
                    'Engagement': engagement
                })

        return pd.DataFrame(sentiment_data)

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()

        # Moving averages
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        return df.fillna(0)

    def make_prediction(self, stock_data, sentiment_data):
        """Make stock prediction"""
        # Simple prediction logic for demo
        recent_price_change = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-5]) / stock_data['Close'].iloc[-5]
        avg_sentiment = sentiment_data['Sentiment'].mean()
        recent_volume = stock_data['Volume'].iloc[-5:].mean()

        # Combine factors
        prediction_score = (recent_price_change * 0.4) + (avg_sentiment * 0.6)

        # Determine signal
        if prediction_score > 0.02:
            signal = "STRONG BUY"
            confidence = min(95, 70 + abs(prediction_score) * 100)
        elif prediction_score > 0.005:
            signal = "BUY"
            confidence = min(85, 60 + abs(prediction_score) * 100)
        elif prediction_score < -0.02:
            signal = "STRONG SELL"
            confidence = min(95, 70 + abs(prediction_score) * 100)
        elif prediction_score < -0.005:
            signal = "SELL"
            confidence = min(85, 60 + abs(prediction_score) * 100)
        else:
            signal = "HOLD"
            confidence = 50 + np.random.randint(0, 20)

        expected_change = prediction_score * 100

        return {
            'signal': signal,
            'confidence': confidence,
            'expected_change': expected_change,
            'current_price': stock_data['Close'].iloc[-1],
            'avg_sentiment': avg_sentiment
        }

# Initialize the system
@st.cache_resource
def load_prediction_system():
    return StockPredictionSystem()

system = load_prediction_system()

# Main app
def main():
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("🤖 AI Stock Prediction System")

    with col2:
        focus_mode = st.button("🎯 Focus Mode", key="focus_btn")

    with col3:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)

    # Stock selection
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_stock = st.selectbox(
            "🔍 Select or Search Stock:",
            options=list(system.stock_list.keys()),
            format_func=lambda x: f"{x} - {system.stock_list[x]}",
            key="stock_select"
        )

    with col2:
        if st.button("📊 Analyze", type="primary"):
            st.rerun()

    # Generate data
    stock_data = system.generate_stock_data(selected_stock)
    sentiment_data = system.generate_sentiment_data(selected_stock)
    stock_data = system.calculate_technical_indicators(stock_data)

    # Make prediction
    prediction = system.make_prediction(stock_data, sentiment_data)

    # Prediction overlay
    overlay_class = "bullish-overlay" if "BUY" in prediction['signal'] else "bearish-overlay"
    st.markdown(f"""
    <div class="prediction-overlay {overlay_class}">
        {prediction['signal']}
    </div>
    """, unsafe_allow_html=True)

    # Main analysis panel
    st.markdown('<div class="analysis-panel">', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence_class = "high-confidence" if prediction['confidence'] > 75 else "medium-confidence" if prediction['confidence'] > 50 else "low-confidence"
        st.markdown(f"""
        <div class="prediction-confidence {confidence_class}">
            Confidence<br>{prediction['confidence']:.1f}%
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Current Price", f"${prediction['current_price']:.2f}", 
                 f"{prediction['expected_change']:+.2f}%")

    with col3:
        st.metric("Avg Sentiment", f"{prediction['avg_sentiment']:.2f}", 
                 "Positive" if prediction['avg_sentiment'] > 0 else "Negative")

    with col4:
        st.metric("Signal", prediction['signal'], 
                 "📈" if "BUY" in prediction['signal'] else "📉" if "SELL" in prediction['signal'] else "➡️")

    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "😊 Sentiment", "📊 Technical", "🚨 Alerts"])

    with tab1:
        # Candlestick chart
        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price"
        ))

        # Add moving averages
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['SMA_20'],
            mode='lines', name='SMA 20', line=dict(color='orange', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['SMA_50'],
            mode='lines', name='SMA 50', line=dict(color='red', width=1)
        ))

        fig.update_layout(
            title=f"{selected_stock} - Candlestick Chart with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Sentiment heatmap
        sentiment_pivot = sentiment_data.pivot_table(
            values='Sentiment', index='Source', 
            columns=sentiment_data['Date'].dt.day, aggfunc='mean'
        )

        fig = px.imshow(
            sentiment_pivot,
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="Social Media Sentiment Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Sentiment trends
        sentiment_trends = sentiment_data.groupby(['Date', 'Source'])['Sentiment'].mean().reset_index()
        fig = px.line(
            sentiment_trends, x='Date', y='Sentiment', color='Source',
            title="Sentiment Trends by Source"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Technical indicators
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
            vertical_spacing=0.1
        )

        # RSI
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['RSI'],
            mode='lines', name='RSI'
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

        # MACD
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['MACD'],
            mode='lines', name='MACD'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['MACD_Signal'],
            mode='lines', name='Signal'
        ), row=2, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['BB_Upper'],
            mode='lines', name='BB Upper', line=dict(color='red', dash='dash')
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['Close'],
            mode='lines', name='Close Price'
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=stock_data['Date'], y=stock_data['BB_Lower'],
            mode='lines', name='BB Lower', line=dict(color='green', dash='dash')
        ), row=3, col=1)

        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Alerts and recommendations
        st.subheader("🚨 Trading Alerts")

        alerts = []

        # Generate alerts based on conditions
        if prediction['confidence'] > 80:
            alerts.append({
                'type': 'HIGH_CONFIDENCE',
                'message': f"High confidence {prediction['signal']} signal detected",
                'urgency': 'HIGH'
            })

        if abs(prediction['expected_change']) > 2:
            alerts.append({
                'type': 'LARGE_MOVE',
                'message': f"Expected large price movement: {prediction['expected_change']:+.1f}%",
                'urgency': 'MEDIUM'
            })

        if prediction['avg_sentiment'] > 0.3:
            alerts.append({
                'type': 'POSITIVE_SENTIMENT',
                'message': "Strong positive sentiment across social media",
                'urgency': 'MEDIUM'
            })
        elif prediction['avg_sentiment'] < -0.3:
            alerts.append({
                'type': 'NEGATIVE_SENTIMENT',
                'message': "Strong negative sentiment across social media",
                'urgency': 'MEDIUM'
            })

        if not alerts:
            st.info("No alerts at this time. Market conditions appear normal.")
        else:
            for alert in alerts:
                if alert['urgency'] == 'HIGH':
                    st.error(f"🔴 {alert['message']}")
                elif alert['urgency'] == 'MEDIUM':
                    st.warning(f"🟡 {alert['message']}")
                else:
                    st.info(f"🔵 {alert['message']}")

        # Risk assessment
        st.subheader("⚖️ Risk Assessment")

        risk_score = abs(prediction['expected_change']) + (100 - prediction['confidence']) / 100

        if risk_score < 1:
            st.success("🟢 LOW RISK - Conservative position recommended")
        elif risk_score < 3:
            st.warning("🟡 MEDIUM RISK - Standard position sizing")
        else:
            st.error("🔴 HIGH RISK - Reduce position size or avoid")

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    with col2:
        st.caption(f"Analyzing: {selected_stock}")

    with col3:
        st.caption("🤖 AI-Powered Predictions")

    # Auto refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
