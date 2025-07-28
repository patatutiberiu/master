import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Stock Prediction Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===================== Utility Functions =====================
TIMEFRAME_OPTIONS = {
    '5 minutes':  ('1d',  '5m'),
    '15 minutes': ('5d',  '15m'),
    '1 hour':     ('1mo', '60m'),
    '1 day':      ('6mo', '1d'),
    '1 week':     ('1y',  '1wk'),
    '1 month':    ('2y',  '1mo'),
    '6 months':   ('6mo', '1d')
}

@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol: str, period: str, interval: str):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data.empty:
        st.warning("No data returned from Yahoo Finance – showing simulated data instead.")
        return None
    data.reset_index(inplace=True)
    data.rename(columns={'Volume': 'Volume'}, inplace=True)
    return data

# ===================== Prediction Engine =====================
class SimplePredictor:
    """Very lightweight heuristic predictor for demo purposes"""
    def predict(self, df: pd.DataFrame):
        if len(df) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 50,
                'expected_change': 0
            }
        close = df['Close']
        sma_short = close.rolling(10).mean().iloc[-1]
        sma_long = close.rolling(30).mean().iloc[-1]
        rsi = self._rsi(close).iloc[-1]

        # Check for NaN values
        if pd.isna(sma_short) or pd.isna(sma_long) or pd.isna(rsi):
            return {
                'signal': 'HOLD',
                'confidence': 50,
                'expected_change': 0
            }

        score = 0
        if sma_short > sma_long:
            score += 0.4
        else:
            score -= 0.4
        if rsi < 30:
            score += 0.3
        elif rsi > 70:
            score -= 0.3
        daily_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        score += daily_change * 0.3

        if score > 0.3:
            signal = 'BUY'
        elif score < -0.3:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        confidence = min(95, 50 + abs(score) * 100)
        expected_change = score * 100
        return {
            'signal': signal,
            'confidence': confidence,
            'expected_change': expected_change
        }

    def _rsi(self, series: pd.Series, period: int = 14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

predictor = SimplePredictor()

# ===================== Portfolio Simulator =====================
class PortfolioSimulator:
    def __init__(self, symbols, cash=100_000):
        self.symbols = symbols
        self.initial_cash = cash
        self.cash = cash
        self.positions = {s: 0 for s in symbols}
        self.history = []

    def run(self, period='6mo'):
        for symbol in self.symbols:
            data = yf.download(symbol, period=period, interval='1d', progress=False)
            if data.empty:
                continue
            data.reset_index(inplace=True)
            for idx in range(30, len(data)):
                window = data.iloc[:idx]
                pred = predictor.predict(window)
                price = data['Close'].iloc[idx]
                # Simple rule-based trading
                if pred['signal'] == 'BUY' and self.cash > price:
                    # buy one share
                    self.cash -= price
                    self.positions[symbol] += 1
                    self.history.append((data['Date'].iloc[idx], symbol, 'BUY', price))
                elif pred['signal'] == 'SELL' and self.positions[symbol] > 0:
                    # sell all shares
                    self.cash += price * self.positions[symbol]
                    self.history.append((data['Date'].iloc[idx], symbol, 'SELL', price, self.positions[symbol]))
                    self.positions[symbol] = 0
        # Final portfolio value
        final_value = self.cash
        for symbol, qty in self.positions.items():
            if qty > 0:
                last_price = yf.download(symbol, period='1d', interval='1d', progress=False)['Close'][-1]
                final_value += last_price * qty
        return final_value

# ===================== Streamlit UI =====================

def main():
    st.title("📈 AI Stock Prediction Pro")

    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
    with col2:
        timeframe_label = st.selectbox(
            "Timeframe:", list(TIMEFRAME_OPTIONS.keys()), index=3)

    period, interval = TIMEFRAME_OPTIONS[timeframe_label]

    # Fetch data
    with st.spinner("Fetching data..."):
        data = fetch_stock_data(symbol, period, interval)
        if data is None:
            st.stop()

    # Prediction
    pred = predictor.predict(data)
    overlay_color = "green" if pred['signal'] == 'BUY' else "red" if pred['signal'] == 'SELL' else "gray"

    # Overlay prediction
    st.markdown(f"""
    <h1 style='position:fixed; top:35%; left:50%; transform:translate(-50%, -50%); 
       font-size:12vw; color:{overlay_color}; opacity:0.1; z-index:0;'>
        {pred['signal']}
    </h1>
    """, unsafe_allow_html=True)

    # Candlestick chart – bigger & clearer
    fig = go.Figure(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Market Data'))
    fig.update_layout(height=700, margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Tabs for analysis & simulator
    tab1, tab2 = st.tabs(["📊 Analysis", "🕹️ Simulator"])

    with tab1:
        colA, colB, colC = st.columns(3)
        colA.metric("Signal", pred['signal'])
        colB.metric("Confidence", f"{pred['confidence']:.1f}%")
        colC.metric("Expected Move", f"{pred['expected_change']:+.2f}%")

        # RSI chart
        rsi = predictor._rsi(data['Close']).fillna(50)
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=data['Date'], y=rsi, name='RSI'))
        rsi_fig.add_hline(y=70, line_color='red', line_dash='dash')
        rsi_fig.add_hline(y=30, line_color='green', line_dash='dash')
        rsi_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(rsi_fig, use_container_width=True)

    with tab2:
        st.write("### Run Simple Portfolio Simulator (100k USD)")
        if st.button("Run Simulator"):
            with st.spinner("Simulating..."):
                sim = PortfolioSimulator(['TSLA', 'NVDA', 'GOOGL', 'MSFT', 'AMZN'])
                final_value = sim.run(period='6mo')
                profit = final_value - sim.initial_cash
                st.success(f"Final Portfolio Value: ${final_value:,.2f}  (Profit: ${profit:,.2f})")
                if profit > 0:
                    st.balloons()
                # Show simple trade log last 10
                if sim.history:
                    hist_df = pd.DataFrame(sim.history, columns=['Date', 'Symbol', 'Action', 'Price', 'Qty'])
                    st.dataframe(hist_df.tail(10))

if __name__ == '__main__':
    main()
