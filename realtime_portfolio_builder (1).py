
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go

# Define tickers and labels
tickers = {
    'U.S. Bonds': 'BND',
    'Global Bonds': 'BNDX',
    'U.S. Stocks': 'VTI',
    'Global Stocks': 'VT',
    'Emerging Market Stocks': 'VWO'
}

# Fetch historical data
data = {}
for label, ticker in tickers.items():
    df = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    data[label] = df['Adj Close'].pct_change().dropna()

# Create return and risk arrays
returns = np.array([data[label].mean() * 252 for label in tickers])
std_devs = np.array([data[label].std() * np.sqrt(252) for label in tickers])

# Create correlation matrix
returns_df = np.array([data[label].values for label in tickers])
correlation_matrix = np.corrcoef(returns_df)
cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix

# Streamlit UI
st.set_page_config(page_title="Real-Time Portfolio Builder", layout="wide")
st.title("📈 Real-Time Portfolio Builder with Yahoo Finance")
st.markdown("Adjust asset weights to build your portfolio and visualize its performance using real market data.")

# Sidebar for weights
st.sidebar.header("Asset Allocation")
weights = []
total_weight = 0
for label in tickers:
    w = st.sidebar.slider(f"{label}", 0.0, 1.0, 0.2, 0.01)
    weights.append(w)
    total_weight += w
weights = np.array(weights)

if total_weight != 1.0:
    st.sidebar.warning(f"Total weights must sum to 1.0. Current total: {total_weight:.2f}")

# Portfolio metrics
portfolio_return = np.dot(weights, returns)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_std_dev = np.sqrt(portfolio_variance)
risk_free_rate = 0.02
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev
individual_risk = np.dot(weights, std_devs)
diversification_benefit = individual_risk - portfolio_std_dev

# Display metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Return", f"{portfolio_return:.2%}")
col2.metric("Standard Deviation", f"{portfolio_std_dev:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col4.metric("Diversification Benefit", f"{diversification_benefit:.2%}")

# Efficient frontier
frontier_returns = []
frontier_risks = []
for w1 in np.linspace(0, 1, 30):
    for w2 in np.linspace(0, 1 - w1, 30):
        w3 = 1 - w1 - w2
        weights_frontier = np.array([w1, w2, w3, 0, 0])
        ret = np.dot(weights_frontier, returns)
        var = np.dot(weights_frontier.T, np.dot(cov_matrix, weights_frontier))
        frontier_returns.append(ret)
        frontier_risks.append(np.sqrt(var))

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=frontier_risks, y=frontier_returns, mode='markers', name='Efficient Frontier'))
fig.add_trace(go.Scatter(x=[portfolio_std_dev], y=[portfolio_return], mode='markers', name='Your Portfolio', marker=dict(color='red', size=10)))
fig.update_layout(title='Portfolio Risk-Return Plot with Real-Time Data', xaxis_title='Risk (Standard Deviation)', yaxis_title='Expected Return')
st.plotly_chart(fig, use_container_width=True)

# Challenge mode
st.subheader("🎯 Target Return Challenge")
target_return = st.slider("Set a target return (%)", 2.0, 12.0, 7.0, 0.1) / 100
if portfolio_return >= target_return:
    st.success("✅ Your portfolio meets the target return!")
else:
    st.error("❌ Your portfolio does not meet the target return.")
