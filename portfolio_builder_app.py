
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Asset data
assets = ['U.S. Bonds', 'Global Bonds', 'U.S. Stocks', 'Global Stocks', 'Emerging Market Stocks']
expected_returns = np.array([0.0211, 0.0412, 0.0808, 0.0983, 0.1105])
std_devs = np.array([0.0351, 0.0847, 0.1514, 0.1761, 0.2300])
correlation_matrix = np.array([
    [1.00, 0.52, -0.05, -0.03, -0.05],
    [0.52, 1.00, 0.17, 0.40, 0.29],
    [-0.05, 0.17, 1.00, 0.85, 0.75],
    [-0.03, 0.40, 0.85, 1.00, 0.87],
    [-0.05, 0.29, 0.75, 0.87, 1.00]
])

cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix
risk_free_rate = 0.02

st.set_page_config(page_title="Portfolio Builder", layout="wide")
st.title("📊 Interactive Portfolio Builder")
st.markdown("Adjust asset weights to build your portfolio and visualize its performance.")

# Sidebar for weights
st.sidebar.header("Asset Allocation")
weights = []
total_weight = 0
for asset in assets:
    w = st.sidebar.slider(f"{asset}", 0.0, 1.0, 0.2, 0.01)
    weights.append(w)
    total_weight += w
weights = np.array(weights)

if total_weight != 1.0:
    st.sidebar.warning(f"Total weights must sum to 1.0. Current total: {total_weight:.2f}")

# Portfolio metrics
portfolio_return = np.dot(weights, expected_returns)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
portfolio_std_dev = np.sqrt(portfolio_variance)
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
        ret = np.dot(weights_frontier, expected_returns)
        var = np.dot(weights_frontier.T, np.dot(cov_matrix, weights_frontier))
        frontier_returns.append(ret)
        frontier_risks.append(np.sqrt(var))

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=frontier_risks, y=frontier_returns, mode='markers', name='Efficient Frontier'))
fig.add_trace(go.Scatter(x=[portfolio_std_dev], y=[portfolio_return], mode='markers', name='Your Portfolio', marker=dict(color='red', size=10)))
fig.update_layout(title='Portfolio Risk-Return Plot', xaxis_title='Risk (Standard Deviation)', yaxis_title='Expected Return')
st.plotly_chart(fig, use_container_width=True)

# Challenge mode
st.subheader("🎯 Target Return Challenge")
target_return = st.slider("Set a target return (%)", 2.0, 12.0, 7.0, 0.1) / 100
if portfolio_return >= target_return:
    st.success("✅ Your portfolio meets the target return!")
else:
    st.error("❌ Your portfolio does not meet the target return.")
