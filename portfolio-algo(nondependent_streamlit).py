import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import CLA, plotting
import matplotlib as mpl
mpl.use('Agg')

st.write("""
# Optimal Portfolio Creator
This app creates Portfolios based on Risk Aversion and Budget
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    risk_aversion = st.sidebar.slider('Risk Aversion', 0.0, 0.99, 0.5)
    budget = st.sidebar.slider('Budget (USD)', 100, 1000000, 50000)
    data = {'Risk Aversion': risk_aversion,
            'Budget': budget}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Introduction')
st.write('''In order to address financial literacy,
Optimal Portfolio Creator allows users to create custom asset portfolios from the world\'s most popular assets
ranging from Gold to Stocks to Crytocurrency. Users are able to enter their risk aversion and their portfolio budget
to create the an asset portfolio that provides the highest expected return for their unique situation.
''')
st.subheader('User Input parameters')
st.write(df)

def calc_portfolio(budget, risk_aversion):
    tickerSymbols = ['GOOGL', 'AAPL', 'MSFT', 'SAR', 'UNH',
                     'WMT', 'NSRGY', 'GOLD', 'BTC-USD', 'ETH-USD']
    stock_prices_df = pd.DataFrame()
    for tickerSymbol in tickerSymbols:
        ticker = yf.Ticker(tickerSymbol)
        tickerDf = ticker.history(period='1d', start='2010-1-1', end='2022-2-26')
        tickerDf = tickerDf.drop(
            columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'])
        if stock_prices_df.index.empty:
            stock_prices_df = pd.DataFrame(index=tickerDf.index)
        # Adding daily closing prices
        stock_prices_df[tickerSymbol] = tickerDf['Close']
    # display(stock_prices_df)
    # stock_prices_df.to_csv('result.csv', index = True, header=True) #Download new csv

    mu = expected_returns.mean_historical_return(
        stock_prices_df)  # expected returns
    S = risk_models.sample_cov(stock_prices_df)  # sample covariances

    ef = EfficientFrontier(mu, S)
    # ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper) #upper lower weights of certain industries/asset types
    volatility = 0.99 - (0.86 * risk_aversion)
    ef.efficient_risk(target_volatility=volatility)  # risk aversion (.13-.99)
    weights = ef.clean_weights()

    print('Risk efficient portfolio:')  # Risk efficient portfolio
    risk_ret_tangent, risk_std_tangent, _ = ef.portfolio_performance(verbose=True)

    cla = CLA(mu, S)
    cla.max_sharpe()
    print('Optimal Portfolio:')  # Optimal Portfolio
    cla.portfolio_performance(verbose=True)

    ax = plotting.plot_efficient_frontier(cla, showfig=False)
    ax.scatter(risk_std_tangent, risk_ret_tangent, marker="*",
               s=100, c="r", label="Risk efficient")
    #plt.savefig('graph.png')  # save image locally
    st.pyplot(plt)

    latest_prices = get_latest_prices(stock_prices_df)

    da = DiscreteAllocation(weights, latest_prices,
                            total_portfolio_value=budget)  # change portfolio value
    allocation, leftover = da.greedy_portfolio()
    portfolio = "Discrete allocation of risk efficient portfolio: " + str(allocation)
    remainer = "Funds remaining: ${:.2f}".format(leftover)
    data = {'Portfolio': portfolio,
            'Remainer': remainer}
    features = pd.DataFrame(data, index=[0])
    return features

output_df = calc_portfolio(df['Budget'][0], df['Risk Aversion'][0])
st.write('''**Key**\n
Optimal - Max Sharpe, or maximum return for every additional unit of risk\n
Star - Optimal Risk Efficient Portfolio
''')
st.subheader('Optimal Risk efficient portfolio')
st.write(output_df['Portfolio'][0])
st.write(output_df['Remainer'][0])
st.write('Available Assets: Gold, Apple, Alphabet, Microsoft, Saudi Aramco, United Health, Walmart, Nestle, Ethereum, and Bitcoin.')
