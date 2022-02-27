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
ef.efficient_risk(target_volatility=0.20)  # risk aversion (.13-.99)
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
plt.savefig('graph.png')  # save image locally

latest_prices = get_latest_prices(stock_prices_df)

da = DiscreteAllocation(weights, latest_prices,
                        total_portfolio_value=100000)  # change portfolio value
allocation, leftover = da.greedy_portfolio()
print("Discrete allocation of risk efficient portfolio:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
