import pandas as pd
import numpy as np
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# suppress warning
import warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy

df = pd.read_csv('result.csv')
df = df.iloc[:,:9]
df['TOTAL'] = df.GOOGL + df.AAPL + df.MSFT + df.SAR + df.UNH + df.WMT + df.NSRGY + df.GOLD
df['year'] = pd.to_datetime(df['Date']).dt.year
category = 'TOTAL'
#plt.plot(df.groupby('year').sum()[category])

series=df[category]
result = adfuller(series)
result = adfuller(series.diff().dropna())


get_ipython().system('pip install pmdarima')
import pmdarima as pm
modl = pm.auto_arima(df.iloc[:,9], start_p=1, start_q=1, start_P=1, start_Q=1,
                     max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                     stepwise=True, suppress_warnings=True, D=10, max_D=10,
                     error_action='ignore')

# best parameters are (5,2,0)

P=0
D=2
Q=5

# fit model
N=2800
model = ARIMA(series[:N], order=(Q,D,P))
model_fit = model.fit(disp = 0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
# plt.figure(figsize = (6,4))
plt.plot(residuals)
plt.title('Residual at each data point')
plot_acf(residuals)
plt.title('Residual autocorrelation')
plt.show()
residuals.plot(kind='kde', legend=False)
plt.title('Residual kernel density estimation')
plt.show()
print(residuals.describe())
k2, p = scipy.stats.normaltest(residuals)
alpha = 0.1
#print('p value is ',p[0])

#print('null hypothesis: residuals comes from a normal distribution')
#if p < alpha:  
#    print("The null hypothesis can be rejected")
#else:
#    print("The null hypothesis cannot be rejected")



### a function for grading
def outputResults(adfuller_p_value, fitted_arima_model, residual_normaltest_p_value):
    import csv
    '''
    please pass your argument in this function
    adfuller_p_value: the p value of the stationary series from adfuller test;
    fitted_arima_model: your fitted ARIMA model;
    residual_normaltest_p_value: the normal test's p value of residual from ARIMA model
    '''
    if type(adfuller_p_value) == np.ndarray:
        adfuller_p_value = adfuller_p_value[0]
    if type(residual_normaltest_p_value) == np.ndarray:
        residual_normaltest_p_value = residual_normaltest_p_value[0]
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['adfuller','AIC','normaltest'])
        writer.writerow([adfuller_p_value, fitted_arima_model.aic, residual_normaltest_p_value])

        
forecast = model_fit.forecast(steps=len(series)-N, alpha=0.05)[0]  # 95% conf
# Forecast
fc, se, conf = model_fit.forecast(steps=len(series)-N, alpha=0.05)  # 95% conf

fc_series = pd.Series(fc, index=range(N,len(series)))
lower_series = pd.Series(conf[:, 0], index=range(N,len(series)))
upper_series = pd.Series(conf[:, 1], index=range(N,len(series)))

plt.rcParams.update({'figure.figsize':(10,5)})
# plt.plot(series)

# plt.plot(series)
plt.plot(df.index[:N+1],series[:N+1],label='train_label')
plt.plot(df.index[N:],series[N:],color='green',label='test_label')
#plt.plot(df.iloc[:N+1].index,model_fit.predict(start=2,end=N+1,dynamic=False,typ='levels'),color='orange',label='in sample predict')
plt.plot(df.iloc[N:].index,fc_series.tolist(), label='forecast', color='red')
#plt.fill_between(df.iloc[N:].index, lower_series, upper_series, color='k', alpha=.15)
plt.legend(loc='upper left')