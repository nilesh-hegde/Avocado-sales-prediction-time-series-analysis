import pandas as pd
import numpy as np
import itertools
import datetime as dt

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Facebook Prophet
from fbprophet import Prophet

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('avocado-updated-2020.csv')
data.head()


data.info()


data['date'] = pd.to_datetime(data['date'])
data['type'].unique()


data['geography'].unique()


# Select US demand for organic avocados
my_data = data[(data['geography'] == 'Total U.S.') &
               (data['type'] == 'organic')][['date', 'total_volume']]
my_data


my_data['date_diff'] = my_data['date'].diff()
my_data['date_diff'].value_counts()


# December 2018 is missing
# Drop off 2019 and 2020 observations
my_data = my_data[(my_data['date'].dt.year != 2019) &
                  (my_data['date'].dt.year != 2020)].set_index('date')
my_data.drop('date_diff', axis=1, inplace=True)


# Since we have one date difference between Jan 1st 2018 and the last date of 2017
# Examine last date of 2017 and first date of 2018
print(my_data[my_data.index.year == 2017].tail(1))
print(my_data[my_data.index.year == 2018].head(1))


# This value is duplicated
# There may be a discrepancy among first days of a week  
my_data['weekday'] = my_data.index.strftime('%a')
my_data['weekday'].value_counts()


# Drop the first date of 2018, weekday
# Rename our target
my_data.drop(pd.Timestamp('2018-01-01'), inplace=True)
my_data.drop('weekday', axis=1, inplace=True)
my_data.rename(columns={'date':'ds', 'total_volume':'y'}, inplace=True)
my_data


my_data.index.nunique()


# Number of observations each year
my_data.index.year.value_counts()


# Split data
train = my_data[:181]
test = my_data[181:]
print(train.shape)
print(test.shape)


# Plot the data (train series)
train.plot(figsize=(12,5), legend=None)
plt.title('Organic Avocado Weekly Sales', fontsize=16)
plt.xlabel(None);


# Plot data by monthly basis
# Use line chart and box plot
train['Month'] = train.index.strftime('%b')
train['Year'] = train.index.year

fig, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
sns.lineplot(data=train,
             hue='Year',
             x='Month',
             y='y',
             palette='Set2',
             ax=ax[0])
ax[0].set_title('Organic Avocado Sales By Year', fontsize=16)
ax[0].set_ylabel(None)

sns.boxplot(data=train,
            x='Month',
            y='y',
            color='lightskyblue',
            ax=ax[1])
ax[1].set_title('Organic Avocado Sales Distribution by Month', fontsize=16)
ax[1].set_ylabel(None)

sns.despine()
train.drop(['Year', 'Month'], axis=1, inplace=True)


# Another way to decompose the series characteristics
decomposition = seasonal_decompose(train['y'])
fig = decomposition.plot()
fig.set_size_inches(10,8)


# Try multiple combination to find a model that has the lowest RSME
# Try different number of weeks for annual seasonality

trend = ['additive', 'multiplicative']
seasonality = ['additive', 'multiplicative']
periods = range(52, 56)

lowest_rmse = None
lowest_rmse_model = None

for model in list(itertools.product(trend, seasonality, periods)):
    # Modeling
    fcast_model = ExponentialSmoothing(train['y'],
                                       trend=model[0],
                                       seasonal=model[1],
                                       seasonal_periods=model[2]).fit()
    y_fcast = fcast_model.forecast(len(test)).rename('y_fcast')
    
    # RSME
    rmse = np.sqrt(np.mean((test['y'] - y_fcast)**2))
    
    # Store results
    current_rmse = rmse
        
    # Set baseline for rmse
    if lowest_rmse == None:
        lowest_rmse = rmse
        
    # Compare results
    if current_rmse <= lowest_rmse:
        lowest_rmse = current_rmse
        lowest_rmse_model = model      
    print('{} trend, {} seasonality, {} week frequency - RSME: {}'.format(model[0], model[1], model[2], rmse))
    
print('--------------------------------------------------------------------------------------')
print('Model that has the lowest RSME:')
print('{} trend, {} seasonality, {} week frequency - RSME: {}'.format(lowest_rmse_model[0], lowest_rmse_model[1],
                                                                      lowest_rmse_model[2], lowest_rmse))
                                                                      
                                                                      
    def error_metrics(y_fcast, y_test):
    """
    Return mean absolute percentage error (MAPE)
           mean percentage error (MPE)
           mean absolute error (MAE)
           root mean square error (RMSE)
           
    """
    print(f'MAPE: {np.mean(np.abs((y_test - y_fcast)/y_test))*100}')
    print(f'MPE:  {np.mean((y_test - y_fcast)/y_test)*100}')
    print(f'MAE:  {np.mean(np.abs(y_test - y_fcast))*100}')
    print(f'RMSE: {np.sqrt(np.mean((y_test - y_fcast)**2))}')
    

def exp_smoothing(y_train,
                  y_test,
                  trend=None,
                  seasonal=None,
                  period=None,
                  freq=None,
                  plot=False,
                  figsize=None):
    """
    Forecast using Holt-Winters exponential smoothing.
    Return a graph and error metrics.
    """
    # Modeling
    fcast_model = ExponentialSmoothing(y_train,
                                       trend=trend,
                                       seasonal=seasonal,
                                       seasonal_periods=period).fit()
    y_est = pd.DataFrame(fcast_model.fittedvalues).rename(columns={0:'y_fitted'}) # In-sample fit
    y_fcast = fcast_model.forecast(len(y_test)).rename('y_fcast') # Out-of-sample fit
    
    # Plot Series
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_title('Observed, Fitted, and Forecasted Series\nTriple Exponential Smoothing',
                     fontsize=16)
        ax.set_ylabel('Organic Avocado Weekly Sales')
        ax.plot(y_train,
                label='In-sample data',
                linestyle='-')
        ax.plot(y_test,
                label='Held-out data',
                linestyle='-')
        ax.plot(y_est,
                label='Fitted values',
                linestyle='--',
                color='g')
        ax.plot(y_fcast,
                label='Forecasts',
                linestyle='--',
                color='k')
        ax.legend(loc='best')
        plt.xticks(rotation = 45)
        plt.show(block = False)
        plt.close()
    
    # Print error metrics
    print('-----------------------------')
    if seasonal != None:
        print('{} trend, {} seasonality, {} {} frequency'.format(trend, seasonal, period, freq))
    error_metrics(y_fcast=y_fcast, y_test=y_test)
    print(f'AIC:  {fcast_model.aic}')
    print(f'BIC:  {fcast_model.bic}')
    
    
    exp_smoothing(train['y'],
              test['y'],
              trend='additive',
              seasonal='multiplicative',
              period=53,
              freq='week',
              plot=True,
              figsize=(12,5))
              
              
    exp_smoothing(train['y'],
              test['y'],
              trend='additive',
              seasonal='multiplicative',
              period=54,
              freq='week',
              plot=True,
              figsize=(12,5))
              
def test_stationarity(y, title, window , figsize=(12,5)):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    # Determing rolling statistics
    rolmean = y.rolling(window=window, center=False).mean()
    rolstd = y.rolling(window=window, center=False).std()
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=figsize)
    orig = plt.plot(y,
                    label='Original')
    mean = plt.plot(rolmean,
                    label='Rolling Mean',
                    color='r')
    std = plt.plot(rolstd,
                   label='Rolling Std',
                   color='orange')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for ' + title, fontsize=16)
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()

    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic',
                                'p-value',
                                '# Lags Used',
                                'Number of Observations Used'])
    for k, v in dftest[4].items():
        dfoutput['Critical Value (%s)'%k]=v
    print(dfoutput)
    
    
def plot_general(y,
                 title='title',
                 lags=None,
                 figsize=(12,8)):
    """
    Examine the patterns of ACF and PACF, along with the time series plot and histogram.
    Source: https://github.com/jeffrey-yau/Pearson-TSA-Training-Beginner/blob/master/1_Intro_and_Overview.ipynb
    """
    fig = plt.figure(figsize=figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0))
    hist_ax = plt.subplot2grid(layout, (0,1))
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_xlabel(None)
    ts_ax.set_title(title)
    
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()
    
    
plot_general(train['y'],
             title='Organic Avocado Weekly Sales',
             lags=60,
             figsize=(12,8))
             
             
 # Let's go ahead and apply a log transformation
# Then take first difference
def plot_diff(y,
              title='title',
              diff=1,
              log=True,
              test=True,
              window=None,
              lags=None):
    if log:
        y = np.log(y)
        y_diff = y.diff(diff)
        y_diff.dropna(inplace=True)
    else:
        y_diff = y.diff(diff)
        y_diff.dropna(inplace=True)
    plot_general(y_diff, title, lags)
    if test:
        test_stationarity(y_diff, title , window)
    else:
        pass
    
    
 plot_diff(train['y'],
          window=4,
          lags=60,
          diff=53,
          title='Log Total Sales\n(53-week Seasonal Difference)')
          
          
 plot_diff(train['y'],
          window=4,
          lags=60,
          diff=54,
          title='Log Total Sales\n(54-week Seasonal Difference)')
          
# Log transformation
log_train = np.log(train['y'])
log_test = np.log(test['y'])

# Search over few models to find a model that has the lowest AIC/BIC
mdl_index = []
mdl_aic = []
mdl_bic = []

p = range(0,2)
d = range(0,2)
q = range(0,2)
P = range(0,2)
D = range(1,2)
Q = range(0,2)
S = range(53,55)

# Set variables to populate
#lowest_aic = None
#lowest_parm_aic = None
#lowest_param_seasonal_aic = None

#lowest_bic = None
#lowest_parm_bic = None
#lowest_param_seasonal_bic = None

# GridSearch the hyperparameters of p, d, q and P, D, Q, S
for param in list(itertools.product(p, d, q)):
    for param_seasonal in list(itertools.product(P, D, Q, S)):
        mdl = sm.tsa.statespace.SARIMAX(log_train,
                                        order=param,
                                        seasonal_order=param_seasonal)
        results = mdl.fit()      
        # Store results
        current_aic = results.aic
        current_bic = results.bic
        mdl_index.append('SARIMA{}x{}'.format(param, param_seasonal))
        mdl_aic.append(current_aic)
        mdl_bic.append(current_bic)
            
  print(pd.DataFrame(index=mdl_index, data=mdl_aic).rename(columns={0:'AIC'}).sort_values(by='AIC').head(5))
print('-----------------------------------------')
print(pd.DataFrame(index=mdl_index, data=mdl_bic).rename(columns={0:'BIC'}).sort_values(by='BIC').head(5))

class Sarima:
    def __init__(self,
                 y_train,
                 y_test,
                 order,
                 seasonal_order):
        self.y_train = y_train
        self.y_test = y_test
        self.order = order
        self.seasonal_order = seasonal_order
        
        # Modeling
        self._model = sm.tsa.statespace.SARIMAX(self.y_train,
                                                order=self.order,
                                                seasonal_order=self.seasonal_order)
        self._results = self._model.fit()
        
        # Construct in-sample fit
        self.y_est = self._results.get_prediction()
        self.y_est_mean = self.y_est.predicted_mean
        self.y_est_ci = self.y_est.conf_int(alpha=0.05)
    
        # Construct out-of-sample forecasts
        self.y_fcast = self._results.get_forecast(steps=len(y_test)).summary_frame()
        self.y_fcast.set_index(y_test.index, inplace=True)
        
    def results(self):
        print(self._results.summary())
    
    def diagnostics(self):
        print(self._results.plot_diagnostics(figsize=(15,8)))
        
    def plot(self):
        # Transform forecast to original scale
        inv_y_fcast = np.exp(self.y_fcast)
        inv_y_est_mean = np.exp(self.y_est_mean)
        inv_y_est_ci = np.exp(self.y_est_ci)
        inv_y_train = np.exp(self.y_train)
        inv_y_test = np.exp(self.y_test)
        
        # Plot the series
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        start_index = self.order[1] + self.seasonal_order[3]
        ax.set_title('Observed, Fitted, and Forecasted Series\nSARIMA{}x{}'.format(self.order, self.seasonal_order),
                     fontsize=16)
        ax.set_ylabel('Organic Avocado Weekly Sales')
        ax.plot(inv_y_train,
                label='In-sample data',
                linestyle='-')
        ax.plot(inv_y_test,
                label='Held-out data',
                linestyle='-')
        ax.plot(inv_y_est_mean[start_index :],
                label='Fitted values',
                linestyle='--',
                color='g')
        ax.plot(inv_y_fcast['mean'],
                label='Forecasts',
                linestyle='--',
                color='k')
        
        # Plot confidence intervals
        ax.fill_between(inv_y_est_mean[start_index :].index,
                        inv_y_est_ci.iloc[start_index :, 0],
                        inv_y_est_ci.iloc[start_index :, 1],
                        color='g', alpha=0.05)
        ax.fill_between(inv_y_fcast.index,
                       inv_y_fcast['mean_ci_lower'],
                       inv_y_fcast['mean_ci_upper'], 
                       color='k',
                        alpha=0.05)
        
        ax.legend(loc='upper left')
        plt.xticks(rotation = 45)
        plt.show(block = False)
        plt.close()
        
        # Return error metrics
        error_metrics(inv_y_fcast['mean'], inv_y_test)
        
Sarima(y_train=log_train,
       y_test=log_test,
       order=(1, 1, 1),
       seasonal_order=(0, 1, 1, 54)).results()
       
Sarima(y_train=log_train,
       y_test=log_test,
       order=(1, 1, 1),
       seasonal_order=(0, 1, 1, 54)).diagnostics()
       
Sarima(y_train=log_train,
       y_test=log_test,
       order=(1, 1, 1),
       seasonal_order=(0, 1, 1, 54)).plot()
       
Sarima(y_train=log_train,
       y_test=log_test,
       order=(0, 1, 1),
       seasonal_order=(0, 1, 1, 54)).results()
       
 Sarima(y_train=log_train,
       y_test=log_test,
       order=(0, 1, 1),
       seasonal_order=(0, 1, 1, 54)).diagnostics()
Sarima(y_train=log_train,
       y_test=log_test,
       order=(0, 1, 1),
       seasonal_order=(0, 1, 1, 54)).plot()
df = train.reset_index()
df.rename(columns={'date':'ds'}, inplace=True)
df = train.reset_index()
df.rename(columns={'date':'ds'}, inplace=True)

error_metrics(y_fcast = forecast[-len(test):]['yhat'].values,
              y_test = test['y'].values)
              
              
