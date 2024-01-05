import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# Feature Selection
from sklearn.feature_selection import SelectKBest
# regression selection
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
# classification selection
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from toolbox import standardizer, normalizer, rescaler

# Plotting
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

from data_forming import events_data, full_data
signal = 'bin'
# print(events_data)
Y = events_data.loc[:, signal]
Y.name = Y.name
X = events_data.loc[:, events_data.columns != signal, ]
Y = events_data.loc[:, Y.name]
X = events_data.loc[:, X.columns]
data = events_data

X = standardizer(X)
# X = normalizer(X)
# X = rescaler(X, (-1, 1))

# data = data.loc[data['bin'] == 0]
# print(data)
#
# # research------------------------------------------------------------------------------------
# print('data.describe()--------------------------------------------------------------------')
# print(data.describe())
# print('data-------------------------------------------------------------------------------')
# print(data)
#
# # 3. Exploratory Data Analysis ---------------------------------------------------------------
# # 3.1. Descriptive Statistics
# # shape
# print('data.shape-------------------------------------------------------------------------')
# print(data.shape)
# # peek at data
# pd.set_option('display.width', 100)
# print('data.head(2)-----------------------------------------------------------------------')
# print(data.head(2))
# # types
# pd.set_option('display.max_rows', 500)
# print('data.dtypes------------------------------------------------------------------------')
# print(data.dtypes)
# print('data.describe()--------------------------------------------------------------------')
# print(data.describe())

# 3.2. Data Visualization --------------------------------------------------------------------
# histograms
# data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 12))
# # density
# data.plot(kind='density', subplots=True, layout=(20, 20), sharex=False, legend=True, fontsize=1, figsize=(15, 15))
# # Box and Whisker Plots
# data.plot(kind='box', subplots=True, layout=(20, 20), sharex=False, sharey=False, figsize=(15, 15))
# # correlation
# correlation = data.corr()
# pyplot.figure(figsize=(15, 15))
# pyplot.title('Correlation Matrix')
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
# # Scatterplot Matrix
# pyplot.figure(figsize=(15, 15))
# scatter_matrix(data, figsize=(12, 12))

# 3.3. Time Series Analysis ------------------------------------------------------------------
# Time series broken down into different time series comonent.
# res = sm.tsa.seasonal_decompose(Y, period=30)
# fig = res.plot()
# fig.set_figheight(8)
# fig.set_figwidth(15)

# pyplot.show()

# 4.2 Feature Selection ---------------------------------------------------------------------
"""Statistical tests can be used to select those features that have the strongest relationship
with the output variable.The scikit-learn library provides the SelectKBest class that can be used with a suite of
different statistical tests to select a specific number of features. The example below uses the chi-squared (chi²)
statistical test for non-negative features to select the best features from the Dataset."""
X.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'ret'], axis=1, inplace=True)
print(X.columns)
bestfeatures = SelectKBest(mutual_info_classif, k='all')
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print('featureScores--------------------------------------------------------------------------')
print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

# bbc != 0
# 4H%K      0.035715
# BTC4H%DS  0.028873
# BTC4H%D   0.021451
# roc20     0.018674
# ema20     0.017499
# ema3      0.017459
# TrD13     0.015830
# adx       0.015808
# %D        0.014730
# 4H_rsi    0.014578
# BTCDema9  0.013545
# 4H%D      0.012211
# %K        0.010757
# BTC4H%K   0.010464
# H4_ema6   0.009665
# TrD3      0.008909
# mom10     0.008812
# mom20     0.008753
# roc10     0.007890
# 4H%DS     0.007479

# full events
# bb_cross  0.201304
# roc10     0.026474
# diff      0.023615
# %K        0.023340
# roc20     0.014551
# 4H%K      0.013499
# 4H_rsi    0.011702
# ema20     0.010961
# BTCTrD13  0.010719
# TrD13     0.009496
# ema3      0.009436
# TrD3      0.007151
# srl_corr  0.007055
# macd      0.006882
# 4Hmacd    0.006814
# adx       0.005870
# mom20     0.005419
# atr       0.004886
# ema13     0.003843
# 4H_atr    0.003161
