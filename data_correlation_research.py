import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# Feature Selection
from sklearn.feature_selection import SelectKBest
# regression selection
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
# classification selection
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

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
bestfeatures = SelectKBest(k='all', score_func=f_classif)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print('featureScores--------------------------------------------------------------------------')
print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

# bbc != 0
# TrD6         256.962609
# TrD3         248.268246
# TrD9         243.180894
# TrD13        226.627497
# TrD20        195.419734
# StD           22.607029
# cusum         10.439449
# roc30          7.588108
# 4Hmacd_diff    7.292761

# bbc == 1
# TrD3         147.683369
# TrD6         136.792228
# TrD9         128.361185
# TrD13        123.603026
# TrD20        111.933259
# 4H%K          39.811147
# 4H%D          33.520830
# 4H_rsi        28.968843
# 4H%DS         23.369553
# 4Hmacd_diff   16.344895

# bbc == -1
# TrD6         123.281671
# TrD3         115.131158
# TrD9         114.728591
# TrD13        101.704586
# TrD20         81.174263
# 4Hmacd_diff   15.116768
# StD           13.807081

# full
# TrD3      120.280969
# TrD6      117.228333
# TrD9      111.061182
# TrD13     106.329219
# TrD20      99.030422
# srl_corr   32.244731
# diff       24.341237
# bb_cross   21.077353
# %K         17.845789
# adx        14.996010
