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

# bb = -1
# Specs
# 4H_rsi       31.561384
# 4H%K         21.853448
# 4H%D         16.996173
# 4H%DS        11.427634
# srl_corr      7.522237
# macd          6.076092
# 4Hmacd_diff   5.826797
# 4M_diff       5.497436
# adx           4.446873
# Volatility    1.230380

#bb = 1
# Specs
# 4H%K         17.188775
# 4H%D         10.148062
# 4Hmacd_diff   9.115304
# 4M_diff       6.767452
# 4H_rsi        6.150274
# 4H%DS         5.617976
# rsi           4.583238
# 1D_Close      3.859218
# Dema3         3.797740
# Dema6         3.727843
# Dema9         3.697064
# Dema13        3.687299
# 1D_High       3.619445
# 4H_Close      3.600882
# Dema20        3.569223
# 4H_High       3.454920
# H4_ema3       3.428998
# 1D_Low        3.405874
# H4_ema6       3.354515
# 4H_Low        3.274641
