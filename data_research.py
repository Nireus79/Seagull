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

from data_forming import X, Y, research_data


data = research_data

# research------------------------------------------------------------------------------------
print('data.describe()--------------------------------------------------------------------')
print(data.describe())
print('data-------------------------------------------------------------------------------')
print(data)

# 3. Exploratory Data Analysis ---------------------------------------------------------------
# 3.1. Descriptive Statistics
# shape
print('data.shape-------------------------------------------------------------------------')
print(data.shape)
# peek at data
pd.set_option('display.width', 100)
print('data.head(2)-----------------------------------------------------------------------')
print(data.head(2))
# types
pd.set_option('display.max_rows', 500)
print('data.dtypes------------------------------------------------------------------------')
print(data.dtypes)
print('data.describe()--------------------------------------------------------------------')
print(data.describe())

# 3.2. Data Visualization --------------------------------------------------------------------
# histograms
# data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(12, 12))
# # density
# data.plot(kind='density', subplots=True, layout=(20, 20), sharex=False, legend=True, fontsize=1, figsize=(15, 15))
# # Box and Whisker Plots
# data.plot(kind='box', subplots=True, layout=(20, 20), sharex=False, sharey=False, figsize=(15, 15))
# # # correlation
# correlation = data.corr()
# pyplot.figure(figsize=(15, 15))
# pyplot.title('Correlation Matrix')
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
# # Scatterplot Matrix
# pyplot.figure(figsize=(15, 15))
# scatter_matrix(data, figsize=(12, 12))
#
# # 3.3. Time Series Analysis ------------------------------------------------------------------
# # Time series broken down into different time series comonent.
# res = sm.tsa.seasonal_decompose(Y, period=365)
# fig = res.plot()
# fig.set_figheight(8)
# fig.set_figwidth(15)
#
# pyplot.show()

# 4. Data Preparation
#
# 4.1. Data Cleaning
# Check for the NAs in the rows, either drop them or fill them with the mean of the column
# Checking for any null values and removing the null values'''
# print('dataset Null Values =', data.isnull().values.any())
# Given that there are null values drop the row containing the null values.
# Drop the rows containing NA
# data.dropna(axis=0)
# Fill na with 0
# dataset.fillna('0')
#
# Filling the NAs with the mean of the column.
# dataset['col'] = dataset['col'].fillna(dataset['col'].mean())
#
# 4.2 Feature Selection ---------------------------------------------------------------------
"""Statistical tests can be used to select those features that have the strongest relationship
with the output variable.The scikit-learn library provides the SelectKBest class that can be used with a suite of
different statistical tests to select a specific number of features. The example below uses the chi-squared (chi²)
statistical test for non-negative features to select 10 of the best features from the Dataset."""

bestfeatures = SelectKBest(k='all', score_func=f_regression)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print('featureScores--------------------------------------------------------------------------')
print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

# raw
# Specs
# Close       139.770717
# 4H%K         59.489339
# Dema13       54.025903
# Volatility   52.684790
# Dema9        47.765246
# 4Hmacd       41.703173
# 4H%D         25.379989
# 4H_rsi       23.312552
# 4H%DS         7.757869

# scaled
# Specs
# Close       139.770717
# 4H%K         59.489339
# Dema13       54.025903
# Volatility   52.684790
# Dema9        47.765246
# 4Hmacd       41.703173
# 4H%D         25.379989
# 4H_rsi       23.312552
# 4H%DS         7.757869
