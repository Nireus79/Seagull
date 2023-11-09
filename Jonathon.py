import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import warnings
from data_forming import spliter, full_data, research_data

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# load dataset
# print(dataset.shape)
# peek at data
# set_option('display.width', 100)
# print(dataset.tail(5))
# describe data
# set_option('precision', 3)
# print(dataset.describe())
dataset = research_data
# Data Cleaning
# print('Null Values =', dataset.isnull().values.any())
dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()

# dataset = dataset.drop(columns=['Timestamp'])
# Preparing the data for classification
# Initialize the `signals` DataFrame with the `signal` column
# datas['PriceMove'] = 0.0

# print(dataset.tail())


# calculation of rate of change


# excluding columns that are not needed for our prediction.


# Data Visualization
# dataset[['Weighted_Price']].plot(grid=True)
# plt.show()
# histograms
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(15, 15))
plt.show()
# fig = plt.figure()
# plot = dataset.groupby(['signal']).size().plot(kind='barh', color='red')
# plt.show()
# correlation
# correlation = dataset.corr()
# plt.figure(figsize=(15, 15))
# plt.title('Correlation Matrix')
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

# Evaluate Algorithms and Models
# Train Test Split
# split out validation dataset for the end
Y = dataset['signal']
X = dataset.loc[:, dataset.columns != 'signal']

seed = 1
X_train, X_validation, Y_train, Y_validation, bt_data = spliter(dataset, 5)
# test options for classification
num_folds = 10
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring ='neg_log_loss'
# scoring = 'roc_auc'

# Compare Models and Algorithms
# spot check the algorithms
models = [('LR', LogisticRegression(n_jobs=-1)),
          ('LDA', LinearDiscriminantAnalysis()),
          ('KNC', KNeighborsClassifier()),
          ('CART_C', DecisionTreeClassifier()),
          ('NB', GaussianNB()),
          ('NN', MLPClassifier()),
          ('AB', AdaBoostClassifier()),
          ('GBM', GradientBoostingClassifier()),
          ('RF', RandomForestClassifier(n_jobs=-1))]

# K-folds cross validation
results = []
names = []
# for name, model in models:
#     kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
#     cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
#
# # compare algorithms
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# fig.set_size_inches(15, 8)
# plt.show()

# Model Tuning and Grid Search
# Grid Search: Random Forest Classifier
'''
n_estimators : int (default=100)
    The number of boosting stages to perform.
    Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
max_depth : integer, optional (default=3)
    maximum depth of the individual regression estimators.
    The maximum depth limits the number of nodes in the tree.
    Tune this parameter for best performance; the best value depends on the interaction of the input variables
criterion : string, optional (default=”gini”)
    The function to measure the quality of a split.
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

'''
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# rescaledX = standardizer(X_train)
max_depth = [50, 200]
n_estimators = [50, 200]
criterion = ["gini", "entropy"]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)
model = RandomForestClassifier(n_jobs=-1)
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X_train, Y_train)

# Print Results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
ranks = grid_result.cv_results_['rank_test_score']
for mean, stdev, param, rank in zip(means, stds, params, ranks):
    print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))

# Finalise the Model
# prepare model
model = RandomForestClassifier(criterion='gini', n_estimators=80, max_depth=10, n_jobs=-1)  # rbf is default kernel
# model = LogisticRegression()
model.fit(X_train, Y_train)

# estimate accuracy on validation set
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

df_cm = pd.DataFrame(confusion_matrix(Y_validation, predictions), columns=np.unique(Y_validation),
                     index=np.unique(Y_validation))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font sizes

# Variable Intuition/Feature Importance
Importance = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.show()

# Backtesting Results
# Create column for Strategy Returns by multiplying the daily returns by the position that was held at close
# of business the previous day
backtestdata = pd.DataFrame(index=X_validation.index)
# backtestdata = pd.DataFrame()
backtestdata['signal_pred'] = predictions
backtestdata['signal_actual'] = Y_validation
backtestdata['Market Returns'] = X_validation['Close'].pct_change()
backtestdata['Actual Returns'] = backtestdata['Market Returns'] * backtestdata['signal_actual'].shift(1)
backtestdata['Strategy Returns'] = backtestdata['Market Returns'] * backtestdata['signal_pred'].shift(1)
backtestdata = backtestdata.reset_index()
backtestdata.head()
backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().hist()
backtestdata[['Strategy Returns', 'Actual Returns']].cumsum().plot(style=['-', '--'])
plt.show()
