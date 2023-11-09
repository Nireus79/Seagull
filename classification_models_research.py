import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

# Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import warnings
from data_forming import research_data, X_train, Y_train, X_test, Y_test

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

dataset = research_data
dataset[dataset.columns.values] = dataset[dataset.columns.values].ffill()

seed = 1
# test options for classification
num_folds = 10
scoring = 'accuracy'
# scoring = 'precision'
# scoring = 'recall'
# scoring ='neg_log_loss'
# scoring = 'roc_auc'

# Compare Models and Algorithms
# spot check the algorithms
models = [
    ('CART_C', DecisionTreeClassifier()),
    ('AB', AdaBoostClassifier()),
    ('GBM', GradientBoostingClassifier()),
    ('RF', RandomForestClassifier(n_jobs=-1)),
    ('LR', LogisticRegression(n_jobs=-1)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNC', KNeighborsClassifier()),
    ('NB', GaussianNB()),
    ('NN', MLPClassifier())
]

# K-folds cross validation
results = []
names = []
test_results = []
train_results = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    res = model.fit(X_train, Y_train)
    print(name, '-----------------------------------------------------------------------------------------------')
    if name == 'CART_C' or name == 'AB' or name == 'GBM' or name == 'RF':
        Importance = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=X_train.columns)
        Importance.sort_values('Importance', axis=0, ascending=True)
        print(Importance)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)
    # Test results
    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
fig.set_size_inches(15, 8)
plt.show()

fig = plt.figure()

ind = np.arange(len(names))  # the x locations for the groups
width = 0.35  # the width of the bars

fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.bar(ind - width / 2, train_results, width=width, label='Train Error')
plt.bar(ind + width / 2, test_results, width=width, label='Test Error')
fig.set_size_inches(15, 8)
plt.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
plt.show()