import numpy as np
import pandas as pd
from matplotlib import pyplot

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

import warnings
from toolbox import spliter
from data_forming import events_data, signal, delta
part = 1
X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, delta)
warnings.filterwarnings('ignore')

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# 5. Evaluate Algorithms and Models --------------------------------------------------------

# 5.2. Test Options and Evaluation Metrics
num_folds = 10
seed = 7
# scikit is moving away from mean_squared_error.
# In order to avoid confusion, and to allow comparison with other models, we invert the final scores
scoring = 'neg_mean_squared_error'
# 5.3. Compare Models and Algorithms ----------------------------------------------------------

# 5.3.1 Machine Learning models-from scikit-learn
# Regression and Tree Regression algorithms
models = [
    ('LR', LinearRegression()),
    ('LASSO', Lasso()),
    ('EN', ElasticNet()),
    ('KNN', KNeighborsRegressor()),
    ('CART_S', DecisionTreeRegressor()),
    ('SVR', SVR()),
    ('MLP', MLPRegressor()),
    ('RFR', RandomForestRegressor()),
    ('ABR', AdaBoostRegressor()),
    ('GBR', GradientBoostingRegressor()),
    ('ETR', ExtraTreesRegressor())
]

# Once we have selected all the models, we loop over each of them. First we run the K-fold analysis.
# Next we run the model on the entire training and testing dataset.

names = []
kfold_results = []
test_results = []
train_results = []

for name, model in models:
    names.append(name)
    # K Fold analysis:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    # converted mean square error to positive. The lower the beter
    cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    kfold_results.append(cv_results)
    # Full Training period
    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)
    # Test results
    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)
    msg = "%s: cv_results.mean(): %f cv_results.std(): (%f) train_result: %f test_result: %f" \
          % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
    print(msg)

# K Fold results
# We being by looking at the K Fold results
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison: Kfold results')
ax = fig.add_subplot(111)
pyplot.boxplot(kfold_results)
ax.set_xticklabels(names)
fig.set_size_inches(15, 8)
pyplot.show()
# We see the linear regression and the regularized regression
# including the Lasso regression (LASSO) and elastic net (EN) seem to do a good job.

# Training and Test error -------------------
# compare algorithms
fig = pyplot.figure()

ind = np.arange(len(names))  # the x locations for the groups
width = 0.35  # the width of the bars

fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.bar(ind - width / 2, train_results, width=width, label='Train Error')
pyplot.bar(ind + width / 2, test_results, width=width, label='Test Error')
fig.set_size_inches(15, 8)
pyplot.legend()
ax.set_xticks(ind)
ax.set_xticklabels(names)
pyplot.show()
