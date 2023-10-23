import matplotlib.pyplot as plt

from data_forming import X_train, X_test, Y_train, Y_test
import pandas as pd
import numpy as np
import warnings
from toolbox import evaluate_LSTM_combinations, create_LSTMmodel, evaluate_arima_models, create_ANN
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

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
from sklearn.compose import ColumnTransformer

# Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from scikeras.wrappers import KerasRegressor

# Libraries for Statistical Models
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Time series Models
from statsmodels.tsa.arima.model import ARIMA

# Error Metrics
from sklearn.metrics import mean_squared_error

# Time series Models
from statsmodels.tsa.arima.model import ARIMA

# Error Metrics
from sklearn.metrics import mean_squared_error

# Saving the Model
from pickle import dump

warnings.filterwarnings('ignore')

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

num_folds = 10
scoring = 'neg_mean_absolute_error'
seed = 7


# 1 Grid search: Linear regression
def GS_Linear_regression():
    """
    fit_intercept : boolean, optional, default True
        whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    """
    param_grid = {'fit_intercept': [True, False]}
    model = LinearRegression()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print(('%f (%f) with: %r' % (mean, stdev, param)))


# 2 Grid search: Lasso
def GS_Lasso():
    """
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.
    """
    param_grid = {'alpha': [0.01, 0.1, 0.3, 0.7, 1, 1.5, 3, 5]}
    model = Lasso()
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print(('%f (%f) with: %r' % (mean, stdev, param)))


# 3. Grid Search : ElasticNet
def GS_ElasticNet():
    """
    alpha : float, optional
        Constant that multiplies the penalty terms. Defaults to 1.0.
        See the notes for the exact mathematical meaning of this
        parameter.``alpha = 0`` is equivalent to an ordinary least square,
        solved by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.
    """
    param_grid = {'alpha': [0.01, 0.1, 0.3, 0.7, 1, 1.5, 3, 5],
                  'l1_ratio': [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]}
    model = ElasticNet()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 4. Grid search : KNeighborsRegressor
def GS_KNeighborsRegressor():
    """
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
    """
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
    model = KNeighborsRegressor()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 5. Grid search : DecisionTreeRegressor
def GS_DecisionTreeRegressor():
    """
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If floated, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    """
    param_grid = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = DecisionTreeRegressor()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 6. Grid search : SVR
def GS_SVR():
    """
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.
    """
    param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                  'gamma': [0.001, 0.01, 0.1, 1]}
    # 'epslion': [0.01, 0.1, 1]}
    model = SVR()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 7. Grid search : MLPRegressor
def GS_MLPRegressor():
    """
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    """
    param_grid = {'hidden_layer_sizes': [(20,), (50,), (20, 20), (20, 30, 20)]}
    model = MLPRegressor()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 8. Grid search : RandomForestRegressor
def GS_RandomForestRegressor():
    """
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    """
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400]}
    model = RandomForestRegressor()
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 9. Grid search : GradientBoostingRegressor
def GS_GradientBoostingRegressor():
    """
    n_estimators:

        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    """
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400]}
    model = GradientBoostingRegressor(random_state=seed)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 10. Grid search : ExtraTreesRegressor
def GS_ExtraTreesRegressor():
    """
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    """
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400]}
    model = ExtraTreesRegressor(random_state=seed)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 11. Grid search : AdaBoostRegressor
def GS_AdaBoostRegressor():
    """
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    """
    param_grid = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
                  'learning_rate': [1, 2, 3]}
    model = AdaBoostRegressor(random_state=seed)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, Y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# 12. Grid search : KerasNNRegressor
def GS_KerasNNRegressor():
    """
    nn_shape : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith
        hidden layer.
    """
    EnableDeepLearningRegreesorFlag = 0
    # Add Deep Learning Regressor
    if EnableDeepLearningRegreesorFlag == 1:
        param_grid = {'nn_shape': [(20,), (50,), (20, 20), (20, 30, 20)]}
        model = KerasRegressor()
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X_train, Y_train)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


def polynomial():
    Deg = range(2, 15)
    results = []
    names = []
    for deg in Deg:
        polynomial_features = PolynomialFeatures(degree=deg)
        x_poly = polynomial_features.fit_transform(X_train)
        model = LinearRegression(fit_intercept=False)
        model.fit(x_poly, Y_train)
        Y_poly_pred = model.predict(x_poly)
        rmse = np.sqrt(mean_squared_error(Y_train, Y_poly_pred))
        r2 = r2_score(Y_train, Y_poly_pred)
        results.append(rmse)  # 4 degrees 0.06 rmse / 9 degrees r2 -10.5
        names.append(deg)
    plt.plot(names, results, 'o')
    plt.xlabel('n-degrees polynomial')
    plt.suptitle('Algorithm comparison')
    plt.show()





# create_ANN(X_train, Y_train, X_test, Y_test)


# print('GS_Linear_regression-------------------------------------------------------------------------------------------')
# GS_Linear_regression()  # Best: -0.048184 using {'fit_intercept': False}
# print('GS_Lasso-------------------------------------------------------------------------------------------------------')
# GS_Lasso()
# print('GS_ElasticNet--------------------------------------------------------------------------------------------------')
# GS_ElasticNet()
# print('GS_SVR---------------------------------------------------------------------------------------------------------')
# GS_SVR()


# names = []
# test_results = []
# train_results = []
#
# # 5. Evaluate Algorithms and Models --------------------------------------------------------
#
# # 5.1. Train Test Split and Evaluation Metrics Next, we start by splitting our data in training and testing chunks.
# # If we are going to use Time series models we have to split the data in continuous series.
# validation_size = 0.2
#
# # In case the data is not dependent on the time series, then train and test split randomly
# # seed = 7
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
#
# # In case the data is not dependent on the time series, then train and test split should be done based on sequential
# # sample This can be done by selecting an arbitrary split point in the ordered list of observations and creating two
# # new datasets.
#
# # 5.3.1 Time Series based models-ARIMA and LSTM --------------------------------------------------
# # Let us first prepare the dataset for ARIMA models,
# # by having only the correlated variables as exogenous variables.
#
# # Time Series Model - ARIMA Model
# print('X_train.columns--------------------------------------------------------------------------')
# print(X_train.columns)
# X_train_ARIMA = X_train.loc[:, ['doteur_close', 'etheur_close', 'ema9', 'volatility', 'cusum']]
# X_test_ARIMA = X_test.loc[:, ['doteur_close', 'etheur_close', 'ema9', 'volatility', 'cusum']]
# tr_len = len(X_train_ARIMA)
# te_len = len(X_test_ARIMA)
# to_len = len(X)
#
# modelARIMA = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=[2, 0, 0])
# model_fit = modelARIMA.fit()
#
# error_Training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
# predicted = model_fit.predict(start=tr_len - 1, end=to_len - 1, exog=X_test_ARIMA)[1:]
# error_Test_ARIMA = mean_squared_error(Y_test, predicted)
# print('error_Test_ARIMA------------------------------------------------------------------------')
# print(error_Test_ARIMA)
#
# # LSTM Model
# seq_len = 2  # Length of the seq for the LSTM
#
# Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len - 1:], np.array(Y_test)
# X_train_LSTM = np.zeros((X_train.shape[0] + 1 - seq_len, seq_len, X_train.shape[1]))
# X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
# for i in range(seq_len):
#     X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0] + i + 1 - seq_len, :]
#     X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0] + i - 1:X.shape[0] + i + 1 - seq_len, :]
#
# # Lstm Network
#
#
# LSTMModel = create_LSTMmodel(X_train_LSTM, 12, learn_rate=0.01, momentum=0)
# LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSTM, Y_test_LSTM), epochs=330,
#                               batch_size=72, verbose=0, shuffle=False)
#
# # Visual plot to check if the error is reducing
# pyplot.plot(LSTMModel_fit.history['loss'], label='train', )
# pyplot.plot(LSTMModel_fit.history['val_loss'], '--', label='test', )
# pyplot.legend()
# pyplot.show()
#
# error_Training_LSTM = mean_squared_error(Y_train_LSTM, LSTMModel.predict(X_train_LSTM))
# predicted = LSTMModel.predict(X_test_LSTM)
# error_Test_LSTM = mean_squared_error(Y_test, predicted)
# print('lenX: ', len(X))
# print('predicted LSTM ------------------------------------------------------------')
# print(predicted)
# print(np.sum(np.array(predicted) >= 0, axis=0))
# print('error_Test_LSTM-------------------------------------------------------------')
# print(error_Test_LSTM)
#
# # Append to previous results
# test_results.append(error_Test_ARIMA)
# test_results.append(error_Test_LSTM)
#
# train_results.append(error_Training_ARIMA)
# train_results.append(error_Training_LSTM)
#
# names.append("ARIMA")
# names.append("LSTM")
#
# # Overall Comparison of all the algorithms (including Time Series Algorithms) -------------------
# # compare algorithms
# fig = pyplot.figure()
#
# ind = np.arange(len(names))  # the x locations for the groups
# width = 0.35  # the width of the bars
#
# fig.suptitle('Comparing the performance of various algorthims on the Train and Test Dataset')
# ax = fig.add_subplot(111)
# pyplot.bar(ind - width / 2, train_results, width=width, label='Train Error')
# pyplot.bar(ind + width / 2, test_results, width=width, label='Test Error')
# fig.set_size_inches(15, 8)
# pyplot.legend()
# ax.set_xticks(ind)
# ax.set_xticklabels(names)
# pyplot.ylabel('Mean Square Error')
# pyplot.show()
#
# # 6. Model Tuning and Grid Search ------------------------------------------------------------------
# """ARIMA model is one of the best mode according to metrics above, so we perform the model tuning of the ARIMA model.
#     The default order of ARIMA model is [1,0,0]. We perform a grid search with different combination p,d and q
#     in the ARIMA model's order. """
#
# # Grid Search for ARIMA Model - LSTM
# # Change p,d and q and check for the best result
#
# # evaluate parameters
# print('ARIMA parameters evaluation ------------------------------------------------------')
# p_values = [0, 1, 2]
# d_values = range(0, 2)
# q_values = range(0, 2)
# warnings.filterwarnings("ignore")
# print('evaluate_models(p_values, d_values, q_values)-------------------------------------------')
# print(evaluate_arima_models(X_train, Y_train, p_values, d_values, q_values))
#
# print('LSTM parameters evaluation -------------------------------------------------------')
# neurons_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 2
# learn_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # 0.4
# momentum_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 0.5
# # dense_list = [1, 5]
# # batch_size_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # verbose_list = [0, 1]
# # epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# evaluate_LSTM_combinations(X_train_LSTM, X_test_LSTM, Y_test_LSTM, neurons_list, learn_rate_list, momentum_list)
#
# # 7. Finalise the Model -------------------------------------------------------------------------
# # 7.1. Results on the Test Dataset
# # prepare model
# modelARIMA_tuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=[2, 0, 0])
# model_fit_tuned = modelARIMA_tuned.fit()
# # estimate accuracy on validation set
# predicted_tuned = model_fit.predict(start=tr_len - 1, end=to_len - 1, exog=X_test_ARIMA)[1:]
# print('mean_squared_error(Y_test, predicted_tuned)---------------------------------------------')
# print(mean_squared_error(Y_test, predicted_tuned))
