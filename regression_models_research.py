import numpy as np
import pandas as pd
from matplotlib import pyplot
from tqdm import tqdm
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
from toolbox import rescaler, normalizer, standardizer
import warnings
from toolbox import spliter, uniqueCombinations
from data_forming import events_data, delta

warnings.filterwarnings('ignore')

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def K_F(Xtr, Ytr, Xts, Yts):
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
        # ('LR', LinearRegression()),
        # ('LASSO', Lasso()),
        # ('EN', ElasticNet()),
        # ('KNN', KNeighborsRegressor()),
        # ('CART_S', DecisionTreeRegressor()),
        # ('SVR', SVR()),
        # ('MLP', MLPRegressor()),
        ('RFR', RandomForestRegressor()),
        # ('ABR', AdaBoostRegressor()),
        ('GBR', GradientBoostingRegressor()),
        # ('ETR', ExtraTreesRegressor())
    ]
    # Once we have selected all the models, we loop over each of them. First we run the K-fold analysis.
    # Next we run the model on the entire training and testing dataset.
    names = []
    kfold_results = []
    test_results = []
    train_results = []
    m = []
    for name, model in models:
        names.append(name)
        # K Fold analysis:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        # converted mean square error to positive. The lower the beter
        cv_results = -1 * cross_val_score(model, Xtr, Ytr, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)
        # Full Training period
        res = model.fit(Xtr, Ytr)
        train_result = mean_squared_error(res.predict(Xtr), Ytr)
        train_results.append(train_result)
        # Test results
        test_result = mean_squared_error(res.predict(Xts), Yts)
        test_results.append(test_result)
        msg = "%s: cv_results.mean(): %f cv_results.std(): (%f) train_result: %f test_result: %f" \
              % (name, cv_results.mean(), cv_results.std(), train_result, test_result)
        ms = test_result
        print(msg)
        m.append(ms)
    # print(m)
    return m
    # K Fold results
    # We being by looking at the K Fold results
    # fig = pyplot.figure()
    # fig.suptitle('Algorithm Comparison: Kfold results')
    # ax = fig.add_subplot(111)
    # pyplot.boxplot(kfold_results)
    # ax.set_xticklabels(names)
    # fig.set_size_inches(15, 8)
    # pyplot.show()
    # We see the linear regression and the regularized regression
    # including the Lasso regression (LASSO) and elastic net (EN) seem to do a good job.
    # Training and Test error -------------------
    # compare algorithms
    # fig = pyplot.figure()
    #
    # ind = np.arange(len(names))  # the x locations for the groups
    # width = 0.35  # the width of the bars
    #
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # pyplot.bar(ind - width / 2, train_results, width=width, label='Train Error')
    # pyplot.bar(ind + width / 2, test_results, width=width, label='Test Error')
    # fig.set_size_inches(15, 8)
    # pyplot.legend()
    # ax.set_xticks(ind)
    # ax.set_xticklabels(names)
    # pyplot.show()


def report_generator(full_feats, standard_feats, pl, trd, tsd):
    combinations = uniqueCombinations(full_feats, standard_feats, pl)
    print('Combinations:', len(combinations))
    print(combinations)
    r = []
    for i in tqdm(combinations):
        print(i)
        Y_tr = trd['ret']
        X_tr = trd.drop(columns=['ret'])
        X_tr = normalizer(X_tr)
        Y_ts = tsd['ret']
        X_ts = tsd.drop(columns=['ret'])
        X_ts = normalizer(X_ts)
        rep = K_F(X_tr, Y_tr, X_ts, Y_ts)
        r.append((i, rep))
        # print(r)
    print(r)
    return r


signal = 'ret'
# features = ['TrD6', 'St4H', 'mom10', 'MAV', signal]
features = ['TrD3', 'TrD6', 'TrD20', 'vrsi', 'mom10', 'TrD6', 'Volume',
            'mom10', 'roc10', '4H%K', 'Volatility', 'St4H', 'MAV', 'bb_cross', signal]
# features = ['TrD3', 'Volatility', 'srl_corr', 'bb_cross', signal]
# X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, features, delta)


train_data = pd.read_csv('csv/synth/synth10000.csv')[features]
test_data = events_data[features]

data = report_generator(features, ['ret'], 2, train_data, test_data)
df = pd.DataFrame(data, columns=['Features', 'Results'])
print(df)
# df['LR'] = df['Results'][1][0]
# df['LASSO'] = df['Results'][1][1]
# df['KNN'] = df['Results'][1][2]
# df['CART_S'] = df['Results'][1][3]
# df['SVR'] = df['Results'][1][4]
# df['MLP'] = df['Results'][1][5]
# df['RFR'] = df['Results'][1][6]
# df['ABR'] = df['Results'][1][7]
# df['GBR'] = df['Results'][1][8]
# df['ETR'] = df['Results'][1][9]
# df = df.drop(columns=['Results'])
#
# print(df.loc[df['LR'].idxmin()])
# print(df.loc[df['LASSO'].idxmin()])
# print(df.loc[df['KNN'].idxmin()])
# print(df.loc[df['CART_S'].idxmin()])
# print(df.loc[df['SVR'].idxmin()])
# print(df.loc[df['MLP'].idxmin()])
# print(df.loc[df['RFR'].idxmin()])
# print(df.loc[df['ABR'].idxmin()])
# print(df.loc[df['GBR'].idxmin()])
# print(df.loc[df['ETR'].idxmin()])

# [TrD3, TrD6, ret]
# LR                   0.00184
# LASSO                0.001946
# KNN                  0.001944
# CART_S               0.001506
# SVR                  0.001873
# MLP                  0.002238
# RFR                  0.106893
# ABR                  0.001001
# GBR                  0.001927
# ETR                  0.001011

# [4H%K, TrD3, TrD6, ret]
# LR                          0.00184
# LASSO                      0.001946
# KNN                        0.001944
# CART_S                     0.001506
# SVR                          0.0018
# MLP                        0.002238
# RFR                        0.336632
# ABR                        0.000978
# GBR                        0.002026
# ETR                        0.001011
