from sklearn.feature_selection import SelectKBest, RFE, f_classif, SelectPercentile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_forming import events_data

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# part = 5

# X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, 'All', delta)
# backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
# X_train_c, X_test_c = X_train.copy(), X_test.copy()
# X_train_n, X_test_n = normalizer(X_train_c), normalizer(X_test_c)
# if 'bb_cross' in X_train.columns:
#     print('bb_cross in X')
#     X_train_n.bb_cross, X_test_n.bb_cross = X_train.bb_cross, X_test.bb_cross


# print('full_data.columns', full_data.columns)

# 'GBC' max precision0: features      [Tr9, Tr20, TrD3, TrD9]
# precision0                   0.760274
# recall0                      0.834586
# precision1                   0.768421
# recall1                      0.675926

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# https://towardsai.net/p/data-science/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
# https://www.youtube.com/watch?v=hCwTDTdYirg&t=11s

#      # features-# Categorical  - # Numerical
# target         -               -
#                -               -
# Categorical    -# Chi squared  - # t-test
#                -# Mutual info  - # Mutual info
# ---------------------------------------------------------------
# Numerical      -# t-test       - # Pearson correlation
#                -# Mutual info  - # Spearman rank correlation
#                -               - # Mutual info
# ---------------------------------------------------------------
# VARIANCE ------------------------------------------------------
def Variance(data):
    print('Var ----------------------------------------------------------------')
    Var = data.var(axis=0)
    print('Var')
    print(Var.sort_values())


# Correlation --------------------------------------------------
def Correlation(data, sig):
    print('Correlation ----------------------------------------------------------')
    # fig = plt.figure()
    data.groupby([sig]).size().plot(kind='barh', color='red')

    correlation = data.corr()
    print(correlation[sig].sort_values())

    # plt.figure(figsize=(100, 100))
    # plt.title('Correlation Matrix')
    # sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    # plt.show()


# K-best ------------------------------------------------------------
def K_best(X_tr, Y_tr, X_ts, Y_ts):
    print('KBest ------------------------------------------------------')
    bestfeatures = SelectKBest(f_classif, k='all')
    fit = bestfeatures.fit(X_tr, Y_tr)
    scores = pd.DataFrame(fit.scores_)
    Xcolumns = pd.DataFrame(X_tr.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([Xcolumns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

    f1_score_list = []
    for k in tqdm(range(1, len(X_tr.columns))):
        model = MLPClassifier()
        selector = SelectKBest(f_classif, k=k)
        fit = selector.fit(X_tr, Y_tr)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X_tr.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        # print('featureScores--------------------------------------------------------------------------')
        # print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

        sel_XtrainK = selector.transform(X_tr)
        sel_XtestK = selector.transform(X_ts)

        model.fit(sel_XtrainK, Y_tr)
        preds = model.predict(sel_XtestK)
        score = f1_score(Y_ts, preds)
        # print(k, score)
        f1_score_list.append(score)

    fig, ax = plt.subplots()

    x = np.arange(1, len(X_tr.columns))
    y = f1_score_list

    ax.bar(x, y, width=0.2)
    ax.set_xlabel('Number of features selected using f_classif')
    ax.set_ylabel('F1-Score (weighted)')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(np.arange(1, len(X_tr.columns)))
    ax.set_xticklabels(np.arange(1, len(X_tr.columns)), fontsize=12)

    for i, v in enumerate(y):
        plt.text(x=i + 1, y=v + 0.05, s=str(v), ha='center')

    plt.tight_layout()
    plt.show()


# Select persentile -----------------------------------------------------------------------------------------
def SelectPrsnt(X_tr, Y_tr):
    """
    percentile (int), default=10
    Percent of features to keep.
    :return:
    """
    print('SelectPercentile ----------------------------------------------------')
    bestfeatures = SelectPercentile(f_classif, percentile=10)
    fit = bestfeatures.fit(X_tr, Y_tr)
    scores = pd.DataFrame(fit.scores_)
    Xcolumns = pd.DataFrame(X_tr.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([Xcolumns, scores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features


def rfe(X_tr, Y_tr, X_ts, Y_ts):
    """
    Recursive feature elimination performs RFE in a cross-validation loop to find the optimal number of features.
    :return:
    """
    gbc = GradientBoostingClassifier(max_depth=5, random_state=42)
    rfe_f1score_list = []
    for k in tqdm(range(1, len(X_tr))):
        RFE_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
        RFE_selector.fit(X_tr, Y_tr)
        sel_XtrainR = RFE_selector.transform(X_tr)
        sel_XtestR = RFE_selector.transform(X_ts)
        gbc.fit(sel_XtrainR, Y_tr)
        RFE_preds = gbc.predict(sel_XtestR)

        f1_score_rfe = round(f1_score(Y_ts, RFE_preds, average='weighted'), 3)
        print(k, f1_score_rfe)
        rfe_f1score_list.append(f1_score_rfe)

    fig, ax = plt.subplots()

    x = np.arange(1, len(X_tr))
    y = rfe_f1score_list

    ax.bar(x, y, width=0.2)
    ax.set_xlabel('Number of features selected using RFE')
    ax.set_ylabel('F1-Score (weighted)')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(np.arange(1, len(X_tr)))
    ax.set_xticklabels(np.arange(1, len(X_tr)), fontsize=12)

    for i, v in enumerate(y):
        plt.text(x=i + 1, y=v + 0.05, s=str(v), ha='center')

    plt.tight_layout()
    plt.show()


def Boruta(X_tr, Y_tr, X_ts, Y_ts):
    gbc = GradientBoostingClassifier(max_depth=5, random_state=42)

    boruta_selector = BorutaPy(gbc, random_state=42)
    boruta_selector.fit(X_tr.values, Y_tr.values.ravel())
    sel_XtrainB = boruta_selector.transform(X_tr.values)
    sel_XtestB = boruta_selector.transform(X_ts.values)
    gbc.fit(sel_XtrainB, Y_tr)
    boruta_preds = gbc.predict(sel_XtestB)
    boruta_f1_score = round(f1_score(Y_ts, boruta_preds, average='weighted'), 3)
    print(boruta_f1_score)
    RFE_selector = RFE(estimator=gbc, n_features_to_select=5, step=10)
    RFE_selector.fit(X_tr, Y_tr)
    selected_features_mask = boruta_selector.support_
    selected_features = X_tr.columns[selected_features_mask]
    print('selected_features:', selected_features)


def MDI(X, Y):
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    names = X.columns
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))


train_data = pd.read_csv('csv/synth/Prelder_standard_1_ev100.csv')
test_data = events_data # pd.read_csv('csv/synth/Prelder_standard_1_ev20.csv')
signal = 'ret'
Y_train = train_data[signal]
X_train = train_data.drop(columns=[signal])
# X_train = normalizer(X_train)
Y_test = test_data[signal]
X_test = test_data.drop(columns=[signal])
# X_test = normalizer(X_test)


X_train_c, X_test_c = X_train.copy(), X_test.copy()
X_train_c.drop(columns=['bb_cross'], axis=1, inplace=True)
X_test_c.drop(columns=['bb_cross'], axis=1, inplace=True)
# X_train_r, X_test_r = normalizer(X_train_c), normalizer(X_test_c)
# X_train_r['bb_cross'], X_test_r['bb_cross'] = X_train.bb_cross, X_test.bb_cross


Correlation(train_data, signal)
Correlation(test_data, signal)
# K_best(X_train, Y_train, X_test, Y_test)
# SelectPrsnt(X_train, Y_train)
# rfe(X_train, Y_train, X_test, Y_test)
# Boruta(X_train, Y_train, X_test, Y_test)
# MDI(X_train_r, Y_train)
# MDI(X_test_r, Y_test)

# Volume       -0.086304
# Vtr20        -0.074799
# atr          -0.036880
# 4H_atr       -0.034183
# Volatility   -0.032739
# VV           -0.029241
# MAV          -0.027062
# bb_sq        -0.026743
# St4H         -0.000787
# vroci         0.001508
# Vol_Vol       0.002309
# bb_l          0.007215
# bb_t          0.011117
# VtrD3         0.016977
# t             0.025272
# cusum         0.025922
# DVol          0.029245
# Close         0.034211
# srl_corr      0.034575
# VtrD6         0.034594
# VtrD9         0.034940
# VtrD13        0.037194
# 4Hmacd        0.040728
# MAV_signal    0.048659
# Tr9           0.060032
# macd          0.060129
# StD           0.066072
# Tr13          0.066608
# Tr20          0.066754
# Tr6           0.068013
# TrD20         0.073629
# TrD13         0.079352
# roci          0.083812
# TrD6          0.085459
# TrD9          0.086478
# TrD3          0.088669
# %D            0.090452
# 4H%K          0.091111
# bb_cross      0.094711
# %K            0.094719
# rsi           0.095081
# 4H%D          0.095699
# 4H_rsi        0.111352
# roc50         0.130287
# diff          0.137621
# roc40         0.140987
# roc30         0.151631
# bin           0.584978
# ret           1.000000

# Correlation ----------------------------------------------------------
# Volume       -0.092885
# Vtr20        -0.076732
# 4H_atr       -0.036621
# atr          -0.034200
# bb_sq        -0.029708
# Volatility   -0.026502
# St4H         -0.019650
# VV           -0.018278
# MAV          -0.018059
# Vol_Vol       0.000395
# bb_l          0.005333
# vroci         0.008556
# bb_t          0.010779
# VtrD3         0.016515
# t             0.022297
# cusum         0.023162
# DVol          0.029391
# srl_corr      0.033463
# Close         0.033841
# VtrD6         0.035793
# VtrD13        0.037748
# VtrD9         0.038017
# 4Hmacd        0.049410
# MAV_signal    0.059515
# macd          0.070345
# TrD20         0.071013
# Tr9           0.074897
# Tr13          0.078704
# StD           0.079996
# Tr6           0.080411
# TrD13         0.083584
# Tr20          0.084315
# TrD6          0.084756
# roci          0.087890
# TrD3          0.090504
# TrD9          0.090866
# 4H%K          0.093019
# %D            0.101553
# 4H%D          0.102815
# rsi           0.105423
# %K            0.107567
# bb_cross      0.107597
# 4H_rsi        0.116528
# diff          0.147765
# roc50         0.148514
# roc40         0.160210
# roc30         0.165716
# bin           0.586398
# ret           1.000000

# Volume       -0.116321
# Volatility   -0.102454
# atr          -0.097538
# Vtr20        -0.096477
# bb_sq        -0.095991
# MAV          -0.094891
# 4H_atr       -0.079143
# St4H         -0.019297
# t            -0.015661
# cusum        -0.014043
# Vol_Vol      -0.006333
# VV           -0.005987
# Close         0.001185
# vroci         0.003976
# bb_l          0.015437
# VtrD3         0.017801
# bb_t          0.017944
# VtrD6         0.033062
# 4Hmacd        0.039465
# VtrD9         0.039797
# srl_corr      0.039884
# MAV_signal    0.043691
# VtrD13        0.049397
# TrD20         0.062645
# TrD13         0.067049
# roci          0.074943
# StD           0.075119
# DVol          0.081449
# TrD9          0.082104
# TrD6          0.095260
# macd          0.098222
# 4H%K          0.100736
# 4H%D          0.105957
# Tr6           0.106723
# %D            0.106881
# Tr9           0.110854
# %K            0.111168
# bb_cross      0.111366
# 4H_rsi        0.117409
# rsi           0.118706
# TrD3          0.118753
# Tr13          0.120022
# Tr20          0.123260
# diff          0.172846
# roc50         0.173188
# roc40         0.190464
# roc30         0.208518
# bin           0.677573
# ret           1.000000