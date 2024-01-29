from toolbox import spliter, normalizer
from data_forming import full_data, events_data, signal
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, f_classif, SelectPercentile
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import seaborn as sns
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm

part = 5
features = ['Tr9', 'Tr20', 'TrD3', 'TrD20', '4H%K', 'bb_sq']

X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feature_columns=features)
backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
X_train_c, X_test_c = X_train.copy(), X_test.copy()
X_train_n, X_test_n = normalizer(X_train_c), normalizer(X_test_c)
if 'bb_cross' in X_train.columns:
    print('bb_cross in X')
    X_train_n.bb_cross, X_test_n.bb_cross = X_train.bb_cross, X_test.bb_cross

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

    x = np.arange(1, len(X_train_n))
    y = rfe_f1score_list

    ax.bar(x, y, width=0.2)
    ax.set_xlabel('Number of features selected using RFE')
    ax.set_ylabel('F1-Score (weighted)')
    ax.set_ylim(0, 1.2)
    ax.set_xticks(np.arange(1, len(X_train_n)))
    ax.set_xticklabels(np.arange(1, len(X_train_n)), fontsize=12)

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
    selected_features = X_train_n.columns[selected_features_mask]
    print('selected_features:', selected_features)


signal = 'bin'
# events_data_c = events_data.copy()
# events_data_n = normalizer(events_data_c)
# events_data_n[signal] = events_data[signal]
# events_data_n.bb_cross = events_data.bb_cross


# Variance(events_data)
# Correlation(events_data, signal)
K_best(X_train_n, Y_train, X_test_n, Y_test)
# SelectPrsnt()
# rfe(X_train_c, Y_train_c, X_test_c, Y_test_c)
# Boruta(X_train_c, Y_train_c, X_test_c, Y_test_c)
