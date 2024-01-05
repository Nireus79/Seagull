from data_forming import X_train, X_test, Y_train, Y_test
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from boruta import BorutaPy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
# XVar = X_train.var(axis=0)
# print(XVar)

# K-best ------------------------------------------------------------
X_trainK, X_testK, Y_trainK, Y_testK = X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()
bestfeatures = SelectKBest(mutual_info_classif, k='all')
fit = bestfeatures.fit(X_trainK, Y_trainK)
scores = pd.DataFrame(fit.scores_)
Xcolumns = pd.DataFrame(X_train.columns)
# concat two dataframes for better visualization
featureScores = pd.concat([Xcolumns, scores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print('featureScores--------------------------------------------------------------------------')
print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features


f1_score_list = []
for k in range(1, len(X_trainK.columns)):
    model = MLPClassifier()
    selector = SelectKBest(mutual_info_classif, k=k)
    fit = selector.fit(X_trainK, Y_trainK)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_trainK.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print('featureScores--------------------------------------------------------------------------')
    print(featureScores.nlargest(20, 'Score').set_index('Specs'))  # print 20 best features

    sel_XtrainK = selector.transform(X_trainK)
    sel_XtestK = selector.transform(X_testK)

    model.fit(sel_XtrainK, Y_trainK)
    preds = model.predict(sel_XtestK)
    score = f1_score(Y_testK, preds)
    print(score)
    f1_score_list.append(score)

fig, ax = plt.subplots()

x = np.arange(1, len(X_trainK.columns))
y = f1_score_list

ax.bar(x, y, width=0.2)
ax.set_xlabel('Number of features selected using mutual information')
ax.set_ylabel('F1-Score (weighted)')
ax.set_ylim(0, 1.2)
ax.set_xticks(np.arange(1, len(X_trainK.columns)))
ax.set_xticklabels(np.arange(1, len(X_trainK.columns)), fontsize=12)

for i, v in enumerate(y):
    plt.text(x=i + 1, y=v + 0.05, s=str(v), ha='center')

plt.tight_layout()
plt.show()

# Recursive feature elimination RFE --------------------------------------------------------------------------
# X_trainR, X_testR, Y_trainR, Y_testR = X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()
# gbc = GradientBoostingClassifier(max_depth=5, random_state=42)
# rfe_f1score_list = []
# for k in range(1, len(X_trainR)):
#     RFE_selector = RFE(estimator=gbc, n_features_to_select=k, step=1)
#     RFE_selector.fit(X_trainR, Y_trainR)
#     sel_XtrainR = RFE_selector.transform(X_trainR)
#     sel_XtestR = RFE_selector.transform(X_testR)
#     gbc.fit(sel_XtrainR, Y_trainR)
#     RFE_preds = gbc.predict(sel_XtestR)
#
#     f1_score_rfe = round(f1_score(Y_testR, RFE_preds, average='weighted'), 3)
#     print(k, f1_score_rfe)
#     rfe_f1score_list.append(f1_score_rfe)
#
# fig, ax = plt.subplots()
#
# x = np.arange(1, len(X_trainR))
# y = rfe_f1score_list
#
# ax.bar(x, y, width=0.2)
# ax.set_xlabel('Number of features selected using RFE')
# ax.set_ylabel('F1-Score (weighted)')
# ax.set_ylim(0, 1.2)
# ax.set_xticks(np.arange(1, len(X_trainR)))
# ax.set_xticklabels(np.arange(1, len(X_trainR)), fontsize=12)
#
# for i, v in enumerate(y):
#     plt.text(x=i + 1, y=v + 0.05, s=str(v), ha='center')
#
# plt.tight_layout()
# plt.show()

# BORUTA --------------------------------------------------------------------------------------------
# X_trainB, X_testB, Y_trainB, Y_testB = X_train.copy(), X_test.copy(), Y_train.copy(), Y_test.copy()
# gbc = GradientBoostingClassifier(max_depth=5, random_state=42)
#
# boruta_selector = BorutaPy(gbc, random_state=42)
# boruta_selector.fit(X_trainB.values, Y_trainB.values.ravel())
# sel_XtrainB = boruta_selector.transform(X_trainB.values)
# sel_XtestB = boruta_selector.transform(X_testB.values)
# gbc.fit(sel_XtrainB, Y_trainB)
# boruta_preds = gbc.predict(sel_XtestB)
# boruta_f1_score = round(f1_score(Y_testB, boruta_preds, average='weighted'), 3)
#
# RFE_selector = RFE(estimator=gbc, n_features_to_select=5, step=10)
# RFE_selector.fit(X_trainB, Y_trainB)
# selected_features_mask = boruta_selector.support_
# selected_features = X_trainB.columns[selected_features_mask]
# print('selected_features:', selected_features)

# full selected_features: '%D', 'rsi', 'diff', 'srl_corr', 'roc20', 'TrD3'
# bb != 0 selected_features: 'macd', '4H%D', 'rsi', 'mom10', 'TrD3'
# side. bb != 0: 'macd', '%K', '4H%K', '%D', 'roc10', 'TrD3'
