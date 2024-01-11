from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, standardizer
from data_forming import events_data, part, signal
import numpy as np
import pandas as pd
import itertools

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def model_test(ftd):
    X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, ftd)
    X_train, X_test = standardizer(X_train), standardizer(X_test)
    print('X.columns', X_train.columns)

    Model = MLPClassifier()
    Model.fit(X_train, Y_train)
    predictions = Model.predict(X_test)
    report = classification_report(Y_test, predictions, target_names=['0', '1'])
    print(report)
    return report


def uniqueCombinations(list_elements, plethos):
    lst = list(itertools.combinations(list_elements, plethos))
    s = set(lst)
    # print('actual', len(l), l)
    return list(s)


features = ['ema6', 'macd', '4Hmacd', '%K', '4H%K', '%D', '4H%D', '%DS', '4H%DS',
            'rsi', '4H_rsi', 'diff', 'roc30', 'TrD3', 'bb_sq', 'bb_l', 'bb_t']

features_to_drop = ['Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Dema3', 'Volatility']

combinations = uniqueCombinations(features, 14)
reports = []
# X.columns Index(['TrD3', 'bb_t'], dtype='object')
#               precision    recall  f1-score   support
#
#            0       0.59      0.95      0.72       130
#            1       0.72      0.17      0.28       105
#
#     accuracy                           0.60       235
#    macro avg       0.65      0.56      0.50       235
# weighted avg       0.65      0.60      0.52       235
# X.columns Index(['bb_l', 'bb_t'], dtype='object')
#               precision    recall  f1-score   support
#
#            0       0.55      0.99      0.71       130
#            1       0.00      0.00      0.00       105
#
#     accuracy                           0.55       235
#    macro avg       0.28      0.50      0.35       235
# weighted avg       0.30      0.55      0.39       235

# X.columns Index(['bb_sq', 'bb_t'], dtype='object')
#               precision    recall  f1-score   support
#
#            0       0.55      0.99      0.71       130
#            1       0.00      0.00      0.00       105
#
#     accuracy                           0.55       235
#    macro avg       0.28      0.50      0.35       235
# weighted avg       0.30      0.55      0.39       235

# X.columns Index(['rsi', 'roc30', 'TrD3', 'bb_t'], dtype='object')
#               precision    recall  f1-score   support
#
#            0       0.73      0.79      0.76       130
#            1       0.71      0.64      0.67       105
#
#     accuracy                           0.72       235
#    macro avg       0.72      0.72      0.72       235
# weighted avg       0.72      0.72      0.72       235
for i in combinations:
    for a in range(len(i) - 1):
        features_to_drop.append(i[a])
    reports.append((i, model_test(features_to_drop)))
    for r in range(len(i) - 1):
        features_to_drop.remove(features_to_drop[-1])

# feats_to_drop_t = ['Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Dema3', 'Volatility',
#                    'macd', '4Hmacd', 'ema6', '%DS', 'bb_t', 'diff', '4H%DS', 'roc30', '4H_rsi']

# bbst 2
#          5                 4                 3                 2                 1
# 4H%K     0.049219 macd     0.034713 bb_l     0.039098 ema20    0.042637 bb_sq    0.035683
# ema20    0.035830 4H%D     0.032897 roc20    0.037382 roc20    0.032365 4H%K     0.032544
# %K       0.032249 roc20    0.031864 %D       0.035171 bb_sq    0.031898 mom10    0.022986
# ema3     0.028960 bb_sq    0.031496 bb_sq    0.034236 ema3     0.028970 roc10    0.022618
# roc20    0.027495 Tr20     0.030354 4H%K     0.027882 %D       0.028580 TrD13    0.021423
# bb_sq    0.024118 %D       0.022615 4Hmacd   0.022074 %DS      0.023525 roc30    0.019424
# %D       0.021553 4H%K     0.019527 rsi      0.015712 adx      0.023427 TrD3     0.018838
# 4H_rsi   0.017729 roc10    0.017473 4H%DS    0.015288 4H_rsi   0.019724 macd     0.018600
# TrD13    0.016955 adx      0.014461 4H%D     0.013341 H4_ema6  0.019213 bb_l     0.017851
# mom20    0.016256 ema3     0.014274 %K       0.012651 %K       0.015168 4H%DS    0.017294

# bbst 3
#             5                        4                  3                  2                  1
# roc30       9.568736e-02 roc30       0.049310 roc30     0.064834 %DS       0.051512 roc30     0.073358
# %DS         5.264076e-02 4H%D        0.049021 %K        0.043826 roc10     0.037872 %K        0.063652
# diff        3.673265e-02 diff        0.048229 4Hmacd    0.041243 cusum     0.036472 bb_t      0.047946
# StD         3.602626e-02 bb_l        0.035527 diff      0.034507 bb_sq     0.028896 bb_l      0.042156
# bb_t        3.499322e-02 %DS         0.032056 bb_sq     0.026264 %K        0.024983 4Hmacd    0.035880
# 4Hmacd      3.041183e-02 bb_t        0.029900 4H_atr    0.023679 4H%K      0.013060 cusum     0.029223
# 4H%D        3.018110e-02 4Hmacd      0.029170 %DS       0.022564 4Hmacd    0.012418 %DS       0.026976
# atr         2.515620e-02 cusum       0.025840 mom10     0.020301 diff      0.011505 4H%D      0.026674
# 4H_atr      2.443527e-02 bb_sq       0.023816 roc10     0.019044 mom10     0.010021 atr       0.026145
# 4H_rsi      2.310832e-02 4H_atr      0.020510 rsi       0.017265 4H_atr    0.006828 StD       0.023579
# srl_corr    2.147385e-02 %K          0.018407 4H%D      0.016186 4H%D      0.003278 srl_corr  0.020388
# mom20       1.877908e-02 macd        0.017509 cusum     0.016144 bb_t      0.000359 4H%K      0.020308
