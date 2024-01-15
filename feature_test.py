from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, standardizer
from data_forming import events_data
import pandas as pd
import itertools

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

signal = 'bin'
part = 5


def model_test(ftd):
    X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, ftd)
    X_train, X_test = standardizer(X_train), standardizer(X_test)
    # print('X.columns', X_train.columns)

    Model = MLPClassifier()
    Model.fit(X_train, Y_train)
    predictions = Model.predict(X_test)
    report = classification_report(Y_test, predictions, target_names=['0', '1'], output_dict=True)
    # print(report)
    return report


def uniqueCombinations(list_elements, plethos):
    lst = list(itertools.combinations(list_elements, plethos))
    combs = []
    for e in lst:
        combs.append(list(e))
    return combs


# full_features = ['Dema3', 'ema6', 'macd', '4Hmacd', '%K', '4H%K', '%D', '4H%D', '%DS', '4H%DS', 'rsi', '4H_rsi',
#                  'diff', 'roc30', 'bb_cross', 'Volatility', 'TrD3', 'bb_sq', 'bb_l',
#                  'bb_t']
def report_generator():
    full_features = ['ema6', 'macd', '4Hmacd', '%K', '4H%K', '%D', '4H%D', '%DS', '4H%DS', 'rsi', '4H_rsi',
                     'diff', 'roc30', 'Volatility', 'TrD3', 'bb_sq', 'bb_l', 'bb_t']

    combinations = uniqueCombinations(full_features, 6)
    print(len(combinations))

    r = []

    for i in combinations:
        rep = model_test(i)
        r.append((i, rep))

    reports = pd.DataFrame({'reports': r})

    reports['features'] = reports.apply(lambda x: x[0][0], axis=1)
    reports['precision0'] = reports.apply(lambda x: x[0][1]['0']['precision'], axis=1)
    reports['recall0'] = reports.apply(lambda x: x[0][1]['0']['recall'], axis=1)
    reports['precision1'] = reports.apply(lambda x: x[0][1]['1']['precision'], axis=1)
    reports['recall1'] = reports.apply(lambda x: x[0][1]['1']['recall'], axis=1)
    reports.drop(columns=['reports'], axis=1, inplace=True)
    print(reports)
    reports.to_csv('6.csv')


# df.loc[df['favcount'].idxmax(), 'sn']
# report_generator()
reps = pd.read_csv('5.csv')
print('max precision0:', reps.loc[reps['precision0'].idxmax()])
print('max recall0:', reps.loc[reps['recall0'].idxmax()])
print('max precision1:', reps.loc[reps['precision1'].idxmax()])
print('max recall1:', reps.loc[reps['recall1'].idxmax()])

# Correlation
# events bb!=0raw            bb!=0 normalized           bb full norm
# BTCVolatility    -0.060824 USDT_cusum       -0.038993 vema13           -0.064105
# BTC_adx          -0.052688 USDT4H_atr       -0.038484 vema20           -0.061622
# USDTVolatility   -0.046719 USDTbb_sq        -0.031173 vema9            -0.060738
# USDT4H_atr       -0.046637 USDT_adx         -0.031116 bb_sq            -0.054016
# Volatility       -0.045244 USDT_atr         -0.030005 vema6            -0.053661
# USDT_atr         -0.043441 Volatility       -0.028296 vema3            -0.045423
# USDT_macd        -0.043059 USDT_roc20       -0.025614 atr              -0.044591
# USDTTrD13        -0.042864 USDTVolatility   -0.024774 4H_atr           -0.036789
# USDTbb_sq        -0.042019 atr              -0.024670 BTC1D_Volume     -0.032610
# USDTTrD9         -0.039902 USDT_mom20       -0.024477 1D_Volume        -0.027897
# USDTTrD6         -0.038757 USDTTrD3         -0.023485 USDT1D_Volume    -0.025164
# USDTTrD3         -0.038034 USDTTrD6         -0.022561 BTC_srl_corr     -0.024978
# adx              -0.037355 USDTTrD9         -0.021857 4H_Volume        -0.023791
# USDT4Hmacd       -0.032253 USDTTrD13        -0.021622 USDT_srl_corr    -0.021420
# USDT_mom10       -0.022924 BTCVolatility    -0.018667 USDT4H_atr       -0.020023
# USDT_roc10       -0.021593 BTC_atr          -0.017679 BTCbb_sq         -0.019713
# USDT_mom20       -0.020701 BTC4Hmacd        -0.017007 BTC_atr          -0.018177
# USDT_roc20       -0.018806 USDT_macd        -0.016544 USDT_cusum       -0.016717
# StD4             -0.017030 BTC4H%K          -0.016349 BTC4Hmacd        -0.016392
# BTCStD4          -0.016072 BTC4H%D          -0.014718 USDT_adx         -0.015746
# USDT_%DS         -0.015262 BTC4H%DS         -0.014601 srl_corr          0.024609
# USDT_%D          -0.014367 BTC_roc10         0.025731 4H%DS             0.025048
# vema20           -0.012116 USDT_%K           0.028119 USDT_%D           0.025073
# BTC_srl_corr     -0.012027 USDT4Hmacd        0.028685 macd              0.026283
# BTC_%D            0.101418 1D_Volume         0.029106 rsi               0.026894
# roc20             0.102903 %K                0.033217 mom10             0.028084
# bb_cross          0.102993 USDT1D_Volume     0.037458 %K                0.028514
# BTC_roc10         0.105403 diff              0.038058 Tr20              0.030001
# rsi               0.106135 StD               0.044528 BTC_cusum         0.030091
# %K                0.107045 4Hmacd            0.045625 USDT4H%K          0.030545
# BTC4H%D           0.107177 roc10             0.048313 USDT4H%DS         0.032531
# %D                0.107432 roc30             0.056910 USDT4H%D          0.032762
# roc30             0.108708 srl_corr          0.057085 roc30             0.033177
# BTC4H_rsi         0.109499 USDTStD           0.057642 mom30             0.034324
# BTC4H%DS          0.110309 roc20             0.064939 roc20             0.036403
# 4H_rsi            0.120573 mom30             0.068567 USDT_%K           0.037066
# roc10             0.120950 Tr20              0.080799 StD               0.037187
# 4H%K              0.124605 mom20             0.081084 mom20             0.037342
# 4H%D              0.131558 mom10             0.081308 USDTStD           0.048767
# 4H%DS             0.132209 macd              0.086425 bb_cross          0.060684
#                            bb_cross          0.102993
# KBest   bb!=0            Percentile bb!=0
# macd           7.469976  macd           7.469976
# mom10          6.733183  mom10          6.733183
# mom20          6.684961  mom20          6.684961
# Tr20           6.612164  Tr20           6.612164
# roc20          5.066358  roc20          5.066358
# mom30          4.805880  mom30          4.805880
# bb_cross       4.404341  bb_cross       4.404341
# USDTStD        4.099248  USDTStD        4.099248
# roc30          3.937385  roc30          3.937385
# srl_corr       3.791461  srl_corr       3.791461
# roc10          2.792061  roc10          2.792061
# StD            2.446352  StD            2.446352
# 4Hmacd         1.847211  4Hmacd         1.847211
# diff           1.752199  diff           1.752199
# USDT4H_atr     1.705298  USDT4H_atr     1.705298
# %K             1.527080  %K             1.527080
# USDT1D_Volume  1.441922  USDT1D_Volume  1.441922
# USDT_%K        1.129477  USDT_%K        1.129477
# USDT_cusum     1.050387  USDT_cusum     1.050387
# USDT_atr       1.031399  USDT_atr       1.031399

# Kbest bbfull
# bb_cross      14.135597
# vema13        9.392936
# vema20        8.912502
# vema9         7.990112
# USDTStD       6.699874
# vema6         6.259626
# bb_sq         6.189401
# atr           5.055012
# vema3         5.027493
# mom20         4.193210
# USDT_%K       4.037766
# StD           3.933027
# roc20         3.746181
# BTC1D_Volume  3.636781
# 4H_atr        3.343666
# mom30         3.276361
# USDT4H%D      3.149400
# roc30         3.115243
# USDT4H%DS     3.098730
# Tr20          2.917827
