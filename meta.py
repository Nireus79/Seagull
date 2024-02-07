from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, normalizer
from data_forming import full_data, events_data, signal
import numpy as np
import pandas as pd

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

part = 5
# https://hudsonthames.org/meta-labeling-a-toy-example/

# Two train sets

events_dataSell = events_data.copy()  # .loc[events_data['bb_cross'] != 0]
SellFeatures = ['bb_cross', 'bb_l', 'TrD3']

events_dataBuy = events_data.copy().loc[events_data['bb_cross'] != 0]
BuyFeatures = ['St4H', 'TrD3']

# Train sell model ---------------------------------------------------------------------------------------
print('Sell model ------------------------------------------------------------------------------------------------')
X_trainSell, X_testSell, Y_trainSell, Y_testSell = spliter(events_dataSell, signal, part, SellFeatures)
X_trainSell_c, X_testSell_c = X_trainSell.copy(), X_testSell.copy()
X_trainSell_n, X_testSell_n = normalizer(X_trainSell_c), normalizer(X_testSell_c)
if 'bb_cross' in X_trainSell.columns:
    print('bb_cross in X')
    X_trainSell_n.bb_cross, X_testSell_n.bb_cross = X_trainSell.bb_cross, X_testSell.bb_cross

X_train_ASell, Y_train_ASell = X_trainSell_n[:int(len(X_trainSell) * 0.5)], Y_trainSell[:int(len(Y_trainSell) * 0.5)]
X_train_BSell, Y_train_BSell = X_trainSell_n[int(len(X_trainSell) * 0.5):], Y_trainSell[int(len(Y_trainSell) * 0.5):]
print('event 0', np.sum(np.array(events_dataSell[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataSell[signal]) == 1, axis=0))
print('X.columns', X_trainSell.columns)

PrimeModelSell = MLPClassifier()
PrimeModelSell.fit(X_train_ASell, Y_train_ASell)
prime_predictionsS = PrimeModelSell.predict(X_train_BSell)
test_set_predS = PrimeModelSell.predict(X_testSell)
X_train_BSell['predA'] = prime_predictionsS
X_testSell['predA'] = test_set_predS

meta_dfSell = pd.DataFrame()
meta_dfSell['actual'] = Y_train_BSell
meta_dfSell['predicted'] = prime_predictionsS
meta_dfSell['meta'] = meta_dfSell.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaS = meta_dfSell.iloc[:, 2]

test_meta_dfS = pd.DataFrame()
test_meta_dfS['actual'] = Y_testSell
test_meta_dfS['predicted'] = test_set_predS
test_meta_dfS['meta'] = test_meta_dfS.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_metaS = test_meta_dfS.iloc[:, 2]

MetaModelSell = MLPClassifier()
MetaModelSell.fit(X_train_BSell, Y_train_metaS)
test_set_meta_predS = MetaModelSell.predict(X_testSell)
print(classification_report(Y_testSell, test_set_predS, target_names=['0', '1']))
print(classification_report(Y_test_metaS, test_set_meta_predS, target_names=['0', '1']))

# Train buy model ------------------------------------------------------------------------------------
print('Buy model ---------------------------------------------------------------------------------------------------')

X_trainBuy, X_testBuy, Y_trainBuy, Y_testBuy = spliter(events_dataBuy, signal, part, BuyFeatures)
X_trainBuy_c, X_testBuy_c = X_trainBuy.copy(), X_testBuy.copy()
X_trainBuy_n, X_testBuy_n = normalizer(X_trainBuy_c), normalizer(X_testBuy_c)
if 'bb_cross' in X_trainBuy.columns:
    print('bb_cross in X')
    X_trainBuy_n.bb_cross, X_testBuy_n.bb_cross = X_trainBuy.bb_cross, X_testBuy.bb_cross

X_train_ABuy, Y_train_ABuy = X_trainBuy_n[:int(len(X_trainBuy) * 0.5)], Y_trainBuy[:int(len(Y_trainBuy) * 0.5)]
X_train_BBuy, Y_train_BBuy = X_trainBuy_n[int(len(X_trainBuy) * 0.5):], Y_trainBuy[int(len(Y_trainBuy) * 0.5):]
print('event 0', np.sum(np.array(events_dataBuy[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataBuy[signal]) == 1, axis=0))
print('X.columns', X_trainBuy.columns)

PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_ABuy, Y_train_ABuy)
prime_predictionsBuy = PrimeModelBuy.predict(X_train_BBuy)
test_set_predBuy = PrimeModelBuy.predict(X_testBuy)
X_train_BBuy['predA'] = prime_predictionsBuy
X_testBuy['predA'] = test_set_predBuy

meta_dfBuy = pd.DataFrame()
meta_dfBuy['actual'] = Y_train_BBuy
meta_dfBuy['predicted'] = prime_predictionsBuy
meta_dfBuy['meta'] = meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaBuy = meta_dfBuy.iloc[:, 2]

test_meta_dfBuy = pd.DataFrame()
test_meta_dfBuy['actual'] = Y_testBuy
test_meta_dfBuy['predicted'] = test_set_predBuy
test_meta_dfBuy['meta'] = test_meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_metaBuy = test_meta_dfBuy.iloc[:, 2]

MetaModelBuy = MLPClassifier()
MetaModelBuy.fit(X_train_BBuy, Y_train_metaBuy)
test_set_meta_predBuy = MetaModelBuy.predict(X_testBuy)
print(classification_report(Y_testBuy, test_set_predBuy, target_names=['0', '1']))
print(classification_report(Y_test_metaBuy, test_set_meta_predBuy, target_names=['0', '1']))

if len(X_testSell) >= len(X_testBuy):
    backtest_data = full_data[X_testBuy.index[0]:X_testBuy.index[-1]]
else:
    backtest_data = full_data[X_testSell.index[0]:X_testSell.index[-1]]

# one train set (over fitting metamodel)
# Train sell model ---------------------------------------------------------------------------------------
# print('Sell model --------------------------------------------------------------------------------------------------')
# feats_to_dropS = ['Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'Dema3',
#                   '%K', '%D', '4H%D', 'rsi', 'Volatility']
# X_trainSell, X_testSell, Y_trainSell, Y_testSell = spliter(events_data, signal, part, feats_to_dropS)
# print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
# print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
# print('X.columns', X_trainSell.columns)
#
# PrimeModelSell = MLPClassifier()
# PrimeModelSell.fit(X_trainSell, Y_trainSell)
#
# test_set_predSell = PrimeModelSell.predict(X_testSell)
# X_testSell['predA'] = test_set_predSell
#
# meta_dfSell = pd.DataFrame()
# meta_dfSell['actual'] = Y_testSell
# meta_dfSell['predicted'] = test_set_predSell
# meta_dfSell['meta'] = meta_dfSell.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# Y_train_metaSell = meta_dfSell.iloc[:, 2]
#
# MetaModelSell = MLPClassifier()
# MetaModelSell.fit(X_testSell, Y_train_metaSell)
# print(classification_report(Y_testSell, test_set_predSell, target_names=['no_trade', 'trade']))
#
# print('Buy model ---------------------------------------------------------------------------------------------------')
# feats_to_dropBuy = ['Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'Dema3',
#                     '%K', '%D', '4H%D', 'rsi', 'Volatility']
# events_dataBuy = events_data.loc[events_data['bb_cross'] != 0]
# X_trainBuy, X_testBuy, Y_trainBuy, Y_testBuy = spliter(events_dataBuy, signal, part, feats_to_dropBuy)
#
# print('event 0', np.sum(np.array(events_dataBuy[signal]) == 0, axis=0))
# print('event 1', np.sum(np.array(events_dataBuy[signal]) == 1, axis=0))
# print('X.columns', X_trainBuy.columns)
#
# PrimeModelBuy = MLPClassifier()
# PrimeModelBuy.fit(X_trainBuy, Y_trainBuy)
# test_set_predBuy = PrimeModelBuy.predict(X_testBuy)
# X_testBuy['predA'] = test_set_predBuy
#
# meta_dfBuy = pd.DataFrame()
# meta_dfBuy['actual'] = Y_testBuy
# meta_dfBuy['predicted'] = test_set_predBuy
# meta_dfBuy['meta'] = meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# Y_train_metaBuy = meta_dfBuy.iloc[:, 2]
#
# MetaModelBuy = MLPClassifier()
# MetaModelBuy.fit(X_testBuy, Y_train_metaBuy)
# print(classification_report(Y_testBuy, test_set_predBuy, target_names=['no_trade', 'trade']))
