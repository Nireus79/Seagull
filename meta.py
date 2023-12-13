from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from toolbox import spliter
from data_forming import events_data, part, signal
import numpy as np
import pandas as pd

# https://hudsonthames.org/meta-labeling-a-toy-example/


# Two train sets
# Train sell model ---------------------------------------------------------------------------------------
print('Sell model --------------------------------------------------------------------------------------------------')
feats_to_dropSell = ['4H_Low', 'Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'Dema9',
                     'T_diff', 'M_diff']
feats_to_dropBuy = ['4H_Low', 'Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'Dema9',
                    '4H%K', '4H%D']
X_trainSell, X_testSell, Y_trainSell, Y_testSell = spliter(events_data, signal, part, feats_to_dropSell)

X_train_ASell, Y_train_ASell = X_trainSell[:int(len(X_trainSell) * 0.5)], Y_trainSell[:int(len(Y_trainSell) * 0.5)]
X_train_BSell, Y_train_BSell = X_trainSell[int(len(X_trainSell) * 0.5):], Y_trainSell[int(len(Y_trainSell) * 0.5):]
print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
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

events_dataBuy = events_data.loc[events_data['bb_cross'] != 0]
X_trainBuy, X_testBuy, Y_trainBuy, Y_testBuy = spliter(events_dataBuy, signal, part, feats_to_dropBuy)

X_train_ABuy, Y_train_ABuy = X_trainBuy[:int(len(X_trainBuy) * 0.5)], Y_trainBuy[:int(len(Y_trainBuy) * 0.5)]
X_train_BBuy, Y_train_BBuy = X_trainBuy[int(len(X_trainBuy) * 0.5):], Y_trainBuy[int(len(Y_trainBuy) * 0.5):]
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

# one train set (over fitting meta model)
# # Train sell model ---------------------------------------------------------------------------------------
# print('Sell model --------------------------------------------------------------------------------------------------')
# feats_to_dropS = ['4H_Low', '4H_atr', 'Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility',
#                   'Dema9']
# X_trainSell, X_testSell, Y_trainSell, Y_testSell = spliter(events_data, signal, part, feats_to_dropS)
# print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
# print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
# print('X.columns', X_trainSell.columns)
#
# PrimeModelSell = KNeighborsClassifier()
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
# MetaModelSell = KNeighborsClassifier()
# MetaModelSell.fit(X_testSell, Y_train_metaSell)
# print(classification_report(Y_testSell, test_set_predSell, target_names=['no_trade', 'trade']))
#
# print('Buy model ---------------------------------------------------------------------------------------------------')
# feats_to_dropBuy = ['4H_Low', '4H_atr', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', '4H%K', '4H%D']
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
