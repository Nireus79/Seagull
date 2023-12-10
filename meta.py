from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from toolbox import spliter
from data_forming import events_data, full_data, part, signal
import numpy as np
import pandas as pd

# https://hudsonthames.org/meta-labeling-a-toy-example/
# Train sell model ---------------------------------------------------------------------------------------
print('Sell model ----------------------------')
feats_to_drop = ['4H_Low', '4H_atr', 'Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility',
                 'Dema9', 'Dema13', '4Hmacd']
X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feats_to_drop)

X_train_A, Y_train_A = X_train[:int(len(X_train) * 0.5)],  Y_train[:int(len(Y_train) * 0.5)]
X_train_B, Y_train_B = X_train[int(len(X_train) * 0.5):], Y_train[int(len(Y_train) * 0.5):]
print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('X.columns', X_train.columns)

PrimeModelSell = KNeighborsClassifier()
PrimeModelSell.fit(X_train_A, Y_train_A)
prime_predictions = PrimeModelSell.predict(X_train_B)
test_set_pred = PrimeModelSell.predict(X_test)

meta_df = pd.DataFrame()
meta_df['actual'] = Y_train_B
meta_df['predicted'] = prime_predictions
meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_meta = meta_df.iloc[:, 2]

test_meta_df = pd.DataFrame()
test_meta_df['actual'] = Y_test
test_meta_df['predicted'] = test_set_pred
test_meta_df['meta'] = test_meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_meta = test_meta_df.iloc[:, 2]

MetaModelSell = KNeighborsClassifier()
MetaModelSell.fit(X_train_B, Y_train_meta)
test_set_meta_pred = MetaModelSell.predict(X_test)
print(classification_report(Y_test, test_set_pred, target_names=['no_trade', 'trade']))
print(classification_report(Y_test_meta, test_set_meta_pred, target_names=['False', 'True']))

# Train buy model ------------------------------------------------------------------------------------
print('Buy model ----------------------------')
feats_to_drop = ['4H_Low', '4H_atr', 'Close', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility',
                 '4H%K', '4H%D']
events_data = events_data.loc[events_data['bb_cross'] != 0]
X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feats_to_drop)

X_train_A, Y_train_A = X_train[:int(len(X_train) * 0.5)],  Y_train[:int(len(Y_train) * 0.5)]
X_train_B, Y_train_B = X_train[int(len(X_train) * 0.5):], Y_train[int(len(Y_train) * 0.5):]
print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('X.columns', X_train.columns)

PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_A, Y_train_A)
prime_predictions = PrimeModelBuy.predict(X_train_B)
test_set_pred = PrimeModelBuy.predict(X_test)

meta_df = pd.DataFrame()
meta_df['actual'] = Y_train_B
meta_df['predicted'] = prime_predictions
meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_meta = meta_df.iloc[:, 2]

test_meta_df = pd.DataFrame()
test_meta_df['actual'] = Y_test
test_meta_df['predicted'] = test_set_pred
test_meta_df['meta'] = test_meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_meta = test_meta_df.iloc[:, 2]

MetaModelBuy = MLPClassifier()
MetaModelBuy.fit(X_train_B, Y_train_meta)
test_set_meta_pred = MetaModelBuy.predict(X_test)
print(classification_report(Y_test, test_set_pred, target_names=['no_trade', 'trade']))
print(classification_report(Y_test_meta, test_set_meta_pred, target_names=['False', 'True']))
# One train set -------------------------------------------------------------------------------------------------------

# Prime_model = KNeighborsClassifier()
# Prime_model.fit(X_train, Y_train)
# prime_predictions = Prime_model.predict(X_test)
#
# meta_df = pd.DataFrame()
# meta_df['actual'] = Y_test
# meta_df['predicted'] = prime_predictions
# meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# Y_meta = meta_df.iloc[:, 2]
#
# Meta_model = KNeighborsClassifier()
# Meta_model.fit(X_test, Y_meta)
# meta_predictions = Meta_model.predict(X_train)
#
# print(classification_report(Y_test, prime_predictions, target_names=['no_trade', 'trade']))
# print(classification_report(Y_train, meta_predictions, target_names=['0', '1']))

# Two train sets ------------------------------------------------------------------------------------------------------
# train_set = events_data[:int(len(events_data) * 0.9)]
# train_data1 = train_set[:int(len(train_set) * 0.5)]
# train_data2 = train_set[int(len(train_set) * 0.5):]
# test_data = events_data[int(len(events_data) * 0.9):]
# print(len(train_data1), len(train_data2), len(test_data))
#
# signal = 'bin'
#
# Y_train_A = train_data1.loc[:, signal]
# Y_train_A.name = Y_train_A.name
# X_train_A = train_data1.loc[:, train_data1.columns != signal]
# Y_train_A = train_data1.loc[:, Y_train_A.name]
# X_train_A = train_data1.loc[:, X_train_A.columns]
#
# Y_train_B = train_data2.loc[:, signal]
# Y_train_B.name = Y_train_B.name
# X_train_B = train_data2.loc[:, train_data2.columns != signal]
# Y_train_B = train_data2.loc[:, Y_train_B.name]
# X_train_B = train_data2.loc[:, X_train_B.columns]
#
# Y_test = test_data.loc[:, signal]
# Y_test.name = Y_test.name
# X_test = test_data.loc[:, test_data.columns != signal]
# Y_test = test_data.loc[:, Y_test.name]
# X_test = test_data.loc[:, X_test.columns]
#
# meta_backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
#
# PrimeModel = KNeighborsClassifier()
# PrimeModel.fit(X_train_A, Y_train_A)
# prime_predictions = PrimeModel.predict(X_train_B)
# test_set_pred = PrimeModel.predict(X_test)
#
#
# # X_train_B['PredA'] = prime_predictions
# # X_test['PredA'] = test_set_pred
#
# meta_df = pd.DataFrame()
# meta_df['actual'] = Y_train_B
# meta_df['predicted'] = prime_predictions
# meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# Y_train_meta = meta_df.iloc[:, 2]
#
# test_meta_df = pd.DataFrame()
# test_meta_df['actual'] = Y_test
# test_meta_df['predicted'] = test_set_pred
# test_meta_df['meta'] = test_meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# Y_test_meta = test_meta_df.iloc[:, 2]
#
# MetaModel = KNeighborsClassifier()
# MetaModel.fit(X_train_B, Y_train_meta)
# test_set_meta_pred = MetaModel.predict(X_test)
#
#
# print(classification_report(Y_test, test_set_pred, target_names=['no_trade', 'trade']))
#
# print(classification_report(Y_test_meta, test_set_meta_pred, target_names=['0', '1']))

# 3 models ------------------------------------------------------------------------------------------------------------
# modelPrime = MLPClassifier()
# # model1 = LogisticRegression()
# modelPrime.fit(X1, Y1)
# prime_predictions = modelPrime.predict(X2)
# X2['Meta'] = prime_predictions
# X20 = X2.loc[X2['Meta'] == 0]
# Y20 = Y2.loc[X20.index]
# X21 = X2.loc[X2['Meta'] == 1]
# Y21 = Y2.loc[X21.index]
#
#
# modelMetaNeg = MLPClassifier()
# # model2 = LogisticRegression()
# modelMetaNeg.fit(X20, Y20)
#
# modelMetaPos = MLPClassifier()
# # model2 = LogisticRegression()
# modelMetaPos.fit(X21, Y21)
#
# X3['Meta'] = modelPrime.predict(X3)
# X30 = X3.loc[X3['Meta'] == 0]
# X31 = X3.loc[X3['Meta'] == 1]
# Y30 = Y3.loc[X30.index]
# Y31 = Y3.loc[X31.index]
#
# negPred = modelMetaNeg.predict(X30)
# posPred = modelMetaPos.predict(X31)
#
#
# print(classification_report(Y2, prime_predictions, target_names=['no_trade', 'trade']))
# print(classification_report(Y30, negPred, target_names=['no_trade', 'trade']))
# print(classification_report(Y31, posPred, target_names=['no_trade', 'trade']))
