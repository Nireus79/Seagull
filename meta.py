from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import normalizer, spliter
from data_forming import full_data, events_data, signal, delta
import numpy as np
import pandas as pd
import joblib

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# https://hudsonthames.org/meta-labeling-a-toy-example/

part = 10
events_dataBuy = events_data.copy().loc[events_data['bb_cross'] != 0]
BuyFeatures = ['TrD3', 'vdiff', 'Volatility', 'bb_cross']
# ['TrD3', 'TrD9', 'St4H', '4H%K', '4H_atr', 'bb_cross']
# ['TrD3', 'macd', 'bb_cross']  # ['TrD3', 'macd', 'bb_cross']

events_dataSell = events_data.copy().loc[events_data['bb_cross'] != 0]
SellFeatures = ['TrD6', '4H_atr', 'MAV', 'bb_cross']
# ['TrD6', 'TrD13', 'Tr13', '%K', 'atr', 'bb_cross']
# ['TrD3', 'TrD13', 'St4H', '4H_atr', 'bb_sq', 'bb_cross']
# ['TrD20', 'diff', 'bb_cross']  # ['TrD9', '4Hmacd', 'vrsi', 'macd']

# Train buy model ------------------------------------------------------------------------------------
print('Buy model ---------------------------------------------------------------------------------------------------')
print('event 0', np.sum(np.array(events_dataBuy[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataBuy[signal]) == 1, axis=0))
X_trainBuy, X_testBuy, Y_trainBuy, Y_testBuy = spliter(events_dataBuy, signal, part, BuyFeatures, delta)
X_trainBuy_c, X_testBuy_c = X_trainBuy.copy(), X_testBuy.copy()

if 'bb_cross' in X_trainBuy_c.columns:
    print('bb_cross in X')
    X_trainBuy_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_testBuy_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_trainBuy_n, X_testBuy_n = normalizer(X_trainBuy_c), normalizer(X_testBuy_c)
    X_trainBuy_n['bb_cross'], X_testBuy_n['bb_cross'] = X_trainBuy.bb_cross, X_testBuy.bb_cross
else:
    X_trainBuy_n, X_testBuy_n = normalizer(X_trainBuy_c), normalizer(X_testBuy_c)

ModelBuy = MLPClassifier()
ModelBuy.fit(X_trainBuy_n, Y_trainBuy)
PredictionsBuy = ModelBuy.predict(X_testBuy_n)
print(classification_report(Y_testBuy, PredictionsBuy, target_names=['0', '1']))
# META ---------------------------------------------------------------------------------------------------------------
print('META ----------------------------------------------------------')
X_train_ABuy, Y_train_ABuy = X_trainBuy_n[:int(len(X_trainBuy) * 0.5)], Y_trainBuy[:int(len(Y_trainBuy) * 0.5)]
X_train_BBuy, Y_train_BBuy = X_trainBuy_n[int(len(X_trainBuy) * 0.5):], Y_trainBuy[int(len(Y_trainBuy) * 0.5):]
PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_ABuy, Y_train_ABuy)
prime_predictionsBuy = PrimeModelBuy.predict(X_train_BBuy)
test_set_predBuy = PrimeModelBuy.predict(X_testBuy_n)
X_train_BBuy['predA'] = prime_predictionsBuy
X_testBuy_n['predA'] = test_set_predBuy

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
test_set_meta_predBuy = MetaModelBuy.predict(X_testBuy_n)
print(classification_report(Y_testBuy, test_set_predBuy, target_names=['0', '1']))
print(classification_report(Y_test_metaBuy, test_set_meta_predBuy, target_names=['0', '1']))

# Train sell model ---------------------------------------------------------------------------------------
print('Sell model ------------------------------------------------------------------------------------------------')
print('event 0', np.sum(np.array(events_dataSell[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataSell[signal]) == 1, axis=0))
X_trainSell, X_testSell, Y_trainSell, Y_testSell = spliter(events_dataSell, signal, part, SellFeatures, delta)
X_trainSell_c, X_testSell_c = X_trainSell.copy(), X_testSell.copy()

if 'bb_cross' in X_trainSell_c.columns:
    print('bb_cross in X')
    X_trainSell_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_testSell_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_trainSell_n, X_testSell_n = normalizer(X_trainSell_c), normalizer(X_testSell_c)
    X_trainSell_n['bb_cross'], X_testSell_n['bb_cross'] = X_trainSell.bb_cross, X_testSell.bb_cross
else:
    X_trainSell_n, X_testSell_n = normalizer(X_trainSell_c), normalizer(X_testSell_c)
ModelSell = MLPClassifier()
ModelSell.fit(X_trainSell_n, Y_trainSell)
PredictionsSell = ModelSell.predict(X_testSell_n)
print(classification_report(Y_testSell, PredictionsSell, target_names=['0', '1']))
# META ----------------------------------------------------------------------------------------------------------------
print('META ----------------------------------------------------------')
X_train_ASell, Y_train_ASell = X_trainSell_n[:int(len(X_trainSell) * 0.5)], Y_trainSell[:int(len(Y_trainSell) * 0.5)]
X_train_BSell, Y_train_BSell = X_trainSell_n[int(len(X_trainSell) * 0.5):], Y_trainSell[int(len(Y_trainSell) * 0.5):]

PrimeModelSell = MLPClassifier()
PrimeModelSell.fit(X_train_ASell, Y_train_ASell)
prime_predictionsS = PrimeModelSell.predict(X_train_BSell)
test_set_predS = PrimeModelSell.predict(X_testSell_n)
X_train_BSell['predA'] = prime_predictionsS
X_testSell_n['predA'] = test_set_predS

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
test_set_meta_predS = MetaModelSell.predict(X_testSell_n)
print(classification_report(Y_testSell, test_set_predS, target_names=['0', '1']))
print(classification_report(Y_test_metaS, test_set_meta_predS, target_names=['0', '1']))

if len(X_testSell) >= len(X_testBuy):
    backtest_data = full_data[X_testBuy.index[0]:X_testBuy.index[-1]]
else:
    backtest_data = full_data[X_testSell.index[0]:X_testSell.index[-1]]

# Save the trained model as a pickle string.
# saved_prime_modelBuy = joblib.dump(PrimeModelBuy, 'PrimeModelBuy.pkl')
# saved_meta_modelBuy = joblib.dump(MetaModelBuy, 'MetaModelBuy.pkl')
# saved_prime_modelSell = joblib.dump(PrimeModelSell, 'PrimeModelSell.pkl')
# saved_meta_modelSell = joblib.dump(MetaModelSell, 'MetaModelSell.pkl')

# Load the pickled model
# knn_from_pickle = pickle.loads(saved_model)
