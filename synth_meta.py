import pandas as pd
from toolbox import normalizer
from data_forming import events_data
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import classification_report

train_data = pd.read_csv('csv/synth/synth10000_00124.csv')
test_data = events_data
# train_data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

features = ['TrD3', 'Volatility', 'TrD6', 'TrD13', 'mom10', 'bb_cross', 'bin', 'ret']

S002612 = ['TrD6', 'TrD13', 'mom10', 'bb_cross']
B002612 = ['TrD3', 'TrD6', 'Volatility', 'bb_cross']
S00124 = ['Volatility', 'TrD3', 'bb_cross', 'srl_corr']
B00124 = ['diff', '4Hmacd', 'srl_corr', 'Tr6', 'TrD3']
R00124 = []
train_data = train_data[features]
test_data = test_data[features]

B = B002612
S = S002612

signalC = 'bin'
signalR = 'ret'
# print(train_data)
# print(test_data)
Y_trainC = train_data[signalC]

Y_trainR = train_data[signalR]
X_train = train_data.drop(columns=[signalC, signalR, 'bb_cross'])
X_train = normalizer(X_train)
X_train['bb_cross'] = train_data['bb_cross']
Y_testC = test_data[signalC]
Y_testR = test_data[signalR]
X_test = test_data.drop(columns=[signalC, signalR, 'bb_cross'])
X_test = normalizer(X_test)
X_test['bb_cross'] = test_data['bb_cross']

# ------------------------------------------------------------------------------------------------------
X_train_AB, Y_train_AB = X_train[:int(len(X_train) * 0.5)][B], Y_trainC[:int(len(Y_trainC) * 0.5)]
X_train_BB, Y_train_BB = X_train[int(len(X_train) * 0.5):][B], Y_trainC[int(len(Y_trainC) * 0.5):]
PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_AB, Y_train_AB)
prime_predictionsBuy = PrimeModelBuy.predict(X_train_BB)
test_set_predBuy = PrimeModelBuy.predict(X_test[B])
X_train_BB['predA'] = prime_predictionsBuy
X_testB = X_test[B]
X_testB['predA'] = test_set_predBuy

meta_dfBuy = pd.DataFrame()
meta_dfBuy['actual'] = Y_train_BB
meta_dfBuy['predicted'] = prime_predictionsBuy
meta_dfBuy['meta'] = meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaBuy = meta_dfBuy.iloc[:, 2]

test_meta_dfBuy = pd.DataFrame()
test_meta_dfBuy['actual'] = Y_testC
test_meta_dfBuy['predicted'] = test_set_predBuy
test_meta_dfBuy['meta'] = test_meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_metaBuy = test_meta_dfBuy.iloc[:, 2]

MetaModelBuy = MLPClassifier()
MetaModelBuy.fit(X_train_BB, Y_train_metaBuy)
test_set_meta_predBuy = MetaModelBuy.predict(X_testB)
# print(classification_report(Y_testC, test_set_predBuy, target_names=['0', '1']))
# print(classification_report(Y_test_metaBuy, test_set_meta_predBuy, target_names=['0', '1']))
# -------------------------------------------------------------------------------------------------------
X_train_AS, Y_train_AS = X_train[:int(len(X_train) * 0.5)][S], Y_trainC[:int(len(Y_trainC) * 0.5)]
X_train_BS, Y_train_BS = X_train[int(len(X_train) * 0.5):][S], Y_trainC[int(len(Y_trainC) * 0.5):]

PrimeModelSell = MLPClassifier()
PrimeModelSell.fit(X_train_AS, Y_train_AS)
prime_predictionsS = PrimeModelSell.predict(X_train_BS)
test_set_predS = PrimeModelSell.predict(X_test[S])
X_train_BS['predA'] = prime_predictionsS
X_testS = X_test[S]
X_testS['predA'] = test_set_predS

meta_dfSell = pd.DataFrame()
meta_dfSell['actual'] = Y_train_BS
meta_dfSell['predicted'] = prime_predictionsS
meta_dfSell['meta'] = meta_dfSell.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaS = meta_dfSell.iloc[:, 2]

test_meta_dfS = pd.DataFrame()
test_meta_dfS['actual'] = Y_testC
test_meta_dfS['predicted'] = test_set_predS
test_meta_dfS['meta'] = test_meta_dfS.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_metaS = test_meta_dfS.iloc[:, 2]

MetaModelSell = MLPClassifier()
MetaModelSell.fit(X_train_BS, Y_train_metaS)
test_set_meta_predS = MetaModelSell.predict(X_testS)
# print(classification_report(Y_testC, test_set_predS, target_names=['0', '1']))
# print(classification_report(Y_test_metaS, test_set_meta_predS, target_names=['0', '1']))
# ---------------------------------------------------------------------------------------------------------
ModelRisk = MLPRegressor()
ModelRisk.fit(X_train[B], Y_trainR)
