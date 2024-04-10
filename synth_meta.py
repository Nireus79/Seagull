import pandas as pd
from toolbox import normalizer
from data_forming import events_data, signal
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

train_data = pd.read_csv('csv/synth/synth1000.csv')
test_data = pd.read_csv('csv/synth/events_data.csv')
# train_data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
train_data = train_data[['TrD3', '4Hmacd', 'mom20', 'bb_l', 'bin']]
# test_data = events_data
test_data = test_data[['TrD3', '4Hmacd', 'mom20', 'bb_l', 'bin']]


# print(train_data)
# print(test_data)
Y_train = train_data[signal]
X_train = train_data.drop(columns=[signal])
X_train = normalizer(X_train)
Y_test = test_data[signal]
X_test = test_data.drop(columns=[signal])
X_test = normalizer(X_test)

X_train_A, Y_train_A = X_train[:int(len(X_train) * 0.5)], Y_train[:int(len(Y_train) * 0.5)]
X_train_B, Y_train_B = X_train[int(len(X_train) * 0.5):], Y_train[int(len(Y_train) * 0.5):]
PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_A, Y_train_A)
prime_predictionsBuy = PrimeModelBuy.predict(X_train_B)
test_set_predBuy = PrimeModelBuy.predict(X_test)
X_train_B['predA'] = prime_predictionsBuy
X_test['predA'] = test_set_predBuy

meta_dfBuy = pd.DataFrame()
meta_dfBuy['actual'] = Y_train_B
meta_dfBuy['predicted'] = prime_predictionsBuy
meta_dfBuy['meta'] = meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaBuy = meta_dfBuy.iloc[:, 2]

test_meta_dfBuy = pd.DataFrame()
test_meta_dfBuy['actual'] = Y_test
test_meta_dfBuy['predicted'] = test_set_predBuy
test_meta_dfBuy['meta'] = test_meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_test_metaBuy = test_meta_dfBuy.iloc[:, 2]

MetaModelBuy = MLPClassifier()
MetaModelBuy.fit(X_train_B, Y_train_metaBuy)
test_set_meta_predBuy = MetaModelBuy.predict(X_test)
print(classification_report(Y_test, test_set_predBuy, target_names=['0', '1']))
print(classification_report(Y_test_metaBuy, test_set_meta_predBuy, target_names=['0', '1']))

