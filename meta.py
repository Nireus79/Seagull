from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_forming import events_data, full_data
import numpy as np
import pandas as pd

events_data.drop(columns=['Close', 'Open', 'High', 'Low', 'Volume', 'Dema9', '4H%K', 'momentum',
                            'elder', 'ret'],
                 axis=1, inplace=True)
train_set = events_data[:int(len(events_data) * 0.9)]
research_data1 = train_set[:int(len(train_set) * 0.5)]
research_data2 = train_set[int(len(train_set) * 0.5):]
test_data = events_data[int(len(events_data) * 0.9):]
print(len(research_data1), len(research_data2), len(test_data))

signal = 'bin'

Y1 = research_data1.loc[:, signal]
Y1.name = Y1.name
X1 = research_data1.loc[:, research_data1.columns != signal]
Y1 = research_data1.loc[:, Y1.name]
X1 = research_data1.loc[:, X1.columns]

Y2 = research_data2.loc[:, signal]
Y2.name = Y2.name
X2 = research_data2.loc[:, research_data2.columns != signal]
Y2 = research_data2.loc[:, Y2.name]
X2 = research_data2.loc[:, X2.columns]

Y3 = test_data.loc[:, signal]
Y3.name = Y3.name
X3 = test_data.loc[:, test_data.columns != signal]
Y3 = test_data.loc[:, Y3.name]
X3 = test_data.loc[:, X3.columns]

meta_backtest_data = full_data[X3.index[0]:X3.index[-1]]

modelPrime = MLPClassifier()
modelPrime.fit(X1, Y1)
prime_predictions = modelPrime.predict(X2)

X2['Pseudo'] = prime_predictions
X2['Actual'] = Y2
X2['Meta'] = X2.apply(lambda x: 1 if x['Pseudo'] == x['Actual'] else 0, axis=1)
X2.drop(columns=['Pseudo', 'Actual'], axis=1, inplace=True)

# X2['Meta'] = prime_predictions

modelMeta = MLPClassifier()
modelMeta.fit(X2, Y2)

X3['Meta'] = modelPrime.predict(X3)
meta_predictions = modelMeta.predict(X3)

print(classification_report(Y2, prime_predictions, target_names=['no_trade', 'trade']))
print(classification_report(Y3, meta_predictions, target_names=['no_trade', 'trade']))

# 3 models
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
