from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from data_forming import events_data, full_data, X_train, X_test, Y_train, Y_test
import numpy as np
import pandas as pd
# https://hudsonthames.org/meta-labeling-a-toy-example/
events_data.drop(columns=['Close', 'Open', 'High', 'Low', 'Volume', 'Dema9', '4H%K', 'momentum',
                          'elder', 'ret'],
                 axis=1, inplace=True)

# modelPrime = KNeighborsClassifier()
# modelPrime.fit(X_train, Y_train)
# prime_predictions = modelPrime.predict(X_test)
#
# meta_df = pd.DataFrame()
# meta_df['actual'] = Y_test
# meta_df['predicted'] = prime_predictions
# meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)
#
# meta_labels = meta_df.iloc[:, 2]

train_set = events_data[:int(len(events_data) * 0.8)]
research_data1 = train_set[:int(len(train_set) * 0.5)]
research_data2 = train_set[int(len(train_set) * 0.5):]
test_data = events_data[int(len(events_data) * 0.8):]
print(len(research_data1), len(research_data2), len(test_data))

signal = 'bin'

Y_meta_train_A = research_data1.loc[:, signal]
Y_meta_train_A.name = Y_meta_train_A.name
X_meta_train_A = research_data1.loc[:, research_data1.columns != signal]
Y_meta_train_A = research_data1.loc[:, Y_meta_train_A.name]
X_meta_train_A = research_data1.loc[:, X_meta_train_A.columns]

Y_meta_train_B = research_data2.loc[:, signal]
Y_meta_train_B.name = Y_meta_train_B.name
X_meta_train_B = research_data2.loc[:, research_data2.columns != signal]
Y_meta_train_B = research_data2.loc[:, Y_meta_train_B.name]
X_meta_train_B = research_data2.loc[:, X_meta_train_B.columns]

Y_meta_test = test_data.loc[:, signal]
Y_meta_test.name = Y_meta_test.name
X_meta_test = test_data.loc[:, test_data.columns != signal]
Y_meta_test = test_data.loc[:, Y_meta_test.name]
X_meta_test = test_data.loc[:, X_meta_test.columns]

meta_backtest_data = full_data[X_meta_test.index[0]:X_meta_test.index[-1]]

modelPrime = KNeighborsClassifier()
modelPrime.fit(X_meta_train_A, Y_meta_train_A)
prime_predictions = modelPrime.predict(X_meta_train_B)

# X_meta_train_B['PredA'] = prime_predictions
meta_df = pd.DataFrame()
meta_df['actual'] = Y_meta_train_B
meta_df['predicted'] = prime_predictions
meta_df['meta'] = meta_df.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

meta_labels = meta_df.iloc[:, 2]

modelMeta = KNeighborsClassifier()
modelMeta.fit(X_meta_train_B, meta_labels)
meta_predictions = modelMeta.predict(X_meta_test)
test_pred = modelPrime.predict(X_meta_test)

predictions = modelPrime.predict(X_meta_test)
print(classification_report(Y_meta_test, predictions, target_names=['no_trade', 'trade']))
print(classification_report(Y_meta_test, meta_predictions, target_names=['no_trade', 'trade']))
print(classification_report(test_pred, meta_predictions, target_names=['prime', 'meta']))

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
