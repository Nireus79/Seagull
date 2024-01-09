from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, standardizer
from data_forming import events_data, part, signal
import numpy as np
import pandas as pd

events_data_t = events_data.loc[events_data['bb_cross'] != 0]

feats_to_drop_t = ['Close', 'Open', 'High', 'Low', 'Volume', 'Dema3', 'bb_cross',
                   '4H%D', 'rsi', 'Volatility', 'TrD3', 'Tr20',
                   'StD4', 'StD', 'bb_sq', 'bb_l', 'bb_t']

# Train sell model ---------------------------------------------------------------------------------------
print('Sell model ------------------------------------------------------------------------------------------------')
X_train, X_test, Y_train, Y_test = spliter(events_data_t, signal, part, feats_to_drop_t)
X_train, X_test = standardizer(X_train), standardizer(X_test)

print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('X.columns', X_train.columns)

Model = MLPClassifier()
Model.fit(X_train, Y_train)
predictions = Model.predict(X_test)

print(classification_report(Y_test, predictions, target_names=['0', '1']))
