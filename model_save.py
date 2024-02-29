from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import normalizer, spliter
from data_forming import full_data, events_data, signal
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
import joblib

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


# https://hudsonthames.org/meta-labeling-a-toy-example/

part = 0
events_dataBuy = events_data.copy()  # .loc[events_data['bb_cross'] != 0]
BuyFeatures = ['TrD3', '4H%K', 'bb_cross']

events_dataSell = events_data.copy()  # .loc[events_data['bb_cross'] != 0]
SellFeatures = ['TrD9', 'St4H', 'bb_cross']
XB, YB = spliter(events_dataBuy, signal, part, BuyFeatures)
XS, YS = spliter(events_dataSell, signal, part, SellFeatures)

# Train buy model ------------------------------------------------------------------------------------
print('Buy model ---------------------------------------------------------------------------------------------------')
print('event 0', np.sum(np.array(events_dataBuy[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataBuy[signal]) == 1, axis=0))
# X_trainBuy, X_testBuy, Y_trainBuy, Y_testBuy = spliter(events_dataBuy, signal, part, BuyFeatures)
X_trainBuy_c, = XB.copy()

if 'bb_cross' in X_trainBuy_c.columns:
    print('bb_cross in X')
    X_trainBuy_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_trainBuy_n = normalizer(X_trainBuy_c)
    X_trainBuy_n['bb_cross'] = XB.bb_cross
else:
    X_trainBuy_n = normalizer(X_trainBuy_c)
ModelBuy = MLPClassifier()
ModelBuy.fit(X_trainBuy_n, YB)

# META ---------------------------------------------------------------------------------------------------------------
print('META ----------------------------------------------------------')
X_train_ABuy, Y_train_ABuy = X_trainBuy_n[:int(len(X_trainBuy_n) * 0.5)], YB[:int(len(YB) * 0.5)]
X_train_BBuy, Y_train_BBuy = X_trainBuy_n[int(len(X_trainBuy_n) * 0.5):], YB[int(len(YB) * 0.5):]
PrimeModelBuy = MLPClassifier()
PrimeModelBuy.fit(X_train_ABuy, Y_train_ABuy)
prime_predictionsBuy = PrimeModelBuy.predict(X_train_BBuy)
X_train_BBuy['predA'] = prime_predictionsBuy

meta_dfBuy = pd.DataFrame()
meta_dfBuy['actual'] = Y_train_BBuy
meta_dfBuy['predicted'] = prime_predictionsBuy
meta_dfBuy['meta'] = meta_dfBuy.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaBuy = meta_dfBuy.iloc[:, 2]

MetaModelBuy = MLPClassifier()
MetaModelBuy.fit(X_train_BBuy, Y_train_metaBuy)


# Train sell model ---------------------------------------------------------------------------------------
print('Sell model ------------------------------------------------------------------------------------------------')
print('event 0', np.sum(np.array(events_dataSell[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_dataSell[signal]) == 1, axis=0))
X_trainSell_c = XS.copy()

if 'bb_cross' in X_trainSell_c.columns:
    print('bb_cross in X')
    X_trainSell_c.drop(columns=['bb_cross'], axis=1, inplace=True)
    X_trainSell_n = normalizer(X_trainSell_c)
    X_trainSell_n['bb_cross'] = XS.bb_cross
else:
    X_trainSell_n = normalizer(X_trainSell_c)

ModelSell = MLPClassifier()
ModelSell.fit(X_trainSell_n, YS)

# META ----------------------------------------------------------------------------------------------------------------
print('META ----------------------------------------------------------')
X_train_ASell, Y_train_ASell = X_trainSell_n[:int(len(X_trainSell_n) * 0.5)], YS[:int(len(YS) * 0.5)]
X_train_BSell, Y_train_BSell = X_trainSell_n[int(len(X_trainSell_n) * 0.5):], YS[int(len(YS) * 0.5):]

PrimeModelSell = MLPClassifier()
PrimeModelSell.fit(X_train_ASell, Y_train_ASell)
prime_predictionsS = PrimeModelSell.predict(X_train_BSell)

X_train_BSell['predA'] = prime_predictionsS

meta_dfSell = pd.DataFrame()
meta_dfSell['actual'] = Y_train_BSell
meta_dfSell['predicted'] = prime_predictionsS
meta_dfSell['meta'] = meta_dfSell.apply(lambda x: 1 if x['actual'] == x['predicted'] else 0, axis=1)

Y_train_metaS = meta_dfSell.iloc[:, 2]

MetaModelSell = MLPClassifier()
MetaModelSell.fit(X_train_BSell, Y_train_metaS)

# Save the trained model as a pickle string.
# saved_prime_modelBuy = joblib.dump(PrimeModelBuy, 'PrimeModelBuy.pkl')
# saved_meta_modelBuy = joblib.dump(MetaModelBuy, 'MetaModelBuy.pkl')
# saved_prime_modelSell = joblib.dump(PrimeModelSell, 'PrimeModelSell.pkl')
# saved_meta_modelSell = joblib.dump(MetaModelSell, 'MetaModelSell.pkl')

# Load the pickled model
# knn_from_pickle = pickle.loads(saved_model)
