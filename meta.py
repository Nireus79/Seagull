from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from data_forming import research_data

research_data1 = research_data[:300]
research_data2 = research_data[300:600]
test_data = research_data[600:]
signal = 'bin'

Y1 = research_data1.loc[:, signal]
Y1.name = Y1.name
X1 = research_data1.loc[:, research_data1.columns != signal]
Y1 = research_data1.loc[:, Y1.name]
X1 = research_data1.loc[:, X1.columns]

Y2 = research_data1.loc[:, signal]
Y2.name = Y2.name
X2 = research_data2.loc[:, research_data2.columns != signal]
Y2 = research_data2.loc[:, Y2.name]
X2 = research_data2.loc[:, X2.columns]

Y3 = test_data.loc[:, signal]
Y3.name = Y3.name
X3 = test_data.loc[:, test_data.columns != signal]
Y3 = test_data.loc[:, Y3.name]
X3 = test_data.loc[:, X3.columns]


model1 = MLPClassifier()
model1.fit(X1, Y1)

X2['Pseudo'] = model1.predict(X2)
X2['True'] = Y2
X2['Meta'] = X2.apply(lambda x: 1 if x['Pseudo'] == x['True'] else 0, axis=1)
X2.drop(columns=['Pseudo', 'True'], axis=1, inplace=True)

model2 = MLPClassifier()
model2.fit(X2, Y2)
X3['Meta'] = model1.predict(X3)

predictions = model2.predict(X3)

print(classification_report(Y3, predictions, target_names=['no_trade', 'trade']))
