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

X2['P'] = model1.predict(X2)
X2['T'] = Y2
X2['M'] = X2.apply(lambda x: 1 if x['P'] == x['T'] else 0, axis=1)
X2.drop(columns=['P', 'T'], axis=1, inplace=True)

model2 = MLPClassifier()
model2.fit(X2, Y2)
predictions2 = model2.predict(X2)
# predictions3 = model2.predict(X3)
# X3['P'] = predictions3

print(classification_report(Y3, predictions2, target_names=['no_trade', 'trade']))
# print(classification_report(Y3, predictions3, target_names=['no_trade', 'trade']))
