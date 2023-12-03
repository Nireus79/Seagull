from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_forming import research_data, full_data

research_data.drop(columns=['Open', 'High', 'Low', 'Volume', 'bb_cross', 'ret'], axis=1, inplace=True)
train_set = research_data[:int(len(research_data) * 0.8)]
research_data1 = train_set[:int(len(train_set) * 0.5)]
research_data2 = train_set[int(len(train_set) * 0.5):]
test_data = research_data[int(len(research_data) * 0.8):]
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


model1 = MLPClassifier(
    activation='tanh',
    alpha=0.0006,
    hidden_layer_sizes=(100,),
    learning_rate='adaptive',
    solver='adam'
)
# model1 = LogisticRegression(solver='saga')
model1.fit(X1, Y1)
predictions1 = model1.predict(X2)
X2['Pseudo'] = predictions1
X2['True'] = Y2
X2['Meta'] = X2.apply(lambda x: 1 if x['Pseudo'] == x['True'] else 0, axis=1)
X2.drop(columns=['Pseudo', 'True'], axis=1, inplace=True)

model2 = MLPClassifier(
    activation='tanh',
    alpha=0.0006,
    hidden_layer_sizes=(100,),
    learning_rate='adaptive',
    solver='adam'
)
# model2 = LogisticRegression(solver='saga')
model2.fit(X2, Y2)
X3['Meta'] = model1.predict(X3)
predictions2 = model2.predict(X3)
print(X1.columns)
print(X2.columns)
print(X3.columns)
print(classification_report(Y2, predictions1, target_names=['no_trade', 'trade']))
print(classification_report(Y3, predictions2, target_names=['no_trade', 'trade']))
