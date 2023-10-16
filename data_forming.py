import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff
from ta.volatility import average_true_range
from toolbox import asset_merger, primary_asset_merger, data_merger, rescaler, normalizer, standardizer, volatility
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, dropLabels, getEvents, getBins, \
    bbands, get_up_cross_bol, get_down_cross_bol, df_rolling_autocorr, returns, applyPtSlOnT1, mpPandasObj

# https://data.binance.vision/
doteur_csv = 'C:/Users/themi/Desktop/Jonathon/doteur1h/'
btceur_csv = 'C:/Users/themi/Desktop/Jonathon/btceur1h/'
etheur_csv = 'C:/Users/themi/Desktop/Jonathon/etheur1h/'
eurusd_csv = 'C:/Users/themi/Desktop/Jonathon/eurusd1h/'

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# merged_doteur = primary_asset_merger(doteur_csv)
# merged_doteur.to_csv('dot_eur_20_23_hours.csv')
# merged_btceur = asset_merger(btceur_csv, 'btceur')
# merged_btceur.to_csv('btc_eur_20_23_hours.csv')
# merged_etheur = asset_merger(etheur_csv, 'etheur')
# merged_etheur.to_csv('eth_eur_20_23_hours.csv')
# merged_eurusd = asset_merger(eurusd_csv, 'eurusd')
# merged_eurusd.to_csv('eur_usd_20_23_hours.csv')

dot = pd.read_csv('csv/dot_eur_20_23_hours.csv')
dot.time = pd.to_datetime(dot.time, unit='ms')
dot.set_index('time', inplace=True)
eth = pd.read_csv('csv/eth_eur_20_23_hours.csv')
eth.time = pd.to_datetime(eth.time, unit='ms')
eth.set_index('time', inplace=True)
bit = pd.read_csv('csv/btc_eur_20_23_hours.csv')
bit.time = pd.to_datetime(bit.time, unit='ms')
bit.set_index('time', inplace=True)
eur = pd.read_csv('csv/eur_usd_20_23_hours.csv')
eur.time = pd.to_datetime(eur.time, unit='ms')
eur.set_index('time', inplace=True)

# print(dot)
# print(eth)
# print(bit)
# print(eur)

cpus = 1
ptsl = [2, 1]  # profit-taking and stop loss limit multipliers
minRet = 0.01  # The minimum target return required for running a triple barrier search
window = 24
asset1 = 'etheur'
asset2 = 'btceur'
asset3 = 'eurusd'

data = dot
data[asset1 + '_close'] = eth[asset1 + '_close']
# data[asset2 + '_close'] = bit[asset2 + '_close']
# data[asset3 + '_close'] = eur[asset3 + '_close']
data['ema9'] = data['Close'].rolling(9).mean()
# data['ema13'] = data['Close'].rolling(13).mean()
# data['ema20'] = data['Close'].rolling(20).mean()
# data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['%K'] = stoch(data['High'], data['Low'], data['Close'],
#                    window=14, smooth_window=3, fillna=False)
# data['%D'] = data['%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'],
#                                  window=14, fillna=False)
data['price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=1)
data.drop(columns=['price'], axis=1, inplace=True)
data['volatility'] = getDailyVol(data['Close'], window, 1)
# data['diff'] = np.log(data['Close']).diff()
# training data
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
# data['spos'] = data.apply(lambda x: 1 if max(0., x['diff']) > data['volatility'].mean() else 0, axis=1)
# data['sneg'] = data.apply(lambda x: -1 if min(0., x['diff']) < -data['volatility'].mean() else 0, axis=1)
# data['bb_up'] = data.apply(lambda x: 1 if x['Close'] > x['upper'] else 0, axis=1)
# data['bb_down'] = data.apply(lambda x: -1 if x['Close'] < x['lower'] else 0, axis=1)

tEvents = getTEvents(data['Close'], h=data['volatility'].mean())
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=1)
events = getEvents(data['Close'], tEvents, ptsl, data['volatility'], minRet, cpus, t1, side=None)

labels = getBins(events, data['Close'])
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
# data['bin'] = clean_labels['bin']
data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]
full_data = data.copy()
# training_data = data.copy()
training_data = data.loc[events.index]  # cusum + bb
training_data.dropna(axis=0, inplace=True)
# data = data.loc[data.apply(lambda x: x.spos == 1 or x.sneg == -1, axis=1)]

# data = standardizer(data)
# data = normalizer(data)  # only negative output ret
# data = rescaler(data, minmax=(-1, 1))

prediction = 'ret'  # 'bin'
Y = training_data.loc[:, prediction]
Y.name = Y.name
X = training_data.loc[:, ('Close', 'etheur_close', 'ema9', 'volatility')]
# dataset = pd.concat([Y, X], axis=1)

Y = training_data.loc[:, Y.name]
X = training_data.loc[:, X.columns]

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
test_data = full_data[X_test.index[0]:]
print(test_data)


# print(data)
# print(training_data)
# print(training_data.isnull().sum())
# print(training_data)
# print('len training_data: ', len(data))
# print('total ret', np.sum(np.array(training_data.ret) != 0, axis=0))
# print('positive ret', np.sum(np.array(training_data.ret) > 0, axis=0))
# print('negative ret', np.sum(np.array(training_data.ret) < 0, axis=0))
# print(X_test)
