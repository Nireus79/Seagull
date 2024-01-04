import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff, macd_signal, macd, adx
from ta.volatility import average_true_range
from toolbox import rescaler, normalizer, standardizer, ROC, MOM, spliter, crossing_elder, crossing3
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, dropLabels, \
    getEvents, bbands, metaBins, df_rolling_autocorr, returns, getBins
import warnings

warnings.filterwarnings('ignore')

# https://data.binance.vision/
# https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/Labeling%20and%20MetaLabeling%20for%20Supervised%20Classification.ipynb

# db = pd.read_csv('csv/db/ETHEUR_10mdb')
# db.set_index('time', inplace=True)
# print(db)


eth5m = pd.read_csv('csv/tb/ETHEUR_5m.csv')
# btc5m = pd.read_csv('csv/tb/BTCEUR_5m.csv')

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

eth5m.time = pd.to_datetime(eth5m.time, unit='ms')
eth5m.set_index('time', inplace=True)
eth5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# btc5m.time = pd.to_datetime(btc5m.time, unit='ms')
# btc5m.set_index('time', inplace=True)
# btc5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

ohlc = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}
eth30m = eth5m.resample('30min').apply(ohlc)
eth4h = eth5m.resample('4H').apply(ohlc)
eth1D = eth5m.resample('D').apply(ohlc)

# btc30m = btc5m.resample('30min').apply(ohlc)
# btc4h = btc5m.resample('4H').apply(ohlc)
# btc1D = btc5m.resample('D').apply(ohlc)

# eth30m['BTC_Close'] = btc30m['Close']
# eth30m['BTC_High'] = btc30m['High']
# eth30m['BTC_Low'] = btc30m['Low']

eth30m['4H_Close'] = eth4h['Close']
eth30m['4H_Low'] = eth4h['Low']
eth30m['4H_High'] = eth4h['High']

# eth30m['BTC4H_Close'] = btc4h['Close']
# eth30m['BTC4H_Low'] = btc4h['Low']
# eth30m['BTC4H_High'] = btc4h['High']

eth30m['1D_Close'] = eth1D['Close']
# eth30m['BTC1D_Close'] = btc1D['Close']

cpus = 1
ptsl = [1, 1]  # profit-taking and stop loss limit multipliers
minRet = c_labels = .01  # The minimum target return(def .01) (volatility) required for running a triple barrier search
delta = 1
span = 100
window = 20
bb_stddev = 2
part = 5

data = eth30m
data.ffill(inplace=True)

# data['ema3'] = data['Close'].rolling(3).mean()
# data['H4_ema3'] = data['4H_Close'].rolling(3).mean()
# data['H4_ema6'] = data['4H_Close'].rolling(6).mean()
data['Dema3'] = data['1D_Close'].rolling(3).mean()
# data['ema6'] = data['Close'].rolling(6).mean()
# data['Dema6'] = data['1D_Close'].rolling(6).mean()
# data['ema9'] = data['Close'].rolling(9).mean()
# data['Dema9'] = data['1D_Close'].rolling(9).mean()

# data['BTCDema3'] = data['BTC1D_Close'].rolling(3).mean()
# data['BTCDema6'] = data['BTC1D_Close'].rolling(6).mean()
# data['BTCDema9'] = data['BTC1D_Close'].rolling(9).mean()

# data['ema13'] = data['Close'].rolling(13).mean()
# data['Dema13'] = data['1D_Close'].rolling(13).mean()
# data['BTCDema13'] = data['BTC1D_Close'].rolling(13).mean()

# data['ema20'] = data['Close'].rolling(20).mean()
# data['Dema20'] = data['1D_Close'].rolling(20).mean()
# data['BTCDema20'] = data['BTC1D_Close'].rolling(20).mean()

# data['adx'] = adx(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['4Hmacd'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
data['%D'] = data['%K'].rolling(3).mean()
data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()
# data['4H%DS'] = data['4H%D'].rolling(3).mean()
# data['BTC4H%K'] = stoch(data['BTC4H_High'], data['BTC4H_Low'], data['BTC4H_Close'],
#                      window=14, smooth_window=3, fillna=False)
# data['BTC4H%D'] = data['BTC4H%K'].rolling(3).mean()
# data['BTC4H%DS'] = data['BTC4H%D'].rolling(3).mean()
data['rsi'] = rsi(data['Close'], window=14, fillna=False)
# data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
# data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
data['diff'] = np.log(data['Close']).diff()
# data['cusum'] = data['Close'].cumsum()
data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
# data['roc10'] = ROC(data['Close'], 10)
data['roc20'] = ROC(data['Close'], 20)
# data['roc30'] = ROC(data['Close'], 30)
data['mom10'] = MOM(data['Close'], 10)
# data['mom20'] = MOM(data['Close'], 20)
# data['mom30'] = MOM(data['Close'], 30)
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)
bb_sides = crossing3(data, 'Close', 'upper', 'lower')
# elder_sides = crossing_elder(data, '4H%K', '4H%D')
data['bb_cross'] = bb_sides
# data['bbc'] = bb_sides
data['Volatility'] = getDailyVol(data['Close'], span, delta).rolling(window).mean()

# data['Tr3_20'] = data.apply(lambda x: x['ema3'] - x['ema20'], axis=1)
data['TrD3'] = data.apply(lambda x: x['Close'] - x['Dema3'], axis=1)
# data['TrD6'] = data.apply(lambda x: x['Close'] - x['Dema6'], axis=1)
# data['TrD9'] = data.apply(lambda x: x['Close'] - x['Dema9'], axis=1)
# data['TrD13'] = data.apply(lambda x: x['Close'] - x['Dema13'], axis=1)
# data['TrD20'] = data.apply(lambda x: x['Close'] - x['Dema20'], axis=1)
# data['BTCTrD3'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema3'], axis=1)
# data['BTCTrD6'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema6'], axis=1)
# data['BTCTrD9'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema9'], axis=1)
# data['BTCTrD13'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema13'], axis=1)
# data['BTCTrD20'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema20'], axis=1)
# data['StD'] = data.apply(lambda x: x['4H%D'] - x['4H%DS'], axis=1)
# data['BTCStD'] = data.apply(lambda x: x['BTC4H%D'] - x['BTC4H%DS'], axis=1)
tEvents = getTEvents(data['Close'], h=data['Volatility'])
t1 = addVerticalBarrier(tEvents, data['Close'], delta)

events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=bb_sides)
# labels = getBins(events, data['Close'])
labels = metaBins(events, data.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())
data.drop(columns=['1D_Close', '4H_Close', '4H_Low', '4H_High', 'Price', 'ave', 'upper', 'lower'],
          axis=1, inplace=True)
# data.drop(columns=['BTC1D_Close', 'BTC4H_Close', 'BTC4H_Low', 'BTC4H_High',
#                    'BTC_High', 'BTC_Low', 'BTC_Close'], axis=1, inplace=True)
feats_to_drop = ['Close', 'Open', 'High', 'Low', 'Volume', 'Volatility', 'bb_cross',
                 'Dema3', 'macd', '%K', '4H%K', '4H%D', 'mom10']

full_data = data.copy()

events_data = data.loc[events.index]
# events_data = events_data.loc[events_data['bb_cross'] != 0]

# signal = 'ret'
signal = 'bin'

X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feats_to_drop)
backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
X_train, X_test = standardizer(X_train), standardizer(X_test)
# X_train, X_test = normalizer(X_train), normalizer(X_test)
# X_train, X_test = rescaler(X_train, (0, 1)), rescaler(X_test, (0, 1))

# BALANCE CLASSES (down sampling)
# minority = events_data[events_data[signal] == 1]
# majority = events_data[events_data[signal] == 0].sample(n=len(minority), replace=True)
# events_data = pd.concat([minority, majority])
# print(events_data)

print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('event data min ret', events_data.ret.min())
print('event data max ret', events_data.ret.max())

# print('full_data.columns', full_data.columns)
print('X.columns', X_train.columns)
