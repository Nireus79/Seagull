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
btc5m = pd.read_csv('csv/tb/BTCEUR_5m.csv')
usdt5m = pd.read_csv('csv/tb/EURUSDT_5m.csv')


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

eth5m.time = pd.to_datetime(eth5m.time, unit='ms')
eth5m.set_index('time', inplace=True)
eth5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

btc5m.time = pd.to_datetime(btc5m.time, unit='ms')
btc5m.set_index('time', inplace=True)
btc5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

usdt5m.time = pd.to_datetime(usdt5m.time, unit='ms')
usdt5m.set_index('time', inplace=True)
usdt5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

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

btc30m = btc5m.resample('30min').apply(ohlc)
btc4h = btc5m.resample('4H').apply(ohlc)
btc1D = btc5m.resample('D').apply(ohlc)

usdt30m = usdt5m.resample('30min').apply(ohlc)
usdt4h = usdt5m.resample('4H').apply(ohlc)
usdt1D = usdt5m.resample('D').apply(ohlc)

eth30m['BTC_Close'] = btc30m['Close']
eth30m['BTC_High'] = btc30m['High']
eth30m['BTC_Low'] = btc30m['Low']

eth30m['USDT_Close'] = usdt30m['Close']
eth30m['USDT_High'] = usdt30m['High']
eth30m['USDT_Low'] = usdt30m['Low']

eth30m['4H_Close'] = eth4h['Close']
eth30m['4H_Low'] = eth4h['Low']
eth30m['4H_High'] = eth4h['High']
eth30m['4H_Volume'] = eth4h['Volume']

eth30m['BTC4H_Close'] = btc4h['Close']
eth30m['BTC4H_Low'] = btc4h['Low']
eth30m['BTC4H_High'] = btc4h['High']

eth30m['USDT4H_Close'] = usdt4h['Close']
eth30m['USDT4H_Low'] = usdt4h['Low']
eth30m['USDT4H_High'] = usdt4h['High']

eth30m['1D_Close'] = eth1D['Close']
eth30m['1D_Volume'] = eth1D['Volume']
eth30m['BTC1D_Close'] = btc1D['Close']
eth30m['BTC1D_Volume'] = btc1D['Volume']
eth30m['USDT1D_Close'] = usdt1D['Close']
eth30m['USDT1D_Volume'] = usdt1D['Volume']

cpus = 1
ptsl = [1, 1]  # profit-taking / stop loss limit multipliers
minRet = .01  # The minimum target return(def .01) (volatility) required for running a triple barrier search
delta = 1
span = 100
window = 20
bb_stddev = 2
part = 3

data = eth30m
data.ffill(inplace=True)

data['ema3'] = data['Close'].rolling(3).mean()
data['ema6'] = data['Close'].rolling(6).mean()
data['ema9'] = data['Close'].rolling(9).mean()
data['ema13'] = data['Close'].rolling(13).mean()
data['ema20'] = data['Close'].rolling(20).mean()

data['vema3'] = data['Volume'].rolling(3).mean()
data['vema6'] = data['Volume'].rolling(6).mean()
data['vema9'] = data['Volume'].rolling(9).mean()
data['vema13'] = data['Volume'].rolling(13).mean()
data['vema20'] = data['Volume'].rolling(20).mean()

data['adx'] = adx(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['%D'] = data['%K'].rolling(3).mean()
data['%DS'] = data['%D'].rolling(3).mean()
data['rsi'] = rsi(data['Close'], window=14, fillna=False)
data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['diff'] = np.log(data['Close']).diff()
data['cusum'] = data['Close'].cumsum()
data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
data['roc10'] = ROC(data['Close'], 10)
data['roc20'] = ROC(data['Close'], 20)
data['roc30'] = ROC(data['Close'], 30)
data['mom10'] = MOM(data['Close'], 10)
data['mom20'] = MOM(data['Close'], 20)
data['mom30'] = MOM(data['Close'], 30)
data['price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)

data['H4_ema3'] = data['4H_Close'].rolling(3).mean()
data['H4_ema6'] = data['4H_Close'].rolling(6).mean()
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
data['4H%D'] = data['4H%K'].rolling(3).mean()
data['4H%DS'] = data['4H%D'].rolling(3).mean()
data['4Hmacd'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)

data['Dema3'] = data['1D_Close'].rolling(3).mean()
data['Dema6'] = data['1D_Close'].rolling(6).mean()
data['Dema9'] = data['1D_Close'].rolling(9).mean()
data['Dema13'] = data['1D_Close'].rolling(13).mean()
data['Dema20'] = data['1D_Close'].rolling(20).mean()

data['TrD3'] = data.apply(lambda x: x['Close'] - x['Dema3'], axis=1)
data['TrD6'] = data.apply(lambda x: x['Close'] - x['Dema6'], axis=1)
data['TrD9'] = data.apply(lambda x: x['Close'] - x['Dema9'], axis=1)
data['TrD13'] = data.apply(lambda x: x['Close'] - x['Dema13'], axis=1)
data['Tr20'] = data.apply(lambda x: x['Close'] - x['ave'], axis=1)
data['StD4'] = data.apply(lambda x: x['4H%K'] - x['4H%D'], axis=1)
data['StD'] = data.apply(lambda x: x['%K'] - x['%D'], axis=1)
data['bb_sq'] = data.apply(lambda x: x['upper'] - x['lower'], axis=1)
# data['bb_l'] = data.apply(lambda x: (x['upper'] - x['price']) / (x['price'] - x['lower']), axis=1)
# data['bb_t'] = data.apply(lambda x: x['bb_l'] / x['bb_sq'], axis=1)

# BTC ----------------------------------------------------------------------------------------------------------
data['BTC_ema3'] = data['BTC_Close'].rolling(3).mean()
data['BTC_ema6'] = data['BTC_Close'].rolling(6).mean()
data['BTC_ema9'] = data['BTC_Close'].rolling(9).mean()
data['BTC_ema13'] = data['BTC_Close'].rolling(13).mean()
data['BTC_ema20'] = data['BTC_Close'].rolling(20).mean()

data['BTC_adx'] = adx(data['BTC_High'], data['BTC_Low'], data['BTC_Close'], window=14, fillna=False)
data['BTC_macd'] = macd_diff(data['BTC_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['BTC_%K'] = stoch(data['BTC_High'], data['BTC_Low'], data['BTC_Close'], window=14, smooth_window=3, fillna=False)
data['BTC_%D'] = data['BTC_%K'].rolling(3).mean()
data['BTC_%DS'] = data['BTC_%D'].rolling(3).mean()
data['BTC_rsi'] = rsi(data['BTC_Close'], window=14, fillna=False)
data['BTC_atr'] = average_true_range(data['BTC_High'], data['BTC_Low'], data['BTC_Close'], window=14, fillna=False)
data['BTC_diff'] = np.log(data['BTC_Close']).diff()
data['BTC_cusum'] = data['BTC_Close'].cumsum()
data['BTC_srl_corr'] = df_rolling_autocorr(returns(data['BTC_Close']), window=window).rename('BTC_srl_corr')
data['BTC_roc10'] = ROC(data['BTC_Close'], 10)
data['BTC_roc20'] = ROC(data['BTC_Close'], 20)
data['BTC_roc30'] = ROC(data['BTC_Close'], 30)
data['BTC_mom10'] = MOM(data['BTC_Close'], 10)
data['BTC_mom20'] = MOM(data['BTC_Close'], 20)
data['BTC_mom30'] = MOM(data['BTC_Close'], 30)
data['BTC_price'], data['BTC_ave'], data['BTC_upper'], data['BTC_lower'] =\
    bbands(data['BTC_Close'], window=window, numsd=bb_stddev)

data['BTCH4_ema3'] = data['BTC4H_Close'].rolling(3).mean()
data['BTCH4_ema6'] = data['BTC4H_Close'].rolling(6).mean()
data['BTC4H%K'] = stoch(data['BTC4H_High'], data['BTC4H_Low'], data['BTC4H_Close'],
                        window=14, smooth_window=3, fillna=False)
data['BTC4H%D'] = data['BTC4H%K'].rolling(3).mean()
data['BTC4H%DS'] = data['BTC4H%D'].rolling(3).mean()
data['BTC4Hmacd'] = macd_diff(data['BTC4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['BTC4H_rsi'] = rsi(data['BTC4H_Close'], window=14, fillna=False)
data['BTC4H_atr'] = average_true_range(data['BTC4H_High'], data['BTC4H_Low'], data['BTC4H_Close'], window=14, fillna=False)

data['BTCDema3'] = data['BTC1D_Close'].rolling(3).mean()
data['BTCDema6'] = data['BTC1D_Close'].rolling(6).mean()
data['BTCDema9'] = data['BTC1D_Close'].rolling(9).mean()
data['BTCDema13'] = data['BTC1D_Close'].rolling(13).mean()
data['BTCDema20'] = data['BTC1D_Close'].rolling(20).mean()

data['BTCTrD3'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema3'], axis=1)
data['BTCTrD6'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema6'], axis=1)
data['BTCTrD9'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema9'], axis=1)
data['BTCTrD13'] = data.apply(lambda x: x['BTC_Close'] - x['BTCDema13'], axis=1)

data['BTCStD4'] = data.apply(lambda x: x['BTC4H%K'] - x['BTC4H%D'], axis=1)
data['BTCStD'] = data.apply(lambda x: x['BTC_%K'] - x['BTC_%D'], axis=1)
data['BTCbb_sq'] = data.apply(lambda x: x['BTC_upper'] - x['BTC_lower'], axis=1)
# data['BTCbb_l'] = data.apply(lambda x: (x['BTC_upper'] - x['BTC_price']) / (x['BTC_price'] - x['BTC_lower']), axis=1)
# data['BTCbb_t'] = data.apply(lambda x: x['BTCbb_l'] / x['BTCbb_sq'], axis=1)
# USDT ----------------------------------------------------------------------------------------------------------------
data['USDT_ema3'] = data['USDT_Close'].rolling(3).mean()
data['USDT_ema6'] = data['USDT_Close'].rolling(6).mean()
data['USDT_ema9'] = data['USDT_Close'].rolling(9).mean()
data['USDT_ema13'] = data['USDT_Close'].rolling(13).mean()
data['USDT_ema20'] = data['USDT_Close'].rolling(20).mean()

data['USDT_adx'] = adx(data['USDT_High'], data['USDT_Low'], data['USDT_Close'], window=14, fillna=False)
data['USDT_macd'] = macd_diff(data['USDT_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['USDT_%K'] = stoch(data['USDT_High'], data['USDT_Low'], data['USDT_Close'],
                        window=14, smooth_window=3, fillna=False)
data['USDT_%D'] = data['USDT_%K'].rolling(3).mean()
data['USDT_%DS'] = data['USDT_%D'].rolling(3).mean()
data['USDT_rsi'] = rsi(data['USDT_Close'], window=14, fillna=False)
data['USDT_atr'] = average_true_range(data['USDT_High'], data['USDT_Low'], data['USDT_Close'], window=14, fillna=False)
data['USDT_diff'] = np.log(data['USDT_Close']).diff()
data['USDT_cusum'] = data['USDT_Close'].cumsum()
data['USDT_srl_corr'] = df_rolling_autocorr(returns(data['USDT_Close']), window=window).rename('USDT_srl_corr')
data['USDT_roc10'] = ROC(data['USDT_Close'], 10)
data['USDT_roc20'] = ROC(data['USDT_Close'], 20)
data['USDT_roc30'] = ROC(data['USDT_Close'], 30)
data['USDT_mom10'] = MOM(data['USDT_Close'], 10)
data['USDT_mom20'] = MOM(data['USDT_Close'], 20)
data['USDT_mom30'] = MOM(data['USDT_Close'], 30)
data['USDT_price'], data['USDT_ave'], data['USDT_upper'], data['USDT_lower'] =\
    bbands(data['USDT_Close'], window=window, numsd=bb_stddev)

data['USDTH4_ema3'] = data['USDT4H_Close'].rolling(3).mean()
data['USDTH4_ema6'] = data['USDT4H_Close'].rolling(6).mean()
data['USDT4H%K'] = stoch(data['USDT4H_High'], data['USDT4H_Low'], data['USDT4H_Close'],
                         window=14, smooth_window=3, fillna=False)
data['USDT4H%D'] = data['USDT4H%K'].rolling(3).mean()
data['USDT4H%DS'] = data['USDT4H%D'].rolling(3).mean()
data['USDT4Hmacd'] = macd_diff(data['USDT4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['USDT4H_rsi'] = rsi(data['USDT4H_Close'], window=14, fillna=False)
data['USDT4H_atr'] = average_true_range(data['USDT4H_High'], data['USDT4H_Low'], data['USDT4H_Close'],
                                        window=14, fillna=False)

data['USDTDema3'] = data['USDT1D_Close'].rolling(3).mean()
data['USDTDema6'] = data['USDT1D_Close'].rolling(6).mean()
data['USDTDema9'] = data['USDT1D_Close'].rolling(9).mean()
data['USDTDema13'] = data['USDT1D_Close'].rolling(13).mean()
data['USDTDema20'] = data['USDT1D_Close'].rolling(20).mean()

data['USDTTrD3'] = data.apply(lambda x: x['USDT_Close'] - x['USDTDema3'], axis=1)
data['USDTTrD6'] = data.apply(lambda x: x['USDT_Close'] - x['USDTDema6'], axis=1)
data['USDTTrD9'] = data.apply(lambda x: x['USDT_Close'] - x['USDTDema9'], axis=1)
data['USDTTrD13'] = data.apply(lambda x: x['USDT_Close'] - x['USDTDema13'], axis=1)

data['USDTStD4'] = data.apply(lambda x: x['USDT4H%K'] - x['USDT4H%D'], axis=1)
data['USDTStD'] = data.apply(lambda x: x['USDT_%K'] - x['USDT_%D'], axis=1)
data['USDTbb_sq'] = data.apply(lambda x: x['USDT_upper'] - x['USDT_lower'], axis=1)
# data['USDTbb_l'] = data.apply(lambda x: (x['USDT_upper'] - x['USDT_price']) / (x['USDT_price'] - x['USDT_lower']),
#                               axis=1)
# data['USDTbb_t'] = data.apply(lambda x: x['USDTbb_l'] / x['USDTbb_sq'], axis=1)

bb_sides = crossing3(data, 'Close', 'upper', 'lower')
# elder_sides = crossing_elder(data, '4H%K', '4H%D')
data['bb_cross'] = bb_sides
data['Volatility'] = getDailyVol(data['Close'], span, delta).rolling(window).mean()
data['BTCVolatility'] = getDailyVol(data['BTC_Close'], span, delta).rolling(window).mean()
data['USDTVolatility'] = getDailyVol(data['USDT_Close'], span, delta).rolling(window).mean()

tEvents = getTEvents(data['Close'], h=data['Volatility'])
t1 = addVerticalBarrier(tEvents, data['Close'], delta)

events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=bb_sides)
labels = metaBins(events, data.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())
data.drop(columns=['1D_Close', '4H_Close', '4H_Low', '4H_High', 'price', 'ave', 'upper', 'lower', 'Open', 'High', 'Low']
          , axis=1, inplace=True)
data.drop(columns=['BTC1D_Close', 'BTC4H_Close', 'BTC4H_Low', 'BTC4H_High', 'BTC_High', 'BTC_Low', 'BTC_Close',
                   'USDT1D_Close', 'USDT4H_Close', 'USDT4H_Low', 'USDT4H_High', 'USDT_High', 'USDT_Low', 'USDT_Close']
          , axis=1, inplace=True)


full_data = data.copy()

events_data = data.loc[events.index]

events_data = events_data.loc[events_data['bb_cross'] != 0]

# signal = 'ret'
signal = 'bin'
print(data.columns)
full_features = ['Dema3', 'ema6', 'macd', '4Hmacd', '%K', '4H%K', '%D', '4H%D', '%DS', '4H%DS', 'rsi', '4H_rsi',
                 'diff', 'roc30', 'bb_cross', 'Volatility', 'Volume', '4H_Volume', 'TrD3', 'bb_sq', 'bb_l', 'bb_t',
                 'BTC1D_Volume', 'srl_corr', 'BTC_diff', 'cusum', 'BTC_cusum', 'USDT_cusum', 'BTCVol']

X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feature_columns=None)
backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
# X_train, X_test = standardizer(X_train), standardizer(X_test)
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
