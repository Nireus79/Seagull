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

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# https://data.binance.vision/
# https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/Labeling%20and%20MetaLabeling%20for%20Supervised%20Classification.ipynb

# db = pd.read_csv('csv/db/ETHEUR_10mdb')
# db.set_index('time', inplace=True)
# print(db)


eth5m = pd.read_csv('csv/tb/ETHEUR_5m.csv')
# btc5m = pd.read_csv('csv/tb/BTCEUR_5m.csv')
usdt5m = pd.read_csv('csv/tb/EURUSDT_5m.csv')

eth5m.time = pd.to_datetime(eth5m.time, unit='ms')
eth5m.set_index('time', inplace=True)
eth5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# btc5m.time = pd.to_datetime(btc5m.time, unit='ms')
# btc5m.set_index('time', inplace=True)
# btc5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

# usdt5m.time = pd.to_datetime(usdt5m.time, unit='ms')
# usdt5m.set_index('time', inplace=True)
# usdt5m.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

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
#
# usdt30m = usdt5m.resample('30min').apply(ohlc)
# usdt4h = usdt5m.resample('4H').apply(ohlc)
# usdt1D = usdt5m.resample('D').apply(ohlc)

# eth30m['BTC_Close'] = btc30m['Close']
# eth30m['BTC_High'] = btc30m['High']
# eth30m['BTC_Low'] = btc30m['Low']
#
# eth30m['USDT_Close'] = usdt30m['Close']
# eth30m['USDT_High'] = usdt30m['High']
# eth30m['USDT_Low'] = usdt30m['Low']

# eth5m['30m_Close'] = eth30m['Close']
# eth5m['30m_Low'] = eth30m['Low']
# eth5m['30m_High'] = eth30m['High']
# eth5m['30m_Volume'] = eth30m['Volume']

eth30m['4H_Close'] = eth4h['Close']
eth30m['4H_Low'] = eth4h['Low']
eth30m['4H_High'] = eth4h['High']
eth30m['4H_Volume'] = eth4h['Volume']

# eth30m['BTC4H_Close'] = btc4h['Close']
# eth30m['BTC4H_Low'] = btc4h['Low']
# eth30m['BTC4H_High'] = btc4h['High']
#
# eth30m['USDT4H_Close'] = usdt4h['Close']
# eth30m['USDT4H_Low'] = usdt4h['Low']
# eth30m['USDT4H_High'] = usdt4h['High']
#
eth30m['1D_Close'] = eth1D['Close']
eth30m['1D_Volume'] = eth1D['Volume']

# eth30m['BTC1D_Close'] = btc1D['Close']
# eth30m['BTC1D_Volume'] = btc1D['Volume']
# eth30m['USDT1D_Close'] = usdt1D['Close']
# eth30m['USDT1D_Volume'] = usdt1D['Volume']

cpus = 1
ptsl = [1, 1]  # profit-taking / stop loss limit multipliers
minRet = .01  # The minimum target return(def .01) (volatility) required for running a triple barrier search
delta = 1
span = 100
window = 20
bb_stddev = 2
#  day event data mean ret 0.0002541716925555011
# 4H event data mean ret   0.00003291562224155382
part = 5

data = eth30m
data.ffill(inplace=True)

data['ema3'] = data['Close'].rolling(3).mean()
data['ema6'] = data['Close'].rolling(6).mean()
data['ema9'] = data['Close'].rolling(9).mean()
data['ema13'] = data['Close'].rolling(13).mean()
data['ema20'] = data['Close'].rolling(20).mean()

data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['%D'] = data['%K'].rolling(3).mean()

data['vema3'] = data['Volume'].rolling(3).mean()
data['vema6'] = data['Volume'].rolling(6).mean()
data['vema9'] = data['Volume'].rolling(9).mean()
data['vema13'] = data['Volume'].rolling(13).mean()
data['vema20'] = data['Volume'].rolling(20).mean()

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
data['bb_sq'] = data.apply(lambda x: x['upper'] - x['lower'], axis=1)
data['bb_l'] = data.apply(lambda x: (x['upper'] - x['Close']) / (x['Close'] - x['lower']) if
x['Close'] - x['lower'] != 0 else 0, axis=1)
data['bb_t'] = data.apply(lambda x: x['bb_l'] / x['bb_sq'] if
x['bb_sq'] != 0 else 0, axis=1)

# data['30m_ema3'] = data['30m_Close'].rolling(3).mean()
# data['30m_ema6'] = data['30m_Close'].rolling(6).mean()
# data['30m_ema9'] = data['30m_Close'].rolling(9).mean()
# data['30m_ema13'] = data['30m_Close'].rolling(13).mean()
# data['30m_ema20'] = data['30m_Close'].rolling(20).mean()

# data['30m_vema3'] = data['30m_Volume'].rolling(3).mean()
# data['30m_vema6'] = data['30m_Volume'].rolling(6).mean()
# data['30m_vema9'] = data['30m_Volume'].rolling(9).mean()
# data['30m_vema13'] = data['30m_Volume'].rolling(13).mean()
# data['30m_vema20'] = data['30m_Volume'].rolling(20).mean()

# data['30m_adx'] = adx(data['30m_High'], data['30m_Low'], data['30m_Close'], window=14, fillna=False)
# data['30m_macd'] = macd_diff(data['30m_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['30m_%K'] = stoch(data['30m_High'], data['30m_Low'], data['30m_Close'], window=14, smooth_window=3, fillna=False)
# data['30m_%D'] = data['30m_%K'].rolling(3).mean()
# data['30m_%DS'] = data['30m_%D'].rolling(3).mean()
# data['30m_rsi'] = rsi(data['30m_Close'], window=14, fillna=False)
# data['30m_atr'] = average_true_range(data['30m_High'], data['30m_Low'], data['30m_Close'], window=14, fillna=False)
# data['30m_diff'] = np.log(data['30m_Close']).diff()
# data['30m_cusum'] = data['30m_Close'].cumsum()
# data['30m_srl_corr'] = df_rolling_autocorr(returns(data['30m_Close']), window=window).rename('30m_srl_corr')
# data['30m_roc10'] = ROC(data['30m_Close'], 10)
# data['30m_roc20'] = ROC(data['30m_Close'], 20)
# data['30m_roc30'] = ROC(data['30m_Close'], 30)
# data['30m_mom10'] = MOM(data['30m_Close'], 10)
# data['30m_mom20'] = MOM(data['30m_Close'], 20)
# data['30m_mom30'] = MOM(data['30m_Close'], 30)

data['4H_ema3'] = data['4H_Close'].rolling(3).mean()
data['4H_ema6'] = data['4H_Close'].rolling(6).mean()
data['4H_ema9'] = data['4H_Close'].rolling(9).mean()
data['4H_ema13'] = data['4H_Close'].rolling(13).mean()
data['4H_ema20'] = data['4H_Close'].rolling(20).mean()
data['4H_roc10'] = ROC(data['4H_Close'], 10)
data['4H_roc20'] = ROC(data['4H_Close'], 20)
data['4H_roc30'] = ROC(data['4H_Close'], 30)
data['4H_mom10'] = MOM(data['4H_Close'], 10)
data['4H_mom20'] = MOM(data['4H_Close'], 20)
data['4H_mom30'] = MOM(data['4H_Close'], 30)

data['4H_ema3'] = data['4H_Close'].rolling(3).mean()
data['4H_ema6'] = data['4H_Close'].rolling(6).mean()
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

data['Dvema3'] = data['1D_Volume'].rolling(3).mean()
data['Dvema6'] = data['1D_Volume'].rolling(6).mean()
data['Dvema9'] = data['1D_Volume'].rolling(9).mean()
data['Dvema13'] = data['1D_Volume'].rolling(13).mean()
data['Dvema20'] = data['1D_Volume'].rolling(20).mean()

data['Tr6'] = data.apply(lambda x: x['Close'] - x['ema6'], axis=1)
data['Tr9'] = data.apply(lambda x: x['Close'] - x['ema9'], axis=1)
data['Tr13'] = data.apply(lambda x: x['Close'] - x['ema13'], axis=1)
data['Tr20'] = data.apply(lambda x: x['Close'] - x['ema20'], axis=1)

# data['Tr30m3'] = data.apply(lambda x: x['30m_Close'] - x['30m_ema3'], axis=1)
# data['Tr30m6'] = data.apply(lambda x: x['30m_Close'] - x['30m_ema6'], axis=1)
# data['Tr30m9'] = data.apply(lambda x: x['30m_Close'] - x['30m_ema9'], axis=1)
# data['Tr30m13'] = data.apply(lambda x: x['30m_Close'] - x['30m_ema13'], axis=1)
# data['Tr30m20'] = data.apply(lambda x: x['30m_Close'] - x['30m_ema20'], axis=1)

data['Tr4h3'] = data.apply(lambda x: x['4H_Close'] - x['4H_ema3'], axis=1)
data['Tr4h6'] = data.apply(lambda x: x['4H_Close'] - x['4H_ema6'], axis=1)
data['Tr4h9'] = data.apply(lambda x: x['4H_Close'] - x['4H_ema9'], axis=1)
data['Tr4h13'] = data.apply(lambda x: x['4H_Close'] - x['4H_ema13'], axis=1)
data['Tr4h20'] = data.apply(lambda x: x['4H_Close'] - x['4H_ema20'], axis=1)

data['TrD3'] = data.apply(lambda x: x['Close'] - x['Dema3'], axis=1)
data['TrD6'] = data.apply(lambda x: x['Close'] - x['Dema6'], axis=1)
data['TrD9'] = data.apply(lambda x: x['Close'] - x['Dema9'], axis=1)
data['TrD13'] = data.apply(lambda x: x['Close'] - x['Dema13'], axis=1)
data['TrD20'] = data.apply(lambda x: x['Close'] - x['Dema20'], axis=1)
data['Vtr3'] = data.apply(lambda x: x['Volume'] - x['vema3'], axis=1)
data['Vtr6'] = data.apply(lambda x: x['Volume'] - x['vema6'], axis=1)
data['Vtr9'] = data.apply(lambda x: x['Volume'] - x['vema9'], axis=1)
data['Vtr13'] = data.apply(lambda x: x['Volume'] - x['vema13'], axis=1)
data['Vtr20'] = data.apply(lambda x: x['Volume'] - x['vema20'], axis=1)
data['VtrD3'] = data.apply(lambda x: x['Volume'] - x['Dvema3'], axis=1)
data['VtrD6'] = data.apply(lambda x: x['Volume'] - x['Dvema6'], axis=1)
data['VtrD9'] = data.apply(lambda x: x['Volume'] - x['Dvema9'], axis=1)
data['VtrD13'] = data.apply(lambda x: x['Volume'] - x['Dvema13'], axis=1)
data['VtrD20'] = data.apply(lambda x: x['Volume'] - x['Dvema20'], axis=1)
data['StD4'] = data.apply(lambda x: x['4H%K'] - x['4H%D'], axis=1)
data['StD'] = data.apply(lambda x: x['%K'] - x['%D'], axis=1)
data['St4H'] = data.apply(lambda x: x['4H%K'] - x['4H%D'], axis=1)

bb_sides = crossing3(data, 'Close', 'upper', 'lower')
# elder_sides = crossing_elder(data, '4H%K', '4H%D')
data['bb_cross'] = bb_sides
data['Volatility'] = getDailyVol(data['Close'], span, delta).rolling(window).mean()

tEvents = getTEvents(data['Close'], h=data['Volatility'])
t1 = addVerticalBarrier(tEvents, data['Close'], delta)

events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=bb_sides)
labels = metaBins(events, data.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

full_data = data.copy()

events_data = data.loc[events.index]

events_data = events_data.loc[events_data['bb_cross'] != 0]

# signal = 'ret'
signal = 'bin'
# print(data.columns)

features = ['Tr9', 'Tr20', 'TrD3',
            '4H%K', '4H%D', 'bb_sq', 'diff',
            'srl_corr', 'bb_cross']
X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feature_columns=features)
backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
X_train_c, X_test_c = X_train.copy(), X_test.copy()
X_train_n, X_test_n = normalizer(X_train_c), normalizer(X_test_c)
if 'bb_cross' in X_train.columns:
    print('bb_cross in X')
    X_train_n.bb_cross, X_test_n.bb_cross = X_train.bb_cross, X_test.bb_cross


# BALANCE CLASSES (down sampling)
minority = events_data[events_data[signal] == 1]
majority = events_data[events_data[signal] == 0].sample(n=len(minority), replace=True)
events_data = pd.concat([minority, majority])
# print(events_data)

print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('event data min ret', events_data.ret.min())
print('event data max ret', events_data.ret.max())
print('event data mean ret', events_data.ret.mean())

# print('full_data.columns', full_data.columns)
# print('X.columns', X_train.columns)
