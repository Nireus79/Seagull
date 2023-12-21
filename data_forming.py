import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff, macd_signal, macd, adx
from ta.volatility import average_true_range
from toolbox import rescaler, normalizer, standardizer, ROC, MOM, spliter, crossing3
from Pradofun import getDailyVol, getDailyVolRows, getTEvents, addVerticalBarrier, addVerticalBarrierRows, dropLabels, \
    getEvents, bbands, metaBins, df_rolling_autocorr, returns, getDailyTimeBarVolatilityRows, getBins
import warnings

warnings.filterwarnings('ignore')

# https://data.binance.vision/
# https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/Labeling%20and%20MetaLabeling%20for%20Supervised%20Classification.ipynb


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

eth = pd.read_csv('csv/time_bars_30min/ETHEUR_full_30m.csv')
eth.time = pd.to_datetime(eth.time, unit='ms')
eth.set_index('time', inplace=True)

# eur_usd1H = pd.read_csv('csv/time_bars_1h/eur_usd_20_23_hours.csv')
# eur_usd1H.rename(columns={'eurusd_open': 'Open', 'eurusd_high': 'High', 'eurusd_low': 'Low', 'eurusd_close': 'Close',
#                           'volume': 'Volume'}, inplace=True)
# eur_usd1H.time = pd.to_datetime(eur_usd1H.time, unit='ms')
# eur_usd1H.set_index('time', inplace=True)
#
# btc_eur1H = pd.read_csv('csv/time_bars_1h/btc_eur_20_23_hours.csv')
# btc_eur1H.rename(columns={'btceur_open': 'Open', 'btceur_high': 'High', 'btceur_low': 'Low', 'btceur_close': 'Close',
#                           'volume': 'Volume'}, inplace=True)
# btc_eur1H.time = pd.to_datetime(btc_eur1H.time, unit='ms')
# btc_eur1H.set_index('time', inplace=True)


ohlc = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}
eth4h = eth.resample('4H').apply(ohlc)
eth['4H_Close'] = eth4h['Close']
eth['4H_Low'] = eth4h['Low']
eth['4H_High'] = eth4h['High']

eth1D = eth.resample('D').apply(ohlc)
eth['1D_Close'] = eth1D['Close']
eth['1D_High'] = eth1D['High']
eth['1D_Low'] = eth1D['Low']
eth = eth.ffill()

# eur_usd4H = eur_usd1H.resample('4H').apply(ohlc)
# eth['4H_eurusd_Close'] = eur_usd4H['Close']
#
# btc_eur4H = btc_eur1H.resample('4H').apply(ohlc)
# eth['4H_btc_eur_Close'] = btc_eur4H['Close']

cpus = 1
ptsl = [1, 2]  # profit-taking and stop loss limit multipliers
minRet = c_labels = .01  # The minimum target return(def .01) (volatility) required for running a triple barrier search
vertical_days = 1
rows = 48
span = 100
window = 40
bb_stddev = 1
part = 5

data = eth
# data['ema3'] = data['Close'].rolling(3).mean()
# data['H4_ema3'] = data['4H_Close'].rolling(3).mean()
# data['H4_ema6'] = data['4H_Close'].rolling(6).mean()
# data['adx'] = adx(data['High'], data['Low'], data['Close'], window=14, fillna=False)
# data['Dema3'] = data['1D_Close'].rolling(3).mean()
# data['ema6'] = data['Close'].rolling(6).mean()
# data['Dema6'] = data['1D_Close'].rolling(6).mean()
# data['ema9'] = data['Close'].rolling(9).mean()
# data['Dema9'] = data['1D_Close'].rolling(9).mean()
# data['ema13'] = data['Close'].rolling(13).mean()
# data['Dema13'] = data['1D_Close'].rolling(13).mean()
# data['ema20'] = data['Close'].rolling(20).mean()
# data['Dema20'] = data['1D_Close'].rolling(20).mean()
# data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['4Hmacd_diff'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
# data['%D'] = data['%K'].rolling(3).mean()
data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()
# data['4H%DS'] = data['4H%D'].rolling(3).mean()
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
# data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
# data['diff'] = np.log(data['Close']).diff()
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)
bb_sides = crossing3(data, 'Close', 'upper', 'lower')
data['bb_cross'] = bb_sides
data['Volatility'] = getDailyVol(data['Close'], span, vertical_days, 'ewm').rolling(window).mean()
# data['Volatility'] = getDailyVolRows(data['Close'], span, rows, 'ewm').rolling(window).mean()

# data['DT_diff'] = data.apply(lambda x: x['Dema9'] - x['Dema13'], axis=1)
# data['4M_diff'] = data.apply(lambda x: x['4H%K'] - x['4H%D'], axis=1)
# data['T_diff'] = data.apply(lambda x: x['ema9'] - x['ema13'], axis=1)
# data['M_diff'] = data.apply(lambda x: x['%K'] - x['%D'], axis=1)
tEvents = getTEvents(data['Close'], h=data['Volatility'])
t1 = addVerticalBarrier(tEvents, data['Close'], vertical_days)
# t1 = addVerticalBarrierRows(tEvents, data['Close'], rows)

events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=bb_sides)

# labels = getBins(events, data['Close'])
labels = metaBins(events, eth.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())
data.drop(columns=['4H_Close', '4H_Low', '4H_High', '1D_Close', '1D_High', '1D_Low',
                   'Price', 'ave', 'upper', 'lower'], axis=1, inplace=True)

data[['4H%K', '4H%D', '4H_rsi', 'Volatility']] = standardizer(data[['4H%K', '4H%D', '4H_rsi', 'Volatility']])
# data[['4H%K', '4H%D', '4H_rsi']] = normalizer(data[['4H%K', '4H%D', '4H_rsi']])
# data[['4H%K', '4H%D', '4H_rsi']] = rescaler(data[['4H%K', '4H%D', '4H_rsi']], (0, 1))
full_data = data.copy()

events_data = data.loc[events.index]
# events_data = events_data.loc[events_data['bb_cross'] == -1]

# signal = 'ret'
signal = 'bin'

feats_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'bb_cross', '4H_rsi', 'Volatility']

X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feats_to_drop)
backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
# BALANCE CLASSES (down sampling)
# minority = events_data[events_data[signal] == 1]
# majority = events_data[events_data[signal] == 0].sample(n=len(minority), replace=True)
# events_data = pd.concat([minority, majority])
# print(events_data)

print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('research_data min ret', events_data.ret.min())
print('research_data max ret', events_data.ret.max())

print('full_data.columns', full_data.columns)
print('X.columns', X_train.columns)
