import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff
from ta.volatility import average_true_range
from toolbox import rescaler, normalizer, standardizer, ROC, MOM, spliter, crossing2, crossing3, meta_spliter
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, dropLabels, getEvents, bbands, metaBins, \
    df_rolling_autocorr, returns
import warnings

warnings.filterwarnings('ignore')

# https://data.binance.vision/


# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# csv = 'D:/crypto_DATA/time/30m/ETHEUR/'
# merged = primary_asset_merger(csv)
# merged.to_csv('ETHEUR_full_30m.csv')

# merged_doteur.to_csv('doteur_20_23_hours.csv')
# merged_btceur = asset_merger(btceur_csv, 'btceur')
# merged_btceur.to_csv('btc_eur_20_23_hours.csv')
# merged_etheur = asset_merger(etheur_csv, 'etheur')
# merged_etheur.to_csv('eth_eur_20_23_hours.csv')
# merged_eurusd = asset_merger(eurusd_csv, 'eurusd')
# merged_eurusd.to_csv('eur_usd_20_23_hours.csv')

# dot = pd.read_csv('csv/time_bars_30min/DOTEUR_full_30m.csv')
# dot.time = pd.to_datetime(dot.time, unit='ms')
# dot.set_index('time', inplace=True)

eth = pd.read_csv('csv/time_bars_30min/ETHEUR_full_30m.csv')
eth.time = pd.to_datetime(eth.time, unit='ms')
eth.set_index('time', inplace=True)

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
eth = eth.ffill()

# dot4h = dot.resample('4H').apply(ohlc)
# dot['4H_Close'] = dot4h['Close']
# dot['4H_Low'] = dot4h['Low']
# dot['4H_High'] = dot4h['High']
# dot1D = dot.resample('D').apply(ohlc)
# dot['1D_Close'] = dot1D['Close']
# dot = dot.ffill()

cpus = 1
ptsl = [1, 1]  # profit-taking and stop loss limit multipliers
minRet = c_labels = .01  # The minimum target return (volatility) required for running a triple barrier search
vertical_days = 1
span = 100
window = 20
bb_stddev = 1

asset1 = 'etheur'
asset2 = 'btceur'
asset3 = 'eurusd'

data = eth
# data['Dot'] = dot['Close']  # .loc[data.index]
# data[asset1 + '_close'] = eth['close']
# data[asset2 + '_close'] = bit['close']
# data[asset3 + '_close'] = eur['close']
# data['ema9'] = data['Close'].rolling(9).mean()
data['Dema9'] = data['1D_Close'].rolling(9).mean()
# data['ema13'] = data['Close'].rolling(13).mean()
# data['Dema13'] = data['1D_Close'].rolling(13).mean()
# data['ema20'] = data['Close'].rolling(20).mean()
# data['Dema20'] = data['1D_Close'].rolling(20).mean()
# data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['4Hmacd'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
# data['%D'] = data['%K'].rolling(3).mean()
data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
# data['4H%DS'] = data['4H%D'].rolling(3).mean()  # Stochastic slow.
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
# data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
# data['diff'] = np.log(data['Close']).diff()
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)
bb_sides = crossing3(data, 'Close', 'upper', 'lower')
data['bb_cross'] = bb_sides
data['Volatility'] = getDailyVol(data['Close'], span, vertical_days, 'ewm').rolling(window).mean()

data['trend'] = data.apply(lambda x: 1 if x['Close'] > x['Dema9'] else 0, axis=1)
data['momentum'] = data.apply(lambda x: 1 if x['4H%K'] > x['4H%D'] else 0, axis=1)
data['elder'] = data.apply(lambda x: 1 if x['trend'] == 1 and x['momentum'] == 1 else 0, axis=1)
elder_sides = data.loc[data['trend'] == 1]

tEvents = getTEvents(data['Close'], h=data['Volatility'])
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=vertical_days)

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
data.drop(columns=['4H_Close', '4H_High', '1D_Close', 'Price', 'ave', 'upper', 'lower'], axis=1, inplace=True)

# data[['Close', 'Dema9', '4H%K', '4H%D']] = standardizer(data[['Close', 'Dema9', '4H%K', '4H%D']])
# data[['Close', 'Dema9', '4H%K', '4H%D']] = normalizer(data[['Close', 'Dema9', '4H%K', '4H%D']])
# data[['Close', 'Dema9', '4H%K', '4H%D']] = rescaler(data[['Close', 'Dema9', '4H%K', '4H%D']], (0, 1))
full_data = data.copy()

events_data = data.loc[events.index]

events_dataB = events_data.loc[events_data['bb_cross'] != 0]

# signal = 'ret'
signal = 'bin'

feats_to_drop = ['4H_Low', '4H_atr', 'Close', 'Open', 'High', 'Low', 'Volume','Dema9', 'bb_cross', 'Volatility',
                 'trend', 'momentum', 'elder']
feats_to_dropB = ['4H_Low', '4H_atr', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'trend', 'momentum',
                  'elder', '4H%K', '4H%D']
feats_to_dropS = ['4H_Low', '4H_atr', 'Open', 'High', 'Low', 'Volume', 'bb_cross', 'Volatility', 'trend', 'momentum',
                  'elder', 'Close', 'Dema9']

part = 5
X, Y, X_train, X_test, Y_train, Y_test, backtest_data = spliter(full_data, events_data, signal, part, feats_to_drop)
# XB, YB, X_trainB, X_testB, Y_trainB, Y_testB, backtest_dataB = \
#     spliter(full_data, events_dataB, signal, part, feats_to_dropB)
# XS, YS, X_trainS, X_testS, Y_trainS, Y_testS, backtest_dataS = \
#     spliter(full_data, events_data, signal, part, feats_to_dropS)

# BALANCE CLASSES (down sampling)
# minority = events_data[events_data[signal] == 1]
# majority = events_data[events_data[signal] == 0].sample(n=len(minority), replace=True)
# events_data = pd.concat([minority, majority])
# print(research_data)

print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('research_data min ret', events_data.ret.min())
print('research_data max ret', events_data.ret.max())

print('full_data.columns', full_data.columns)
print('X.columns', X.columns)
