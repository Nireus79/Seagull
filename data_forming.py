import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff
from ta.volatility import average_true_range
from toolbox import asset_merger, primary_asset_merger, data_merger, rescaler, normalizer, standardizer
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, dropLabels, getEvents, getBins, \
    bbands, get_up_cross_bol, get_down_cross_bol, df_rolling_autocorr, returns, applyPtSlOnT1, mpPandasObj, \
    getDailyVolCGPT
from sklearn.linear_model import LinearRegression
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

cpus = 1
ptsl = [2, 1]  # profit-taking and stop loss limit multipliers
minRet = .01  # The minimum target return (volatility) required for running a triple barrier search
vertical_days = 1
span = 20
window = 20
c_labels = .01

asset1 = 'etheur'
asset2 = 'btceur'
asset3 = 'eurusd'

data = eth  # [:'2023-09-30 00:00:00']
# data['Dot'] = dot['Close']  # .loc[data.index]
# data[asset1 + '_close'] = eth['close']
# data[asset2 + '_close'] = bit['close']
# data[asset3 + '_close'] = eur['close']
# data['ema9'] = data['Close'].rolling(9).mean()
# data['Dema9'] = data['1D_Close'].rolling(9).mean()
# data['ema13'] = data['Close'].rolling(13).mean()
# data['Dema13'] = data['1D_Close'].rolling(13).mean()
# data['ema20'] = data['Close'].rolling(20).mean()
# data['Dema20'] = data['1D_Close'].rolling(20).mean()
# data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['4Hmacd'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
# data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
# data['%D'] = data['%K'].rolling(3).mean()
data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
# data['4H%DS'] = data['4H%D'].rolling(3).mean()  # Stochastic slow.
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
# data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=1)
data.drop(columns=['Price'], axis=1, inplace=True)
# data['Volatility_prcnt'] = getDailyVol(data['Close'], span, vertical_days, 'p')
data['Volatility'] = getDailyVol(data['Close'], span, vertical_days, 'ewm')
bb_down = get_down_cross_bol(data, 'Close')
bb_up = get_up_cross_bol(data, 'Close')
bb_side_up = pd.Series(-1, index=bb_up.index)  # sell on up cross for mean reversion
bb_side_down = pd.Series(1, index=bb_down.index)  # buy on down cross for mean reversion
bb_side_raw = pd.concat([bb_side_up, bb_side_down]).sort_index()
data['bb_cross'] = bb_side_raw

# data['diff'] = np.log(data['close']).diff()
# training data
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
# data['bol_up_cross'] = get_up_cross_bol(data, 'Close')
# data['bol_down_cross'] = get_down_cross_bol(data, 'Close')

threshold = data['Volatility'].mean()
tEvents = getTEvents(data['Close'], h=threshold)
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=vertical_days)
events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=None)
labels = getBins(events, data['Close'])
clean_labels = dropLabels(labels, c_labels)
data['ret'] = clean_labels['ret']
# data['bin'] = clean_labels['bin']
data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())

# data = standardizer(data)
# data = normalizer(data)
# data = rescaler(data, minmax=(-1, 1))

full_data = data.copy()
# cusum events
research_data = data.loc[events.index]
# cusum + bb events
# research_data = research_data.loc[research_data.apply(lambda x: x.bol_up_cross != 0 or x.bol_down_cross != 0, axis=1)]

ret = 'ret'  # 'Close'  # 'ret' 'bin'
Y = research_data.loc[:, ret]
Y.name = Y.name
X = research_data.loc[:, ('4H_rsi', '4H%K', '4H%D', '4Hmacd')]

Y = research_data.loc[:, Y.name]
X = research_data.loc[:, X.columns]

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
test_size = int(len(X) * validation_size)
# part = 2
# X_test, X_train = X[test_size:test_size*part], X[test_size*part:]
# Y_test, Y_train = Y[test_size:test_size*part], Y[test_size*part:]
# backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]
backtest_data = full_data[X_test.index[0]:]

# print(data)
# print(full_data)
# print(full_data.isnull().sum())
# print(research_data)
# print(research_data.isnull().sum())
# print('len research_data: ', len(data))
# print(backtest_data)
data['p_ret'] = research_data.apply(lambda x: x.ret / x.Close if x.ret != 0 else 0, axis=1)
print('total events', np.sum(np.array(data.ret) != 0, axis=0))
print('positive ret', np.sum(np.array(data.ret) > 0, axis=0))
print('over comm ret', np.sum(np.array(data.p_ret) > .045, axis=0))
print('negative ret', np.sum(np.array(data.ret) < 0, axis=0))
# print('%vol > 0.045', np.sum(np.array(data.Volatility_prcnt) > 0.045, axis=0))
print('mean % ret', data.p_ret.mean())
print('max % ret', data.p_ret.max())
print('min % ret', data.p_ret.min())
print('bb 1', np.sum(np.array(data.bb_cross) > 0, axis=0))
print('bb -1', np.sum(np.array(data.bb_cross) < 0, axis=0))
# print(backtest_data)


# eth4h = eth.resample('4H').apply(ohlc)
# eth['4H_Close'] = eth4h['Close']
# eth['4H_Low'] = eth4h['Low']
# eth['4H_High'] = eth4h['High']
# eth1D = eth.resample('D').apply(ohlc)
# eth['1D_Close'] = eth1D['Close']
# eth = eth.ffill()
#
# eth['Dema13'] = eth['1D_Close'].rolling(13).mean()
# eth['4H%K'] = stoch(eth['4H_High'], eth['4H_Low'], eth['4H_Close'], window=14, smooth_window=3, fillna=False)
# eth['Volatility'] = getDailyVol(eth['Close'], window, 2)
#
# etEvents = getTEvents(eth['Close'], h=eth['Volatility'].mean())
# et1 = addVerticalBarrier(etEvents, eth['Close'], numDays=1)
# eevents = getEvents(eth['Close'], etEvents, ptsl, eth['Volatility'], minRet, cpus, t1, side=None)
#
# elabels = getBins(eevents, eth['Close'])
# eclean_labels = dropLabels(elabels, .05)
# eth['ret'] = eclean_labels['ret']
# # data['bin'] = clean_labels['bin']
# eth = eth.fillna(0)
# eth = eth.loc[~eth.index.duplicated(keep='first')]
