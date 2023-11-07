import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff
from ta.volatility import average_true_range
from toolbox import asset_merger, primary_asset_merger, data_merger, rescaler, normalizer, standardizer, ROC, MOM
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
span = window = 30
c_labels = .01

asset1 = 'etheur'
asset2 = 'btceur'
asset3 = 'eurusd'

data = eth
# data['Dot'] = dot['Close']  # .loc[data.index]
# data[asset1 + '_close'] = eth['close']
# data[asset2 + '_close'] = bit['close']
# data[asset3 + '_close'] = eur['close']
data['ema9'] = data['Close'].rolling(9).mean()
data['Dema9'] = data['1D_Close'].rolling(9).mean()
data['ema13'] = data['Close'].rolling(13).mean()
data['Dema13'] = data['1D_Close'].rolling(13).mean()
data['ema20'] = data['Close'].rolling(20).mean()
data['Dema20'] = data['1D_Close'].rolling(20).mean()
data['macd'] = macd_diff(data['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['4Hmacd'] = macd_diff(data['4H_Close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
data['%K'] = stoch(data['High'], data['Low'], data['Close'], window=14, smooth_window=3, fillna=False)
data['4H%K'] = stoch(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, smooth_window=3, fillna=False)
data['%D'] = data['%K'].rolling(3).mean()
data['4H%D'] = data['4H%K'].rolling(3).mean()
data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
data['4H%DS'] = data['4H%D'].rolling(3).mean()  # Stochastic slow.
data['rsi'] = rsi(data['Close'], window=14, fillna=False)
data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=1)
data['roc10'] = ROC(data['Close'], 10)
data['roc30'] = ROC(data['Close'], 30)
data['mom10'] = MOM(data['Close'], 10)
data['mom30'] = MOM(data['Close'], 30)
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
data['cusum'] = data['Close'].cumsum()
data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
data['bol_up_cross'] = get_up_cross_bol(data, 'Close')
data['bol_down_cross'] = get_down_cross_bol(data, 'Close')

threshold = data['Volatility'].mean()
tEvents = getTEvents(data['Close'], h=threshold)
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=vertical_days)
events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=None)
labels = getBins(events, data['Close'])
clean_labels = dropLabels(labels, c_labels)
# data['signal'] = clean_labels['ret']
data['signal'] = clean_labels['bin']
data.drop(columns=['Price'], axis=1, inplace=True)
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

signal = 'signal'  # 'ret'
Y = research_data.loc[:, signal]
Y.name = Y.name
X = research_data.loc[:, research_data.columns != signal]
Y = research_data.loc[:, Y.name]
X = research_data.loc[:, X.columns]


def spliter(dataset, part):
    validation_size = 0.2
    test_size = int(len(X) * validation_size)
    if part == 1:
        X_tst, X_tr = X[:test_size], X[test_size:]
        Y_tst, Y_tr = Y[:test_size], Y[test_size:]
        bt_data = dataset[X_tst.index[0]:X_tst.index[-1]]
        return X_tr, X_tst, Y_tr, Y_tst, bt_data
    elif part == 2:
        X_tst, X_tr = X[test_size:test_size * 2], pd.concat([X[:test_size], X[test_size * 2:]])
        Y_tst, Y_tr = Y[test_size:test_size * 2], pd.concat([Y[:test_size], Y[test_size * 2:]])
        bt_data = dataset[X_tst.index[0]:X_tst.index[-1]]
        return X_tr, X_tst, Y_tr, Y_tst, bt_data
    elif part == 3:
        X_tst, X_tr = X[test_size * 2:test_size * 3], pd.concat([X[:test_size * 2], X[test_size * 3:]])
        Y_tst, Y_tr = Y[test_size * 2:test_size * 3], pd.concat([Y[:test_size * 2], Y[test_size * 3:]])
        bt_data = dataset[X_tst.index[0]:X_tst.index[-1]]
        return X_tr, X_tst, Y_tr, Y_tst, bt_data
    elif part == 4:
        X_tst, X_tr = X[test_size * 3:test_size * 4], pd.concat([X[:test_size * 3], X[test_size * 4:]])
        Y_tst, Y_tr = Y[test_size * 3:test_size * 4], pd.concat([Y[:test_size * 3], Y[test_size * 4:]])
        bt_data = dataset[X_tst.index[0]:X_tst.index[-1]]
        return X_tr, X_tst, Y_tr, Y_tst, bt_data
    elif part == 5:
        X_tst, X_tr = X[test_size * 4:], X[:test_size * 4]
        Y_tst, Y_tr = Y[test_size * 4:], Y[:test_size * 4]
        bt_data = dataset[X_tst.index[0]:X_tst.index[-1]]
        return X_tr, X_tst, Y_tr, Y_tst, bt_data

