import pandas as pd
import numpy as np
from ta.momentum import rsi, stoch
from ta.trend import macd_diff
from ta.volatility import average_true_range
from toolbox import asset_merger, primary_asset_merger, data_merger, rescaler, normalizer, standardizer, ROC, MOM, \
    spliter, crossing2, crossing3, meta_spliter
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, dropLabels, getEvents, getBins, \
    bbands, get_up_cross_bol, get_down_cross_bol, df_rolling_autocorr, returns, applyPtSlOnT1, mpPandasObj, \
    getDailyVolCGPT, metaBins
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
# data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
# data['4H%DS'] = data['4H%D'].rolling(3).mean()  # Stochastic slow.
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
# data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)
# data['roc10'] = ROC(data['Close'], 10)
# data['roc30'] = ROC(data['Close'], 30)
# data['mom10'] = MOM(data['Close'], 10)
# data['mom30'] = MOM(data['Close'], 30)
# data['Volatility_prcnt'] = getDailyVol(data['Close'], span, vertical_days, 'p')
data['Volatility'] = getDailyVol(data['Close'], span, vertical_days, 'ewm').rolling(20).mean()
# bb_down = get_down_cross_bol(data, 'Close')
# bb_up = get_up_cross_bol(data, 'Close')
# bb_side_up = pd.Series(-1, index=bb_up.index)  # sell on up cross for mean reversion
# bb_side_down = pd.Series(1, index=bb_down.index)  # buy on down cross for mean reversion
# bb_sides = pd.concat([bb_side_up, bb_side_down]).sort_index()
bb_sides = crossing3(data, 'Close', 'upper', 'lower')
# stoch_sides = crossing2(data, 'Close', '%K', '%D')
data['bb_cross'] = bb_sides
# print(bb_sides)

# data['diff'] = np.log(data['Close']).diff()
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
# data['bol_up_cross'] = get_up_cross_bol(data, 'Close')
# data['bol_down_cross'] = get_down_cross_bol(data, 'Close')
# data['trend'] = data.apply(lambda x: 1 if x['Close'] > x['Dema9'] else 0, axis=1)
# data['momentum'] = data.apply(lambda x: 1 if x['4H%K'] > x['4H%D'] else 0, axis=1)
threshold = data['Volatility']
tEvents = getTEvents(data['Close'], h=threshold)
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=vertical_days)
events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=bb_sides)
# labels = getBins(events, data['Close'])
labels = metaBins(events, eth.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
# data['event'] = data.apply(lambda x: True if x['ret'] != 0 else False, axis=0)
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())
data.drop(columns=['4H_Close', '4H_Low', '4H_High', '1D_Close', 'Price', 'ave', 'upper', 'lower'],
          axis=1, inplace=True)

full_data = data.copy()

# cusum events
research_data = data.loc[events.index]

# cusum + bb events
research_data = research_data[research_data['bb_cross'] != 0]


# signal = 'ret'
signal = 'bin'

# BALANCE CLASSES
# minority = research_data[research_data[signal] == 0]
# majority = research_data[research_data[signal] == 1].sample(n=len(minority), replace=True)
# research_data = pd.concat([minority, majority])
# print(research_data)


X, Y, X_train, X_test, Y_train, Y_test, backtest_data = spliter(full_data, research_data, signal, 4)
# X1, Y1, X2, Y2, X3, Y3, backtest_data = meta_spliter(full_data, research_data, 'bin', 5)
# X = standardizer(X)
# X_train = standardizer(X_train)
# backtest_data = standardizer(backtest_data)
# X_test = standardizer(X_test)
# X = normalizer(X)
# X = rescaler(X, (0, 1))

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, shuffle=False)

# seq_len = 2
# Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len - 1:], np.array(Y_test)
# X_train_LSTM = np.zeros((X_train.shape[0] + 1 - seq_len, seq_len, X_train.shape[1]))
# X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
# for i in range(seq_len):
#     X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0] + i + 1 - seq_len, :]
#     X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0] + i - 1:X.shape[0] + i + 1 - seq_len, :]

print('event 1', np.sum(np.array(research_data[signal]) == 1, axis=0))
print('event 0', np.sum(np.array(research_data[signal]) == 0, axis=0))

# print('event -1', np.sum(np.array(research_data[signal]) == -1, axis=0))
print('full_data.columns', full_data.columns)
print('X.columns', X.columns)
# print(research_data['Volatility'].mean())
# print(research_data['Volatility'].min())
# print(research_data['Volatility'].max())
# print(research_data['Volatility'])
