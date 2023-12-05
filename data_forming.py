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
data['4H%D'] = data['4H%K'].rolling(3).mean()
# data['%DS'] = data['%D'].rolling(3).mean()  # Stochastic slow.
data['4H%DS'] = data['4H%D'].rolling(3).mean()  # Stochastic slow.
# data['rsi'] = rsi(data['Close'], window=14, fillna=False)
# data['4H_rsi'] = rsi(data['4H_Close'], window=14, fillna=False)
# data['atr'] = average_true_range(data['High'], data['Low'], data['Close'], window=14, fillna=False)
# data['4H_atr'] = average_true_range(data['4H_High'], data['4H_Low'], data['4H_Close'], window=14, fillna=False)
# data['diff'] = np.log(data['Close']).diff()
# data['cusum'] = data['Close'].cumsum()
# data['srl_corr'] = df_rolling_autocorr(returns(data['Close']), window=window).rename('srl_corr')
data['Price'], data['ave'], data['upper'], data['lower'] = bbands(data['Close'], window=window, numsd=bb_stddev)

data['Volatility'] = getDailyVol(data['Close'], span, vertical_days, 'ewm').rolling(20).mean()
bb_sides = crossing3(data, 'Close', 'upper', 'lower')
data['bb_cross'] = bb_sides

data['trend'] = data.apply(lambda x: 1 if x['Close'] > x['Dema9'] else 0, axis=1)
data['momentum'] = data.apply(
    lambda x: 1 if x['4H%D'] > x['4H%DS'] else (-1 if x['4H%D'] < x['4H%DS'] else 0), axis=1)
data['elder'] = data.apply(lambda x: 1 if x['trend'] == 1 and x['momentum'] == 1 else 0, axis=1)
elder_sides = data['elder']

threshold = data['Volatility']
tEvents = getTEvents(data['Close'], h=threshold)
t1 = addVerticalBarrier(tEvents, data['Close'], numDays=vertical_days)
events = getEvents(data['Close'], tEvents, ptsl, data['Volatility'], minRet, cpus, t1, side=elder_sides)
# labels = getBins(events, data['Close'])
labels = metaBins(events, eth.Close, t1)
clean_labels = dropLabels(labels, .05)
data['ret'] = clean_labels['ret']
data['bin'] = clean_labels['bin']

data = data.fillna(0)
data = data.loc[~data.index.duplicated(keep='first')]

# print(data)
# print(data.isnull().sum())
data.drop(columns=['4H_Close', '4H_Low', '4H_High', '1D_Close', 'Price', 'ave', 'upper', 'lower'],
          axis=1, inplace=True)

data[['4H%D', '4H%DS', 'Volatility', 'trend']] = standardizer(data[['4H%D', '4H%DS', 'Volatility', 'trend']])
# data[['4H%D', '4H%DS']] = normalizer(data[['4H%D', '4H%DS']])
# data[['4H%D', '4H%DS']] = rescaler(data[['4H%D', '4H%DS']], (0, 1))
full_data = data.copy()

research_data = data.loc[events.index]

# signal = 'ret'
signal = 'bin'

# BALANCE CLASSES
# minority = research_data[research_data[signal] == 0]
# majority = research_data[research_data[signal] == 1].sample(n=len(minority), replace=True)
# research_data = pd.concat([minority, majority])
# print(research_data)


X, Y, X_train, X_test, Y_train, Y_test, backtest_data = spliter(full_data, research_data, signal, 2)

print('event 1', np.sum(np.array(research_data[signal]) == 1, axis=0))
print('event 0', np.sum(np.array(research_data[signal]) == 0, axis=0))

print('full_data.columns', full_data.columns)
print('X.columns', X.columns)
