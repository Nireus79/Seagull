import pandas as pd
import numpy as np
from tqdm import tqdm
# from numba import jit
import glob
import winsound

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# https://data.binance.vision/
def csv_merger(path):
    names = glob.glob(path + "*.csv")  # get names of all CSV files under path
    # If your CSV files use commas to split fields, then the sep argument can be omitted or set to ","
    # columns = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', '7'] # TICK columns
    columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Open time',
               '0', '1', '2', '3', '4']
    #     1499040000000,      // Open time # Klines columns
    #     "0.01634790",       // Open
    #     "0.80000000",       // High
    #     "0.01575800",       // Low
    #     "0.01577100",       // Close
    #     "148976.11427815",  // Volume
    #     1499644799999,      // Close time
    #     "2434.19055334",    // Quote asset volume
    #     308,                // Number of trades
    #     "1756.87402397",    // Taker buy base asset volume
    #     "28.46694368",      // Taker buy quote asset volume
    #     "17928899.62484339" // Ignore.
    data = pd.concat([pd.read_csv(filename, names=columns, sep=",") for filename in names])
    data.drop(columns=['Open time', '0', '1', '2', '3', '4'], axis=1, inplace=True)
    # save the DataFrame to a file
    # data.to_csv("BTCEUR_5m.csv")
    return data


# print(csv_merger('D:/crypto_DATA/time/BTCEUR/5m/'))


def mad_outlier(y, thresh=3.):
    """
    compute outliers based on mad
    # args
        y: assumed to be arrayed with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    """
    median = np.median(y)
    diff = np.sum((y - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def returns(s):
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])


def tick_bars(df, price_column, m):
    """
    compute tick bars
    # args
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # returns
        idx: list of indices
    """
    t = df[price_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def tick_bar_df(df, price_column, m):
    idx = tick_bars(df, price_column, m)
    return df.iloc[idx]


# ========================================================
def volume_bars(df, volume_column, m):
    """
    compute volume bars
    # args
        df: pd.DataFrame()
        column: name for volume data
        m: int(), threshold value for volume returns
        idx: list of indices
    """
    t = df[volume_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def volume_bar_df(df, volume_column, m):
    idx = volume_bars(df, volume_column, m)
    return df.iloc[idx]


# ========================================================
def dollar_bars(df, value_column, m):
    """
    compute dollar bars
    # args
        df: pd.DataFrame()
        column: name for dollar volume data
        m: int(), threshold value for dollars returns
        idx: list of indices
    """
    t = df[value_column]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= m:
            idx.append(i)
            ts = 0
            continue
    return idx


def dollar_bar_df(df, value_column, m):
    idx = dollar_bars(df, value_column, m)
    return df.iloc[idx]


# ========================================================

# @jit(nopython=True)
# def numba_isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
#     return np.fabs(a - b) <= np.fmax(rel_tol * np.fmax(np.fabs(a), np.fabs(b)), abs_tol)
#
#
# @jit(nopython=True)
# def bt(p0, p1, bs):
#     # if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
#     if numba_isclose((p1 - p0), 0.0, abs_tol=0.001):
#         b = bs[-1]
#         return b
#     else:
#         b = np.abs(p1 - p0) / (p1 - p0)
#         return b
#
#
# @jit(nopython=True)
# def get_imbalance(t):
#     bs = np.zeros_like(t)
#     for i in np.arange(1, bs.shape[0]):
#         t_bt = bt(t[i - 1], t[i], bs[:i - 1])
#         bs[i - 1] = t_bt
#     return bs[:-1]  # remove last value


# def bar(x, y):
#     return np.int64(x / y) * y
#
#
# def datetime_indexing(source):
#     """
#     Takes a string path to a csv file and reindexing to datetime
#     :param source: string path to csv file
#     :return: datetime indexed data frame
#     """
#     df = pd.read_csv(source)
#     df.time = pd.to_datetime(df.time, unit='ms')
#     df.set_index('time', inplace=True)
#     return df
#
# def form_log_returns(time_bars_price):
#     """
#     Takes time bars data frame and return time_bars_price.close / time_bars_price.close.shift(1)
#     :param time_bars_price: time bars data frame
#     :return: time_bars_price.close / time_bars_price.close.shift(1)
#     """
#     log_return = np.log(time_bars_price.close / time_bars_price.close.shift(1)).dropna()
#     return log_return
#
#
# def form_tick_bars(data, transactions):
#     """
#     :param data: tick by tick data frame
#     :param transactions: number of transactions given as int
#     :return: data frame structured by number of transactions
#     When constructing tick bars, you need to be aware of outliers. Many exchanges
#     carry out an auction at the open and an auction at the close. This means that for a
#     period of time, the order book accumulates bids and offers without matching them.
#     When the auction concludes, a large trade is published at the clearing price, for an
#     outsized amount. This auction trade could be the equivalent of thousands of ticks,
#     even though it is reported as one tick.
#     """
#     tick_bars = data.groupby(bar(np.arange(len(data)), transactions)).agg({'price': 'ohlc', 'qty': 'sum'})
#     return tick_bars
#
#
# def form_vol_bars(data, traded_volume):
#     """
#     :param data: tick by tick data frame
#     :param traded_volume: volume of trades given as int
#     :return:data frame structured by volume of trades
#     Several market microstructure theories study the interaction between
#     prices and volume. Sampling as a function of one of these variables is a convenient
#     artifact for these analyses
#     """
#     volume_bars = data.groupby(bar(np.cumsum(data['qty']), traded_volume)).agg({'price': 'ohlc', 'qty': 'sum'})
#     volume_bars = volume_bars.loc[:, 'price']
#     return volume_bars
#
#
# def form_dollar_bars(data, market_value):
#     """
#     :param data:tick by tick data frame
#     :param market_value: volume of traded money given as int
#     :return:data frame structured by volume of money
#     Range and speed of variation will be reduced once you compute the number of dollar bars per day
#     over the years for a constant dollar bar size.
#     A second argument that makes dollar bars more interesting than time tick or
#     volume bars is that the number of outstanding shares often changes multiple times
#     over the course of a security life, as a result of corporate actions. Even after adjusting
#     for splits and reverse splits, there are other actions that will impact the amount of ticks
#     and volumes, like issuing new shares or buying back existing shares (a very common
#     practice since the Great Recession of 2008). Dollar bars tend to be robust in the face
#     of those actions. Still, you may want to sample dollar bars where the size of the bar is
#     not kept constant over time.
#     WARNING Lopez de Prado suggests about 50 dollar bars per day structure of data frequency.
#     That can be adjusted by setting market_value parameter accordingly.
#     """
#     data['value'] = data['qty'] * data['price']
#     dollar_bars = data.groupby(bar(np.cumsum(data['value']), market_value)).agg({'price': 'ohlc', 'qty': 'sum'})
#     dollar_bars_price = dollar_bars.loc[:, 'price']
#     return dollar_bars_price


def form_dollar_bars(csv, vol):
    data = pd.read_csv(csv)
    data.time = pd.to_datetime(data.time, unit='ms')
    data.set_index('time', inplace=True)
    data['value'] = data['price'] * data['qty']
    # mad = mad_outlier(data.price.values.reshape(-1, 1))
    # data = data.loc[~mad]
    # print(data)
    dbars = dollar_bar_df(data, 'value', vol).dropna()
    dbars.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    # dbars.to_csv('DOTEUR_1mdb')
    return dbars


# print(db_creator('D:/crypto_DATA/tick/DOTEUR/DOTEUR_full_tick.csv', 1000000))


def form_time_bars(csv, frequency):
    """
    Takes tick to tick data frame and structures data by given time freq.
    param data:
    :param csv:
    :param data: tick to tick data frame
    :param frequency: string ex: '5min'
    see doc https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    :return: time bars of given frequency
    Time bars oversample information during low-activity periods and undersample
    information during high-activity periods.
    Time-sampled series often exhibit poor
    statistical properties, like serial correlation, heteroscedasticity, and non-normality of
    returns.
    WARNING (works with datetime index only datetime_indexing function must be used as arg)
    WARNING Lopez de Prado suggests 1min bars as substitute for constructing market microstructural futures
    """
    data = pd.read_csv(csv)
    data.time = pd.to_datetime(data.time, unit='ms')
    data.set_index('time', inplace=True)
    data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    data = data.loc[~data.index.duplicated(keep='first')]
    time_bars = data.groupby(pd.Grouper(freq=frequency)).agg({'price': 'ohlc', 'qty': 'sum'})
    time_bars_price = time_bars.loc[:, 'price']
    time_bars_price.ffill(inplace=True)
    # time_bars_price.to_csv('EURUSDT_30m.csv')
    return time_bars_price

# print(pd.read_csv('D:/crypto_DATA/tick/DOTEUR/DOTEUR_full_tick.csv'))

# print(form_time_bars('D:/crypto_DATA/tick/EURUSDT/EURUSDT_full_tick.csv', '30min'))
