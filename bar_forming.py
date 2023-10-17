import pandas as pd
import numpy as np
from tqdm import tqdm
# from numba import jit
import glob


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


def form_time_bars(data, frequency):
    """
    Takes tick to tick data frame and structures data by given time freq.
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
    data.time = pd.to_datetime(data.time, unit='ms')
    data.set_index('time', inplace=True)
    time_bars = data.groupby(pd.Grouper(freq=frequency)).agg({'price': 'ohlc', 'qty': 'sum'})
    time_bars_price = time_bars.loc[:, 'price']
    return time_bars_price


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
        m: int(), threshold value for volume
    # returns
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
        m: int(), threshold value for dollars
    # returns
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
#

#
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


def db_creator(csv, vol):
    columns = ['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker', '7']
    data = pd.read_csv(csv, header=None, names=columns)
    data = data.drop(columns=['id', 'base_qty', 'is_buyer_maker', '7'], axis=1)
    # data.time = pd.to_datetime(data.time, unit='ms')
    # data.set_index('time', inplace=True)
    data['value'] = data['price'] * data['qty']
    # print(data)
    # mad = mad_outlier(data.price.values.reshape(-1, 1))
    # data = data.loc[~mad]
    # print(data)
    dbars = dollar_bar_df(data, 'value', vol).dropna()
    return dbars


def csv_merger(path):
    # path = "E:/T/ETHUSDT/10mdb/"  # set this to the folder containing CSVs
    names = glob.glob(path + "*.csv")  # get names of all CSV files under path
    # If your CSV files use commas to split fields, then the sep
    # argument can be ommitted or set to ","
    file_list = pd.concat([pd.read_csv(filename, sep=",") for filename in names])
    # save the DataFrame to a file
    # csv = file_list.to_csv("sample.csv")
    return file_list


def db_csv(csv, m, output):
    db_creator(csv, m).to_csv(output)


def sequence(trades, sample):
    read = trades + '/ETHUSDT-trades-2020-01.csv'
    out = 'ETHUSDT-20mdb-2020-01.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-02.csv'
    out = 'ETHUSDT-20mdb-2020-02.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-03.csv'
    out = 'ETHUSDT-20mdb-2020-03.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-04.csv'
    out = 'ETHUSDT-20mdb-2020-04.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-05.csv'
    out = 'ETHUSDT-20mdb-2020-05.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-06.csv'
    out = 'ETHUSDT-20mdb-2020-06.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-07.csv'
    out = 'ETHUSDT-20mdb-2020-07.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-08.csv'
    out = 'ETHUSDT-20mdb-2020-08.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-09.csv'
    out = 'ETHUSDT-20mdb-2020-09.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-10.csv'
    out = 'ETHUSDT-20mdb-2020-10.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-11.csv'
    out = 'ETHUSDT-20mdb-2020-11.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2020-12.csv'
    out = 'ETHUSDT-20mdb-2020-12.csv'
    db_csv(read, sample, out)

    read = trades + '/ETHUSDT-trades-2021-01.csv'
    out = 'ETHUSDT-20mdb-2021-01.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-02.csv'
    out = 'ETHUSDT-20mdb-2021-02.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-03.csv'
    out = 'ETHUSDT-20mdb-2021-03.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-04.csv'
    out = 'ETHUSDT-20mdb-2021-04.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-05.csv'
    out = 'ETHUSDT-20mdb-2021-05.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-06.csv'
    out = 'ETHUSDT-20mdb-2021-06.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-07.csv'
    out = 'ETHUSDT-20mdb-2021-07.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-08.csv'
    out = 'ETHUSDT-20mdb-2021-08.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-09.csv'
    out = 'ETHUSDT-20mdb-2021-09.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-10.csv'
    out = 'ETHUSDT-20mdb-2021-10.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-11.csv'
    out = 'ETHUSDT-20mdb-2021-11.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2021-12.csv'
    out = 'ETHUSDT-20mdb-2021-12.csv'
    db_csv(read, sample, out)

    read = trades + '/ETHUSDT-trades-2022-01.csv'
    out = 'ETHUSDT-20mdb-2022-01.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-02.csv'
    out = 'ETHUSDT-20mdb-2022-02.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-03.csv'
    out = 'ETHUSDT-20mdb-2022-03.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-04.csv'
    out = 'ETHUSDT-20mdb-2022-04.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-05.csv'
    out = 'ETHUSDT-20mdb-2022-05.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-06.csv'
    out = 'ETHUSDT-20mdb-2022-06.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-07.csv'
    out = 'ETHUSDT-20mdb-2022-07.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-08.csv'
    out = 'ETHUSDT-20mdb-2022-08.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-09.csv'
    out = 'ETHUSDT-20mdb-2022-09.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-10.csv'
    out = 'ETHUSDT-20mdb-2022-10.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-11.csv'
    out = 'ETHUSDT-20mdb-2022-11.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2022-12.csv'
    out = 'ETHUSDT-20mdb-2022-12.csv'
    db_csv(read, sample, out)

    read = trades + '/ETHUSDT-trades-2023-01.csv'
    out = 'ETHUSDT-20mdb-2023-01.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2023-02.csv'
    out = 'ETHUSDT-20mdb-2023-02.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2023-03.csv'
    out = 'ETHUSDT-20mdb-2023-03.csv'
    db_csv(read, sample, out)
    read = trades + '/ETHUSDT-trades-2023-04.csv'
    out = 'ETHUSDT-20mdb-2023-04.csv'
    db_csv(read, sample, out)


raw = 'D:/crypto_DATA/raw/ETHUSDT/trades/csv'
dbs = 'D:/crypto_DATA/raw/ETHUSDT/dbs/20mdb/'

# sequence(raw, 20000000)
# c = csv_merger(dbs)
# c.to_csv('ETHUSDT_20mdb_2020_2023.csv')
