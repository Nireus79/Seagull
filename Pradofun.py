import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime as dt
from tqdm import tqdm
import time
import sys
from dask import dataframe as dd


# Chapter 3 ------------------------------------------------------------------------------------------------------------
def linParts(numAtoms, numThreads):
    """partition of atoms with a single loop"""
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nestedParts(numAtoms, numThreads, upperTriang=False):
    """partition of atoms with an inner loop"""
    parts, numThreads_ = [0], min(numThreads, numAtoms)
    for num in range(numThreads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part ** .5) / 2.
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upperTriang:  # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


def expandCall(kargs):
    """Expand the arguments of a callback function, kargs['func']"""
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


def processJobs_(jobs):
    """Run jobs sequentially, for debugging"""
    out = []
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out


def reportProgress(jobNum, numJobs, time0, task):
    """Report progress as asynch jobs are completed"""
    msg = [float(jobNum) / numJobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    if jobNum < numJobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')
    return


def processJobs(jobs, task=None, numThreads=24):
    """Run in parallel. jobs must contain a 'func' callback, for expandCall"""
    if task is None:
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes=numThreads)
    outputs, out, time0 = pool.imap_unordered(expandCall, jobs), [], time.time()
    # Process asyn output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return out


def mpPandasObj(func, pdObj, numThreads=24, mpBatches=1, linMols=True, **kargs):
    """
    if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    """

    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads=numThreads)
    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    for i in out:
        # df0 = df0.append(i) #  As of pandas 2.0, append (previously deprecated)
        df0 = pd.concat([df0, pd.DataFrame(i)], ignore_index=True)
    df0 = df0.sort_index()
    return df0


def applyPtSlOnT1(close, events, ptSl, molecule):
    """
    TRIPLE-BARRIER LABELING METHOD
    apply stop loss/profit taking, if it takes place before t1 (end of event)
    :param close: A pandas series of prices.
    :param events: A pandas dataframe, with columns,
         t1: The timestamp of vertical barrier. When the value is nan, there will not be a vertical barrier.
         trgt: The unit width of the horizontal barriers.
    :param ptSl: A list of two non-negative float values
        ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
        If 0, there will not be an upper barrier
        ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
        If 0, there will not be a lower barrier
    :param molecule: A list with the subset of event indices that will be processed by a single thread
        (a list consists of indices of events in dataframe "events".
         It is to remove unnecessary events such as rare events.)
    :return: a pandas dataframe containing the timestamps (if any) at which each barrier was touched
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def getDailyVol(close, span0, days):
    """
    Daily Volatility Estimator [3.1]
    daily vol re-indexed to close
    Original df0 = df0[df0 > 0] does not include first day indexes
    was changed to df0 = df0[df0 >= 0]
    :param days:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days))
    df0 = df0[df0 >= 0]  # df0 >= 0 includes first day in index
    # TODO check if original df0 = df0[df0 > 0] is correct
    df0 = (pd.Series(close.index[df0 - days], index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - days  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def getDailyVolCGPT(close, span0, rows):
    """
    The formula used in the function to calculate daily volatility is based on the exponential moving standard
    deviation (EMS). The function calculates daily volatility as an estimate of the standard deviation of daily
    returns.

Here's a breakdown of the formula and the steps involved:

Calculate the Number of Rows/Periods: The function takes a parameter called rows, which represents the number of
previous rows or periods to consider for calculating daily volatility.

Find Start Date for Volatility Calculation: The function searches for the index position where the calculation should
start. It does this by finding the index position where the time series (in this case, the close prices) is a certain
number of rows before the current time.

Create a Reindexed Series: The function creates a new Series called df0, which contains the index positions from the
original close Series corresponding to the start date determined in step 2.

Calculate Daily Returns: It calculates daily returns as the percentage change in the close prices between the start
date and the current date. This is done by taking the ratio of the close prices on the current date and the close
prices on the start date and subtracting 1. This step helps calculate the daily returns of the asset.

Calculate Exponential Moving Standard Deviation: Finally, it calculates the daily volatility by applying an
exponential moving standard deviation (EMS) to the daily returns. The ewm method is used for this purpose with a
given span0 parameter. The span0 parameter controls the weighting of the data points in the moving standard
deviation. It essentially determines how quickly the influence of older data diminishes. A smaller span0 value will
give more weight to recent data, while a larger span0 value will give more weight to historical data.

The resulting Series, named 'dailyVol', contains the estimated daily volatility. It measures the variability of daily
returns, providing an indication of the asset's risk and price fluctuations over time.

In summary, the formula is based on calculating the daily returns over a specified number of rows/periods and then
smoothing these returns using an exponential moving standard deviation to estimate the asset's daily volatility.
Daily Volatility Estimator [3.1] daily vol re-indexed to close Original df0 = df0[df0 > 0] does not include first day
indexes was changed to df0 = df0[df0 >= 0] :param rows: Number of rows to consider for calculating daily volatility
:param close: Series of close prices :param span0: Span parameter for exponential moving average (EMA) :return:
Series containing daily volatility
    """
    df0 = close.index.searchsorted(close.index - pd.DateOffset(rows))
    df0 = df0[df0 >= 0]  # df0 >= 0 includes the first row in the index

    df0 = (pd.Series(close.index[df0 - rows], index=close.index[close.shape[0] - df0.shape[0]:]))

    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception as e:
        print(f'Error: {e}\nPlease confirm there are no duplicate indices.')

    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    print(df0)
    return df0


def getTEvents(gRaw, h):
    """Symmetric CUSUM Filter [2.5.2.1]
    T events are the moments that a shift in
    the mean value of a measured quantity away from a target value.
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < -h.loc[i]:  # .loc[i] gives threshold relative to data['Volatility'].rolling(window).mean()
            sNeg = 0
            tEvents.append(i)
        elif sPos > h.loc[i]:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1, side):
    """
    we accept a new side optional argument (with default None), which contains the side of our bets
    as decided by the primary model. When side is not None, the function understands
    that meta-labeling is in play. Second, because now we know the side, we can effectively discriminate between
    profit taking and stop loss. The horizontal barriers do not
    need to be symmetric, as in Section 3.5. Argument ptSl is a list of two non-negative
    float values, where ptSl[0] is the factor that multiplies trgt to set the width of
    the upper barrier, and ptSl[1] is the factor that multiplies trgt to set the width
    of the lower barrier. When either is 0, the respective barrier is disabled. Snippet 3.6
    implements these enhancements.
    When side is given, its length appears greater than trgt and error appears as
    side_ = side.loc[trgt.index] cannot detect missing indexes.
    loc[trgt.index] was changed to loc[side.index]
    :param close:
    :param tEvents:
    :param ptSl:
    :param trgt:
    :param minRet:
    :param numThreads:
    :param t1:
    :param side:
    :return: df with event timestamp t1 timestamp target and side
    """
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        common_indexes = set(side).intersection(trgt.index)
        common_indexes = list(common_indexes)
        print(common_indexes)
        # control of common indexes between target and side before filtering
        side_, ptSl_ = trgt.loc[common_indexes], ptSl[:2]
        # TODO check if original side_ = side.loc[trgt.index] is correct
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt']))
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                      numThreads=numThreads, close=close, events=events,
                      ptSl=ptSl_)
    # events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan (in pandas 2.0 does not!!!)
    if side is None:
        events = events.drop('side', axis=1)
    return events


def addVerticalBarrier(tEvents, close, numDays):
    """For each index in tEvents,
it finds the timestamp of the next price bar at or immediately after a number
of days numDays. This vertical barrier can be passed as optional argument t1
in getEvents."""
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1


def getBins(events, close):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


def metaBins(events, close, t1):
    """
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    -t1 is original vertical barrier series
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    """
    # 1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    # print(events_)
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    # print(px)
    # 2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        out['ret'] *= events_['side']  # meta-labeling
    out['bin'] = np.sign(out['ret'])

    if 'side' not in events_:
        """only applies when not meta-labeling
        to update bin to 0 when vertical barrier is touched, we need the original
        vertical barrier series since the events['t1'] is the time of first
        touch of any barrier and not the vertical barrier specifically.
        The index of the intersection of the vertical barrier values and the
        events['t1'] values indicate which bin labels needs to be turned to 0"""
        vtouch_first_idx = events[events['t1'].isin(t1.values)].index
        out.loc[vtouch_first_idx, 'bin'] = 0.
    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # meta-labeling
    return out


def dropLabels(events, minPct):
    """apply weights, drop labels with insufficient examples"""
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label: ', df0.argmin(), df0.min())
        events = events[events['bin'] != df0.argmin()]
        break  # TODO check if added break is correct (break not in snippet)
    return events


def getDailyTimeBarVolatility(close, span0):
    """
    DYNAMIC THRESHOLDS for time bars
    daily vol, reindexed to close
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


def getDollarBarVolatilityByCandle(close, span0, market_value):
    """
    DYNAMIC THRESHOLDS for dollar bars
    volatility by market value, reindexed to close
    :param market_value:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - market_value)
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=span0).std()
    return df0


# Chapter 4 ------------------------------------------------------------------------------------------------------------

def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed

    Any event that starts before t1[molecule].max() impacts the count.
    """
    # 1) find events that span the period [molecule[0],molecule[-1]]
    t1 = t1.fillna(closeIdx[-1])  # unclosed events still must impact other weights
    t1 = t1[t1 >= molecule[0]]  # events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()]  # events that start at or before t1[molecule].max()
    # 2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])
    for tIn, tOut in t1.items(): count.loc[tIn:tOut] += 1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1, numCoEvents, molecule):
    """Derive avg. uniqueness over the events lifespan"""
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()
    return wght


def getIndMatrix(barIx, t1):
    """Get Indicator matrix"""
    indM = (pd.DataFrame(0, index=barIx, columns=range(t1.shape[0])))
    for i, (t0, t1) in enumerate(t1.items()):
        indM.loc[t0:t1, i] = 1.
    return indM


def getAvgUniqueness(indM):
    """Average uniqueness from indicator matrix"""
    c = indM.sum(axis=1)  # concurrency
    u = indM.div(c, axis=0)  # uniqueness
    avgU = u[u > 0].mean()  # avg. uniqueness
    return avgU


def seqBootstrap(indM, sLength=None):
    """Generate a sample via sequential bootstrap"""
    if sLength is None: sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # draw prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


def mpSampleW(t1, numCoEvents, close, molecule):
    """Derive sample weight by return attribution"""
    ret = np.log(close).diff()  # log-returns, so that they are additive
    wght = pd.Series(index=molecule)
    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()
    return wght.abs()


def getTimeDecay(tW, clfLastW=1.):
    """apply piecewise-linear decay to observed uniqueness (tW)
    the newest observation gets weight=1, the oldest observation gets weight=clfLastW"""
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print(const, slope)
    return clfW


def dask_resample(ser, freq='L'):
    dds = dd.from_pandas(ser, chunksize=len(ser) // 100)
    tdf = (dds
           .resample(freq)
           .mean()
           .dropna()
           ).compute()
    return tdf


def pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return unpickle_method, (func_name, obj, cls)


def unpickle_method(func_name, obj, cls):
    # TODO function modified. Check function
    func = cls.__dict__[func_name]
    for cls in cls.mro():
        try:
            func
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


def main():
    np.random.seed(12121)  # fix seed as results are unstable
    t1 = pd.Series([2, 3, 5], index=[0, 2, 4])  # t0,t1 for each feature obs
    barIx = range(t1.max() + 1)  # index of bars
    indM = getIndMatrix(barIx, t1)
    phi_random = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi_random)
    print(f'Standard uniqueness: {getAvgUniqueness(indM[phi_random]).mean():.4f}')
    phi_seq = seqBootstrap(indM)
    print(phi_seq)
    print(f'Sequential uniqueness: {getAvgUniqueness(indM[phi_seq]).mean():.4f}')


def get_z_down_cross(df, lim):
    crit1 = df.z.shift(1) < -lim
    crit2 = df.z > -lim
    return df.z[crit1 & crit2]


def get_z_up_cross(df, lim):
    crit1 = df.z.shift(1) < lim
    crit2 = df.z > lim
    return df.z[crit1 & crit2]


def get_up_cross(df):
    crit1 = df.fast.shift(1) < df.slow.shift(1)
    crit2 = df.fast > df.slow
    return df.fast[crit1 & crit2]


def get_down_cross(df):
    crit1 = df.fast.shift(1) > df.slow.shift(1)
    crit2 = df.fast < df.slow
    return df.fast[crit1 & crit2]


def get_up_cross_bol(df, col):
    # col is price column
    crit1 = df[col].shift(1) < df.upper.shift(1)
    crit2 = df[col] > df.upper
    return df[col][crit1 & crit2]


def get_down_cross_bol(df, col):
    # col is price column
    crit1 = df[col].shift(1) > df.lower.shift(1)
    crit2 = df[col] < df.lower
    return df[col][crit1 & crit2]


def bbands(price, window=None, width=None, numsd=None):
    """ returns average, upper band, and lower band"""
    ave = price.rolling(window).mean()
    sd = price.rolling(window).std(ddof=0)
    if width:
        upband = ave * (1 + width)
        dnband = ave * (1 - width)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)
    if numsd:
        upband = ave + (sd * numsd)
        dnband = ave - (sd * numsd)
        return price, np.round(ave, 3), np.round(upband, 3), np.round(dnband, 3)


def returns(s):
    arr = np.diff(np.log(s))
    return pd.Series(arr, index=s.index[1:])


def df_rolling_autocorr(df, window, lag=1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""

    return (df.rolling(window=window)
            .corr(df.shift(lag)))  # could .dropna() here
