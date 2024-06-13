import pandas as pd
import numpy as np
import multiprocessing as mp
import datetime as dt
from tqdm import tqdm
import time
import sys
# from dask import dataframe as dd


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
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + str(round(msg[1], 2)) + \
          ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
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


def getDailyVol(close, span0, delta):
    """
    Daily Volatility Estimator [3.1]
    daily vol re-indexed to close
    Original df0 = df0[df0 > 0] does not include first day indexes
    was changed to df0 = df0[df0 >= 0]
    :param delta:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=delta))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - delta], index=close.index[close.shape[0] - df0.shape[0]:]))
    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - delta  # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def getTEvents(gRaw, h, ptsl):
    """Symmetric CUSUM Filter [2.5.2.1]
    T events are the moments that a shift in
    the mean value of a measured quantity away from a target value.

    The getTEvents function is used to identify "T events" in a time series. T events are moments when there is a
    significant shift in the mean value of a measured quantity away from a target value. This function uses a
    symmetric CUSUM (cumulative sum) filter to detect these events. Here are the steps and the formula involved in
    this function:

Initialize Variables: Initialize three empty lists, tEvents, sPos, and sNeg to keep track of the identified events
and the cumulative sums of positive and negative changes.

Calculate Differences: Compute the differences between consecutive values of the logarithm of the input series gRaw.
This is done using np.log(gRaw).diff().dropna(). The diff variable now contains the log returns.

Iterate Through Differences: Iterate through the differences in log returns, starting from the second index (index 1)
because the first value in diff is NaN.

Cumulative Sum of Positive and Negative Changes:

sPos represents the cumulative sum of positive changes in log returns. sNeg represents the cumulative sum of negative
changes in log returns. At each step, sPos and sNeg are updated by adding the current value of the log return. The
float function is used to convert the cumulative sums to floats. The max(0., pos) and min(0., neg) functions ensure
that these cumulative sums never go below zero. If a positive cumulative sum becomes negative, it's set to zero,
and if a negative cumulative sum becomes positive, it's set to zero as well. Event Detection:

Check if sNeg goes below a threshold -h.loc[i]. The threshold is relative to h, which is a pandas Series containing
some form of volatility measurement. If sNeg crosses below the threshold, it indicates a downward shift in the mean
value, so sNeg is reset to zero, and the current index i is added to the tEvents list. Similarly, if sPos goes above
the threshold, it indicates an upward shift in the mean value, and sPos is reset to zero, and i is added to the
tEvents list. Return T Events: The function returns a pd.DatetimeIndex object containing the timestamps of the
identified T events.

In summary, the function detects T events in a time series by monitoring changes in log returns. When the cumulative
sums of these changes cross certain thresholds (h.loc[i]), it signifies a shift in the mean value,
and the corresponding timestamp is recorded as a T event. This is a common technique used in event-driven finance and
signal processing to detect significant changes in time series data.
    """
    h['pt'] = h.apply(lambda x: x * ptsl[0] if ptsl[0] > 0 else x)
    h['sl'] = h.apply(lambda x: -x * ptsl[1] if ptsl[1] > 0 else -x)
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw.astype('float64')).diff().dropna()
    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        sPos, sNeg = max(0., pos), min(0., neg)
        if sNeg < h['sl'].loc[i]:  # .loc[i] # gives threshold relative to data['Volatility'] not a fixed mean
            sNeg = 0
            tEvents.append(i)
        elif sPos > h['pt'].loc[i]:
            sPos = 0
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


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


def getEvents(tEvents, ptSl, trgt, minRet, t1, side):
    """
    we accept a new side optional argument (with default None), which contains the side of our bets
    as decided by the primary model. When side is not None, the function understands
    that meta-labeling is in play. Second, because now we know the side, we can effectively discriminate between
    profit taking and stop loss. The horizontal barriers do not
    need to be symmetric, as in Section 3.5. Argument ptSl is a list of two non-negative
    float values, where ptSl[0] is the factor that multiplies trgt to set the width of
    the upper barrier, and ptSl[1] is the factor that multiplies trgt to set the width
    of the lower barrier. When either is 0, the respective barrier is disabled. (Snippet 3.6)

    The getEvents function appears to be a part of a financial analysis or trading system. It's responsible for
    generating events, which include information about timestamps, target levels, and sides (buy or sell) for trading
    signals. It makes use of several parameters and logic to form these events. Let me break down the function step
    by step:

Get the Target Levels (trgt): The function first extracts the target levels from the trgt series for the timestamps
in tEvents. It then filters out target levels that are less than a specified minimum return threshold (minRet).

Get the T1 Timestamps (Max Holding Period): If t1 is not provided as an argument (i.e., t1 is set to False),
it initializes a Series with NaN values for the timestamps in tEvents. This represents the maximum holding period for
each event.

Determine Sides and Profit-Taking/Stop Loss Factors:

If the side parameter is not provided (i.e., it's None), it assumes a default side of 1 (e.g., buy) and sets
profit-taking and stop-loss factors to ptSl[0] for both the upper and lower barriers. If the side parameter is
provided, it attempts to match the timestamps in side with the timestamps in trgt. It keeps only the common
timestamps. It creates a side_ Series containing the sides for the common timestamps, and it sets the profit-taking
and stop-loss factors based on ptSl[:2]. Form Event Objects:

The function concatenates the t1, trgt, and side_ Series (or side if side is None) to form an events DataFrame. It
drops rows with NaN values in the 'trgt' column. Parallel Processing of Events: The mpPandasObj function appears to
be used for parallel processing. It applies the applyPtSlOnT1 function to each row of the events DataFrame in
parallel. The parameters passed include close prices, the events DataFrame, and profit-taking/stop-loss factors.

Post-Processing: It appears that the result of parallel processing is used to calculate the 't1' column for the
events DataFrame. It might be determining when a position is exited based on the trading strategy.

Final Output and Cleanup: If the side parameter was not provided initially (i.e., it's None), the 'side' column is
dropped from the events DataFrame. The function returns the events DataFrame, which contains event timestamps ('t1'),
target levels ('trgt'), and side information.

In summary, the getEvents function is responsible for generating events for a trading strategy based on a combination
of target levels, sides (buy/sell), and profit-taking/stop-loss factors. It also supports parallel processing for
efficiency. The specific logic for trading and event generation would depend on the larger context of the trading
strategy and the specific implementation of applyPtSlOnT1. :param close: :param tEvents: :param ptSl: :param trgt:
    :param side:
    :param t1:
    :param trgt:
    :param ptSl:
    :param tEvents:
:param minRet: :param numThreads: :param t1: :param side: :return: df with event timestamp t1 timestamp target and side
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
        common_indexes = set(side.index).intersection(trgt.index)
        common_indexes = list(common_indexes)
        # control of common indexes between target and side before filtering
        side_, ptSl_ = side.loc[common_indexes], ptSl[:2]
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt']))
    # df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
    #                   numThreads=numThreads, close=close, events=events,
    #                   ptSl=ptSl_)
    # events['t1'] = df0.dropna(how='all').min(axis=1)
    if side is None:
        events = events.drop('side', axis=1)
    return events  # .dropna()


def addVerticalBarrier(tEvents, close, delta):
    """For each index in tEvents,
it finds the timestamp of the next price bar at or immediately after a number
of days numDays. This vertical barrier can be passed as optional argument t1
in getEvents."""
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=delta))
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


def getDailyTimeBarVolatility(close, span0, delta):
    """
    DYNAMIC THRESHOLDS for time bars
    daily vol, reindexed to close
    :param delta:
    :param close:
    :param span0:
    :return:
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=delta))
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


# def dask_resample(ser, freq='L'):
#     dds = dd.from_pandas(ser, chunksize=len(ser) // 100)
#     tdf = (dds
#            .resample(freq)
#            .mean()
#            .dropna()
#            ).compute()
#     return tdf


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


def getRndT1(numObs, numBars, maxH):
    # random t Series
    t = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t.loc[ix] = val
    return t.sort_index()


def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t = getRndT1(numObs, numBars, maxH)
    barIx = range(t.max() + 1)
    indM = getIndMatrix(barIx, t)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}


def man():
    t = pd.Series([2, 3, 5], index=[0, 2, 4])  # t0,t1 for each feature obs
    barIx = range(t.max() + 1)  # index of bars
    indM = getIndMatrix(barIx, t)
    phi = np.random.choice(indM.columns, size=indM.shape[1])
    print(phi)
    print('Standard uniqueness:', getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print(phi)
    print('Sequential uniqueness:', getAvgUniqueness(indM[phi]).mean())
    return phi


def mainMC(numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24):
    # Monte Carlo experiments
    jobs = []
    for i in range(int(numIters)):
        job = {'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
        if numThreads == 1:
            out = processJobs_(jobs)
        else:
            out = processJobs(jobs, numThreads=numThreads)
        print(pd.DataFrame(out).describe())
        return pd.DataFrame(out).describe()
