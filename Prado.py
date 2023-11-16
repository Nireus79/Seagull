import pandas as pd
from Pradofun import getDailyVol, getTEvents, addVerticalBarrier, getEvents, getBins, metaBins, dropLabels,\
    get_up_cross, get_down_cross, bbands, get_up_cross_bol, get_down_cross_bol, df_rolling_autocorr, returns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report
import matplotlib.pyplot as plt
from data_forming import eth

# import platform
# from multiprocessing import cpu_count

# https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises/blob/master/notebooks/Labeling%20and%20MetaLabeling%20for%20Supervised%20Classification.ipynb

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# eth = pd.read_csv('csv/db/ETHUSDT_mdb_2020_2023.csv')
# eth = eth.drop(columns=['Unnamed: 0'], axis=1)
# eth.time = pd.to_datetime(eth.time, unit='ms')
# eth.set_index('time', inplace=True)
# eth = eth[~eth.index.duplicated(keep='first')]
# print(eth)

vertical_days = 1
span = 100
window = 20
bb_stddev = 2

# (a) Run cusum filter with threshold equal to std dev of daily returns
close = eth.Close.copy()
dailyVol = getDailyVol(close, span, vertical_days, 'ewm')
# print('dailyVol -----')
# print(dailyVol)
tEvents = getTEvents(close, h=dailyVol.mean())
# print('tEvents -----')
# print(tEvents)

# (b) Add vertical barrier
t1 = addVerticalBarrier(tEvents, close, numDays=vertical_days)
# print('t1 -----')
# print(t1)

# (c) Apply triple-barrier method where ptSl = [1,1] and t1 is the series created in 1.b
ptsl = [1, 1]  # profit-taking and stop loss limit multipliers
minRet = .01  # The minimum target return required for running a triple barrier search

# if platform.system() == "Windows":
#     cpus = 1
# else:
#     cpus = cpu_count() - 1
# print(platform.system())
# print(cpus)
cpus = 1
target = dailyVol

events = getEvents(close, tEvents, ptsl, target, minRet, cpus, t1, side=None)
# print('events -----')
# print(events)

# (d) Apply getBins to generate labels
labels = getBins(events, close)
# print('labels -----')
# print(labels)
# print('labels.bin.value_counts() -----')
# print(labels.bin.value_counts())

# ----------------------------------------------------------------------------------------------------------------------
# [3.2] Use snippet 3.8 to drop under-populated labels
clean_labels = dropLabels(labels, .05)
# print('clean_labels -----')
# print(clean_labels)
# print('clean_labels.bin.value_counts() -----')
# print(clean_labels.bin.value_counts())

# ----------------------------------------------------------------------------------------------------------------------
# [3.3] Adjust the getBins function to return a 0 whenever the vertical barrier is the one touched first.
labels_new = metaBins(events, close, t1)
# print('labels_new -----')
# print(labels_new)

# ----------------------------------------------------------------------------------------------------------------------
# [3.4] Develop moving average crossover strategy. For each obs. the model suggests a side but not size of the bet

print('MA STRATEGY -----')
fast_window = 9
slow_window = 20

close_df = (pd.DataFrame()
            .assign(price=close)
            .assign(fast=close.ewm(fast_window).mean())
            .assign(slow=close.ewm(slow_window).mean()))
# print('close_df -----')
# print(close_df)

up = get_up_cross(close_df)
down = get_down_cross(close_df)
f, ax = plt.subplots(figsize=(11, 8))

close_df.loc[:].plot(ax=ax, alpha=.5)
up.loc[:].plot(ax=ax, ls='', marker='^', markersize=7,
               alpha=0.75, label='upcross', color='g')
down.loc[:].plot(ax=ax, ls='', marker='v', markersize=7,
                 alpha=0.75, label='downcross', color='r')

ax.legend()

side_up = pd.Series(1, index=up.index)
side_down = pd.Series(-1, index=down.index)
side = pd.concat([side_up, side_down]).sort_index()
# print('side -----')
# print(side)

minRet = .01
ptsl = [1, 1]

dailyVol = getDailyVol(close_df['price'], span, vertical_days, 'ewm')
tEvents = getTEvents(close_df['price'], h=dailyVol.mean())
t1 = addVerticalBarrier(tEvents, close_df['price'], numDays=vertical_days)

# print('side -----')
# print(side)
# print('target -----')
# print(target)

ma_events = getEvents(close_df['price'], tEvents, ptsl, target, minRet, cpus, t1, side)
# print('ma_events -----')
# print(ma_events)
# print('ma_events.side.value_counts -----')
# print(ma_events.side.value_counts())

ma_side = ma_events.dropna().side
ma_bins = metaBins(ma_events, close_df['price'], t1).dropna()
# print('ma_bins -----')
# print(ma_bins)

Xx = pd.merge_asof(ma_bins, side.to_frame().rename(columns={0: 'side'}),
                   left_index=True, right_index=True, direction='forward')
# print('Xx -----')
# print(Xx)

# (b) Train Random Forest to decide whether to trade or not {0,1}
# since underlying model (crossing m.a.) has decided the side, {-1,1}
X = ma_side.values.reshape(-1, 1)
# X = Xx.side.values.reshape(-1,1)
y = ma_bins.bin.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_estimator = 10000
RANDOM_STATE = 777
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator, criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print('classification_report -----')
print(classification_report(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# [3.5] Develop mean-reverting Bollinger Band Strategy.
# For each obs. model suggests a side but not size of the bet.

print('BB STRATEGY')
bb_df = pd.DataFrame()
bb_df['price'], bb_df['ave'], bb_df['upper'], bb_df['lower'] = bbands(close, window=window, numsd=vertical_days)
bb_df.dropna(inplace=True)
# print('bb_df -----')
# print(bb_df)

bb_down = get_down_cross_bol(bb_df, 'price')
bb_up = get_up_cross_bol(bb_df, 'price')

# (a) Derive meta-labels for ptSl=[0,2] and t1 where numdays=1. Use as trgt dailyVol.
bb_side_up = pd.Series(-1, index=bb_up.index)  # sell on up cross for mean reversion
bb_side_down = pd.Series(1, index=bb_down.index)  # buy on down cross for mean reversion
bb_side_raw = pd.concat([bb_side_up, bb_side_down]).sort_index()
# print('bb_side_raw -----')
# print(bb_side_raw)

minRet = .01
ptsl = [2, 1]

bb_events = getEvents(close, tEvents, ptsl, target, minRet, cpus, t1=t1, side=bb_side_raw)
# print('bb_events -----')
# print(bb_events)

bb_side = bb_events.dropna().side
# print('bb_side -----')
# print(bb_side)
# print('bb_side.value_counts -----')
# print(bb_side.value_counts())

bb_bins = getBins(bb_events, close).dropna()
# print('bb_bins -----')
# print(bb_bins)
# print('bb_bins.bin.value_counts -----')
# print(bb_bins.bin.value_counts())


# (b) train random forest to decide to trade or not. Use features: volatility, serial correlation, and the crossing
# moving averages from exercise 2.

# df_rolling_autocorr(d1, window=21).dropna().head()
srl_corr = df_rolling_autocorr(returns(close), window=window).rename('srl_corr')
# print('srl_corr -----')
# print(srl_corr)

features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
# print('features -----')
# print(features)

Xy = (pd.merge_asof(features, bb_bins[['bin']],
                    left_index=True, right_index=True,
                    direction='forward').dropna())
# print('Xy -----')
# print(Xy)
# print('Xy.bin.value_counts -----')
# print(Xy.bin.value_counts())

X = Xy.drop('bin', axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_estimator = 10000
RANDOM_STATE = 777
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator, criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print('classification_report -----')
print(classification_report(y_test, y_pred, target_names=['no_trade', 'trade']))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# (c) What is accuracy of predictions from primary model if the secondary model does not filter bets?
# What is classification report?
minRet = .01
ptsl = [0, 2]
bb_events = getEvents(close, tEvents, ptsl, target, minRet, cpus, t1=t1, side=None)
# print('bb_events -----')
# print(bb_events)

bb_bins = getBins(bb_events, close).dropna()
# print('bb_bins -----')
# print(bb_bins)

features = (pd.DataFrame()
            .assign(vol=bb_events.trgt)
            .assign(ma_side=ma_side)
            .assign(srl_corr=srl_corr)
            .drop_duplicates()
            .dropna())
# print('features -----')
# print(features)

Xy = (pd.merge_asof(features, bb_bins[['bin']],
                    left_index=True, right_index=True,
                    direction='forward').dropna())
# print('Xy -----')
# print(Xy)

# run model
X = Xy.drop('bin', axis=1).values
y = Xy['bin'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_estimator = 10000
RANDOM_STATE = 777
rf = RandomForestClassifier(max_depth=2, n_estimators=n_estimator, criterion='entropy', random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print('classification_report -----')
print(classification_report(y_test, y_pred, target_names=['no_trade', 'trade']))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
