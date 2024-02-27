from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, normalizer
from data_forming import events_data, signal
import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def model_test(comb, X_tr, X_ts, Y_tr, Y_ts, md):
    X_trc, X_tsc = X_tr.copy(), X_ts.copy()
    X_trs, X_tss = X_trc[comb], X_tsc[comb]
    X_trn, X_tsn = normalizer(X_trs), normalizer(X_tss)
    if 'bb_cross' in comb:
        X_trn.bb_cross, X_tsn.bb_cross = X_tr.bb_cross, X_ts.bb_cross
    if md == 'MLP':
        Model = MLPClassifier()
        Model.fit(X_trn, Y_tr)
        predictions = Model.predict(X_tsn)
        report = classification_report(Y_ts, predictions, target_names=['0', '1'], output_dict=True)
        return report
    elif md == 'GBC':
        Model = GradientBoostingClassifier(max_depth=len(comb), random_state=42)
        Model.fit(X_trn, Y_tr)
        predictions = Model.predict(X_tsn)
        report = classification_report(Y_ts, predictions, target_names=['0', '1'], output_dict=True)
        return report


def uniqueCombinations(full_elements, std_elements, plethos):
    c = []
    combinations = []
    full_lst = list(itertools.combinations(full_elements, plethos))
    if std_elements is not None:
        for i in full_lst:
            c.append(list(i) + std_elements)
    else:
        for i in full_lst:
            c.append(list(i))
    for i in c:
        u = []
        for e in i:
            if e not in u:
                u.append(e)
        combinations.append(u)
    return combinations


def report_generator(plethos, md, X_tr, X_ts, Y_tr, Y_ts, full_feats, std_feats):
    """
    Takes a full features list and a standard features list and tests X combinations of the features in ml models.
    In case of X combinations from full list only give plethos X int (num of feats) and std_feats = None
    In case of standard features give std_feats list and feature_columns=None to splitter
    :param plethos:
    :param md:
    :param X_tr:
    :param X_ts:
    :param Y_tr:
    :param Y_ts:
    :param full_feats:
    :param std_feats:
    :return:
    """
    combinations = uniqueCombinations(full_feats, std_feats, plethos)
    print('Combinations:', len(combinations))
    r = []
    for i in tqdm(combinations):
        rep = model_test(i, X_tr, X_ts, Y_tr, Y_ts, md)
        r.append((i, rep))
    reports = pd.DataFrame({'reports': r})
    reports['features'] = reports.apply(lambda x: x[0][0], axis=1)
    reports['precision0'] = reports.apply(lambda x: x[0][1]['0']['precision'], axis=1)
    reports['recall0'] = reports.apply(lambda x: x[0][1]['0']['recall'], axis=1)
    reports['precision1'] = reports.apply(lambda x: x[0][1]['1']['precision'], axis=1)
    reports['recall1'] = reports.apply(lambda x: x[0][1]['1']['recall'], axis=1)
    reports.drop(columns=['reports'], axis=1, inplace=True)
    return reports


def research_features(selected_features, eligible_features, plethos, mode, prt, events):
    X_train, X_test, Y_train, Y_test = spliter(events, signal, prt, feature_columns=eligible_features)

    full_features = X_train.columns

    reps = report_generator(plethos, mode, X_train, X_test, Y_train, Y_test, full_features, selected_features)
    print('max precision0:', reps.loc[reps['precision0'].idxmax()])
    print('max recall0:', reps.loc[reps['recall0'].idxmax()])
    print('max precision1:', reps.loc[reps['precision1'].idxmax()])
    print('max recall1:', reps.loc[reps['recall1'].idxmax()])
    print('max precision0:')
    print(reps.tail(10).sort_values(by=['precision0']))
    print('max recall0:')
    print(reps.tail(10).sort_values(by=['recall0']))
    print('max precision1:')
    print(reps.tail(10).sort_values(by=['precision1']))
    print('max recall1:')
    print(reps.tail(10).sort_values(by=['recall1']))


s = ['TrD9', 'bb_cross']
b = ['TrD3', 'bb_cross']

research_features(None, 'All', 2, 'MLP', 1, events_data)

# 5
# tEvents / minRet 0.014
# [TrD9, bb_cross] 0.949192 0.958042 0.843478 0.815126
# [TrD3, bb_cross] 0.948718 0.948718 0.815126 0.815126

# Sell / tEvents / minRet 0
# [TrD9, bb_cross] 0.936667 0.962329 0.837037 0.748344
# [TrD3, bb_cross] 0.936134 0.953767 0.807143 0.748344
# [bb_cross, MAV_signal, TrD9] 0.936667 0.962329 0.837037 0.748344
# [TrD3, St4H, TrD9, bb_cross] 0.942664 0.957192 0.823944 0.774834
# [Dema9, VtrD6, TrD9, bb_cross] 0.843796 0.989726 0.88 0.291391
# [srl_corr, TrD3, TrD9, bb_cross] 0.942568 0.955479 0.818182 0.774834

# Buy bb_cross != 0 / minRet 0
# [TrD3, bb_cross]  0.795699 0.850575 0.80303 0.736111
# [roc10, TrD3] 0.795699 0.850575 0.80303 0.736111
# [St4H, TrD3, bb_cross]    0.802139  0.862069    0.816794  0.743056
# [%K, TrD3, bb_cross]  0.803191  0.867816 0.823077 0.743056
# [4H%DS, TrD3, bb_cross]  0.807292 0.890805 0.849206 0.743056
# [%K, roc10, TrD3, bb_cross] 0.807487 0.867816 0.824427 0.75

# 4
# tEvents / minRet 0
# [TrD3, bb_cross]
# precision0             0.95082
# recall0               0.926941
# precision1            0.728814
# recall1               0.803738

# 3
# [4Hmacd, bb_cross]
# precision0              0.962466
# recall0                 0.831019
# precision1              0.575581
# recall1                 0.876106
# [bb_cross, Vol_Vol]    0.911111  0.854167    0.550000  0.681416
# [bb_cross, MAV_signal]    0.882619  0.905093    0.598039  0.539823

# 2
# [TrD3, bb_cross]
# precision0            0.952273
# recall0               0.961009
# precision1            0.838095
# recall1               0.807339
# [TrD13, bb_cross]
# precision0             0.941964
# recall0                 0.96789
# precision1              0.85567
# recall1                0.761468

# 1
# [atr, bb_cross]
# precision0           0.962536
# recall0              0.747204
# precision1           0.429293
# recall1              0.867347

# [4Hmacd, bb_cross]
# precision0              0.836466
# recall0                 0.995526
# precision1              0.846154
# recall1                 0.112245
# [bb_cross, MAV_signal]    0.911628  0.876957    0.521739  0.612245
# [bb_cross, MAV]    0.912037  0.881432    0.530973  0.612245
