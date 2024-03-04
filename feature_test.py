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

    if 'bb_cross' in comb:
        X_trc.drop(columns=['bb_cross'], axis=1, inplace=True)
        X_tsc.drop(columns=['bb_cross'], axis=1, inplace=True)
        X_trn, X_tsn = normalizer(X_trs), normalizer(X_tss)
        X_trn['bb_cross'], X_tsn['bb_cross'] = X_tr.bb_cross, X_ts.bb_cross
    else:
        X_trn, X_tsn = normalizer(X_trs), normalizer(X_tss)
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


events_data = events_data.loc[events_data['bb_cross'] != 0]
print('Feature test events')
print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('event data min ret', events_data.ret.min())
print('event data max ret', events_data.ret.max())
print('event data mean ret', events_data.ret.mean())


# [TrD3, bb_cross] 0.87037 0.770492 0.762712 0.865385
# [TrD3, TrD9, bb_cross] 0.872727 0.786885 0.775862 0.865385
b0 = ['macd', 'TrD3', 'TrD13', 'bb_cross']  # 0.877193 0.819672 0.803571 0.865385
b1 = ['macd', 'vdiff', 'TrD6', 'bb_cross']  # 0.875 0.803279 0.789474 0.865385
research_features(None, 'All', 3, 'MLP', 5, events_data)
