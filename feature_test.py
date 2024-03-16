from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from toolbox import spliter, normalizer
from data_forming import events_data, signal, delta
import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np

# pd.set_option('display.max_rows', None)
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
    reports['f1-score0'] = reports.apply(lambda x: x[0][1]['0']['f1-score'], axis=1)
    reports['precision1'] = reports.apply(lambda x: x[0][1]['1']['precision'], axis=1)
    reports['recall1'] = reports.apply(lambda x: x[0][1]['1']['recall'], axis=1)
    reports['f1-score1'] = reports.apply(lambda x: x[0][1]['1']['f1-score'], axis=1)
    reports.drop(columns=['reports'], axis=1, inplace=True)
    return reports


def research_features(selected_features, eligible_features, plethos, mode, prt, events):
    X_train, X_test, Y_train, Y_test = spliter(events, signal, prt, eligible_features, delta)
    full_features = X_train.columns
    reps = report_generator(plethos, mode, X_train, X_test, Y_train, Y_test, full_features, selected_features)
    # print('max precision0:', reps.loc[reps['precision0'].idxmax()])
    # print('max recall0:', reps.loc[reps['recall0'].idxmax()])
    # print('max f1 0:', reps.loc[reps['f1-score0'].idxmax()])
    # print('max precision1:', reps.loc[reps['precision1'].idxmax()])
    # print('max recall1:', reps.loc[reps['recall1'].idxmax()])
    # print('max f1 1:', reps.loc[reps['f1-score1'].idxmax()])
    # print('max precision0:')
    # print(reps.tail(1).sort_values(by=['precision0']))
    # print('max recall0:')
    # print(reps.tail(5).sort_values(by=['recall0']))
    # print('max f1 0:')
    # print(reps.tail(5).sort_values(by=['f1-score0']))
    # print('max precision1:')
    # print(reps.tail(5).sort_values(by=['precision1']))
    # print('max recall1:')
    # print(reps.tail(5).sort_values(by=['recall1']))
    # print('max f1 1:')
    # print(reps.tail(5).sort_values(by=['f1-score1']))
    # print(reps)
    return reps


events_data = events_data.loc[events_data['bb_cross'] != 0]
print('Feature test events')
print('event 0', np.sum(np.array(events_data[signal]) == 0, axis=0))
print('event 1', np.sum(np.array(events_data[signal]) == 1, axis=0))
print('event data min ret', events_data.ret.min())
print('event data max ret', events_data.ret.max())
print('event data mean ret', events_data.ret.mean())


def cross_elimination(selected_features, eligible_features, plethos, mode, events):
    cross = pd.DataFrame()
    for c in tqdm(range(1, 6)):
        res = research_features(selected_features, eligible_features, plethos, mode, c, events)
        cross['feats_' + str(c)] = res['features']
        cross['f1-score0_' + str(c)] = res['f1-score0']
        cross['f1-score1_' + str(c)] = res['f1-score1']
    cross['f1_0_mean'] = cross.apply(lambda x:
                                     (x['f1-score0_1'] + x['f1-score0_2']
                                      + x['f1-score0_3'] + x['f1-score0_4'] + x['f1-score0_5']) / 5, axis=1)
    cross['f1_1_mean'] = cross.apply(lambda x:
                                     (x['f1-score1_1'] + x['f1-score1_2']
                                      + x['f1-score1_3'] + x['f1-score1_4'] + x['f1-score1_5']) / 5, axis=1)
    cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
    print(cross.loc[cross['f1_0_mean'].idxmax()])
    print(cross.loc[cross['f1_1_mean'].idxmax()])
    print(cross.tail(5).sort_values(by=['f1_0_mean']))
    print(cross.tail(5).sort_values(by=['f1_1_mean']))


def MDI():
    X, Y = spliter(events_data, signal, 0, 'All', delta)
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    names = X.columns

    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))


# MDI()
# MDI importance
MDIB = ['TrD3', 'TrD6', 'TrD9', 'TrD13', 'TrD20',
        '4H%K', '4H%D', '4H_rsi',
        'rsi', '%D', 'vroc10', 'vsrl_corr',
        'macd', 'bb_l', 'bb_cross']
MDIF = ['bb_cross', 'TrD6', 'TrD9', 'TrD3', 'bb_l', 'bb_t', 'TrD13', 'TrD20', 'rsi', 'mom30', 'vsrl_corr']


cross_elimination(None, MDIB, 2, 'MLP', events_data)
# feats_0      [mom10, TrD6]
# f1_0_mean         0.844105
# f1_1_mean         0.778296
# feats_1      [TrD3, bb_cross]
# f1_0_mean            0.828591
# f1_1_mean            0.780924

# feats_0      [mom20, bb_sq, TrD9]
# f1_0_mean                0.847182
# f1_1_mean                0.763432
# feats_1      [roc20, TrD3, bb_cross]
# f1_0_mean                   0.840276
# f1_1_mean                   0.786383
# feats_0      [mom10, TrD6, bb_cross]
# f1_0_mean                   0.854347
# f1_1_mean                   0.790783
# feats_1      [vdiff, TrD6, bb_cross]
# f1_0_mean                   0.842361
# f1_1_mean                   0.796199

# feats_0      [TrD6, 4H_atr, MAV, bb_cross]
# f1_0_mean                         0.847973
# f1_1_mean                          0.76961
# feats_1      [TrD3, vdiff, Volatility, bb_cross]
# f1_0_mean                               0.833384
# f1_1_mean                               0.773562
