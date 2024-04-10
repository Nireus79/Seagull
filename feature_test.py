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


def k_mean(selected_features, eligible_features, plethos, mode, events, repetition):
    repetition += 1
    cross = pd.DataFrame()
    for c in tqdm(range(1, repetition)):
        res = research_features(selected_features, eligible_features, plethos, mode, c, events)
        cross[f'feats_{c}'] = res['features']
        cross[f'f1-score0_{c}'] = res['f1-score0']
        cross[f'f1-score1_{c}'] = res['f1-score1']
    cross['f1_0_mean'] = cross[[f'f1-score0_{i}' for i in range(1, repetition)]].mean(axis=1)
    cross['f1_1_mean'] = cross[[f'f1-score1_{i}' for i in range(1, repetition)]].mean(axis=1)
    cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
    print(cross.loc[cross['f1_0_mean'].idxmax()])
    print(cross.loc[cross['f1_1_mean'].idxmax()])
    print(cross.tail(5).sort_values(by=['f1_0_mean']))
    print(cross.tail(5).sort_values(by=['f1_1_mean']))


# def k_mean(selected_features, eligible_features, plethos, mode, events):
#     cross = pd.DataFrame()
#     for c in tqdm(range(1, 11)):
#         res = research_features(selected_features, eligible_features, plethos, mode, c, events)
#         cross['feats_' + str(c)] = res['features']
#         cross['f1-score0_' + str(c)] = res['f1-score0']
#         cross['f1-score1_' + str(c)] = res['f1-score1']
#     cross['f1_0_mean'] = cross.apply(lambda x:
#                                      (x['f1-score0_1'] + x['f1-score0_2'] + x['f1-score0_3'] + x['f1-score0_4']
#                                       + x['f1-score0_5'] + x['f1-score0_6'] + x['f1-score0_7'] + x['f1-score0_8']
#                                       + x['f1-score0_9'] + x['f1-score0_10']) / 10, axis=1)
#     cross['f1_1_mean'] = cross.apply(lambda x:
#                                      (x['f1-score1_1'] + x['f1-score1_2'] + x['f1-score1_3'] + x['f1-score1_4']
#                                       + x['f1-score1_5'] + x['f1-score1_6'] + x['f1-score1_7'] + x['f1-score1_8']
#                                       + x['f1-score1_9'] + x['f1-score1_10']) / 10, axis=1)
#     cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
#     print(cross.loc[cross['f1_0_mean'].idxmax()])
#     print(cross.loc[cross['f1_1_mean'].idxmax()])
#     print(cross.tail(5).sort_values(by=['f1_0_mean']))
#     print(cross.tail(5).sort_values(by=['f1_1_mean']))


def MDI(X, Y):
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    names = X.columns
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))


F003124 = ['TrD3', 'TrD6', 'TrD20', 'bb_t', '4H%K', 'srl_corr', 'rsi', 'roc30', 'roc10', '4H%D', 'bb_l', 'roc20',
           'diff',
           '4H_rsi', 'StD', 'vroc30', 'vsrl_corr', '%K', 'mom20', 'vmom30', '%D', '%DS', 'mom30', 'vroc10', 'vmom20',
           'vrsi', 'Volatility', 'event', 'MAV_signal', 'macd', 'MAV', 'TrD9', 'vdiff', 'Vol_Vol', 'TrD13', 'mom10',
           'vmacd']
S003124 = ['TrD3', '4Hmacd', 'srl_corr', 'mom20', 'bb_l']
B003124 = ['TrD6', 'TrD3', 'macd', 'vmacd', 'bb_cross']
F002624 = ['TrD3', 'bb_t', 'TrD6', 'TrD20', '4H%D', 'bb_l', 'srl_corr', 'rsi', 'vroc20', '%DS', 'TrD9', 'StD', 'mom10',
           'diff', 'vroc30', 'MAV', '%D', 'vsrl_corr', 'roc30', 'MAV_signal', 'Vol_Vol', 'roc20', '4H%DS', 'roc10',
           '4Hmacd', 'vdiff', 'vmacd', '4H%K', 'vroc10', 'mom20', 'vmom20', 'macd', 'Vtr6', 'TrD13', 'vmom30', 'event',
           '4H_rsi', '%K', 'vrsi', 'mom30']
S002624 = ['TrD6', 'mom20', 'roc30', 'MAV']
B002624 = ['TrD6', 'Tr6', 'vsrl_corr', 'srl_corr']
F002612 = ['TrD3', 'TrD9', 'TrD6', '4H%K', '4H_rsi', 'rsi', 'macd', '%D', '4Hmacd', 'vroc10', 'vsrl_corr', 'bb_l',
           'TrD13', '4H%D', 'roc10', 'TrD20', 'mom20', 'mom30', 'diff', 'bb_t', 'Vol_Vol', 'srl_corr', 'MAV_signal',
           '%K', 'MAV', 'StD', 'vdiff', 'roc20', '%DS', 'roc30', 'vmom20', 'vroc30', '4H%DS', 'vrsi', 'vroc20',
           'vmom30', 'St4H', 'vmacd']
S002612 = ['TrD6', 'St4H', 'mom20', 'macd', 'MAV']
B002612 = ['TrD3', '4Hmacd', 'mom20', 'Tr6', 'bb_l']

# MDI()
k_mean(B002612, 'All', 1, 'MLP', events_data, 10)

# 0.026 12
# feats_0      [TrD6, mom20]
# f1_0_mean         0.829747
# f1_1_mean         0.736691
# feats_0      [macd, TrD6, mom20]
# f1_0_mean               0.842513
# f1_1_mean               0.760887
# feats_      [St4H, TrD6, mom20, macd]
# f1_0_mean                     0.843465
# f1_1_mean                     0.756247
# feats_0       [MAV, TrD6, St4H, mom20, macd]
# f1_0_mean                           0.84818
# f1_1_mean                          0.757271

# feats_1      [TrD3, mom20]
# f1_0_mean         0.828354
# f1_1_mean         0.742622
# feats_1      [Tr6, TrD3, mom20]
# f1_0_mean              0.836615
# f1_1_mean              0.762914
# feats_1      [bb_l, TrD3, mom20, mom30, Tr6]
# f1_0_mean                           0.848538
# f1_1_mean                           0.784576
# feats_1      [4Hmacd, TrD3, mom20, Tr6, bb_l]
# f1_0_mean                            0.843267
# f1_1_mean                            0.771803

# --------------------------------------------------------------------------
# 0.026 24
# feats_0      [mom20, TrD6]
# f1_0_mean         0.762189
# f1_1_mean          0.65013
# feats_0      [roc30, TrD6, mom20]
# f1_0_mean                0.769173
# f1_1_mean                0.647389
# feats_0      [MAV, TrD6, mom20, roc30]
# f1_0_mean                      0.76689
# f1_1_mean                     0.641328

# feats_1      [Tr6, TrD6]
# f1_0_mean       0.762034
# f1_1_mean       0.661165
# feats_1      [vsrl_corr, TrD6, Tr6]
# f1_0_mean                  0.771196
# f1_1_mean                  0.672233
# feats_1      [srl_corr, TrD6, Tr6, vsrl_corr]
# f1_0_mean                            0.775404
# f1_1_mean                            0.675028
# -------------------------------------------------------------------------
# 0,031 24
# feats_0      [mom20, TrD3]
# f1_0_mean         0.764943
# f1_1_mean         0.627874
# feats_0      [bb_l, mom20, TrD3]
# f1_0_mean               0.776589
# f1_1_mean               0.629337
# feats_0      [4Hmacd, TrD3, mom20, bb_l]
# f1_0_mean                       0.781378
# f1_1_mean                        0.63296
# feats_0      [srl_corr, TrD3, 4Hmacd, mom20, bb_l]
# f1_0_mean                                  0.78583
# f1_1_mean                                 0.656225

# feats_1      [TrD3, bb_cross]
# f1_0_mean            0.758881
# f1_1_mean            0.658297
# eats_1      [macd, TrD3, bb_cross]
# f1_0_mean                   0.76471
# f1_1_mean                  0.663179
# feats_1      [TrD6, TrD3, macd, bb_cross]
# f1_0_mean                        0.762143
# f1_1_mean                        0.666735
# feats_1      [vmacd, TrD6, TrD3, macd, bb_cross]
# f1_0_mean                               0.774178
# f1_1_mean                               0.677318
