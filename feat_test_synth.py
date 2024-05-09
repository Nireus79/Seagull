from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from toolbox import normalizer, uniqueCombinations
# from data_forming import events_data
import pandas as pd

from tqdm import tqdm
import numpy as np
import warnings

# warnings.filterwarnings('ignore')

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
    # if md == 'MLP':
    #     model = MLPClassifier()
    #     model.fit(X_tr, Y_tr)
    # elif md == 'GBC':
    #     model = GradientBoostingClassifier(max_depth=len(comb), random_state=42)
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


def research_features(X_tr, X_tst, Y_tr, Y_tst, selected_features, plethos, mode):
    full_features = X_tr.columns
    reps = report_generator(plethos, mode, X_tr, X_tst, Y_tr, Y_tst, full_features, selected_features)
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


# def k_mean(X_tr, X_tst, Y_tr, Y_tst, selected_features, plethos, mode):
#     cross = pd.DataFrame()
#     for c in tqdm(range(1, 6)):
#         res = research_features(X_tr, X_tst, Y_tr, Y_tst, selected_features, plethos, mode)
#         cross['feats_' + str(c)] = res['features']
#         cross['f1-score0_' + str(c)] = res['f1-score0']
#         cross['f1-score1_' + str(c)] = res['f1-score1']
#     cross['f1_0_mean'] = cross.apply(lambda x:
#                                      (x['f1-score0_1'] + x['f1-score0_2']
#                                       + x['f1-score0_3'] + x['f1-score0_4'] + x['f1-score0_5']) / 5, axis=1)
#     cross['f1_1_mean'] = cross.apply(lambda x:
#                                      (x['f1-score1_1'] + x['f1-score1_2']
#                                       + x['f1-score1_3'] + x['f1-score1_4'] + x['f1-score1_5']) / 5, axis=1)
#     cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
#     print(cross.loc[cross['f1_0_mean'].idxmax()])
#     print(cross.loc[cross['f1_1_mean'].idxmax()])
#     print(cross.tail(5).sort_values(by=['f1_0_mean']))
#     print(cross.tail(5).sort_values(by=['f1_1_mean']))
def k_mean(X_tr, X_tst, Y_tr, Y_tst, selected_features, plethos, mode, repetition):
    repetition += 1
    cross = pd.DataFrame()
    for c in tqdm(range(1, repetition)):
        res = research_features(X_tr, X_tst, Y_tr, Y_tst, selected_features, plethos, mode)
        cross[f'feats_{c}'] = res['features']
        cross[f'f1-score0_{c}'] = res['f1-score0']
        cross[f'f1-score1_{c}'] = res['f1-score1']
    cross['f1_0_mean'] = cross[[f'f1-score0_{i}' for i in range(1, repetition)]].mean(axis=1)
    cross['f1_1_mean'] = cross[[f'f1-score1_{i}' for i in range(1, repetition)]].mean(axis=1)
    cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
    print(cross.loc[cross['f1_1_mean'].idxmax()])
    print("Tail 5 sorted by f1-score0_mean:")
    print("Max f1-score0_mean:")
    print(cross.loc[cross['f1_0_mean'].idxmax()])
    print("Max f1-score1_mean:")
    print(cross.tail(5).sort_values(by='f1_0_mean'))
    print("Tail 5 sorted by f1-score1_mean:")
    print(cross.tail(5).sort_values(by='f1_1_mean'))


def MDI(X, Y):
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    names = X.columns
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
                 reverse=True))


# MDI()

train_data = pd.read_csv('csv/synth/synth_ev100000_30m4H2624.csv')
test_data = pd.read_csv('csv/synth/synth_ev20000_30m4H2624.csv')

signal = 'bin'
Y_train = train_data[signal]
X_train = train_data.drop(columns=[signal, 'ret'])
Y_test = test_data[signal]
X_test = test_data.drop(columns=[signal, 'ret'])

B3042624 = ['TrD20', 'TrD3', '4Hmacd', 'rsi', 'bb_cross']  # 0.679292
S3042624 = ['TrD6', 'TrD3', 'Tr13', 'roci', 'bb_cross']  # 0.774514
B26120 = ['TrD3', 'TrD6', 'Volatility', 'bb_cross']  # 0.77 0.72
S26120 = ['TrD6', 'TrD13', 'mom10', 'bb_cross']  # 0.78 0.85
B26121 = ['TrD13', 'TrD6', 'TrD3', 'MAV', 'bb_cross']  # 0.77 0.73
S26121 = ['TrD13', 'TrD6', 'TrD3', 'Tr20', 'bb_cross']  # 0.78 0.87

B26241 = ['TrD9', 'TrD6', 'TrD3', 'diff', 'bb_cross']  # 0.69 0.73
S26241 = ['TrD20', 'TrD9', 'macd', 'Volatility', 'bb_cross']  # 0.73 0.80
B26242 = ['TrD3', '4Hmacd', 'momi', 'rsi', 'bb_cross']  # 0.72 0.70 G
S26242 = ['TrD6', 'St4H', 'Tr6', 'Volatility', 'bb_cross']  # 0.74 0.80 G
B26243 = ['TrD3', '4Hmacd', 'TrD20', 'Volatility', 'VV', 'roc30', 'srl_corr', 'rsi', 'bb_cross']  # 0.735257
S26243 = ['TrD20', 'TrD3', '4H%D', '4Hmacd', 'Tr6', 'roc30', 'bb_l', 'rsi', 'bb_cross']  # 0.792629
k_mean(X_train, X_test, Y_train, Y_test, B26243, 0, 'MLP', 1)

# feats_1      [TrD3, bb_cross]
# f1_0_mean            0.758037
# f1_1_mean             0.69792
# feats_1      [VV, TrD3, bb_cross]
# f1_0_mean                0.752507
# f1_1_mean                0.708381
# feats_1      [Volatility, TrD3, VV, bb_cross]
# f1_0_mean                            0.754124
# f1_1_mean                            0.710169
# feats_1      [roc30, TrD3, VV, Volatility, bb_cross]
# f1_0_mean                                   0.758776
# f1_1_mean                                   0.711265
# feats_1      [srl_corr, TrD3, VV, Volatility, bb_cross, roc30]
# f1_0_mean                                             0.761742
# f1_1_mean                                             0.717618
# feats_1      [rsi, TrD3, VV, Volatility, bb_cross, roc30, srl_corr]
# f1_0_mean                                             0.765328
# f1_1_mean                                             0.723321
# feats_1      [4Hmacd, TrD3, VV, Volatility, bb_cross, roc30...
# f1_0_mean                                             0.778072
# f1_1_mean                                              0.73175
# feats_1      [Vol_Vol, 4Hmacd, TrD3, VV, Volatility, bb_cro...
# f1_0_mean                                             0.772953
# f1_1_mean                                             0.735257


# feats_0      [Tr6, TrD3]
# f1_0_mean       0.772625
# f1_1_mean       0.675326
# feats_0      [bb_l, Tr6, TrD3]
# f1_0_mean             0.773085
# f1_1_mean              0.68184
# feats_0      [rsi, Tr6, TrD3, bb_l]
# f1_0_mean                  0.777227
# f1_1_mean                  0.692834
# feats_0      [bb_cross, Tr6, TrD3, bb_l, rsi]
# f1_0_mean                            0.779813
# f1_1_mean                            0.669169
# feats_0      [TrD20, Tr6, TrD3, bb_cross, bb_l, rsi]
# f1_0_mean                                   0.784599
# f1_1_mean                                   0.697143
# feats_0      [roc30, Tr6, TrD20, TrD3, bb_cross, bb_l, rsi]
# f1_0_mean                                          0.788877
# f1_1_mean                                          0.691915
# feats_0      [4Hmacd, roc30, Tr6, TrD20, TrD3, bb_cross, bb_l, rsi]
# f1_0_mean                                             0.792451
# f1_1_mean                                             0.719285
# feats_1      [4H%D, 4Hmacd, Tr6, TrD20, TrD3, bb_cross, bb_...
# f1_0_mean                                             0.792629
# f1_1_mean                                             0.722264
