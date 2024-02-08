from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, normalizer
from data_forming import events_data
import pandas as pd
import itertools
from tqdm import tqdm

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


signal = 'bin'
part = 5


def research_features(selected_features, eligible_features, plethos, mode):
    X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feature_columns=eligible_features)

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


eligible_full = ['bb_sq', 'bb_l', 'TrD3', 'bb_cross', 'mom30', 'Tr9', '4H%K', '4H%D', '4H%DS',
                 'atr', 'diff', 'srl_corr', 'ave', '4H_roc30',
                 'Volatility', 'VtrD20', 'StD', 'St4H', 'Vol_Vol']
# full bb
c1 = ['bb_l', 'TrD3']  # 82/86
c2 = ['bb_cross', 'bb_l', 'TrD3']  # 94/90 - 71/82
c22 = ['bb_cross', 'Volatility', 'bb_l', 'TrD3']  # 94/90 - 71/82
c6 = ['TrD3', '4H%K', 'bb_sq', 'diff', 'bb_cross']  # 95/90 - 71/84
c7 = ['Tr9', 'TrD3', '4H%K', '4H%D']  # 74/99

c23 = ['USDT_Volatility', 'USDT_Vol_Vol', 'bb_cross', 'bb_l', 'TrD3']  # 93/90 - 72/80
c3 = ['srl_corr', 'ave', '4H_roc30', 'bb_l', 'TrD3', 'bb_cross']  # 94/83 - 69/82
c4 = ['mom30', 'bb_l', 'TrD3']  # 83/90
c5 = ['4H%DS', 'bb_l', 'TrD3', 'bb_cross']  # 94/90 - 71/83

c8 = ['atr', 'bb_l', 'bb_cross', 'TrD3']  # 94/90 - 72/82
c9 = ['TrD3', '4H%K', 'bb_sq', 'diff', 'bb_cross']  # 94/90 - 71/82
c10 = ['Volume', 'Tr9', 'TrD3', 'TrD20', 'diff', 'St4H']  # 76/99
c11 = ['Volatility', 'TrD3', 'Tr20', 'roc30']  # 77/100
c12 = ['TrD3', '4H%K', 'bb_sq', 'diff', 'bb_cross']  # 94/90 - 72/82
c14 = ['TrD3', 'Dema6', '4H%D', 'diff']  # 76/ 100
c15 = ['ema13', 'Dema6', '4H%K', 'vema20']  # 76/ 100
c16 = ['Volume', 'TrD3', '4H%D', 'mom20']  # 77/98
c17 = ['TrD3', '%K', '4Hmacd', 'diff', 'mom20']  # 77/98
c18 = ['TrD3', '4H%K', '4Hmacd', 'St4H', 'bb_cross', '4H_mom20']  # 95/89 - 70/84

# bb != 0
fa = ['TrD13']  # 0.752381  0.681034    0.727941    0.792
fb = ['TrD3']  # 0.77451  0.681034    0.733813    0.816
f0 = ['TrD13', 'St4H']  # 0.757282  0.672414    0.724638      0.8
f1 = ['srl_corr', 'TrD13']  # 73 / 79
f2 = ['St4H', 'TrD3']  # 74/82
f3 = ['vdiff', 'Vol_Vol', 'srl_corr', 'TrD3']  # 75/80
f4 = ['Vol_Vol', 'USDT_Volatility', 'srl_corr', 'TrD3']  # 73/82
f5 = ['macd', 'TrD3', 'TrD6']  # 75/80

# [USDT_Vol_Vol, St4H, TrD3]    0.794118  0.698276    0.748201    0.832
# [USDT_Volatility, St4H, TrD3]    0.788462  0.706897    0.751825    0.824
# [Vol_Vol, St4H, TrD3]    0.794118  0.698276    0.748201    0.832
# [Vol_Vol, St4H, TrD3]    0.794118  0.698276    0.748201    0.832
# [USDT_Vol_Vol, St4H, TrD3]    0.794118  0.698276    0.748201    0.832
# [USDT_Vol_Vol, St4H, TrD3]    0.786408  0.698276    0.746377    0.824
# [USDT_Volatility, St4H, TrD3]    0.780952  0.706897    0.750000    0.816
# [StD, St4H, TrD3]    0.781250  0.646552    0.717241    0.832
# [price, Vtr20, bb_cross, bb_l, TrD3] 0.934732 0.932558 0.776923 0.782946

eligible_bb0 = ['USDT_Vol_Vol', 'St4H', 'TrD3', 'TrD13', 'StD', 'VtrD9', 'Vtr20', 'bb_cross', 'bb_l', 'macd', 'vdiff',
                'Vol_Vol', 'srl_corr']
research_features(None, eligible_bb0, 3, 'MLP')
finf = ['bb_cross', 'bb_l', 'TrD3']  # 94/90 - 71/82
fin0 = ['St4H', 'TrD3']  # 0.741007/0.824
