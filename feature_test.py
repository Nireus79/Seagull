from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, normalizer
from data_forming import full_data, events_data
import pandas as pd
import itertools
from tqdm import tqdm

# pd.set_option('display.max_rows', None)
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
selected1 = ['Tr9', 'TrD3', '4H%K', '4H%D']
selected2 = ['bb_l', 'TrD3']


def research_features(selected_features, eligible_features, plethos, mode):
    X_train, X_test, Y_train, Y_test = spliter(events_data, signal, part, feature_columns=eligible_features)

    full_features = X_train.columns

    reps = report_generator(plethos, mode, X_train, X_test, Y_train, Y_test, full_features, selected_features)
    print('max precision0:', reps.loc[reps['precision0'].idxmax()])
    print('max recall0:', reps.loc[reps['recall0'].idxmax()])
    print('max precision1:', reps.loc[reps['precision1'].idxmax()])
    print('max recall1:', reps.loc[reps['recall1'].idxmax()])


research_features(selected2, 'All', 3, 'GBC')

# max precision0: features      [bb_l, TrD3]
# precision0        0.767606
# recall0           0.819549
# precision1        0.757576
# recall1           0.694444

# max precision1: features      [mom30, bb_l, TrD3]
# precision0               0.767123
# recall0                  0.842105
# precision1               0.778947
# recall1                  0.685185

# max precision1: features      [ema6, 4H_ema9, bb_l, TrD3]
# precision0                       0.746835
# recall0                          0.887218
# precision1                       0.819277
# recall1                           0.62963

# max precision1: features      [Tr20, TrD3, mom20, 4H_mom10]
# precision0                         0.771242
# recall0                            0.887218
# precision1                         0.829545
# recall1                            0.675926

# max recall0: features      [TrD3, 4H%K, bb_sq, diff, bb_cross]
# precision0                               0.787671
# recall0                                  0.864662
# precision1                               0.810526
# recall1                                  0.712963

# max precision0: features      [Tr9, TrD3, 4H%K, 4H%D]
# precision0                   0.810606
# recall0                      0.804511
# precision1                   0.761468
# recall1                      0.768519

# max precision0: features      [4H_roc20, Dema9, bb_l, TrD3]
# precision0                         0.805085
# recall0                            0.714286
# precision1                         0.691057
# recall1                            0.787037


# max recall1: features      [4H_roc20, Dema9, bb_l, TrD3]
# precision0                         0.805085
# recall0                            0.714286
# precision1                         0.691057
# recall1                            0.787037

# max precision0: features      [4H_roc20, Dema9, bb_l, TrD3]
# precision0                         0.805085
# recall0                            0.714286
# precision1                         0.691057
# recall1                            0.787037

# max precision1: features      [ema6, 4H_ema9, bb_l, TrD3]
# precision0                       0.746835
# recall0                          0.887218
# precision1                       0.819277
# recall1                           0.62963

# max recall1: features      [4H_roc20, Dema9, bb_l, TrD3]
# precision0                         0.805085
# recall0                            0.714286
# precision1                         0.691057
# recall1                            0.787037

# max recall1: features      [Tr4h9, bb_l, TrD3]
# precision0               0.780303
# recall0                  0.774436
# precision1               0.724771
# recall1                  0.731481

# max precision0: features      [roc10, Tr6, TrD13]
# precision0               0.821782
# recall0                   0.62406
# precision1               0.642857
# recall1                  0.833333

# max recall1: features      [roc10, Tr6, TrD13]
# precision0               0.821782
# recall0                   0.62406
# precision1               0.642857
# recall1                  0.833333
