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


# 5 [Volatility, St4H, TrD3]    0.804878  0.868421    0.883721  0.826087
# 4 [Volatility, St4H, TrD3]    0.714286  0.789474    0.804878  0.733333
# 3 [Volatility, St4H, TrD3]    0.558824  0.633333    0.775510  0.716981
#   [Vol_Vol, St4H, TrD3]    0.533333  0.533333    0.735849  0.735849
# 2 [StD, St4H, TrD3]    0.785714   0.6875    0.818182  0.882353
#   [Vol_Vol, St4H, TrD3]    0.774194   0.7500    0.846154  0.862745
#   [St4H, TrD3]    0.774194   0.7500    0.846154  0.862745
# 1 [StD, St4H, TrD3]    0.807692  0.617647    0.771930  0.897959

full0 = ['bb_cross', 'bb_l', 'TrD3']  # 0.975  0.939759    0.811321  0.914894
full1 = ['bb_cross', 'St4H', 'TrD3']  # 0.959368  0.913978    0.716312  0.848739
bb0 = ['St4H', 'TrD3']  # 0.815789  0.815789    0.847826  0.847826
bb1 = ['Volatility', 'St4H', 'TrD3']  # 0.804878  0.868421    0.883721  0.826087
bb2 = ['Vol_Vol', 'St4H', 'TrD3']  # 0.810811  0.789474    0.829787  0.847826
eligible = ['bb_cross', 'Volatility', 'StD', 'Vol_Vol', 'VtrD6', 'VtrD3', 'MAV', 'MAV_signal']
# events_data = events_data.loc[events_data['bb_cross'] != 0]
research_features(None, 'All', 3, 'MLP', 5, events_data)
# max recall0: features      [4H_ema3, Vtr4h20, St4H, TrD3]
# precision0                             0.808
# recall0                                0.808
# precision1                           0.79661
# recall1                              0.79661

