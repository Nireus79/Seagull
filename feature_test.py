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

bb0 = ['TrD3', 'bb_cross']
bb1 = ['St4H', 'TrD3']
bb2 = ['bb_l', 'TrD3']
bbt = ['TrD3', 'bb_cross', 'St4H', 'MAV_signal', 'Vol_Vol', 'bb_l']

eligible = ['bb_cross', 'Volatility', 'StD', 'Vol_Vol', 'VtrD6', 'VtrD3', 'MAV', 'MAV_signal', 'TrD9',
            'Vtr4h20', '4H_roc30', 'Tr4h3', 'St4H']
# events_data = events_data.loc[events_data['bb_cross'] != 0]
research_features(None, bbt, 3, 'MLP', 5, events_data)

# bb
# TrD3 30
# bbc 27
# St4H 6
# MAV_signal 6
# Vol_Vol 6
# bb_l 6

# 5
# max precision0: features      [TrD3, bb_cross, MAV_signal]
# precision0                        0.767742
# recall0                            0.82069
# precision1                        0.742574
# recall1                           0.675676


# 5
# [TrD3, bb_cross] 0.756579 0.815603 0.74 0.666667
# [St4H, TrD3, bb_cross]    0.772727  0.820690    0.745098  0.684685
# [Vol_Vol, TrD3, bb_cross]    0.769231  0.827586    0.750000  0.675676
# [TrD3, bb_cross, Volatility] 0.756579 0.815603 0.74 0.666667
# [MAV_signal, TrD3, bb_cross]    0.756579  0.815603    0.740000  0.666667
# [MAV_signal, Vol_Vol, TrD3, bb_cross]    0.770701  0.834483   0.757576  0.675676
# [bb_cross, bb_l, TrD3]    0.738095  0.855172    0.761364  0.603604

# 4
# [bb_cross, TrD3]    0.812500  0.776119    0.758065  0.796610
# [bb_t, TrD3, bb_cross] 0.821138 0.753731 0.744186 0.813559
# [Volatility, TrD3, bb_cross]    0.812500  0.776119    0.758065  0.796610
# [MAV_signal, TrD3, bb_cross]    0.818898  0.776119    0.760000  0.805085
# [StD, TrD3, bb_cross]    0.796992  0.791045    0.764706  0.771186
# [Vol_Vol, TrD3, bb_cross]    0.806202  0.776119    0.756098  0.788136
# [bb_cross, St4H, TrD3]    0.809524  0.761194    0.746032  0.796610
# [bb_cross, bb_l, TrD3]    0.779412  0.791045    0.758621  0.745763

# 3
# [bb_cross, St4H, TrD3]    0.768000  0.738462    0.732283  0.762295
# [Vol_Vol, TrD3, bb_cross]    0.768000  0.738462    0.732283  0.762295
# [MAV_signal, TrD3, bb_cross]    0.768000  0.738462    0.732283  0.762295
# [StD, TrD3, bb_cross]    0.765625  0.753846    0.741935  0.754098
# [Vol_Vol, TrD3, bb_cross]    0.768000  0.738462    0.732283  0.762295

# 2
# [MAV_signal, TrD3, bb_cross]    0.721429  0.759398    0.714286  0.672269
# [Vol_Vol, TrD3, bb_cross]    0.724138  0.789474    0.738318  0.663866
# [St4H, TrD3, bb_cross]    0.725352  0.774436    0.727273  0.672269
# [bb_cross, bb_l, TrD3]    0.720280  0.774436    0.724771  0.663866

# 1
# [bb_cross, TrD3]    0.725926  0.690141    0.623932  0.663636
# [MAV, TrD3, bb_cross]    0.734266  0.739437    0.660550  0.654545
# [bb_cross, St4H, TrD3]    0.732394  0.732394    0.654545  0.654545
# [bb_l, St4H, TrD3] 0.741935 0.809859 0.721649 0.636364
# [Volatility, bb_l, TrD3]    0.754839  0.823944    0.742268  0.654545
# [MAV_signal, bb_l, TrD3]    0.753165  0.838028    0.755319  0.645455
