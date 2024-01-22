from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from toolbox import spliter, standardizer, normalizer
from data_forming import X_train_n, Y_train, X_test_n, Y_test, features
import pandas as pd
import itertools
from tqdm import tqdm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

signal = 'bin'
part = 5


def model_test(X_tr, Y_tr, X_ts, Y_ts, pl, md):
    if md == 'MLP':
        Model = MLPClassifier()
        Model.fit(X_tr, Y_tr)
        predictions = Model.predict(X_ts)
        report = classification_report(Y_ts, predictions, target_names=['0', '1'], output_dict=True)
        return report
    elif md == 'GBC':
        Model = GradientBoostingClassifier(max_depth=pl, random_state=42)
        Model.fit(X_tr, Y_tr)
        predictions = Model.predict(X_ts)
        report = classification_report(Y_ts, predictions, target_names=['0', '1'], output_dict=True)
        # print(report)
        return report


def uniqueCombinations(list_elements, plethos):
    lst = list(itertools.combinations(list_elements, plethos))
    combs = []
    for e in lst:
        combs.append(list(e))
    return combs


def report_generator(feats, plethos, md, X_tr, Y_tr, X_ts, Y_ts):
    combinations = uniqueCombinations(feats, plethos)
    print('Combinations:', len(combinations))

    r = []

    for i in tqdm(combinations):
        rep = model_test(X_tr, Y_tr, X_ts, Y_ts, i, md)
        r.append((i, rep))

    reports = pd.DataFrame({'reports': r})

    reports['features'] = reports.apply(lambda x: x[0][0], axis=1)
    reports['precision0'] = reports.apply(lambda x: x[0][1]['0']['precision'], axis=1)
    reports['recall0'] = reports.apply(lambda x: x[0][1]['0']['recall'], axis=1)
    reports['precision1'] = reports.apply(lambda x: x[0][1]['1']['precision'], axis=1)
    reports['recall1'] = reports.apply(lambda x: x[0][1]['1']['recall'], axis=1)
    reports.drop(columns=['reports'], axis=1, inplace=True)
    # print(reports)
    return reports
    # reports.to_csv(str(plethos)+'.csv')


reps = report_generator(features, 4, 'MLP', X_train_n, Y_train, X_test_n, Y_test)
print('max precision0:', reps.loc[reps['precision0'].idxmax()])
print('max recall0:', reps.loc[reps['recall0'].idxmax()])
print('max precision1:', reps.loc[reps['precision1'].idxmax()])
print('max recall1:', reps.loc[reps['recall1'].idxmax()])

# max precision0: features      [Tr9, TrD3, 4H%K, 4H%D]
# precision0                   0.810606
# recall0                      0.804511
# precision1                   0.761468
# recall1                      0.768519

# max precision1: features      [TrD3, 4H%K, bb_sq, diff]
# precision0                     0.761589
# recall0                        0.864662
# precision1                          0.8
# recall1                        0.666667

# max precision0: features      [VtrD13, Tr9, Tr20, TrD3, 4H%D, diff]
# precision0                                 0.813953
# recall0                                    0.789474
# precision1                                     0.75
# recall1                                    0.777778

# max precision1: features      [VtrD3, VtrD9, VtrD13, Tr9, TrD3, 4H%D]
# precision0                                   0.772727
# recall0                                      0.894737
# precision1                                    0.83908
# recall1                                      0.675926

# max recall1: features      [VtrD3, Tr9, Tr20, TrD3, diff, St4H]
# precision0                                0.808696
# recall0                                   0.699248
# precision1                                 0.68254
# recall1                                   0.796296

# bb full
# max precision1: features      [Tr9, Tr20, TrD3, bb_sq, diff]
# precision0                              0.78
# recall0                             0.672414
# precision1                          0.730496
# recall1                                0.824
