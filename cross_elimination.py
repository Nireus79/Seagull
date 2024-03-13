from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from toolbox import spliterC, normalizer
from data_forming import events_data, signal
import pandas as pd
import itertools
from tqdm import tqdm

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def model_test(comb, X_tr, X_ts, Y_tr, Y_ts, md):
    X_trs, X_tss = X_tr[comb], X_ts[comb]
    X_trn, X_tsn = normalizer(X_trs), normalizer(X_tss)

    if 'bb_cross' in comb:
        X_trn['bb_cross'], X_tsn['bb_cross'] = X_tr['bb_cross'], X_ts['bb_cross']

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
    combinations = list(itertools.combinations(full_elements, plethos))
    if std_elements is not None:
        combinations = [list(comb) + std_elements for comb in combinations]

    unique_combinations = [list(dict.fromkeys(comb)) for comb in combinations]
    return unique_combinations


def report_generator(plethos, md, X_tr, X_ts, Y_tr, Y_ts, full_feats, std_feats):
    combinations = uniqueCombinations(full_feats, std_feats, plethos)
    print('Combinations:', len(combinations))

    reports = []
    for comb in tqdm(combinations):
        rep = model_test(comb, X_tr, X_ts, Y_tr, Y_ts, md)
        reports.append((comb, rep))

    reports_df = pd.DataFrame(reports, columns=['features', 'report'])
    reports_df['precision0'] = reports_df['report'].apply(lambda x: x['0']['precision'])
    reports_df['recall0'] = reports_df['report'].apply(lambda x: x['0']['recall'])
    reports_df['f1-score0'] = reports_df['report'].apply(lambda x: x['0']['f1-score'])
    reports_df['precision1'] = reports_df['report'].apply(lambda x: x['1']['precision'])
    reports_df['recall1'] = reports_df['report'].apply(lambda x: x['1']['recall'])
    reports_df['f1-score1'] = reports_df['report'].apply(lambda x: x['1']['f1-score'])

    reports_df.drop(columns=['report'], inplace=True)
    return reports_df


def research_features(selected_features, eligible_features, plethos, mode, prt, events):
    X_train, X_test, Y_train, Y_test = spliterC(events, signal, prt, feature_columns=eligible_features)
    full_features = X_train.columns

    reps = report_generator(plethos, mode, X_train, X_test, Y_train, Y_test, full_features, selected_features)
    return reps


def cross_elimination(selected_features, eligible_features, plethos, mode, events):
    cross = pd.DataFrame()
    for c in tqdm(range(1, 6)):
        res = research_features(selected_features, eligible_features, plethos, mode, c, events)
        cross['feats_' + str(c)] = res['features']
        cross['f1-score0_' + str(c)] = res['f1-score0']
        cross['f1-score1_' + str(c)] = res['f1-score1']

    cross['f1_0_mean'] = cross.filter(like='f1-score0').mean(axis=1)
    cross['f1_1_mean'] = cross.filter(like='f1-score1').mean(axis=1)

    cross = cross[['feats_1', 'f1_0_mean', 'f1_1_mean']]
    print(cross.loc[cross['f1_0_mean'].idxmax()])
    print(cross.loc[cross['f1_1_mean'].idxmax()])
    print(cross.tail(5).sort_values(by=['f1_0_mean']))
    print(cross.tail(5).sort_values(by=['f1_1_mean']))


cross_elimination(None, 'All', 4, 'MLP', events_data)
