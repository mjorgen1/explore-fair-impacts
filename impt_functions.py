#!/usr/bin/env python
from evaluation import evaluating_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import pandas as pd
import numpy as np
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
from raiwidgets import FairnessDashboard
import matplotlib.pyplot as plt
import csv



def get_data(file):
    data = pd.read_csv(file)
    data[['score', 'race']] = data[['score', 'race']].astype(int)
    #print(data)
    return data


def prep_data(data, test_size, weight_index):
    # might need to include standardscaler here

    x = data[['score', 'race']].values
    y = data['repay_indices'].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    #print('Here are the x values: ', x, '\n')
    #print('Here are the y values: ', y)

    # collect our sensitive attribute
    race_train = X_train[:, 1]
    race_test = X_test[:, 1]

    # weight_index: 1 means all equal weights
    if weight_index:
        #print('Sample weights are all equal.')
        sample_weight_train = np.ones(shape=(len(y_train),))
        sample_weight_test = np.ones(shape=(len(y_test),))
    # weight_index: 0 means use sample weights
    elif weight_index:
        print('Sample weights are NOT all equal.')
        # TODO
        #print('TODO')
    return X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test





def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        'Overall selection rate': (
            lambda x: selection_rate(y_true, x), True),
        'Demographic parity difference': (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        'Demographic parity ratio': (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        '------': (lambda x: '', True),
        'Overall balanced error rate': (
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
        'Balanced error rate difference': (
            lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), True),
        ' ------': (lambda x: '', True),
        'True positive rate difference': (
            lambda x: true_positive_rate_difference(y_true, x, sensitive_features=group), True),
        'True negative rate difference': (
            lambda x: true_negative_rate_difference(y_true, x, sensitive_features=group), True),
        'False positive rate difference': (
            lambda x: false_positive_rate_difference(y_true, x, sensitive_features=group), True),
        'False negative rate difference': (
            lambda x: false_negative_rate_difference(y_true, x, sensitive_features=group), True),
        'Equalized odds difference': (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        '  ------': (lambda x: '', True),
        'Overall AUC': (
            lambda x: roc_auc_score(y_true, x), False),
        'AUC difference': (
            lambda x: MetricFrame(metrics=roc_auc_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups'), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores)
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient='index', columns=models_dict.keys())

def get_classifier(model_name):

    if model_name == 'dt':
        # Initialize classifier:
        classifier = DecisionTreeClassifier()
    elif model_name == 'gnb':
        classifier = GaussianNB()
    elif model_name == 'lgr':
        # Reference: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
        classifier = LogisticRegression()
    elif model_name == 'gbt':
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        # Note: max_depth default is 3 but tune this parameter for best performance
        classifier = GradientBoostingClassifier(n_estimators=100)
    else:
        print('PROBLEM: input a specified classifier above')

    return classifier


def get_constraint(constraint_str):
    #set seed for consistent results with ExponentiatedGradient
   np.random.seed(0)
   if constraint_str == 'DP':
       constraint = DemographicParity()
   elif constraint_str == 'EO':
       constraint = EqualizedOdds()
   elif constraint_str == 'TPRP':
       constraint = TruePositiveRateParity()
   elif constraint_str == 'FPRP':
       constraint = FalsePositiveRateParity()
   elif constraint_str == 'ERP':
       constraint = ErrorRateParity()
   else:
       print('Error: Not a valid constraint_str')
   return constraint

def get_reduction_algo(model,constraint,reduction_alg):
    if reduction_alg == 'EG':
        mitigator = ExponentiatedGradient(model, constraint)
    elif reduction_alg == 'GS':
        mitigator = GridSearch(model, constraint)
    else:
        raise ValueError('unvalid reduction_alg parameter {"EG","GS"}')
    return mitigator

def get_new_scores(X_test, y_predict, y_test, race_test):
    black_scores = []
    black_type = []
    white_scores = []
    white_type = []
    up_bound = 850
    low_bound = 300
    reward = 75
    penalty = -150

    for index, label in enumerate(y_predict):

        # first check for TP or FP
        if label == 1 and y_test[index] == 1:  # if it's a TP
            if race_test[index] == 0:  # black
                black_type.append('TP')
                new_score = X_test[index][0] + reward
                if new_score <= up_bound:
                    black_scores.append(new_score)
                else:
                    black_scores.append(up_bound)
            elif race_test[index] == 1:  # white
                white_type.append('TP')
                new_score = X_test[index][0] + reward
                if new_score <= up_bound:
                    white_scores.append(new_score)
                else:
                    white_scores.append(up_bound)
        elif label == 1 and y_test[index] == 0:  # if it's a FP
            if race_test[index] == 0:  # black
                black_type.append('FP')
                new_score = X_test[index][0] + penalty
                if new_score < low_bound:
                    black_scores.append(low_bound)
                else:
                    black_scores.append(new_score)
            elif race_test[index] == 1:  # white
                white_type.append('FP')
                new_score = X_test[index][0] + penalty
                if new_score < low_bound:
                    white_scores.append(low_bound)
                else:
                    white_scores.append(new_score)
        elif label == 0 and y_test[index] == 0:  # TN, no change to credit score
            if race_test[index] == 0:  # black
                black_type.append('TN')
                black_scores.append(X_test[index][0])
            elif race_test[index] == 1:  # white
                white_type.append('TN')
                white_scores.append(X_test[index][0])
        elif label == 0 and y_test[index] == 0:  # FN, no change to credit score
            if race_test[index] == 0:  # black
                black_type.append('FN')
                black_scores.append(X_test[index][0])
            elif race_test[index] == 1:  # white
                white_type.append('FN')
                white_scores.append(X_test[index][0])

    return black_scores, white_scores

# Reference: https://thispointer.com/python-dictionary-with-multiple-values-per-key/
def add_values_in_dict(sample_dict, key, list_of_values):
    """Append multiple values to a key in the given dictionary"""
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict

# Reference: https://stackoverflow.com/questions/53013274/writing-data-to-csv-from-dictionaries-with-multiple-values-per-key
def save_dict_in_csv(results_dict, fieldnames, name_csv):
    # Example for fieldnames:
    # overall_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB/DIW', 'DP Diff', 'EO Diff']
    # byrace_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB/DIW']

    # the dictionary needs to be formatted like: {'Run1': [acc, f1, ...], 'Run2': [acc, f1, ...]}
    with open(name_csv, mode='w') as csv_file:
        writer = csv.writer((csv_file))
        writer.writerow(fieldnames)

        for run in results_dict.items():
            ##print(run)
            ##print([row[0]])
            ##print(row[1])
            row = list(run)
            row = [row[0]] + row[1]
            writer.writerow(row)

        csv_file.close()


##### ONLY USD FOR NOTEBOOK #########
def add_constraint(model, constraint_str, reduction_alg, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool):
    # set seed for consistent results with ExponentiatedGradient
    np.random.seed(0)
    constraint = get_constraint(constraint_str)
    mitigator = get_reduction_algo(model,constraint, reduction_alg)
    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    y_pred_mitigated = mitigator.predict(X_test) #y_pred_mitigated

    results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_pred_mitigated, sample_weight_test,race_test)

    if dashboard_bool:
        pass
        #FairnessDashboard(sensitive_features=race_test,y_true=y_test,y_pred={"initial model": y_predict, "mitigated model": y_pred_mitigated})

    return mitigator, results_overall, results_black, results_white, y_pred_mitigated
