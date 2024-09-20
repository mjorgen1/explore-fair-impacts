import pandas as pd
import numpy as np
import csv
import yaml
from yaml.loader import SafeLoader

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, TruePositiveRateParity,FalsePositiveRateParity, ErrorRateParity
from fairlearn.metrics import *

from scripts.evaluation_utils import evaluating_model


def load_args(file):
    """
    Load args and run some basic checks.
        Args:
            - file <str>: full path to .yaml config file
        Returns:
            - data <dict>: all args from file
    """
    with open(file, "r") as stream:
        try:
            data = yaml.load(stream, Loader=SafeLoader)
            print('Arguments: ',data)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def create_original_set_ratios(x_data, y_data, set_size_upper_bound):
    """
    Create a subset with to the original FICO ratios.
        Args:
            - x_data <numpy.ndarray>: ['score','race'] -> array of samples
            - y_data <numpy.ndarray>: ['repay_indices'] -> array of samples
            - set_size_upper_bound <int>: absolute upper bound of the size for the dataset (e.g 100,000)
        Returns:
            subset of x_data and y_data
    """
    # Black = 0; White = 1
    race_ratio = [0.12,0.88]
    black_label_ratio = [0.66,0.34]
    white_label_ratio = [0.24,0.76]
    # limits the absolute test_size if necessary
    if len(y_data) > set_size_upper_bound:
        set_size = set_size_upper_bound
    else:
        set_size = len(y_data)

    # set sizes for each race
    set_size_0 = int(set_size * race_ratio[0])
    set_size_1 = int(set_size * race_ratio[1])

    # number of samples for the Black and White group, according to the label ratios from Fico data
    num_0P = int(set_size_0 * black_label_ratio[1])
    num_0N = int(set_size_0 * black_label_ratio[0])
    num_1P = int(set_size_1 * 0.76)
    num_1N = int(set_size_1 * 0.24)

    # getting the indices of each samples for each group
    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]

    idx_1 = np.where(x_data[:, 1] == 1)[0]

    # if group size numbers are larger than the available samples for that group adjust the number of samples
    if len(idx_0P) < num_0P:
        num_0P = len(idx_0P)
        num_0N = int(num_0P/black_label_ratio[1] * black_label_ratio[0])
        num_1P =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * white_label_ratio[1])
        num_1N =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * white_label_ratio[0])
    if len(idx_0N) < num_0N:
        num_0N = len(idx_0N)
        num_0P = int(num_0N/black_label_ratio[0] * black_label_ratio[1])
        num_1P =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * white_label_ratio[1])
        num_1N =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * white_label_ratio[0])
    if len(idx_1P) < num_1P:
        num_1P = len(idx_1P)
        num_1N = int(num_1P/white_label_ratio[1] * white_label_ratio[0])
        num_0P =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * black_label_ratio[1])
        num_0N =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * black_label_ratio[0])
    if len(idx_1N) < num_1N:
        num_1N = len(idx_1N)
        num_1P = int(num_1N/white_label_ratio[0] * white_label_ratio[1])
        num_0P =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * black_label_ratio[1])
        num_0N =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * black_label_ratio[0])


    # take the amount of samples, by getting the amount of indices
    idx_0N = idx_0N[:num_0N]
    idx_0P = idx_0P[:num_0P]
    idx_1N = idx_1N[:num_1N]
    idx_1P = idx_1P[:num_1P]

    # concatenate indices
    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))
    # retuen subsets
    return x_data[idx,:], y_data[idx]


def balance_test_set_ratios(x_data, y_data, set_upper_bound):
    """
    Create a subset with balanced group and label ratios
        Args:
            - x_data <numpy.ndarray>: ['score','race'] -> array of samples
            - y_data <numpy.ndarray>: ['repay_indices'] -> array of samples
            - set_upper_bound <int>: absolute upper bound of the size for the dataset (e.g 100,000)
        Returns:
            subset of x_data and y_data
    """
    # get the indexes subsets for each group and label
    idx_0N = np.where((x_data[:, -1] == 0) & (y_data == 0))[0]
    idx_1N = np.where((x_data[:, -1] == 1) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, -1] == 0) & (y_data == 1))[0]
    idx_1P = np.where((x_data[:, -1] == 1) & (y_data == 1))[0]

    # get the size of the smallest of those subsets
    num = min(len(idx_0N),len(idx_0P),len(idx_1N),len(idx_1P))
    # if the set would be to large reduce the size
    if num > set_upper_bound * 0.25:
        num = int(set_upper_bound * 0.25)
    # samples the equal amount of samples from each subset
    idx_0N = idx_0N[:num]
    idx_1N = idx_1N[:num]
    idx_0P = idx_0P[:num]
    idx_1P = idx_1P[:num]
    # merge the group indexes together to one list
    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))
    # retuen subsets
    return x_data[idx,:], y_data[idx]


def print_type_ratios(x_data,y_data):
    """
    Print the size of the subgroups (demo-group and label) of the set
        Args:
            - x_data <numpy.ndarray>: ['score','repay_probability','race'] -> array of samples
            - y_data <numpy.ndarray>: ['repay_indices'] -> array of samples
    """
    # get the samples for each subgroup
    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]
    print('Black N/P:',len(idx_0N),'/',len(idx_0P),'White N/P:',len(idx_1N),'/',len(idx_1P))


def prep_data(data, test_size, test_set_variant, test_set_bound,weight_index):
    """
    Prepare training and test set (a few extra steps, like changing test set composition)
        Args:
            - data <numpy.ndarray>: ['score','repay_probability','race','repay_indices'] -> array of samples
            - test_size <int>: percentage of testset from the dataset
            - test_set_variant <int>: 1-> balancing test set ; 2-> adjust testset to orignal FICO distributions
            - test_set_bound <int>: absolute upper bound of the testset (e.g.30,000)
            - weight_index <int>: sample weights

        Returns:
            - X_train <numpy.ndarray>: samples for training
            - X_test <numpy.ndarray>: samples for tesing
            - y_train <numpy.ndarray>: labels of training samples
            - y_test <numpy.ndarray>: labels of test samples
            - race_train <numpy.ndarray>: sensitive attribute of train samples
            - race_test <numpy.ndarray>: sensitive attribute of test samples
            - sample_weight_train <numpy.ndarray>: weights for training samples
            - sample_weight_test <numpy.ndarray>: weights for test samples
    """
    # get important cols from our data_add
    x = data[['score', 'race']].values
    y = data['repay_indices'].values

    print(' Whole set:', len(y))
    print_type_ratios(x,y)

    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    print('Training set:', len(y_train))
    print_type_ratios(X_train,y_train)

    # adjust test set if necessary
    if test_set_variant == 1: # balanced set
        X_test, y_test = balance_test_set_ratios(X_test, y_test, test_set_bound)
    if test_set_variant == 2: # original FICO set ratios
        X_test, y_test = create_original_set_ratios(X_test, y_test, test_set_bound)

    print('Testing set:', len(y_test))
    print_type_ratios(X_test,y_test)

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

    return X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test


def get_metrics_df(models_dict, y_true, group):
    """
    Create dataframe with numerous metrics
        Args:
            - models_dict <dict>:
            - y_true <numpy.ndarray>:  true labels
            - group <numpy.ndarray>:  sensitive feature group of each sample
        Returns:
            dataframe containing results of numerous metrics for our data
    """

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


#TODO: right now, if I want to add randomness to runs, I have to hardcode it here! later, add argument for randomness in the yaml
def get_classifier(model_name):
    """
        Picks the classification method
        Args:
            - model_name <str>: shortcut name for classification method
        Returns:
            classifier <object>: classification method
    """
    if model_name == 'dt':
        # Initialize classifier:
        classifier = DecisionTreeClassifier()  # NOTE: for aies paper, random state was not an included parameter; it can be included though by random_seed = 0
    elif model_name == 'gnb':
        classifier = GaussianNB()
    elif model_name == 'lgr':
        # Reference: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
        classifier = LogisticRegression(max_iter=100000)  # NOTE: for aies paper, random state was not an included parameter; it can be included though by random_seed = 0
    elif model_name == 'gbt':
        # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        # Note: max_depth default is 3 but tune this parameter for best performance
        classifier = GradientBoostingClassifier(n_estimators=100)
    else:
         raise ValueError('unvalid classisfier {dt,gnb,lgr,gbt}')

    return classifier


def get_constraint(constraint_str):
    """
        Picks the fairness contraint.
        Args:
            - constraint_str <str>: shortcut name for fairness constraint
        Returns:
            constraint <object>: fairness  method
    """
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
        raise ValueError('unvalid constraint_str')
    return constraint

def get_new_scores_updated(X_test, y_predict, y_test, di_means, di_stds, race_test):
    """
    Calculate the new scores and retun the types (Tp,TP,FN,TN), after applying penalty (to FP) and reward (to TP) to positive samples
    Args:
        - X_test <numpy.ndarray>: samples for testing
        - y_predict <numpy.ndarray>: predicted test set labels a model
        - y_test <numpy.ndarray>: predicted labels
        - di_means <tuple>: means of the delayed impact distributions
        - di_stds <tuple>: deviation of delayed impact distributions
        - race_test <numpy.ndarray>: sensitive attribute of test samples

    Returns:
        - black_scores <list>: updated credit scores
        - white_scores <list>: updated credit score  of testset samples of the advantaged group
        - black_types <list>: type classification (TP,TP;FN,TN) of testset samples of the disadvantaged group
        - white_types <list>: type classification (TP,TP;FN,TN) of testset samples of the advantaged group
    """
    black_scores = []
    black_types = []
    white_scores = []
    white_types = []
    # credit score bounds
    up_bound = 850
    low_bound = 300
    # reward and penalty distribution parameters
    reward_mu, penalty_mu = di_means
    reward_std, penalty_std = di_stds

    for index, label in enumerate(y_predict):

        # first check for TP or TP
        if label == 1 and y_test[index] == 1:  # if it's a TP
            if race_test[index] == 0:  # black
                black_types.append('TP')
                new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std, 1))
                if new_score <= up_bound:
                    black_scores.append(new_score)
                else:
                    black_scores.append(up_bound)
            elif race_test[index] == 1:  # white
                white_types.append('TP')
                new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std, 1))
                if new_score <= up_bound:
                    white_scores.append(new_score)
                else:
                    white_scores.append(up_bound)
        elif label == 1 and y_test[index] == 0:  # if it's a FP
            if race_test[index] == 0:  # black
                black_types.append('FP')
                new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std, 1))
                if new_score < low_bound:
                    black_scores.append(low_bound)
                else:
                    black_scores.append(new_score)
            elif race_test[index] == 1:  # white
                white_types.append('FP')
                new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std, 1))
                if new_score < low_bound:
                    white_scores.append(low_bound)
                else:
                    white_scores.append(new_score)
        elif label == 0 and y_test[index] == 0:  # TN, no change to credit score
            if race_test[index] == 0:  # black
                black_types.append('TN')
                black_scores.append(X_test[index][0])
            elif race_test[index] == 1:  # white
                white_types.append('TN')
                white_scores.append(X_test[index][0])
        elif label == 0 and y_test[index] == 1:  # FN drop in credit score
            if race_test[index] == 0:  # black
                black_types.append('FN')
                new_score = X_test[index][0] - int(np.random.normal(reward_mu, reward_std, 1))
                if new_score < low_bound:
                    black_scores.append(low_bound)
                else:
                    black_scores.append(new_score)
            elif race_test[index] == 1:  # white
                white_types.append('FN')
                new_score = X_test[index][0] - int(np.random.normal(reward_mu, reward_std, 1))
                if new_score < low_bound:
                    white_scores.append(low_bound)
                else:
                    white_scores.append(new_score)

    return black_scores, white_scores, black_types, white_types


def get_new_scores(X_test, y_predict, y_test, di_means, di_stds, race_test):
    """
        Calculate the new scores and retun the types (Tp,TP,FN,TN), after applying penalty (to FP) and reward (to TP) to positive samples
        Args:
            - X_test <numpy.ndarray>: samples for testing
            - y_predict <numpy.ndarray>: predicted test set labels a model
            - y_test <numpy.ndarray>: predicted labels
            - di_means <tuple>: means of the delayed impact distributions
            - di_stds <tuple>: deviation of delayed impact distributions
            - race_test <numpy.ndarray>: sensitive attribute of test samples

        Returns:
            - black_scores <list>: updated credit scores
            - white_scores <list>: updated credit score  of testset samples of the advantaged group
            - black_types <list>: type classification (TP,TP;FN,TN) of testset samples of the disadvantaged group
            - white_types <list>: type classification (TP,TP;FN,TN) of testset samples of the advantaged group
    """
    black_scores = []
    black_types = []
    white_scores = []
    white_types = []
    # credit score bounds
    up_bound = 850
    low_bound = 300
    # reward and penalty distribution parameters
    reward_mu, penalty_mu = di_means
    reward_std, penalty_std = di_stds

    for index, label in enumerate(y_predict):

        # first check for TP or TP
        if label == 1 and y_test[index] == 1:  # if it's a TP
            if race_test[index] == 0:  # black
                black_types.append('TP')
                new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std,1))
                if new_score <= up_bound:
                    black_scores.append(new_score)
                else:
                    black_scores.append(up_bound)
            elif race_test[index] == 1:  # white
                white_types.append('TP')
                new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std,1))
                if new_score <= up_bound:
                    white_scores.append(new_score)
                else:
                    white_scores.append(up_bound)
        elif label == 1 and y_test[index] == 0:  # if it's a FP
            if race_test[index] == 0:  # black
                black_types.append('FP')
                new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std,1))
                if new_score < low_bound:
                    black_scores.append(low_bound)
                else:
                    black_scores.append(new_score)
            elif race_test[index] == 1:  # white
                white_types.append('FP')
                new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std,1))
                if new_score < low_bound:
                    white_scores.append(low_bound)
                else:
                    white_scores.append(new_score)
        elif label == 0 and y_test[index] == 0:  # TN, no change to credit score
            if race_test[index] == 0:  # black
                black_types.append('TN')
                black_scores.append(X_test[index][0])
            elif race_test[index] == 1:  # white
                white_types.append('TN')
                white_scores.append(X_test[index][0])
        elif label == 0 and y_test[index] == 1:  # FN, no change to credit score
            if race_test[index] == 0:  # black
                black_types.append('FN')
                black_scores.append(X_test[index][0])
            elif race_test[index] == 1:  # white
                white_types.append('FN')
                white_scores.append(X_test[index][0])

    return black_scores, white_scores, black_types, white_types


def get_types(X_test, y_predict, y_test, race_test):
    """
        Return the types (FP;TP;FN;TN) of all testset samples
        Args:
            - X_test <numpy.ndarray>: samples for testing
            - y_predict <numpy.ndarray>: predicted test set labels a model
            - y_test <numpy.ndarray>: predicted labels
            - race_test <numpy.ndarray>: sensitive attribute of test samples

        Returns:
            - black_types <list>: type classification (TP,TP;FN,TN) of testset samples of the disadvantaged group
            - white_types <list>: type classification (TP,TP;FN,TN) of testset samples of the advantaged group
    """
    black_types = []
    white_types = []

    for index, label in enumerate(y_predict):

        # first check for TP or TP
        if label == 1 and y_test[index] == 1:  # if it's a TP
            if race_test[index] == 0:  # black
                black_types.append('TP')
            elif race_test[index] == 1:  # white
                white_types.append('TP')
        elif label == 1 and y_test[index] == 0:  # if it's a FP
            if race_test[index] == 0:  # black
                black_types.append('FP')
            elif race_test[index] == 1:  # white
                white_types.append('FP')
        elif label == 0 and y_test[index] == 0:  # TN
            if race_test[index] == 0:  # black
                black_types.append('TN')
            elif race_test[index] == 1:  # white
                white_types.append('TN')
        elif label == 0 and y_test[index] == 1:  # FN
            if race_test[index] == 0:  # black
                black_types.append('FN')
            elif race_test[index] == 1:  # white
                white_types.append('FN')

    return black_types, white_types


def add_values_in_dict(sample_dict, key, list_of_values):
    """
        Appending multiple values to a key in the given dictionary.
        # Reference: https://thispointer.com/python-dictionary-with-multiple-values-per-key/
        Args:
            - sample_dict <dict>
            - key <str>: kex for dict
            - list_of_values <list>: values for dict
        Returns:
            - sample_dict <dict>: extended dictionary
    """
    if key not in sample_dict:
        sample_dict[key] = list()
    sample_dict[key].extend(list_of_values)
    return sample_dict


def save_dict_in_csv(results_dict, fieldnames, name_csv):
    """
        Save dictionary as csv.
        # Reference: https://stackoverflow.com/questions/53013274/writing-data-to-csv-from-dictionaries-with-multiple-values-per-key
        Args:
            - results_dict <dict> results that should be saved
            - fieldnames <list>: the col names for csv
            - name_csv <str>: name and path of csv file
    """

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


# The below function uses exponentiated gradient as the reduction algorithm of choice
def add_constraint_and_evaluate(model, constraint_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool=False, di_means=(0,0),di_stds=(0,0)):
    """
    Applying fairness constraint and exponentiated gradient reduction algorithm on the model.
        Args:
            - model <object>: classifier
            - constraint str <str>: indicator for fairness constraint
            - X_train <numpy.ndarray>: samples for training
            - y_train <numpy.ndarray>: labels of training samples
            - race_train <numpy.ndarray>: sensitive attribute of train samples
            - race_test <numpy.ndarray>: sensitive attribute of test samples
            - X_test <numpy.ndarray>: samples for testing
            - y_test <numpy.ndarray>: predicted labels
            - y_predicted <numpy.ndarray>: uncontrained model prediction of labels from test set
            - sample_weight_test <numpy.ndarray>: weights for test samples
            - dashboard_bool <bool>: can be used to display FairnessDashboard
            - di_means <tuple>:means of the delayed impact distributions (only necessary when distributions are used)
            - di_stds <tuple>: deviation of delyed impact distributions (only necessary when distributions are used)

        Returns:
            - mitigator <object>: model trained with contraint
            - results_overall <list>: wrapper list with eval results overall
            - results_0 <list>: wrapper list with eval results disadvantaged group
            - results_1 <list>: wrapper list with eval results advantaged group
            - y_pred_mitigated <numpy.ndarray>: predicted test set labels of mitigated,contrained model
    """

    constraint = get_constraint(constraint_str)

    mitigator = ExponentiatedGradient(model, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    y_pred_mitigated = mitigator.predict(X_test) #y_pred_mitigated

    results_overall, results_0, results_1 = evaluating_model(constraint_str,X_test,y_test, y_pred_mitigated, di_means,di_stds, sample_weight_test,race_test)

    if dashboard_bool:
        pass
        #FairnessDashboard(sensitive_features=race_test,y_true=y_test,y_pred={"initial model": y_predict, "mitigated model": y_pred_mitigated})
    return mitigator, results_overall, results_0, results_1, y_pred_mitigated


# This function
def add_constraint_and_evaluate_gs(model, constraint_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool=False, di_means=(0,0),di_stds=(0,0)):
    """
    Applying fairness constraint and grid search reduction algorithm on the model.
        Args:
            - model <object>: classifier
            - constraint str <str>: indicator for fairness constraint
            - X_train <numpy.ndarray>: samples for training
            - y_train <numpy.ndarray>: labels of training samples
            - race_train <numpy.ndarray>: sensitive attribute of train samples
            - race_test <numpy.ndarray>: sensitive attribute of test samples
            - X_test <numpy.ndarray>: samples for testing
            - y_test <numpy.ndarray>: predicted labels
            - y_predicted <numpy.ndarray>: uncontrained model prediction of labels from test set
            - sample_weight_test <numpy.ndarray>: weights for test samples
            - dashboard_bool <bool>: can be used to display FairnessDashboard
            - di_means <tuple>:means of the delayed impact distributions (only necessary when distributions are used)
            - di_stds <tuple>: deviation of delyed impact distributions (only necessary when distributions are used)

        Returns:
            - mitigator <object>: model trained with contraint
            - results_overall <list>: wrapper list with eval results overall
            - results_0 <list>: wrapper list with eval results disadvantaged group
            - results_1 <list>: wrapper list with eval results advantaged group
            - y_pred_mitigated <numpy.ndarray>: predicted test set labels of mitigated,contrained model
    """

    constraint = get_constraint(constraint_str)

    mitigator = GridSearch(model, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    y_pred_mitigated = mitigator.predict(X_test) #y_pred_mitigated

    results_overall, results_0, results_1 = evaluating_model(constraint_str,X_test,y_test, y_pred_mitigated, di_means,di_stds, sample_weight_test,race_test)

    if dashboard_bool:
        pass
        #FairnessDashboard(sensitive_features=race_test,y_true=y_test,y_pred={"initial model": y_predict, "mitigated model": y_pred_mitigated})
    return mitigator, results_overall, results_0, results_1, y_pred_mitigated