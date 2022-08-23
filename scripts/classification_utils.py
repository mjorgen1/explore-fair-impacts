import pandas as pd
import numpy as np
from itertools import zip_longest
import csv
import os
import warnings
import yaml
from yaml.loader import SafeLoader

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
from raiwidgets import FairnessDashboard

from scripts.evaluation_utils_crossval import evaluating_model
from scripts.visualization_utils import visual_repay_dist, visual_scores_by_race

def load_args(file):
    """
    Load args and run some basic checks.
        Args:
            - file <str>: full path to .yaml config file
        Returns:
            - data <dict>: dictionary with all args from file
    """
    with open(file, "r") as stream:
        try:
            data = yaml.load(stream, Loader=SafeLoader)
            print('Arguments: ',data)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_data(file):
    data = pd.read_csv(file)
    data[['score', 'race']] = data[['score', 'race']].astype(int)
    #print(data)
    return data

def create_original_set_ratios(x_data, y_data, race_ratio, set_size_upper_bound):
    """
    Changes the proportions of samples in the set. Proportion of each group (race) and proportion of labels for the Black (0) group.
        Args:
            - x_data <numpy.ndarray>: ['score','repay_probability','race'] -> array of samples
            - y_data <numpy.ndarray>: ['repay_indices'] -> array of samples
            - race_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of black to white samples generated (Black,White)
            - set_size_upper_bound <int>: absolute upper bound of the size for the dataset (e.g 100,000)
        Returns:
            subset of x_data and y_data
    """
    # Black = 0; White = 1
    # limits the absolute test_size if necessary

    if len(y_data) > set_size_upper_bound:
        set_size = set_size_upper_bound
    else:
        set_size = len(y_data)
    # set set sizes for each race
    set_size_0 = int(set_size * race_ratio[0])
    set_size_1 = int(set_size * race_ratio[1])

    # number of samples for the Black group, according to the label ratio
    num_0P = int(set_size_0 * 0.34)
    num_0N = int(set_size_0 * 0.66)
    num_1P = int(set_size_1 * 0.76)
    num_1N = int(set_size_1 * 0.24)

    # getting the indices of each samples for each group
    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]

    idx_1 = np.where(x_data[:, 1] == 1)[0]

    # if group size numbers are larger than the available samples for that group adjust it
    if len(idx_0P) < num_0P:
        num_0P = len(idx_0P)
        num_0N = int(num_0P/0.34 * 0.66)
        num_1P =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.76)
        num_1N =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.24)
    if len(idx_0N) < num_0N:
        num_0N = len(idx_0N)
        num_0P = int(num_0N/0.66 * 0.34)
        num_1P =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.76)
        num_1N =  int((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.24)
    if len(idx_1P) < num_1P:
        num_1P = len(idx_1P)
        num_1N = int(num_1P/0.76 * 0.24)
        num_0P =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * 0.34)
        num_0N =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * 0.66)
    if len(idx_1N) < num_1N:
        num_1N = len(idx_1N)
        num_1P = int(num_1N/0.24 * 0.76)
        num_0P =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * 0.34)
        num_0N =  int((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * 0.66)


    # take the amount of samples, by getting the amount of indices
    idx_0N = idx_0N[:num_0N]
    idx_0P = idx_0P[:num_0P]
    idx_1N = idx_1N[:num_1N]
    idx_1P = idx_1P[:num_1P]
    # concatenate indices
    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))

    return x_data[idx,:], y_data[idx]

def balance_test_set_ratios(x_data, y_data, test_set_bound):

    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]
    num = min(len(idx_0N),len(idx_0P),len(idx_1N),len(idx_1P))
    if num > test_set_bound * 0.25:
        num = int(test_set_bound * 0.25)

    idx_0N = idx_0N[:num]
    idx_1N = idx_1N[:num]
    idx_0P = idx_0P[:num]
    idx_1P = idx_1P[:num]

    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))

    return x_data[idx,:], y_data[idx]


def print_type_ratios(x_data,y_data):
    idx_0N = np.where((x_data[:, 1] == 0) & (y_data == 0))[0]
    idx_1N = np.where((x_data[:, 1] == 1) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 1] == 0) & (y_data == 1))[0]
    idx_1P = np.where((x_data[:, 1] == 1) & (y_data == 1))[0]
    print('Black N/P:',len(idx_0N),'/',len(idx_0P),'White N/P:',len(idx_1N),'/',len(idx_1P))


def prep_data(data, test_size, demo_ratio,label_ratio,test_set_variant, set_bound,weight_index):
    # might need to include standardscaler here

    x = data[['score', 'race']].values
    y = data['repay_indices'].values
    print(' Whole set:', len(y))
    print_type_ratios(x,y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    #print('Here are the x values: ', x, '\n')
    #print('Here are the y values: ', y)
    print('Training set:', len(y_train))
    print_type_ratios(X_train,y_train)
    #X_train, y_train = adjust_train_set_ratios(X_train, y_train, label_ratio, demo_ratio[0], set_bound[0])
    #print('Training set:', len(y_train))
    #print_type_ratios(X_train,y_train)

    #print_type_ratios(X_test,y_test)
    if test_set_variant == 1:
        X_test, y_test = balance_test_set_ratios(X_test, y_test, set_bound[1])
    if test_set_variant == 2:
        X_test, y_test = create_original_set_ratios(X_test, y_test, [0.12,0.88], set_bound[1])

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
         raise ValueError('unvalid classisfier {dt,gnb,lgr,gbt}')

    return classifier


def get_constraint(constraint_str):
  #set seed for consistent results with ExponentiatedGradient
   #np.random.seed(0)
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
    black_types = []
    white_scores = []
    white_types = []
    up_bound = 850
    low_bound = 300
    reward = 75
    penalty = -150

    for index, label in enumerate(y_predict):

        # first check for TP or TP
        if label == 1 and y_test[index] == 1:  # if it's a TP
            if race_test[index] == 0:  # black
                black_types.append('TP')
                new_score = X_test[index][0] + reward
                if new_score <= up_bound:
                    black_scores.append(new_score)
                else:
                    black_scores.append(up_bound)
            elif race_test[index] == 1:  # white
                white_types.append('TP')
                new_score = X_test[index][0] + reward
                if new_score <= up_bound:
                    white_scores.append(new_score)
                else:
                    white_scores.append(up_bound)
        elif label == 1 and y_test[index] == 0:  # if it's a FP
            if race_test[index] == 0:  # black
                black_types.append('FP')
                new_score = X_test[index][0] + penalty
                if new_score < low_bound:
                    black_scores.append(low_bound)
                else:
                    black_scores.append(new_score)
            elif race_test[index] == 1:  # white
                white_types.append('FP')
                new_score = X_test[index][0] + penalty
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


def add_constraint(model, constraint_str, reduction_alg, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool):
    # set seed for consistent results with ExponentiatedGradient
    #np.random.seed(0)
    constraint = get_constraint(constraint_str)
    mitigator = get_reduction_algo(model,constraint, reduction_alg)
    mitigator.fit(X_train, y_train, sensitive_features=race_train)
    y_pred_mitigated = mitigator.predict(X_test) #y_pred_mitigated

    results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_pred_mitigated, sample_weight_test,race_test)

    if dashboard_bool:
        pass
        #FairnessDashboard(sensitive_features=race_test,y_true=y_test,y_pred={"initial model": y_predict, "mitigated model": y_pred_mitigated})
    return mitigator, results_overall, results_black, results_white, y_pred_mitigated


def classify(data_path,results_dir,weight_idx,testset_size,demo_ratio,label_ratio, test_set_variant, set_bound, models,constraints,reduction_algorithms,save):

    warnings.filterwarnings('ignore', category=FutureWarning)
    # Load and Prepare data
    data = get_data(data_path)
    x = data[['score', 'race']].values
    y = data['repay_indices'].values


    X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test = prep_data(data, testset_size,demo_ratio, label_ratio,test_set_variant,set_bound, weight_idx)

    visual_scores_by_race(results_dir,'all',x)
    visual_repay_dist(results_dir,'all',x,y)
    visual_scores_by_race(results_dir,'train',X_train)
    visual_scores_by_race(results_dir,'test',X_test)
    visual_repay_dist(results_dir,'train',X_train, y_train)
    visual_repay_dist(results_dir,'test',X_test,y_test)
    # split up X_test by race
    X_test_b = []
    X_test_w = []
    y_test_b = []
    y_test_w = []

    for index in range(len(X_test)):
        if race_test[index] == 0:  # black
            X_test_b.append(X_test[index][0])
            y_test_b.append(y_test[index])
        elif race_test[index] == 1:  # white
            X_test_w.append(X_test[index][0])
            y_test_w.append(y_test[index])

    for model_str in models.values():
        print(model_str)
        results_path = results_dir
        results_path += f'{model_str}/'
        os.makedirs(results_path, exist_ok=True)

        models_dict = {}
        overall_results_dict = {}
        black_results_dict = {}
        white_results_dict = {}
        all_scores = []
        all_types = []
        scores_names = []

        T_test_b = ['TP' if e==1 else "TN" for e in y_test_b]
        T_test_w = ['TP' if e==1 else "TN" for e in y_test_w]

        all_types.extend([T_test_b,T_test_w])
        all_scores.extend([X_test_b,X_test_w])
        scores_names.extend(['testB', 'testW'])

        # Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
        # train unconstrained model
        classifier = get_classifier(model_str)
        #np.random.seed(0)
        model = classifier.fit(X_train,y_train, sample_weight_train)
        y_predict = model.predict(X_test)

        # Scores on test set
        test_scores = model.predict_proba(X_test)[:, 1]
        models_dict = {"Unmitigated": (y_predict, test_scores)}

        # given predictions+outcomes, I'll need to do the same
        x = data[['score', 'race']].values
        y = data['repay_indices'].values
        scores = cross_val_score(model, x, y, cv=5, scoring='f1_weighted')

        #save scores and types (TP,FP,TN,FN) in list
        X_b, X_w, T_b, T_w = get_new_scores(X_test, y_predict, y_test, race_test)
        all_types.extend([T_b,T_w])
        all_scores.extend([X_b,X_w])
        scores_names.extend(['unmitB', 'unmitW'])

        # evaluate model
        constraint_str = 'Un-'
        results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, sample_weight_test,race_test)

        # adding results to dict
        run_key = f'{model_str} Unmitigated'
        overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
        black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
        white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # train all constrained model for this model type
        for algo_str in reduction_algorithms.values():
            for constraint_str in constraints.values():

                print(algo_str,constraint_str)
                mitigator, results_overall, results_black, results_white, y_pred_mitigated = add_constraint(model, constraint_str, algo_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool=False)
                # Visualisation
                if algo_str ==' GS':
                    pass
                    # We can examine the values of lambda_i chosen for us:
                    #lambda_vecs = mitigator.lambda_vecs_
                    #print(lambda_vecs[0])
                    #models_dict = grid_search_show(mitigator, demographic_parity_difference, y_pred_mitigated, X_test, y_test, race_test, 'DemParityDifference','GS DPD', models_dict, 0.3)
                    #models_dict.pop('GS DPD')

                #save scores in list
                X_b, X_w, T_b, T_w = get_new_scores(X_test, y_pred_mitigated, y_test, race_test)
                all_types.extend([T_b,T_w])
                all_scores.extend([X_b,X_w])
                scores_names.extend([f'{algo_str.lower()}{constraint_str.lower()}B', f'{algo_str.lower()}{constraint_str.lower()}W'])

                run_key = f'{model_str} {algo_str} {constraint_str} Mitigated'
                overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
                black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
                white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # save evaluations
        if save == True:
            overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB','DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
            byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
            save_dict_in_csv(overall_results_dict, overall_fieldnames, results_path+model_str+'_overall_results.csv')
            save_dict_in_csv(black_results_dict, byrace_fieldnames, results_path+model_str+'_black_results.csv')
            save_dict_in_csv(white_results_dict, byrace_fieldnames, results_path+model_str+'_white_results.csv')

            # Save overall score results
            columns_data_scores = zip_longest(*all_scores)
            columns_data_types = zip_longest(*all_types)

            with open(results_path+model_str+'_all_scores.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_scores)
                f.close()
            with open(results_path+model_str+'_all_types.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_types)
                f.close()





def average_results_dict(list_with_dicts):
    for i,dict in enumerate(list_with_dicts):
        print(i)
        if i == 0:
            results = pd.DataFrame(dict)
        else:
            second = pd.DataFrame(dict)
            results = results.add(second, fill_value=0)
    #results = {k: [x/len(list_with_dicts) for x in v] for k, v in results.items()}
    #results = {k: [x for x in v] for k, v in results.items()}
    results = results.div(len(list_with_dicts))
    results = results.to_dict()
    for k,v in results.items():
        results[k] = list(v.values())
    return results

def average_score_lists(label_lists,value_lists):
    label_list_flat = [item for sublist in label_lists for item in sublist]
    value_lists_flat = [item for sublist in value_lists for item in sublist]
    labels = set(label_list_flat)
    results = []
    result_l = []
    for i,l in enumerate(labels):
        idx = [i for i,label in enumerate(label_list_flat) if label==l]
        v_lists = [value_lists_flat[i] for i in idx]
        num = len(v_lists)
        results.append([sum(x)/num for x in zip(*v_lists)])
        result_l.append(l)
    return results, result_l



def classify_crossval(data_path,results_dir,weight_idx,testset_size,demo_ratio,label_ratio, test_set_variant, set_bound, models,constraints,reduction_algorithms,save):

    warnings.filterwarnings('ignore', category=FutureWarning)
    k = 3
    # Load and Prepare data
    data = get_data(data_path)
    X = data[['score', 'race']].values
    y = data['repay_indices'].values

    #prep_data(data, testset_size,demo_ratio, label_ratio,test_set_variant,set_bound, weight_idx)
    # might need to include standardscaler here

    print(' Whole set:', len(y))
    print_type_ratios(X,y)

    skf = StratifiedKFold(n_splits=k)
    for model_str in models.values():
        print(model_str)
        results_path = results_dir
        results_path += f'{model_str}/'
        os.makedirs(results_path, exist_ok=True)

        # Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
        # train unconstrained model
        classifier = get_classifier(model_str)

        overall_results_k = []
        black_results_k = []
        white_results_k = []
        all_scores_k = []
        #all_types_k = []
        scores_names_k = []
        k_counter = 0
        for train_index, test_index in skf.split(X,y):
            overall_results_dict = {}
            black_results_dict = {}
            white_results_dict = {}
            all_scores = []
            #all_types = []
            scores_names = []
            k_counter += 1
            print(k_counter)

            X_train, X_test = X[train_index,:], X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            #print('Here are the x values: ', x, '\n')
            #print('Here are the y values: ', y)
            #print('Training set:', len(y_train))
            #print_type_ratios(X_train,y_train)
            #X_train, y_train = adjust_train_set_ratios(X_train, y_train, label_ratio, demo_ratio[0], set_bound[0])
            #print('Training set:', len(y_train))
            #print_type_ratios(X_train,y_train)

            #print_type_ratios(X_test,y_test)
            if test_set_variant == 1:
                X_test, y_test = balance_test_set_ratios(X_test, y_test, set_bound[1])
            if test_set_variant == 2:
                X_test, y_test = create_original_set_ratios(X_test, y_test, [0.12,0.88], set_bound[1])
                #X_train, y_train = create_original_set_ratios(X_train, y_train, [0.12,0.88], set_bound[0])

            print('Training set:', len(y_train))
            print_type_ratios(X_train,y_train)
            print('Testing set:', len(y_test))
            print_type_ratios(X_test,y_test)

            # collect our sensitive attribute
            race_train = X_train[:, 1]
            race_test = X_test[:, 1]
            # weight_index: 1 means all equal weights
            if weight_idx:
                #print('Sample weights are all equal.')
                sample_weight_train = np.ones(shape=(len(y_train),))
                sample_weight_test = np.ones(shape=(len(y_test),))
            # weight_index: 0 means use sample weights
            elif weight_idx:
                print('Sample weights are NOT all equal.')
                # TODO
                #print('TODO')
            #return X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test

            visual_scores_by_race(results_dir,'all',X)
            visual_repay_dist(results_dir,'all',X,y)
            visual_scores_by_race(results_dir,'train',X_train)
            visual_scores_by_race(results_dir,'test',X_test)
            visual_repay_dist(results_dir,'train',X_train, y_train)
            visual_repay_dist(results_dir,'test',X_test,y_test)
            # split up X_test by race
            X_test_b = []
            X_test_w = []
            y_test_b = []
            y_test_w = []

            for index in range(len(X_test)):
                if race_test[index] == 0:  # black
                    X_test_b.append(X_test[index][0])
                    y_test_b.append(y_test[index])
                elif race_test[index] == 1:  # white
                    X_test_w.append(X_test[index][0])
                    y_test_w.append(y_test[index])
            #T_test_b = ['TP' if e==1 else "TN" for e in y_test_b]
            #T_test_w = ['TP' if e==1 else "TN" for e in y_test_w]

            #all_types.extend([T_test_b,T_test_w])
            all_scores.extend([X_test_b,X_test_w])
            scores_names.extend([f'testB', f'testW'])

            #np.random.seed(0)
            model = classifier.fit(X_train,y_train, sample_weight_train)
            y_predict = model.predict(X_test)

            # Scores on test set
            test_scores = model.predict_proba(X_test)[:, 1]
            models_dict = {"Unmitigated": (y_predict, test_scores)}

            # given predictions+outcomes, I'll need to do the same
            x = data[['score', 'race']].values
            y = data['repay_indices'].values
            scores = cross_val_score(model, x, y, cv=5, scoring='f1_weighted')

            #save scores and types (TP,FP,TN,FN) in list
            X_b, X_w, T_b, T_w = get_new_scores(X_test, y_predict, y_test, race_test)
            #all_types.extend([T_b,T_w])
            all_scores.extend([X_b,X_w])
            scores_names.extend([f'unmitB', f'unmitW'])

            # evaluate model
            constraint_str = 'Un-'
            results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, sample_weight_test,race_test)

            # adding results to dict
            run_key = f'{model_str} Unmitigated'
            overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
            black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
            white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

            # train all constrained model for this model type
            for algo_str in reduction_algorithms.values():
                for constraint_str in constraints.values():

                    print(algo_str,constraint_str)
                    mitigator, results_overall, results_black, results_white, y_pred_mitigated = add_constraint(model, constraint_str, algo_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool=False)

                    #save scores in list
                    X_b, X_w, T_b, T_w = get_new_scores(X_test, y_pred_mitigated, y_test, race_test)
                    #all_types.extend([T_b,T_w])
                    all_scores.extend([X_b,X_w])
                    scores_names.extend([f'{algo_str.lower()}{constraint_str.lower()}B', f'{algo_str.lower()}{constraint_str.lower()}W'])

                    run_key = f'{model_str} {algo_str} {constraint_str} Mitigated'
                    overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
                    black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
                    white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

            overall_results_k.append(overall_results_dict)
            black_results_k.append(black_results_dict)
            white_results_k.append(white_results_dict)

            scores_names_k.append(scores_names)
            all_scores_k.append(all_scores)
            #all_types_k.append(all_types)


        o_results = average_results_dict(overall_results_k)
        b_results = average_results_dict(black_results_k)
        w_results = average_results_dict(white_results_k)
        scores_results,scores_labels = average_score_lists(scores_names_k,all_scores_k)
        # save evaluations
        if save == True:
            overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB','DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
            byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
            save_dict_in_csv(o_results, overall_fieldnames, results_path+model_str+'_overall_results.csv')
            save_dict_in_csv(b_results, byrace_fieldnames, results_path+model_str+'_black_results.csv')
            save_dict_in_csv(w_results, byrace_fieldnames, results_path+model_str+'_white_results.csv')

            # Save overall score results
            columns_data_scores = zip_longest(*scores_results)
            #columns_data_types = zip_longest(*all_types)

            with open(results_path+model_str+'_all_scores.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_labels)
                writer.writerows(columns_data_scores)
                f.close()
        #    with open(results_path+model_str+'_all_types.csv',mode='w') as f:
            #    writer = csv.writer(f)
            #    writer.writerow(scores_names)
            #    writer.writerows(columns_data_types)
            #    f.close()
