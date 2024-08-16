from itertools import zip_longest
import argparse
import warnings
import pandas as pd
import os
import numpy as np
import sys
sys.path.append('../')
from scripts.classification_utils import load_args
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scripts.evaluation_utils import evaluating_model_german
from scripts.classification_utils import get_classifier,add_values_in_dict, save_dict_in_csv, get_constraint
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds, TruePositiveRateParity,FalsePositiveRateParity, ErrorRateParity




def ml_pipeline(data_path, results_dir, weight_idx, testset_size, mitigated, balanced, fp_weight, fn_weight, model_name, constraint_name, save):
    """
    Classification and evaluation function for the german datasets
    Args:
        data_path <str>: path to the dataset csv-file
        results_dir <str>: directory to save the results
        weight_idx <int>: weight index for samples (1 in our runs)
        testset_size <float>: proportion of testset samples in the dataset (e.g. 0.3)
        balanced <boolean>: cost-sensitive classification weight
        fp_weight <int>: cost-sensitive classification value for FPs
        fn_weight <int>: cost-sensitive classification value for FNs
        model_name <str>: classifier name in acronym form used for training
        constraint <str>: fairness constraint used for training
        save <bool>: indicator if the results should be saved
    """

    """
    DATA PREPARATION
    """
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Load and extract data
    data = pd.read_csv(data_path)

    #print(data)
    # print(type(data))
    # print(data.shape)
    #print(data.columns)
    # print(data['age'])
    # print(data['credit'])

    # drop credit and then re-add it to the end of the dataframe
    x = data.drop(['credit'], axis=1)

    # Y labels needed to be 0s and 1s
    # target label is credit, 1 (Good)-->0 or 2 (Bad)-->1
    y = data['credit']
    # print(y)
    y_changed_0s = y.replace(to_replace=1, value=0)
    # print(y_changed_0s)
    y = y_changed_0s.replace(to_replace=2, value=1)
    # print(y)

    """
    PARAMETER SETTING
    """
    if mitigated == 0:  #unmitigated
        constraint_str = 'Unmit'
    elif mitigated == 1:  # mitigated by reductions
        # 'DP': DemographicParity, 'EO': EqualizedOdds, 'TPRP': TruePositiveRateParity, 'FPRP': FalsePositiveRateParity, 'ERP': ErrorRateParity
        constraint = get_constraint(constraint_name)
        constraint_str = constraint_name
    elif mitigated == 2:    # mitigated by cost-sensitive
        constraint_str = 'Cost'

    results_path = results_dir  # directory to save the results
    #models = {'Decision Tree': 'dt', 'Logistic Regression': 'lgr'}
    if model_name == 'lgr':
        max_iterations = 100000

    if mitigated == 2:
        if balanced:
            run_key = f'{model_name}cost-balance'
            results_path_full = results_path + model_name + f'/cost-balance/'
        else:
            run_key = f'{model_name}cost-fp{fp_weight}-fn{fn_weight}'
            results_path_full = results_path + model_name + f'/cost-fp{fp_weight}-fn{fn_weight}/'
    else:
        run_key = f'{model_name + constraint_name}'
        results_path_full = results_path + model_name + constraint_str + '/'

    os.makedirs(results_path_full, exist_ok=True)
    # old bit:
    # os.makedirs(f'{results_path}{model_name}{constraint_str}', exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testset_size, random_state=42)
    # NOTE: the labels are 1 or 2
    # print(y_train)
    # print(y_test)
    X_train = X_train.reset_index().drop(['index'], axis=1)
    X_test = X_test.reset_index().drop(['index'], axis=1)
    y_train = y_train.reset_index().drop(['index'], axis=1)
    y_test = y_test.reset_index().drop(['index'], axis=1)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # weight_index: 1 means all equal weights
    if weight_idx == 1:
        # print('Sample weights are all equal.')
        sample_weight_train = np.ones(shape=(len(y_train),))
        sample_weight_test = np.ones(shape=(len(y_test),))
    # weight_index: 0 means use sample weights
    elif weight_idx == 0:
        print('Sample weights are NOT all equal.')

    # NOTE:
    # Adult (advantaged) is 1
    # Youth (disadvantaged) is 0
    train_age = X_train['age']
    test_age = X_test['age']

    # NOTE: these are all pandas series
    train_credit = X_train['credit_amount']
    test_credit = X_test['credit_amount']

    # use x to check month data
    #months = x['month']
    # print(months.max())     # 72
    # print(months.min())     # 4
    # print(months.mean())    # 20.903
    # print(months.median()) # 18 median


    """
    MODEL TRAINING
    """

    # UNMITIGATED MODEL TRAINING / REDUCTION
    if mitigated == 0 or mitigated == 1:
        print("made it in the unmitigated or reduction thread")
        print('The classifier trained below is: ', model_name)
        if model_name == 'dt':
            classifier = DecisionTreeClassifier(random_state=0)
        elif model_name == 'lgr':
            classifier = LogisticRegression(max_iter=100000, random_state=0)
        else:
            print("error: input an acceptable model name acronoym")

        # Train the classifier:
        model = classifier.fit(X_train, y_train, sample_weight_train)

        # Make predictions with the classifier:
        y_predict = model.predict(X_test)

        # Scores on test set
        #test_scores = model.predict_proba(X_test)[:, 1]
        if mitigated == 1:
            print("made it in the reductions thread")
            mitigator = ExponentiatedGradient(model, constraint)

            mitigator.fit(X_train, y_train, sensitive_features=train_age)
            y_pred_mitigated = mitigator.predict(X_test, random_state = 0)  # y_pred_mitigated

    # COST-SENSITIVE CLASSIFICATION
    elif mitigated == 2:
        print('in cost-sensitive thread')
        # Resource: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html
        # {0:c10 (FP), 1:c01 (FN)}: The misclassification costs are explicitly set for the two classes by means of a dictionary.
        # Conf matrix: [c00,     c01(FN)]
        #              [c10(FP), c11]

        if model_name == 'lgr':
            if not balanced:
                classifier = LogisticRegression(class_weight={0: fp_weight, 1: fn_weight},
                                                max_iter=max_iterations, random_state=0)  # so I can add in weights
            else:
                print('lgr and balanced')
                classifier = LogisticRegression(class_weight='balanced', max_iter=max_iterations, random_state=0)
        elif model_name == 'dt':
            if not balanced:
                classifier = DecisionTreeClassifier(class_weight={0: fp_weight, 1: fn_weight}, random_state=0)
            else:
                classifier = DecisionTreeClassifier(class_weight='balanced', random_state=0)
        else:
            print("Error: we shouldn't end up here")

        # Train the classifier:
        model = classifier.fit(X_train, y_train, sample_weight_train)

        # Make predictions with the classifier:
        y_predict = model.predict(X_test)
    else:
        print("Logic error: we shouldn't end up here")


    """
    SAVING RESULTS
    """
    overall_results_dict = {}
    young_results_dict = {}
    old_results_dict = {}
    combined_results_dict = {}

    if mitigated == 0 or mitigated == 2:  # if unmitigated or cost then...
        print('made it in the correct evaluations line for unmit and cost-sensitive')
        results_overall, results_young, results_old, impact_focused_results_young, impact_focused_results_old = evaluating_model_german(constraint_str, X_test, y_test, y_predict,
                                                                          test_credit, sample_weight_test,
                                                                          test_age)
    elif mitigated == 1:  # if reductions alg then...
        print("made it in the correct evaluations line for reductions")
        results_overall, results_young, results_old, impact_focused_results_young, impact_focused_results_old = evaluating_model_german(constraint_str, X_test, y_test, y_pred_mitigated,
                                                                          test_credit, sample_weight_test,
                                                                          test_age)
    else:
        print("you shouldn't end up here...")

    # TESTING THE NEW PANDAS dfs
    #print(impact_focused_results_young)
    #print(impact_focused_results_old)

    # TODO: add some analysis of the pandas df like I did INSIDE of the calculate impact german function
    # ^Or make that a separate script!!
    # calculate median values of each group after taking out the 0s
    #print('score_youth', score_youth)
    '''
    score_youth_nozeros = [i for i in score_youth if i != 0]
    score_old_nozeros = [i for i in score_old if i != 0]

    print('# of youths: ', len(score_youth))
    score_youth_nozeros = [i for i in score_youth if i != 0]
    print('# of youths WITHOUT 0: ', len(score_youth_nozeros))
    print('median of score_youth updated', median(score_youth_nozeros))
    print('score_youth updated: ', score_youth_nozeros)

    print('# of olds: ', len(score_old))
    score_old_nozeros = [i for i in score_old if i != 0]
    print('# of olds WITHOUT 0: ', len(score_old_nozeros))
    print('median of score_old updated: ', median(score_old_nozeros))
    print('score_old updated: ', score_old_nozeros)
    '''

    #  results_overall  =  [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, i_youth, i_old, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]
    #  results_young    =  [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1, impact]
    #  results_old    =  [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0, impact]
    #
    #
    # added in f1_weighted, results_overall[3] after accuracy
    # overall_accuracy, f1_weighted, sr, tnr, tpr, fner, fper, i_young, i_old, tnr_b, tpr_b, fner_b, b_fper, w_tnr, w_tpr, w_fner, w_fper
    combined_results = [results_overall[3], results_overall[0], results_overall[5], results_overall[6],
                        results_overall[7], results_overall[8], results_overall[9], results_overall[10],
                        results_overall[11], results_young[6], results_young[7], results_young[8], results_young[9],
                        results_old[6], results_old[7], results_old[8], results_old[9]]

    overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
    young_results_dict = add_values_in_dict(young_results_dict, run_key, results_young)
    old_results_dict = add_values_in_dict(old_results_dict, run_key, results_old)
    combined_results_dict = add_values_in_dict(combined_results_dict, run_key, combined_results)

    if save == True:
        print('Saving results.')
        overall_fieldnames = ['Run', 'Acc', 'ConfMatrix', 'F1micro', 'F1weighted', 'F1binary', 'SelectionRate',
                              'TNR rate', 'TPR rate', 'FNER', 'FPER', 'ImpactYouth', 'ImpactOld', 'DP Diff', 'EO Diff',
                              'TPR Diff', 'FPR Diff', 'ER Diff']
        byage_fieldnames = ['Run', 'Acc', 'ConfMatrix', 'F1micro', 'F1weighted', 'F1binary', 'SelectionRate',
                            'TNR rate', 'TPR rate', 'FNER', 'FPER', 'Impact']
        combined_fieldnames = ['Run', 'F1_weighted', 'Acc', 'SelectionRate', 'TNR', 'TPR', 'FNER', 'FPER',
                               'Youth Impact', 'Old Impact', 'TNR_Y', 'TPR_Y', 'FNER_Y', 'FPER_Y', 'TNR_O', 'TPR_O',
                               'FNER_O', 'FPER_O']
        save_dict_in_csv(overall_results_dict, overall_fieldnames,
                         results_path_full + model_name + '_overall_results.csv')
        save_dict_in_csv(young_results_dict, byage_fieldnames, results_path_full + model_name + '_young_results.csv')
        save_dict_in_csv(old_results_dict, byage_fieldnames, results_path_full + model_name + '_old_results.csv')
        save_dict_in_csv(combined_results_dict, combined_fieldnames,
                         results_path_full + model_name + '_combined_results.csv')
        impact_focused_results_young.to_csv(results_path_full + model_name + '_young_impact_results.csv', index=False)
        impact_focused_results_young.to_csv(results_path_full + model_name + '_old_impact_results.csv', index=False)

if __name__ == '__main__':
    #load arguments
    parser = argparse.ArgumentParser(description='Specify the path to your config file.')
    parser.add_argument('-config', type=str, help="Path to where your config yaml file is stored.")
    args = parser.parse_args()

    try:
        args = load_args(f'configs/{args.config}.yaml')
    except:
        print(f'File does not exist: configs/{args.config}.yaml')
    #run classification (incl. evaluation)
    ml_pipeline(**args)