from statistics import median

import numpy as np
import pandas as pd
import sys
#print(sys.executable)
#sys.path.append('/home/kenz/git-workspace/explore-fair-impacts/.venv/lib/python3.11/site-packages/fairlearn')
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from fairlearn.metrics import *


def inspect_MinMax(samples_A_probs,samples_B_probs):
    """
    Prints the lowest and highest repay probability for each group.
        Args:
            - samples_A_probs <numpy.ndarray>: probabilitys of the samples of group A
            - samples_B_probs <numpy.ndarray>: probabilitys of the samples of group B
    """
    max_val_A = np.max(samples_A_probs)
    min_val_A = np.min(samples_A_probs)
    print('the range of the Group A (Black) repay probabilities is: ', max_val_A-min_val_A)
    print('the min value is: ', min_val_A)
    print('the max value is: ', max_val_A)

    max_val_B = np.max(samples_B_probs)
    min_val_B = np.min(samples_B_probs)
    print('the range of the Group B (White) repay probabilities is: ', max_val_B-min_val_B)
    print('the min value is: ', min_val_B)
    print('the max value is: ', max_val_B)


def delayed_impact_csv(data_path= 'data/results/', group= 0, folders= ['dt','gnb','lgr','gbt']):
    """
    Extracting the Delayed Impact by group out of multiple files and saving it in an extra csv.
    Works only if delayed impact was properly calculated when classification.
        Args:
            - data_path <str>: data dir of the results
            - group <int>: group indicator (0,1)
            - folders <list>: result folders (classifier)
    """

    col_names_df = []
    # collect the impact from all folders
    for i,f in enumerate(folders):
        if group == 0:
            path = f'{data_path}{f}/{f}_0_results.csv'
        else:
            path = f'{data_path}{f}/{f}_1_results.csv'

        df = pd.read_csv(path,index_col=0)
        df = df.reset_index()
        col_names_df.append(f'{f.upper()}')
        if i == 0:
            joined_df = df.iloc[:,-1]
        else:
            joined_df = pd.concat([joined_df, df.iloc[:,-1]], axis=1)

    joined_df.set_axis(folders, axis=1)
    df = joined_df
    # set new index
    df['Constraint'] = ['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP']
    df.set_index('Constraint',inplace=True)
    df.columns = col_names_df
    print('Group: ',group,'\n DataFrame: \n',df)
    df.to_csv(f'{data_path}/{group}_DI.csv')


def types_csvs(data_path= 'data/results/', folders= ['dt','gnb','lgr','gbt']):
    """
    Collect the types of the predicted samples (FP;TP;FN;TN) for all model save them in an extra csv.
    No scores needed
        Args:
            - data_path <str>: data dir of the results
            - folders <list>: result folders (classifier)
    """

    pd.set_option('display.max_columns', None)
    for i,f in enumerate(folders):
        path = f'{data_path}{f}/{f}_all_types.csv'
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)
        df = df.melt(var_name="ID",value_name="Category")

        # save ratios 
        df_norm = df.groupby('ID')['Category'].value_counts(normalize=True)
        df_norm = df_norm.rename('Ratio')
        df_norm = pd.DataFrame(df_norm)
        df_norm = df_norm.reset_index()
        
        df_norm = df_norm.pivot(index='Category', columns='ID')['Ratio']
        df_norm = df_norm.fillna(0)
        df_norm = df_norm.round(decimals = 3)
        df_norm.to_csv(f'{data_path}{f}/{f}_type_ratios.csv')

        # save absolute nombers
        df = df.groupby('ID')['Category'].value_counts()
        df = df.rename('Number')
        df = pd.DataFrame(df)
        df = df.reset_index()
        df = df.pivot(index='Category', columns='ID')['Number']
        df = df.fillna(0)
        df.to_csv(f'{data_path}{f}/{f}_type_absolute.csv')


def calculate_delayed_impact(X_test, y_true, y_pred, di_means, di_stds, sensitive_attr_test):
        """
        Calculate the Delayed Impact (DI) (average score change of each group) (considering TP,FP)
            Args:
                - X_test <numpy.ndarray>: samples (scores) of the test set
                - y_true <numpy.ndarray>: true labels of the test set
                - y_pred <numpy.ndarray>: predicted labels for the test set
                - di_means <tuple>:means of the delayed impact distributions
                - di_stds <tuple>: deviation of delyed impact distributions
                - sensitive_attr_test <numpy.ndarray>: indicator of the group/race (Black is 0 and White it 1)
            Returns:
                - di_black <float>: DI for group Black
                - di_white <float>: DI for group White
        """

        # split mean and std for score change distributions (reward fot TP, penalty for FP)
        reward_mu, penalty_mu = di_means
        reward_std, penalty_std = di_stds

        # bounds
        up_bound = 850
        low_bound = 300

        di_black, di_white = 0, 0
        score_diff_black, score_diff_white = [], []
        scores = X_test[:, 0]

        for index, true_label in enumerate(y_true):
            # check for TPs
            if true_label == y_pred[index] and true_label == 1:
                new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std, 1))
                if sensitive_attr_test[index] == 0:  # black borrower
                    if new_score >= up_bound:
                        score_diff_black.append(up_bound - X_test[index][0])
                    else:
                        score_diff_black.append(new_score - X_test[index][0])
                elif sensitive_attr_test[index] == 1:  # white borrower
                    if new_score > up_bound:
                        score_diff_white.append(up_bound - X_test[index][0])
                    else:
                        score_diff_white.append(new_score - X_test[index][0])
            # check for FPs
            elif true_label == 0 and y_pred[index] == 1:
                new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std, 1))
                if sensitive_attr_test[index] == 0:  # black borrower
                    if new_score < low_bound:
                        score_diff_black.append(low_bound - X_test[index][0])
                    else:
                        score_diff_black.append(new_score - X_test[index][0])
                elif sensitive_attr_test[index] == 1:  # white borrower
                    if new_score < low_bound:
                        score_diff_white.append(low_bound - X_test[index][0])
                    else:
                        score_diff_white.append(new_score - X_test[index][0])
            # check for FNs and TNs
            else:
                if sensitive_attr_test[index] == 0:  # black indiv
                    score_diff_black.append(0)
                elif sensitive_attr_test[index] == 1:  # white indiv
                    score_diff_white.append(0)

        # calculate mean score difference or delayed impact of each group
        di_black = sum(score_diff_black) / len(score_diff_black)
        di_white = sum(score_diff_white) / len(score_diff_white)

        return di_black, di_white


# Only focusing on TP and FP impacts based on the loan amt they'd get
def calculate_impact_german(y_test, y_pred, loan_val, sensitive_attr_test):
    """
    Calculate the Impact (I) (average loan amt gained/lost of each group) (considering TP and FP)
        Args:
            - X_test <numpy.ndarray>: samples (scores) of the test set
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - credit_amount_4months <pd.series>: four months worth of credit
            - sensitive_attr_test <numpy.ndarray>: indicator of the group (Youth is 0 and Old is 1)
        Returns:
            - i_youth <float>: impact for group youth
            - i_old <float>: impact for group old
    """
    # TODO: update the comments above

    score_youth, score_old = [], []
    outcome_youth, outcome_old = [], []
    loan_request_youth, loan_request_old = [], []

    for index, true_label in enumerate(y_test):
        # the credit requested by a given applicant
        score = loan_val[index]

        if sensitive_attr_test[index] == 0:  # young borrower
            loan_request_youth.append(score)
        elif sensitive_attr_test[index] == 1:  # old borrower
            loan_request_old.append(score)
        # check for TPs
        if true_label == y_pred[index] and true_label == 1:
            if sensitive_attr_test[index] == 0:  # young borrower
                    score_youth.append(score)
                    outcome_youth.append('TP')
            elif sensitive_attr_test[index] == 1:  # old borrower
                    score_old.append(score)
                    outcome_old.append('TP')
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            if sensitive_attr_test[index] == 0:  # young borrower
                score_youth.append(-score*1.5)
                outcome_youth.append('FP')
            elif sensitive_attr_test[index] == 1:  # old borrower
                score_old.append(-score*1.5)
                outcome_old.append('FP')
        # check for TN
        elif true_label == 0 and y_pred[index] == 0:  # the rest are TNs or FNs
            if sensitive_attr_test[index] == 0:  # youth indiv
                score_youth.append(0)
                outcome_youth.append('TN')
            elif sensitive_attr_test[index] == 1:  # old indiv
                score_old.append(0)
                outcome_old.append('TN')
        # check for FN
        elif true_label == 1 and y_pred[index] == 0:
            if sensitive_attr_test[index] == 0:  # youth indiv
                score_youth.append(0)
                outcome_youth.append('FN')
            elif sensitive_attr_test[index] == 1:  # old indiv
                score_old.append(0)
                outcome_old.append('FN')

    # calculate median values of each group after taking out the 0s
    #print('score_youth', score_youth)
    #print('# of youths: ', len(score_youth))
    score_youth_nozeros = [i for i in score_youth if i != 0]
    #print('# of youths WITHOUT 0: ', len(score_youth_nozeros))
    #print('median of score_youth updated', median(score_youth_nozeros))
    #print('score_youth updated: ', score_youth_nozeros)

    #print('# of olds: ', len(score_old))
    score_old_nozeros = [i for i in score_old if i != 0]
    #print('# of olds WITHOUT 0: ', len(score_old_nozeros))
    #print('median of score_old updated: ', median(score_old_nozeros))
    #print('score_old updated: ', score_old_nozeros)

    #print('score_old', score_old)
    i_youth = median(score_youth_nozeros)
    i_old = median(score_old_nozeros)


    results_youth = {'Model outcome': outcome_youth, 'Loan Requested': loan_request_youth, 'Impact Amt': score_youth}
    results_old = {'Model outcome': outcome_old, 'Loan Requested': loan_request_old, 'Impact Amt': score_old}
    results_youth_df = pd.DataFrame(results_youth)
    results_old_df = pd.DataFrame(results_old)
    # TODO: return a pandas(?? or array by group, model outcomes, loan requested, and impact

    return i_youth, i_old, results_youth_df, results_old_df

def evaluation_by_group_german(X_test, y_test, sensitive_attr_test, y_predict, credit_amount_4months, sample_weight):
    """
    Splits the data by sensitive group and computes evaluation for each group.
        Args:
            - X_test <numpy.ndarray>: samples (input) of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - sensitive_attr_test <numpy.ndarray>: group indicator for samples in the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
            - credit_amount_4months<pd.series>: credit amount received in 4 months
            - sample_weight <numpy.ndarray>:
        Returns:
            - results_0 <list>: eval results for group 0, youth (disadvantaged)
            - results_1 <list>: eval results for group 1, old (advantaged)
    """
    y_test_0, y_pred_0, sw_0, y_test_1, y_pred_1, sw_1 = [], [], [], [], [], []

    # splitting up the y_test and y_pred values by race to then use for age specific classification reports
    for index, age in enumerate(sensitive_attr_test):
        if (age == 0):  # 0, youth
            y_test_0.append(y_test[index])
            y_pred_0.append(y_predict[index])
            sw_0.append(sample_weight[index])
        elif (age == 1):  # 1, old
            y_test_1.append(y_test[index])
            y_pred_1.append(y_predict[index])
            sw_1.append(sample_weight[index])
        else:
            print('You should not end up here...')

    accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, tnr_0, tpr_0, fner_0, fper_0 = analysis(y_test_0, y_pred_0, sw_0)
    accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, tnr_1, tpr_1, fner_1, fper_1 = analysis(y_test_1, y_pred_1, sw_1)
    sr_bygroup = get_selection_rates(y_test, y_predict, sensitive_attr_test, 1)  #sr_bygroup is a pandas series
    sr_0 = round(sr_bygroup.values[0]*100, 2)
    sr_1 = round(sr_bygroup.values[1]*100, 2)
    i_0, i_1, finegrained_i_results_0, finegrained_i_results_1 = calculate_impact_german(y_test, y_predict,credit_amount_4months, sensitive_attr_test)
    results_youth = [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0, round(i_0, 2)]
    results_old = [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1, round(i_1, 2)]

    return results_youth, results_old


def evaluating_model_german(constraint_str, X_test, y_test, y_pred, credit_amount_4months, sample_weight_test, sensitive_attr_test):
    """
    Wrapper function which returns the eval results overall and by sensitive attribute group
        Args:
            - constraint_str <numpy.ndarray>: true labels of the test set
            - X_test <numpy.ndarray>: samples of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - credit_amount_4months<pd.series>: credit amount received in 4 months
            - sample_weight_test <>: all equal
            - sensitive_attr_test <numpy.ndarray>: age indicator for samples in the test set (1: ADULT, 0: YOUTH)
        Returns:
            - results_overall <list>: wrapper list with eval results overall
            - results_0 <list>: wrapper list with eval results disadvantaged group
            - results_1 <list>: wrapper list with eval results advantaged group
    """
    overall_message = 'Evaluation of ' + constraint_str + '-constrained classifier overall:'
    accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, tnr, tpr, fner, fper = analysis(y_test, y_pred,
                                                                                           sample_weight_test)
    sr = get_selection_rates(y_test, y_pred, sensitive_attr_test, 0)
    # print('\n')
    i_youth, i_old, finegrained_i_results_0, finegrained_i_results_1 = calculate_impact_german(y_test, y_pred, credit_amount_4months, sensitive_attr_test)
    # print('\nFairness metric evaluation of ', constraint_str, '-constrained classifier')
    dp_diff, eod_diff, eoo_dif, fpr_dif, er_dif = print_fairness_metrics(y_true=y_test, y_pred=y_pred,
                                                                         sensitive_features=sensitive_attr_test,
                                                                         sample_weight=sample_weight_test)
    results_overall = [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr * 100, 2), tnr, tpr, fner, fper,
                       i_youth, i_old, round(dp_diff * 100, 2), round(eod_diff * 100, 2), round(eoo_dif * 100, 2),
                       round(fpr_dif * 100, 2), round(er_dif * 100, 2)]
    # print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    results_0, results_1 = evaluation_by_group_german(X_test, y_test, sensitive_attr_test, y_pred, credit_amount_4months,
                                               sample_weight_test)
    # print('\n')
    # results 0 = youth, results 1 = old
    return results_overall, results_0, results_1, finegrained_i_results_0, finegrained_i_results_1


def calculate_delayed_impact(X_test, y_true, y_pred,di_means,di_stds, sensitive_attr_test):
    """
    Calculate the Delayed Impact (DI) (average score change of each group) (considering TP,FP)
        Args:
            - X_test <numpy.ndarray>: samples (scores) of the test set
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - di_means <tuple>:means of the delayed impact distributions
            - di_stds <tuple>: deviation of delyed impact distributions
            - sensitive_attr_test <numpy.ndarray>: indicator of the group/race (Black is 0 and White it 1)
        Returns:
            - di_black <float>: DI for group Black
            - di_white <float>: DI for group White
    """

    # split mean and std for score change distributions (reward fot TP, penalty for FP)
    reward_mu, penalty_mu = di_means
    reward_std, penalty_std = di_stds

    # bounds
    up_bound = 850
    low_bound = 300

    di_black, di_white = 0, 0
    score_diff_black, score_diff_white = [], []
    scores = X_test[:,0]

    for index, true_label in enumerate(y_true):
        # check for TPs
        if true_label == y_pred[index] and true_label==1:
            new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std,1))
            if sensitive_attr_test[index] == 0:  # black borrower
                if new_score >= up_bound:
                    score_diff_black.append(up_bound-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif sensitive_attr_test[index] == 1:  # white borrower
                if new_score > up_bound:
                    score_diff_white.append(up_bound-X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std,1))
            if sensitive_attr_test[index] == 0:  # black borrower
                if new_score < low_bound:
                    score_diff_black.append(low_bound-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif sensitive_attr_test[index] == 1:  # white borrower
                if new_score < low_bound:
                    score_diff_white.append(low_bound-X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        elif (true_label == y_pred[index] and true_label == 0) or (true_label == 1 and y_pred[index] == 0):
            if sensitive_attr_test[index] == 0:  # black indiv
                score_diff_black.append(0)
            elif sensitive_attr_test[index] == 1:  # white indiv
                score_diff_white.append(0)

    # calculate mean score difference or delayed impact of each group
    di_black = sum(score_diff_black)/len(score_diff_black)
    di_white = sum(score_diff_white)/len(score_diff_white)

    return di_black, di_white


def print_fairness_metrics(y_true, y_pred, sensitive_features, sample_weight):
    """
    Camputing and and printing of numerous fairness metrics.
        Args:
            - y_true <>: true labels
            - y_pred <>: predicted labels
            - sensitive_features <>:
            - samples_weight <>:
        Returns:
            - dp_diff <float>: dp disparity
            - eod_diff <float>: eo disparity
            - eoo_diff <float>: eoo disparity
            - fpr_dif <float>: fper disparity
            - er_dif <floate>: erp disparity
    """
    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('DP Difference: ', dp_diff)
    #print('-->difference of 0 means that all groups have the same selection rate')
    eod_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('EOD Difference: ', eod_diff)
    #print('-->difference of 0 means that all groups have the same TN, TN, FP, and FN rates')
    eoo_diff = tpr_diff(y_true, y_pred, sensitive_features, sample_weight)
    #print('EOO/TPR Difference: ', eoo_diff)
    fpr_dif = fpr_diff(y_true, y_pred, sensitive_features, sample_weight)
    #print('FPR Difference: ', fpr_dif)
    er_dif = er_diff(y_true, y_pred, sensitive_features)
    #print('ER Difference: ', er_dif)
    return dp_diff, eod_diff, eoo_diff, fpr_dif, er_dif


def get_selection_rates(y_true, y_pred, sensitive_features, type_index):
    """
    Comutes selection rate.
        Args:
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sensitive_features <>:
            - type_index <int>: indicator if overall selection rate (0) or by group (1)
        Returns:
            - sr_return <numpy.ndarray>: Selection rate
    """

    sr_mitigated = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred,
                               sensitive_features=sensitive_features)
    sr_return = -1
    if type_index == 0:
        sr_return = sr_mitigated.overall
        #print('Selection Rate Overall: ', sr_mitigated.overall)
    elif type_index == 1:
        sr_return = sr_mitigated.by_group
        #print('Selection Rate By Group: ', sr_mitigated.by_group, '\n')
    else:
        print('ISSUE: input 0 or 1 as 4th parameter')
    return sr_return


def evaluation_outcome_rates(y_true, y_pred, sample_weight):
    """
    Camputing and and printing of numerous outcome rates.
        Args:
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sample_weight <numpy.ndarray>:
        Returns:
            - tnr <float>: TN rate
            - tpr <float>: TP rate
            - fner <float>: FN rate
            - fper <float>: FP rate
    """

    tnr = true_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    #print('TNR=TN/(TN+FP)= ', tnr)
    tpr = true_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    #print('TPR=TP/(FP+FN)= ', tpr)
    fner = false_negative_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    #print('FNER=FN/(FN+TP)= ', fner)
    fper = false_positive_rate(y_true, y_pred, pos_label=1, sample_weight=sample_weight)
    #print('FPER=FP/(FP+TN)= ', fper)
    return tnr, tpr, fner, fper


def tpr_diff(y_true, y_pred, sensitive_features, sample_weight=None):
    tpr = MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features,
                      sample_params={'sample_weight': sample_weight})
    result = tpr.difference()
    return result


def fpr_diff(y_true, y_pred, sensitive_features, sample_weight=None):
    fpr = MetricFrame(metrics=false_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features,
                sample_params={'sample_weight': sample_weight})
    result = fpr.difference()
    return result


def er_diff(y_true, y_pred, sensitive_features):
    result = MetricFrame(metrics=accuracy_score, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features).difference(method='between_groups')
    return result


def get_f1_scores(y_test, y_predict):
    """
    Calculation of f1-scores.
    # Resource for below: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        Args:
            - y_test <numpy.ndarray>: true labels of the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
        Returns:
            f1_micro <float>, f1_weighted <float>, f1_binary <float>
    """
    # F1 score micro: calculate metrics globally by counting the total true positives, false negatives and false positives
    #print('F1 score micro: ')
    f1_micro = f1_score(y_test, y_predict, average='micro')
    #print(f1_micro)
    # F1 score weighted: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    #print('F1 score weighted: ')
    f1_weighted = f1_score(y_test, y_predict, average='weighted')
    #print(f1_weighted)
    # F1 score binary: Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred}) are binary.
    #print('F1 score binary: ')
    f1_binary = f1_score(y_test, y_predict, average='binary')
    #print(f1_binary)
    return f1_micro, f1_weighted, f1_binary


def analysis(y_test, y_pred, sample_weights):
    """
    Calculation of numerous model results: confusion matrix, accuracy, f1-scores, outcome rates and returning its rounded values.
        Args:
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sample_weights <numpy.ndarray>:
        Returns:
            Numerous rounded variables, cumputed below.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    results_dict = classification_report(y_test, y_pred, output_dict=True)
    #print(classification_report(y_test, y_pred))
    f1_micro, f1_weighted, f1_binary = get_f1_scores(y_test, y_pred)
    #f1_str = str(round(f1_micro * 100, 2)) + '/' + str(round(f1_weighted * 100, 2)) + '/' + str(round(f1_binary * 100, 2))
    tnr, tpr, fner, fper = evaluation_outcome_rates(y_test, y_pred, sample_weights)
    return round(results_dict['accuracy']*100, 2), str(conf_matrix), round(f1_micro * 100, 2), round(f1_weighted * 100, 2), round(f1_binary * 100, 2), round(tnr*100, 2), round(tpr*100, 2), round(fner*100, 2), round(fper*100, 2)

def evaluation_by_group(X_test, y_test, sensitive_attr_test, y_predict,di_means,di_stds, sample_weight):
    """
    Splits the data by sensitive group and computes evaluation for each group.
        Args:
            - X_test <numpy.ndarray>: samples (input) of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - sensitive_attr_test <numpy.ndarray>: group indicator for samples in the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
            - di_means <tuple>:means of the delayed impact distributions
            - di_stds <tuple>: deviation of delyed impact distributions
            - sample_weight <numpy.ndarray>:
        Returns:
            - results_0 <list>: eval results for group 0
            - results_1 <list>: eval results for group 1
    """
    y_test_0, y_pred_0, sw_0, y_test_1, y_pred_1, sw_1 = [], [], [], [], [], []

    # splitting up the y_test and y_pred values by race to then use for race specific classification reports
    for index, race in enumerate(sensitive_attr_test):
        if (race == 0):  # 0
            y_test_0.append(y_test[index])
            y_pred_0.append(y_predict[index])
            sw_0.append(sample_weight[index])
        elif (race == 1):  # 1
            y_test_1.append(y_test[index])
            y_pred_1.append(y_predict[index])
            sw_1.append(sample_weight[index])

        else:
            print('You should not end up here...')

    accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, tnr_0, tpr_0, fner_0, fper_0 = analysis(y_test_0, y_pred_0, sw_0)
    accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, tnr_1, tpr_1, fner_1, fper_1 = analysis(y_test_1, y_pred_1, sw_1)
    sr_bygroup = get_selection_rates(y_test, y_predict, sensitive_attr_test, 1)  #sr_bygroup is a pandas series
    sr_0 = round(sr_bygroup.values[0]*100, 2)
    sr_1 = round(sr_bygroup.values[1]*100, 2)
    di_0, di_1 = calculate_delayed_impact(X_test, y_test, y_predict,di_means,di_stds, sensitive_attr_test)
    results_0 = [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0, round(di_0, 2)]
    results_1 = [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1, round(di_1, 2)]

    return results_0, results_1


def evaluating_model(constraint_str,X_test,y_test, y_pred,di_means, di_stds, sample_weight_test,sensitive_attr_test):
    """
    Wrapper function which returns the eval results overall and by sensitive attribute group
        Args:
            - constraint_str <numpy.ndarray>: true labels of the test set
            - X_test <numpy.ndarray>: samples of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - di_means <tuple>:means of the delayed impact distributions
            - di_stds <tuple>: deviation of delyed impact distributions
            - sample_weight_test <>:
            - sensitive_attr_test <numpy.ndarray>: race indicator for samples in the test set
        Returns:
            - results_overall <list>: wrapper list with eval results overall
            - results_0 <list>: wrapper list with eval results disadvantaged group
            - results_1 <list>: wrapper list with eval results advantaged group
    """
    overall_message = 'Evaluation of '+ constraint_str + '-constrained classifier overall:'
    accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, tnr, tpr, fner, fper = analysis(y_test, y_pred, sample_weight_test)
    sr = get_selection_rates(y_test, y_pred, sensitive_attr_test, 0)
    #print('\n')
    di_B, di_W = calculate_delayed_impact(X_test, y_test, y_pred,di_means,di_stds, sensitive_attr_test)
    #print('\nFairness metric evaluation of ', constraint_str, '-constrained classifier')
    dp_diff, eod_diff, eoo_dif, fpr_dif, er_dif = print_fairness_metrics(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_attr_test, sample_weight=sample_weight_test)
    results_overall = [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, di_B,di_W, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]
    #print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    results_0, results_1 = evaluation_by_group(X_test, y_test, sensitive_attr_test, y_pred,di_means,di_stds, sample_weight_test)
    #print('\n')
    return results_overall, results_0, results_1


def evaluation_by_group_obermyer(X_test, y_test, sensitive_attr_test, y_predict, sample_weight):
    """
    Splits the data by sensitive group and computes evaluation for each group.
        Args:
            - X_test <numpy.ndarray>: samples (input) of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - sensitive_attr_test <numpy.ndarray>: group indicator for samples in the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
            - di_means <tuple>:means of the delayed impact distributions
            - di_stds <tuple>: deviation of delyed impact distributions
            - sample_weight <numpy.ndarray>:
        Returns:
            - results_0 <list>: eval results for group 0
            - results_1 <list>: eval results for group 1
    """
    y_test_0, y_pred_0, sw_0, y_test_1, y_pred_1, sw_1 = [], [], [], [], [], []

    # splitting up the y_test and y_pred values by race to then use for race specific classification reports
    for index, race in enumerate(sensitive_attr_test):
        if (race == 1):  # BLACK GROUP
            y_test_1.append(y_test[index])
            y_pred_1.append(y_predict[index])
            sw_1.append(sample_weight[index])
        elif (race == 0):  # WHITE GROUP
            y_test_0.append(y_test[index])
            y_pred_0.append(y_predict[index])
            sw_0.append(sample_weight[index])

        else:
            print('You should not end up here...')

    accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, tnr_0, tpr_0, fner_0, fper_0 = analysis(y_test_0, y_pred_0, sw_0)
    accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, tnr_1, tpr_1, fner_1, fper_1 = analysis(y_test_1, y_pred_1, sw_1)
    sr_bygroup = get_selection_rates(y_test, y_predict, sensitive_attr_test, 1)  #sr_bygroup is a pandas series
    sr_0 = round(sr_bygroup.values[0]*100, 2)
    sr_1 = round(sr_bygroup.values[1]*100, 2)
    results_white = [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0]
    results_black = [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1]

    return results_black, results_white



def evaluating_model_obermyer(constraint_str,X_test,y_test, y_pred, sample_weight_test,sensitive_attr_test):
    """
    Wrapper function which returns the eval results overall and by sensitive attribute group
        Args:
            - constraint_str <numpy.ndarray>: true labels of the test set
            - X_test <numpy.ndarray>: samples of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sample_weight_test <>:
            - sensitive_attr_test <numpy.ndarray>: race indicator for samples in the test set
        Returns:
            - results_overall <list>: wrapper list with eval results overall
            - results_0 <list>: wrapper list with eval results disadvantaged group
            - results_1 <list>: wrapper list with eval results advantaged group
        Note: in sensitive_attribute_test 0 is white and 1 is black, need to make sure I'm returning what I mean to
    """
    X_test = X_test.to_numpy()
    sensitive_attr_test = sensitive_attr_test.to_numpy()

    overall_message = 'Evaluation of '+ constraint_str + '-constrained classifier overall:'
    accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, tnr, tpr, fner, fper = analysis(y_test, y_pred, sample_weight_test)
    sr = get_selection_rates(y_test, y_pred, sensitive_attr_test, 0)
    #print('\nFairness metric evaluation of ', constraint_str, '-constrained classifier')
    dp_diff, eod_diff, eoo_dif, fpr_dif, er_dif = print_fairness_metrics(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_attr_test, sample_weight=sample_weight_test)
    results_overall = [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]
    #print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    results_black, results_white = evaluation_by_group_obermyer(X_test, y_test, sensitive_attr_test, y_pred,sample_weight_test)
    #print('\n')
    return results_overall, results_black, results_white


""" 
# NOTE: no longer used updated impact function which doesnt only consider TP and FP outcomes 
def calculate_delayed_impact_updated(X_test, y_true, y_pred,di_means,di_stds, sensitive_attr_test):

    #Calculate the Delayed Impact (DI) (average score change of each group) (considering TP,FP, FN)
    #    Args:
    #        - X_test <numpy.ndarray>: samples (scores) of the test set
    #        - y_true <numpy.ndarray>: true labels of the test set
    #        - y_pred <numpy.ndarray>: predicted labels for the test set
    #        - di_means <tuple>:means of the delayed impact distributions
    #        - di_stds <tuple>: deviation of delyed impact distributions
    #        - sensitive_attr_test <numpy.ndarray>: indicator of the group/race (Black is 0 and White it 1)
    #    Returns:
    #        - di_black <float>: DI for group Black
    #        - di_white <float>: DI for group White

    # split mean and std for score change distributions (reward fot TP, penalty for FP)
    reward_mu, penalty_mu = di_means
    reward_std, penalty_std = di_stds

    # bounds
    up_bound = 850
    low_bound = 300

    di_black, di_white = 0, 0
    score_diff_black, score_diff_white = [], []
    scores = X_test[:, 0]

    for index, true_label in enumerate(y_true):
        # check for TPs
        if true_label == y_pred[index] and true_label == 1:
            new_score = X_test[index][0] + int(np.random.normal(reward_mu, reward_std, 1))
            if sensitive_attr_test[index] == 0:  # black borrower
                if new_score >= up_bound:
                    score_diff_black.append(up_bound - X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif sensitive_attr_test[index] == 1:  # white borrower
                if new_score > up_bound:
                    score_diff_white.append(up_bound - X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std, 1))
            if sensitive_attr_test[index] == 0:  # black borrower
                if new_score < low_bound:
                    score_diff_black.append(low_bound - X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif sensitive_attr_test[index] == 1:  # white borrower
                if new_score < low_bound:
                    score_diff_white.append(low_bound - X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        # check for FNs
        elif true_label == 1 and y_pred[index] == 0:
            new_score = X_test[index][0] - int(np.random.normal(reward_mu, reward_std, 1))
            if sensitive_attr_test[index] == 0:  # black borrower
                if new_score < low_bound:
                    score_diff_black.append(low_bound - X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif sensitive_attr_test[index] == 1:  # white borrower
                if new_score < low_bound:
                    score_diff_white.append(low_bound - X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        else:  # the rest are TNs
            if sensitive_attr_test[index] == 0:  # black indiv
                score_diff_black.append(0)
            elif sensitive_attr_test[index] == 1:  # white indiv
                score_diff_white.append(0)

    # calculate mean score difference or delayed impact of each group
    di_black = sum(score_diff_black) / len(score_diff_black)
    di_white = sum(score_diff_white) / len(score_diff_white)

    return di_black, di_white
"""