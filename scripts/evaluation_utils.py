import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
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

def delayed_impact_csv(data_path= 'data/results/',b_or_w = 'Black', folders= ['dt','gnb','lgr','gbt']):

    col_names_df = []

    for i,f in enumerate(folders):
        if b_or_w == 'Black':
            path = f'{data_path}{f}/{f}_black_results.csv'
        else:
            path = f'{data_path}{f}/{f}_white_results.csv'

        df = pd.read_csv(path,index_col=0)
        df = df.reset_index()

        col_names_df.append(f'{f.upper()}')

        if i == 0:
            joined_df = df.iloc[:,-1]
        else:
            joined_df = pd.concat([joined_df, df.iloc[:,-1]], axis=1)

    joined_df.set_axis(folders, axis=1)

    # split dataframe after the two reduction algorithms
    df = joined_df.iloc[:6,:]

    # set new index
    df['Constraint'] = ['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP']
    df.set_index('Constraint',inplace=True)

    df.columns = col_names_df

    print('Group: ',b_or_w,'\n DataFrame: \n',df)
    df.to_csv(f'{data_path}/{b_or_w}_DI.csv')

def immediate_impact_csv(data_path= 'data/results/',b_or_w = 'Black', folders= ['dt','gnb','lgr','gbt']):

    col_names_df = []

    for i,f in enumerate(folders):
        if b_or_w == 'Black':
            path = f'{data_path}{f}/{f}_type_ratios.csv'
            df = pd.read_csv(path,index_col=0)
            df = df.filter(like='B')

        else:
            path = f'{data_path}{f}/{f}_type_ratios.csv'
            df = pd.read_csv(path,index_col=0)
            df = df.filter(like='W')
        col_names_df.append(f'{f.upper()}')
        df = df.iloc[3,[6,0,1,5,3,2]] - df.iloc[0,[6,0,1,5,3,2]]

        if i == 0:
            joined_df = df.iloc[:]
        else:
            joined_df = pd.concat([joined_df, df.iloc[:]], axis=1)

    joined_df.set_axis(folders, axis=1)

    # split dataframe after the two reduction algorithms
    df = joined_df.iloc[:6,:]

    # set new index
    df['Constraint'] = ['Unmitigated', 'DP', 'EO', 'EOO','FPER','ERP']
    df.set_index('Constraint',inplace=True)

    df.columns = col_names_df

    print('Group: ',b_or_w,'\n DataFrame: \n',df)

    df.to_csv(f'{data_path}/{b_or_w}_I.csv')




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


def calculate_delayed_impact(X_test, y_true, y_pred,di_means,di_stds, race_test):
    """
    Calculate the Delayed Impact (DI) (average score change of each group) (considering TP,FP)
        Args:
            - X_test <numpy.ndarray>: samples (scores) of the test set
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - race_test <numpy.ndarray>: indicator of the group/race (Black is 0 and White it 1)
        Returns:
            - di_black <float>: DI for group Black
            - di_white <float>: DI for group White
    """

    # mean and std for score change distributions
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
            if race_test[index] == 0:  # black borrower
                if new_score >= up_bound:
                    score_diff_black.append(up_bound-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif race_test[index] == 1:  # white borrower
                if new_score > up_bound:
                    score_diff_white.append(up_bound-X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            new_score = X_test[index][0] + int(np.random.normal(penalty_mu, penalty_std,1))
            if race_test[index] == 0:  # black borrower
                if new_score < low_bound:
                    score_diff_black.append(low_bound-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif race_test[index] == 1:  # white borrower
                if new_score < low_bound:
                    score_diff_white.append(low_bound-X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        elif (true_label == y_pred[index] and true_label == 0) or (true_label == 1 and y_pred[index] == 0):
            if race_test[index] == 0:  # black indiv
                score_diff_black.append(0)
            elif race_test[index] == 1:  # white indiv
                score_diff_white.append(0)

    # calculate mean score difference or delayed impact of each group
    di_black = sum(score_diff_black)/len(score_diff_black)
    di_white = sum(score_diff_white)/len(score_diff_white)

    #print('The average delayed impact of the black group is: ', di_black)
    #print('The average delayed impact of the white group is: ', di_white)
    return di_black, di_white


def get_selection_rates(y_true, y_pred, sensitive_features, type_index):
    """
    ...
        Args:
            - y_true <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sensitive_features <>:
            - type_index <int>: indicator if overall selection rate (0) or by group (1)
        Returns:
            - sr_return <numpy.ndarray>:
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
            - sample_weight <>:
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


# Resource for below: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
def get_f1_scores(y_test, y_predict):
    """
    Calculation of f1-scores.
    # Resource for below: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        Args:
            - y_test <numpy.ndarray>: true labels of the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
        Returns:
            - f1_micro <float>:
            - f1_weighted <float>:
            - f1_binary <float>:
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
    #print('')
    return f1_micro, f1_weighted, f1_binary


def analysis(y_test, y_pred, sample_weights):
    """
    Calculation of numerous model results: confusion matrix, accuracy, f1-scores, outcome rates and returning its rounded values.
        Args:
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sample_weights <>:
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


def evaluation_by_race(X_test, y_test, race_test, y_predict,di_means,di_stds, sample_weight):
    """
    Splits the data into race and computes evaluation for each race.
        Args:
            - X_test <numpy.ndarray>: samples(scores) of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - race test <numpy.ndarray>: race indicator for samples in the test set
            - y_predict <numpy.ndarray>: predicted labels for the test set
            - sample_weight <c>:
        Returns:
            - results_black <>: eval results for group/race black
            - results_white <>: eval results for group/race white
    """
    y_test_black, y_pred_black, sw_black, y_test_white, y_pred_white, sw_white = [], [], [], [], [], []

    # splitting up the y_test and y_pred values by race to then use for race specific classification reports
    for index, race in enumerate(race_test):
        if (race == 0):  # black
            y_test_black.append(y_test[index])
            y_pred_black.append(y_predict[index])
            sw_black.append(sample_weight[index])
        elif (race == 1):  # white
            y_test_white.append(y_test[index])
            y_pred_white.append(y_predict[index])
            sw_white.append(sample_weight[index])

        else:
            print('You should not end up here...')

    accuracy_black, cs_m_black, f1_m_black, f1_w_black, f1_b_black, tnr_black, tpr_black, fner_black, fper_black = analysis(y_test_black, y_pred_black, sw_black)
    accuracy_white, cs_m_white, f1_m_white, f1_w_white, f1_b_white, tnr_white, tpr_white, fner_white, fper_white = analysis(y_test_white, y_pred_white, sw_white)
    sr_bygroup = get_selection_rates(y_test, y_predict, race_test, 1)  #sr_bygroup is a pandas series
    sr_black = round(sr_bygroup.values[0]*100, 2)
    sr_white = round(sr_bygroup.values[1]*100, 2)
    di_black, di_white = calculate_delayed_impact(X_test, y_test, y_predict,di_means,di_stds, race_test)
    results_black = [accuracy_black, cs_m_black, f1_m_black, f1_w_black, f1_b_black, sr_black, tnr_black, tpr_black, fner_black, fper_black, round(di_black, 2)]
    results_white = [accuracy_white, cs_m_white, f1_m_white, f1_w_white, f1_b_white, sr_white, tnr_white, tpr_white, fner_white, fper_white, round(di_white, 2)]

    return results_black, results_white


def evaluating_model(constraint_str,X_test,y_test, y_pred,di_means,di_stds, sample_weight_test,race_test):
    """
    Wrapper function which returns the eval results overall and by race
        Args:
            - constraint_str <numpy.ndarray>: true labels of the test set
            - X_test <numpy.ndarray>: samples(scores) of the test set
            - y_test <numpy.ndarray>: true labels of the test set
            - y_pred <numpy.ndarray>: predicted labels for the test set
            - sample_weight_test <>:
            - race test <numpy.ndarray>: race indicator for samples in the test set
        Returns:
            - results_overall <list>: wrapper list with eval results overall
            - results_black <list>: wrapper list with eval results black
            - results_white <list>: wrapper list with eval results white
    """
    overall_message = 'Evaluation of '+ constraint_str + '-constrained classifier overall:'
    accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, tnr, tpr, fner, fper = analysis(y_test, y_pred, sample_weight_test)
    sr = get_selection_rates(y_test, y_pred, race_test, 0)
    #print('\n')
    di_B, di_W = calculate_delayed_impact(X_test, y_test, y_pred,di_means,di_stds, race_test)
    #di_str = str(round(di_black, 2)) + '/' + str(round(di_white, 2))
    #print('\nFairness metric evaluation of ', constraint_str, '-constrained classifier')
    dp_diff, eod_diff, eoo_dif, fpr_dif, er_dif = print_fairness_metrics(y_true=y_test, y_pred=y_pred, sensitive_features=race_test, sample_weight=sample_weight_test)

    results_overall = [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, di_B,di_W, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]

    #print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    results_black, results_white = evaluation_by_race(X_test, y_test, race_test, y_pred,di_means,di_stds, sample_weight_test)
    #print('\n')

    return results_overall, results_black, results_white
