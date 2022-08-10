from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
import numpy as np



def inspect_MinMax(samples_A_probs,samples_B_probs):
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


def print_fairness_metrics(y_true, y_pred, sensitive_features, sample_weight):
    #sr_mitigated = MetricFrame(metric=selection_rate, y_true=y_true, y_pred=y_pred,
    #                           sensitive_features=sensitive_features)
    ##print('Selection Rate Overall: ', sr_mitigated.overall)
    ##print('Selection Rate By Group: ', sr_mitigated.by_group, '\n')

    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('DP Difference: ', dp_diff)
    #print('-->difference of 0 means that all groups have the same selection rate')
    dp_ratio = demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('DP Ratio:', dp_ratio)
    #print('-->ratio of 1 means that all groups have the same selection rate \n')

    eod_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('EOD Difference: ', eod_diff)
    #print('-->difference of 0 means that all groups have the same TN, TN, FP, and FN rates')
    eod_ratio = equalized_odds_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    #print('EOD Ratio:', eod_ratio)
    #print('-->ratio of 1 means that all groups have the same TN, TN, FP, and FN rates rates \n')

    eoo_diff = tpr_diff(y_true, y_pred, sensitive_features, sample_weight)
    #print('EOO/TPR Difference: ', eoo_diff)
    fpr_dif = fpr_diff(y_true, y_pred, sensitive_features, sample_weight)
    #print('FPR Difference: ', fpr_dif)
    er_dif = er_diff(y_true, y_pred, sensitive_features)
    #print('ER Difference: ', er_dif)
    #print('')

    return dp_diff, eod_diff, eoo_diff, fpr_dif, er_dif


def calculate_delayed_impact(X_test, y_true, y_pred, race_test):
    # TPs --> score increase by 75
    # FPs --> score drop of 150
    # TNs and FNs do not change (in this case)
    # Delayed Impact (DI) is the average score change of each group
    # In race_test array, Black is 0 and White it 1

    di_black, di_white = 0, 0
    score_diff_black, score_diff_white = [], []
    scores = X_test[:,0]

    for index, true_label in enumerate(y_true):
        # check for TPs
        if true_label == y_pred[index] and true_label==1:
            new_score = X_test[index][0] + 75
            if race_test[index] == 0:  # black borrower
                if new_score >= 850:
                    score_diff_black.append(850-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif race_test[index] == 1:  # white borrower
                if new_score > 850:
                    score_diff_white.append(850-X_test[index][0])
                else:
                    score_diff_white.append(new_score - X_test[index][0])
        # check for FPs
        elif true_label == 0 and y_pred[index] == 1:
            new_score = X_test[index][0] - 150
            if race_test[index] == 0:  # black borrower
                if new_score < 300:
                    score_diff_black.append(300-X_test[index][0])
                else:
                    score_diff_black.append(new_score - X_test[index][0])
            elif race_test[index] == 1:  # white borrower
                if new_score < 300:
                    score_diff_white.append(300-X_test[index][0])
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


def analysis(y_test, y_pred, sample_weights, print_statement):
    #print(#print_statement)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results_dict = classification_report(y_test, y_pred, output_dict=True)
    #print(classification_report(y_test, y_pred))
    f1_micro, f1_weighted, f1_binary = get_f1_scores(y_test, y_pred)
    #f1_str = str(round(f1_micro * 100, 2)) + '/' + str(round(f1_weighted * 100, 2)) + '/' + str(round(f1_binary * 100, 2))
    tnr, tpr, fner, fper = evaluation_outcome_rates(y_test, y_pred, sample_weights)
    return round(results_dict['accuracy']*100, 2), str(conf_matrix), round(f1_micro * 100, 2), round(f1_weighted * 100, 2), round(f1_binary * 100, 2), round(tnr*100, 2), round(tpr*100, 2), round(fner*100, 2), round(fper*100, 2)


def evaluation_by_race(X_test, y_test, race_test, y_predict, sample_weight):
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

    accuracy_black, cs_m_black, f1_m_black, f1_w_black, f1_b_black, tnr_black, tpr_black, fner_black, fper_black = analysis(y_test_black, y_pred_black, sw_black, 'EVALUATION FOR BLACK GROUP')
    accuracy_white, cs_m_white, f1_m_white, f1_w_white, f1_b_white, tnr_white, tpr_white, fner_white, fper_white = analysis(y_test_white, y_pred_white, sw_white, '\nEVALUATION FOR WHITE GROUP')
    sr_bygroup = get_selection_rates(y_test, y_predict, race_test, 1)  #sr_bygroup is a pandas series
    sr_black = round(sr_bygroup.values[0]*100, 2)
    sr_white = round(sr_bygroup.values[1]*100, 2)
    di_black, di_white = calculate_delayed_impact(X_test, y_test, y_predict, race_test)
    results_black = [accuracy_black, cs_m_black, f1_m_black, f1_w_black, f1_b_black, sr_black, tnr_black, tpr_black, fner_black, fper_black, round(di_black, 2)]
    results_white = [accuracy_white, cs_m_white, f1_m_white, f1_w_white, f1_b_white, sr_white, tnr_white, tpr_white, fner_white, fper_white, round(di_white, 2)]
    return results_black, results_white


def evaluating_model(constraint_str,X_test,y_test, y_pred, sample_weight_test,race_test):

    overall_message = 'Evaluation of '+ constraint_str + '-constrained classifier overall:'
    accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, tnr, tpr, fner, fper = analysis(y_test, y_pred, sample_weight_test, overall_message)
    sr = get_selection_rates(y_test, y_pred, race_test, 0)
    #print('\n')
    di_B, di_W = calculate_delayed_impact(X_test, y_test, y_pred, race_test)
    #di_str = str(round(di_black, 2)) + '/' + str(round(di_white, 2))
    #print('\nFairness metric evaluation of ', constraint_str, '-constrained classifier')
    dp_diff, eod_diff, eoo_dif, fpr_dif, er_dif = print_fairness_metrics(y_true=y_test, y_pred=y_pred, sensitive_features=race_test, sample_weight=sample_weight_test)

    results_overall = [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, di_B,di_W, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]

    #print('Evaluation of ', constraint_str, '-constrained classifier by race:')
    results_black, results_white = evaluation_by_race(X_test, y_test, race_test, y_pred, sample_weight_test)
    #print('\n')

    return results_overall, results_black, results_white
