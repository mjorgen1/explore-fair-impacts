import pandas as pd
import numpy as np
import sys
sys.path.append('../')
import os

from sklearn.model_selection import train_test_split
from scripts.classification_utils import load_args,prep_data,get_classifier, get_new_scores, add_constraint_and_evaluate,add_values_in_dict, save_dict_in_csv
from scripts.evaluation_utils import evaluating_model_obermyer


# NOTE: this script works when running via cmd line inside the "obermyer_experiments" folder when I have the following
# above:
# import sys
# sys.path.append('../')


"""
DATA PREPARATION
"""
data_original = pd.read_csv('/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_original.csv')   # same as data_df in obermyer's code
data_processed = pd.read_csv('/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_processed.csv') # same as all_X_y_df in obermyer's code
data_risk_scores = pd.read_csv("/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_risk_scores.csv")


# investigating this variable and potentially turn it into a new label
comorbid_sum = data_risk_scores['gagne_sum_tm1']
#print('min', comorbid_sum.min())                       min: 0
#print('max', comorbid_sum.max())                       max 18
#print('average', comorbid_sum.mean())                  average 1.4431370941292228
#print('median', comorbid_sum.median())                 median 1.0
#print('q1', comorbid_sum.quantile(0.25))               q1 0.0
#print('50th percentile', comorbid_sum.quantile(0.5))   50th percentile 1.0
#print('q3', comorbid_sum.quantile(0.75))               q3 2.0
#print('90th percentile', comorbid_sum.quantile(0.9))   90th percentile 4.0

# target label: predicting if someone has more than 2 comorbidities
over2_comorbids = []

for indices, row in data_risk_scores.iterrows():
    if data_risk_scores.at[indices, 'gagne_sum_tm1'] > 2.0:
        over2_comorbids.append(1)
    else:
        over2_comorbids.append(0)

count_black_pos_class = 0
count_black_neg_class = 0
count_white_pos_class = 0
count_white_neg_class = 0

#black samples: 5582
#Black positive class:  1775 (31% of the black class is in the positive class)
#Black negative class:  3807

#white samples: 43202
#White positive class:  7846 (18% of the white class is in the positive class)
#White negative class:  35356

#Positive class total: 9621 (18% is the black class and 82% is the white class)
#Negative class total: 39163
#Total samples: 48784

# demo_race_black value of 1 is BLACK and 0 for WHITE
# TODO: make sure the above flows throughout
for indices, row in data_processed.iterrows():
    if data_risk_scores.at[indices, 'gagne_sum_tm1'] > 2.0:
        if data_risk_scores.at[indices, 'dem_race_black'] ==1:
            count_black_pos_class += 1
        else:
            count_white_pos_class += 1
    else:
        if data_risk_scores.at[indices, 'dem_race_black'] ==1:
            count_black_neg_class += 1
        else:
            count_white_neg_class += 1

#print('Black positive class: ', count_black_pos_class)
#print('Black negative class: ', count_black_neg_class)
#print('White positive class: ', count_white_pos_class)
#print('White negative class: ', count_white_neg_class)

over2_comorbids_df = pd.Series(over2_comorbids, name='over2_comorbids_binary')
count_risk_high = over2_comorbids_df[over2_comorbids_df == 1].shape[0]
count_risk_low = over2_comorbids_df[over2_comorbids_df == 0].shape[0]
#print(count_risk_high)
#print(count_risk_low)

new_x_y_df = pd.concat([data_risk_scores, over2_comorbids_df], axis=1)
#print(new_x_y_df.head)
y = new_x_y_df['over2_comorbids_binary']
x = data_risk_scores.drop(columns=['risk_score_t', 'risk_score_binary', 'gagne_sum_tm1'])

"""
PARAMETER SETTING
"""
race_blind = True


results_path = 'results-comorbid/obermyer-unmit-race-blind-gbt/'  # directory to save the results
weight_idx = 1 # weight index for samples (1 in our runs)
test_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
test_set_variant = 0 # 0= default (testset like trainset), 1= balanced testset, 2= original,true FICO distribution
test_set_bound = 30000 # absolute upper bound for test_set size
#di_means = [100,-100] # means for delayed impact distributions (rewardTP,penaltyFP)
#di_stds = [15,15] # standard deviations for delayed impact distributions (rewardTP,penaltyFP)
save = True # indicator if the results should be saved

models = {'Decision Tree': 'dt', 'Gaussian Naive Bayes':'gnb','Logistic Regression': 'lgr', 'Gradient Boosted Trees': 'gbt'}
model_name = models['Gradient Boosted Trees']

os.makedirs(f'{results_path}{model_name}', exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
#print(type(X_train))  # pandas dataframe
X_train = X_train.reset_index().drop(['index'], axis=1)
# collect race data for training set
race_train = X_train['dem_race_black']
#print(race_train)
race_test = X_test['dem_race_black']

if race_blind:
    X_train = X_train.drop(['dem_race_black'], axis=1)
    X_test = X_test.drop(['dem_race_black'], axis=1)

X_test = X_test.reset_index().drop(['index'], axis=1)
y_train = y_train.reset_index().drop(['index'], axis=1)
y_test = y_test.reset_index().drop(['index'], axis=1)

y_train_flattened = np.ravel(y_train)
y_test_flattened = np.ravel(y_test)

# weight_index: 1 means all equal weights
if weight_idx == 1:
    # print('Sample weights are all equal.')
    sample_weight_train = np.ones(shape=(len(y_train),))
    sample_weight_test = np.ones(shape=(len(y_test),))
# weight_index: 0 means use sample weights
elif weight_idx == 0:
    print('Sample weights are NOT all equal.')

"""
MODEL TRAINING
"""

print('The classifier trained below is: ', model_name)
classifier = get_classifier(model_name)

# Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
np.random.seed(0)

# Train the classifier:
model = classifier.fit(X_train,y_train_flattened, sample_weight_train)

# Make predictions with the classifier:
y_predict = model.predict(X_test)

# Scores on test set
test_scores = model.predict_proba(X_test)[:, 1]

"""
SAVING RESULTS
"""

constraint_str = 'Un-'
overall_results_dict = {}
black_results_dict = {}
white_results_dict = {}
combined_results_dict = {}

results_overall, results_black, results_white = evaluating_model_obermyer(constraint_str,X_test,y_test_flattened, y_predict, sample_weight_test,race_test)
#  results_overall  =  [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]
#  results_black    =  [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1]
#  results_white    =  [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0]
#
#
# added in f1_weighted, results_overall[3] after accuracy
#overall_accuracy, f1_weighted, sr, tnr, tpr, fner, fper, --,--, tnr_b, tpr_b, fner_b, b_fper, w_tnr, w_tpr, w_fner, w_fper
combined_results = [results_overall[3], results_overall[0], results_overall[5], results_overall[6], results_overall[7], results_overall[8], results_overall[9], 'na', 'na', results_black[6], results_black[7], results_black[8], results_black[9], results_white[6], results_white[7], results_white[8], results_white[9]]

run_key = f'{model_name} Unmitigated'
overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)
combined_results_dict = add_values_in_dict(combined_results_dict, run_key, combined_results)

if save == True:
    overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
    byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER']
    combined_fieldnames = ['Run', 'F1_weighted','Acc', 'SelectionRate', 'TNR', 'TPR', 'FNER', 'FPER', 'Black Impact', 'White Impact', 'TNR_B', 'TPR_B', 'FNER_B', 'FPER_B', 'TNR_W', 'TPR_W', 'FNER_W', 'FPER_W']
    save_dict_in_csv(overall_results_dict, overall_fieldnames,  results_path+model_name+'_overall_results.csv')
    save_dict_in_csv(black_results_dict, byrace_fieldnames,  results_path+model_name+'_black_results.csv')
    save_dict_in_csv(white_results_dict, byrace_fieldnames,  results_path+model_name+'_white_results.csv')
    save_dict_in_csv(combined_results_dict, combined_fieldnames, results_path+model_name+'_combined_results.csv')