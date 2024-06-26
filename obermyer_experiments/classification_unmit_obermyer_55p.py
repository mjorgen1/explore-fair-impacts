import fairlearn
import csv
import pandas as pd
import numpy as np
from itertools import zip_longest
import sys
# so the jupyter nbs can find the scripts folder if I have the sys and append ../../ bit added on
# but when running from cmd line, it can do it fine
sys.path.append('../')
#print(sys.path)
import os
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from scripts.classification_utils import load_args,prep_data,get_classifier, get_new_scores, add_constraint_and_evaluate,add_values_in_dict, save_dict_in_csv
from scripts.evaluation_utils import evaluating_model_obermyer



"""
DATA PREPARATION
"""
data_original = pd.read_csv('/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_original.csv')   # same as data_df in obermyer's code
data_processed = pd.read_csv('/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_processed.csv') # same as all_X_y_df in obermyer's code
data_risk_scores = pd.read_csv("/home/kenz/git-workspace/explore-fair-impacts/data/obermyer_data/data_risk_scores.csv")

# code to get 55p instances and add to data_risk_scores
#  55th percentile 3.2614234754543907
#  count_above_55p: 21990
#  count_above_55p_b: 2641 (12% of the count above 55)    (47% of the total blacks)
#  count_above_55p_w: 19349 (88% of the count above 55)   (44% of the total whites)
#  count_above_55p_f: 13772  (62% of the count above 55)  (44% of the total females)
#  count_above_55p_m: 8218   (38% of the count above 55)  (45% of the total males)

risk_score_55p_binary = []

for indices, row in data_risk_scores.iterrows():
    if data_risk_scores.at[indices, 'risk_score_t'] >= 3.2614234754543907:
        risk_score_55p_binary.append(1)
    else:
        risk_score_55p_binary.append(0)

risk_score_55p_binary_df = pd.Series(risk_score_55p_binary, name='risk_score_55p_binary')
#print('there should be ', 21990, ' instances with 1s for risk_score_55p_binary')
count_risk_high = risk_score_55p_binary_df[risk_score_55p_binary_df == 1].shape[0]
#print(count_risk_high)
#print(risk_score_55p_binary_df)

new_x_y_df = pd.concat([data_risk_scores, risk_score_55p_binary_df], axis=1)

#print(new_x_y_df.head())
y = new_x_y_df['risk_score_55p_binary']
x = new_x_y_df.drop(columns=['risk_score_t','risk_score_binary','risk_score_55p_binary'])
#print(x.head())
#print(y.head())


"""
PARAMETER SETTING
"""
race_blind = True


results_path = 'results-55p/obermyer-unmit-race-blind-gbt/'  # directory to save the results
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