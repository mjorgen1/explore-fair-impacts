import pandas as pd
import os
import numpy as np
import sys
sys.path.append('../')

from sklearn.model_selection import train_test_split
from scripts.evaluation_utils import evaluating_model_german
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scripts.classification_utils import load_args,prep_data,get_classifier, get_new_scores, add_constraint_and_evaluate,add_values_in_dict, save_dict_in_csv




"""
DATA PREPARATION
"""
german_data = pd.read_csv(filepath_or_buffer='german_data.csv')

print(german_data)
print(type(german_data))
print(german_data.shape)

print(german_data.columns)

#print(german_data['age'])

#print(german_data['credit'])

# drop credit and then re-add it to the end of the dataframe
x = german_data.drop(['credit'], axis=1)

# Y labels needed to be 0s and 1s
# target label is credit, 1 (Good)-->0 or 2 (Bad)-->1
y = german_data['credit']
#print(y)
y_changed_0s = y.replace(to_replace=1, value=0)
#print(y_changed_0s)
y = y_changed_0s.replace(to_replace=2, value=1)
#print(y)


"""
PARAMETER SETTING
"""

fp_weight = 5
fn_weight = 1
balanced = False

results_path = 'german_results/german_cost/' # directory to save the results
weight_idx = 1 # weight index for samples (1 in our runs)
test_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
save = True # indicator if the results should be saved
models = {'Decision Tree': 'dt', 'Gaussian Naive Bayes':'gnb','Logistic Regression': 'lgr', 'Gradient Boosted Trees': 'gbt'}
model_name = models['Logistic Regression']

os.makedirs(f'{results_path}{model_name}', exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
# NOTE: the labels are 1 or 2
#print(y_train)
#print(y_test)
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
train_month = X_train['month']
test_credit = X_test['credit_amount']
test_month = X_test['month']

# use x to check month data
months = x['month']
#print(months.max())     # 72
#print(months.min())     # 4
#print(months.mean())    # 20.903
#print(months.median()) # 18 median

# array to collect the values
test_4month_credit_arr = []

# calculate 4 months worth of credit for ppl
for index, credit_amt in enumerate(test_credit):
    print(credit_amt)
    test_4month_credit_arr.append((credit_amt / test_month.loc[index]) * 4)

test_4months_credit = pd.Series(data = test_4month_credit_arr)
# NOTE: the below pandas series contains the numbers that we're adding / subtracting depending on the outcome
print(test_4months_credit)


"""
MODEL TRAINING
"""

if not balanced:
    classifier = LogisticRegression(class_weight={0:fp_weight, 1:fn_weight})  # so I can add in weights
else:
    classifier = LogisticRegression(class_weight='balanced')
# Resource: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html
# {0:c10 (FP), 1:c01 (FN)}: The misclassification costs are explicitly set for the two classes by means of a dictionary.
# Conf matrix: [c00,     c01(FN)]
#              [c10(FP), c11]

print('The classifier trained below is: ', model_name)

# Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
np.random.seed(0)

# Train the classifier:
model = classifier.fit(X_train,y_train, sample_weight_train)

# Make predictions with the classifier:
y_predict = model.predict(X_test)

# Scores on test set
test_scores = model.predict_proba(X_test)[:, 1]

"""
SAVING RESULTS
"""

constraint_str = 'Cost-'
overall_results_dict = {}
young_results_dict = {}
old_results_dict = {}
# TODO: check if combined actually works??
#combined_results_dict = {}

results_overall, results_young, results_old = evaluating_model_german(constraint_str,X_test,y_test, y_predict, test_4months_credit, sample_weight_test,test_age)
#  results_overall  =  [accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr*100, 2), tnr, tpr, fner, fper, round(dp_diff*100, 2), round(eod_diff*100, 2), round(eoo_dif*100, 2), round(fpr_dif*100, 2), round(er_dif*100, 2)]
#  results_black    =  [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1]
#  results_white    =  [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0]
#
#
# added in f1_weighted, results_overall[3] after accuracy
#overall_accuracy, f1_weighted, sr, tnr, tpr, fner, fper, --,--, tnr_b, tpr_b, fner_b, b_fper, w_tnr, w_tpr, w_fner, w_fper
#combined_results = [results_overall[3], results_overall[0], results_overall[5], results_overall[6], results_overall[7], results_overall[8], results_overall[9], 'na', 'na', results_black[6], results_black[7], results_black[8], results_black[9], results_white[6], results_white[7], results_white[8], results_white[9]]

results_path_full = results_path+model_name+'/'
#print(results_path_full)

run_key = f'{model_name}cost-fp{fp_weight}-fn{fn_weight}'
overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
young_results_dict = add_values_in_dict(young_results_dict, run_key, results_young)
old_results_dict = add_values_in_dict(old_results_dict, run_key, results_old)
#combined_results_dict = add_values_in_dict(combined_results_dict, run_key, combined_results)

if save == True:
    overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'ImpactYouth','ImpactOld','DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
    byage_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'Impact']
    #combined_fieldnames = ['Run', 'F1_weighted','Acc', 'SelectionRate', 'TNR', 'TPR', 'FNER', 'FPER', 'Black Impact', 'White Impact', 'TNR_B', 'TPR_B', 'FNER_B', 'FPER_B', 'TNR_W', 'TPR_W', 'FNER_W', 'FPER_W']
    save_dict_in_csv(overall_results_dict, overall_fieldnames,  results_path_full+model_name+'_overall_results.csv')
    save_dict_in_csv(young_results_dict, byage_fieldnames,  results_path_full+model_name+'_young_results.csv')
    save_dict_in_csv(old_results_dict, byage_fieldnames,  results_path_full+model_name+'_old_results.csv')
    #save_dict_in_csv(combined_results_dict, combined_fieldnames, results_path+model_name+'_combined_results.csv')