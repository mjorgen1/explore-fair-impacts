import pandas as pd
import numpy as np
import csv
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scripts.classification_utils import prep_data, add_values_in_dict
from scripts.evaluation_utils import evaluating_model



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
    #mode = 'a' if os.path.exists(name_csv) else 'w+'
    #print(mode)
    #if mode == 'w+':
    #    os.makedirs(name_csv, exist_ok=True)
    # the dictionary needs to be formatted like: {'Run1': [acc, f1, ...], 'Run2': [acc, f1, ...]}
    with open(name_csv, mode='w+') as csv_file:
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


"""
PARAMETER SETTING
"""

data_path = 'data/synthetic_datasets/Demo-0-Lab-0.csv'# path to the dataset csv-file
results_path = 'fico-results/random_seed_0/lgr/cost-mit-fp6-fn5/'  # directory to save the results
fp_weight = 6
fn_weight = 5
balanced = False
random_bool = True
weight_idx = 1 # weight index for samples (1 in our runs)
testset_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
test_set_variant = 0 # 0= default (testset like trainset), 1= balanced testset, 2= original,true FICO distribution
test_set_bound = 30000 # absolute upper bound for test_set size
constraint_str = 'Cost-'
di_means = [75,-150] # means for delayed impact distributions (rewardTP,penaltyFP)
di_stds = [15,15] # standard deviations for delayed impact distributions (rewardTP,penaltyFP)
save = True # indicator if the results should be saved
models = {'Decision Tree': 'dt', 'Logistic Regression': 'lgr'}
model_name = models['Logistic Regression']


overall_results_dict = {}
black_results_dict = {}
white_results_dict = {}
combined_results_dict = {}
all_types = []
all_scores = []
scores_names = []


data = pd.read_csv(data_path)
data[['score', 'race']] = data[['score', 'race']].astype(int)
x = data[['score', 'race']].values
y = data['repay_indices'].values

X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test = prep_data(data, testset_size,test_set_variant,test_set_bound, weight_idx)

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

# NOTE: I DIDN'T INCLUDE THE SAVING OF SCORES AND TYPES TO A LIST

results_path += f'{model_name}/'
print(results_path)
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)


"""
MODEL TRAINING
"""
print('The classifier trained below is: ', model_name)

if model_name == 'lgr':
    if random_bool:
        if not balanced:
            classifier = LogisticRegression(class_weight={0:fp_weight, 1:fn_weight}, random_state = 0)
        else:
            classifier = LogisticRegression(class_weight='balanced', random_state = 0)
    else:
        if not balanced:
            classifier = LogisticRegression(class_weight={0:fp_weight, 1:fn_weight})
        else:
            classifier = LogisticRegression(class_weight='balanced')
elif model_name == 'dt':
    if random_bool:
        if not balanced:
            classifier = DecisionTreeClassifier(class_weight={0:fp_weight, 1:fn_weight}, random_state = 0)
        else:
            classifier = DecisionTreeClassifier(class_weight='balanced', random_state = 0)
    else:
        if not balanced:
            classifier = DecisionTreeClassifier(class_weight={0:fp_weight, 1:fn_weight})
        else:
            classifier = DecisionTreeClassifier(class_weight='balanced')
else:
    print('error: input an available model, lgr or dt')

# Resource: https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/CostSensitive.html
# {0:c10 (FP), 1:c01 (FN)}: The misclassification costs are explicitly set for the two classes by means of a dictionary.
# Conf matrix: [c00,     c01(FN)]
#              [c10(FP), c11]

# Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
np.random.seed(0)

# Train the classifier:
model = classifier.fit(X_train,y_train, sample_weight_train)

# Make predictions with the classifier:
y_predict = model.predict(X_test)

# Scores on test set
test_scores = model.predict_proba(X_test)[:, 1]

# results_overall = accuracy, cs_matrix, f1_micro, f1_weighted, f1_binary, round(sr * 100, 2), tnr, tpr, fner, fper,
#                        di_B, di_W, round(dp_diff * 100, 2), round(eod_diff * 100, 2), round(eoo_dif * 100, 2),
#                        round(fpr_dif * 100, 2), round(er_dif * 100, 2)]
# results_0 = [accuracy_0, cs_m_0, f1_m_0, f1_w_0, f1_b_0, sr_0, tnr_0, tpr_0, fner_0, fper_0, round(di_0, 2)]
# results_1 = [accuracy_1, cs_m_1, f1_m_1, f1_w_1, f1_b_1, sr_1, tnr_1, tpr_1, fner_1, fper_1, round(di_1, 2)]
results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, di_means,di_stds, sample_weight_test,race_test)
combined_results = [results_overall[3], results_overall[0], results_overall[5], results_overall[6],
                    results_overall[7], results_overall[8], results_overall[9], results_overall[10],
                    results_overall[11], results_black[6], results_black[7], results_black[8], results_black[9],
                    results_white[6], results_white[7], results_white[8], results_white[9]]


"""
SAVING RESULTS
"""

run_key = f'{model_name}cost-fp{fp_weight}-fn{fn_weight}'
overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)
combined_results_dict = add_values_in_dict(combined_results_dict, run_key, combined_results)

# To use below!!
if save == True:
    overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB','DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
    byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
    combined_fieldnames = ['Run', 'F1_weighted', 'Acc', 'SelectionRate', 'TNR', 'TPR', 'FNER', 'FPER',
                           'Black Impact', 'White Impact', 'TNR_B', 'TPR_B', 'FNER_B', 'FPER_B', 'TNR_W', 'TPR_W',
                           'FNER_W', 'FPER_W']
    save_dict_in_csv(overall_results_dict, overall_fieldnames,  results_path+model_name+'_overall_results.csv')
    save_dict_in_csv(black_results_dict, byrace_fieldnames,  results_path+model_name+'_black_results.csv')
    save_dict_in_csv(white_results_dict, byrace_fieldnames,  results_path+model_name+'_white_results.csv')
    save_dict_in_csv(combined_results_dict, combined_fieldnames,
                     results_path + model_name + '_combined_results.csv')