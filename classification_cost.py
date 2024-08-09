import pandas as pd
import numpy as np
import csv
import os
from itertools import zip_longest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scripts.classification_utils import load_args, prep_data, get_classifier, get_new_scores_updated, \
    add_constraint_and_evaluate, add_values_in_dict
from scripts.evaluation_utils import evaluating_model_updated



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

data_path = 'data/synthetic_datasets/Demo-0-Lab-0.csv'# path to the dataset csv-file
results_path = 'results-updated-impact-func/cost-mit-fp-10-fn1/' # directory to save the results

fp_weight = 10
fn_weight = 1
balanced = False

weight_idx = 1 # weight index for samples (1 in our runs)
testset_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
test_set_variant = 0 # 0= default (testset like trainset), 1= balanced testset, 2= original,true FICO distribution
test_set_bound = 30000 # absolute upper bound for test_set size

di_means = [100,-100] # means for delayed impact distributions (rewardTP,penaltyFP)
di_stds = [15,15] # standard deviations for delayed impact distributions (rewardTP,penaltyFP)

save = True # indicator if the results should be saved

models = {'Decision Tree': 'dt', 'Gaussian Naive Bayes':'gnb','Logistic Regression': 'lgr', 'Gradient Boosted Trees': 'gbt'}
model_name = models['Logistic Regression']

overall_results_dict = {}
black_results_dict = {}
white_results_dict = {}
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

print('The classifier trained below is: ', model_name)

results_path += f'{model_name}/'
print(results_path)
if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

if not balanced:
    classifier = LogisticRegression(class_weight={0:fp_weight, 1:fn_weight})  # so I can add in weights
else:
    classifier = LogisticRegression(class_weight='balanced')
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

X_unmit_b, X_unmit_w,T_unmit_b, T_unmit_w = get_new_scores_updated(X_test, y_predict, y_test, di_means, di_stds, race_test)

constraint_str = 'Cost-'
results_overall, results_black, results_white = evaluating_model_updated(constraint_str,X_test,y_test, y_predict, di_means,di_stds, sample_weight_test,race_test)

run_key = f'{model_name}cost-fp{fp_weight}-fn{fn_weight}'
overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

# To use below!!
if save == True:
    overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB','DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
    byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
    save_dict_in_csv(overall_results_dict, overall_fieldnames,  results_path+model_name+'_overall_results.csv')
    save_dict_in_csv(black_results_dict, byrace_fieldnames,  results_path+model_name+'_black_results.csv')
    save_dict_in_csv(white_results_dict, byrace_fieldnames,  results_path+model_name+'_white_results.csv')

"""
if save == True:
    # Save overall score results
    columns_data_scores = zip_longest(*all_scores)
    columns_data_types = zip_longest(*all_types)

    with open(results_path+model_name+'_all_scores.csv',mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(scores_names)
            writer.writerows(columns_data_scores)
            f.close()
    with open(results_path+model_name+'_all_types.csv',mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(scores_names)
        writer.writerows(columns_data_types)
        f.close()
"""