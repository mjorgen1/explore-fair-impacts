import pandas as pd
import numpy as np
from scripts.classification_utils import prep_data, add_values_in_dict
from sklearn.utils.class_weight import compute_class_weight


"""
PARAMETER SETTING
"""

data_path = 'data/synthetic_datasets/Demo-0-Lab-0.csv'# path to the dataset csv-file
results_path = 'fico-results/mit_cost/'  # directory to save the results
weight_idx = 1 # weight index for samples (1 in our runs)
testset_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
test_set_variant = 0 # 0= default (testset like trainset), 1= balanced testset, 2= original,true FICO distribution
test_set_bound = 30000 # absolute upper bound for test_set size

data = pd.read_csv(data_path)
data[['score', 'race']] = data[['score', 'race']].astype(int)
x = data[['score', 'race']].values
y = data['repay_indices'].values

X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test = prep_data(data, testset_size,test_set_variant,test_set_bound, weight_idx)

# get the class weights
fico_class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(fico_class_weights)
# [1.7167803  0.70546026]