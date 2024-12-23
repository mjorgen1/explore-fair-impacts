import pandas as pd
import os
import numpy as np
import sys
import warnings
sys.path.append('../')

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


warnings.filterwarnings('ignore', category=FutureWarning)

"""
DATA PREPARATION
"""

german_data = pd.read_csv(filepath_or_buffer='german_data.csv')


# drop credit and then re-add it to the end of the dataframe
x = german_data.drop(['credit'], axis=1)

# Y labels needed to be 0s and 1s
# target label is credit, 1 (Good)-->1 or 2 (Bad)-->0
y = german_data['credit']
y = y.replace(to_replace=2, value=0)
#print('updated labels',y)
# NOTE: The below lines with replace weren't quite right actually
# y_changed_0s = y.replace(to_replace=1, value=0)
# y = y_changed_0s.replace(to_replace=2, value=1)

#os.makedirs(f'{results_path}{model_name}/cost-fp{fp_weight}-fn{fn_weight}/', exist_ok=True)

test_size = 0.3 # proportion of testset samples in the dataset (e.g. 0.3)
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


german_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(german_weights)
# [1.67464115 0.71283096] so I'm assuming that is for fp weight first and then fn weight next
