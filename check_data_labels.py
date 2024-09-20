import pandas as pd
import numpy as np
import csv


fico_data_path = 'data/synthetic_datasets/Demo-0-Lab-0.csv'
german_data_path = 'german_credit/german_data.csv'



# Load and extract data
data_fico = pd.read_csv(fico_data_path)
data_fico[['score', 'race']] = data_fico[['score', 'race']].astype(int)
x_fico = data_fico[['score', 'race']].values
y_fico = data_fico['repay_indices'].values

data_german = pd.read_csv(german_data_path)
# drop credit and then re-add it to the end of the dataframe
x_german = data_german.drop(['credit'], axis=1)
# Y labels needed to be 0s and 1s
# target label is credit, 1 (Good)-->1 or 2 (Bad)-->0
y_german = data_german['credit']

# print(y)
y_changed_0s = y_german.replace(to_replace=1, value=0) # og bit in my code
y_german = y_german.replace(to_replace=2, value=0)
# print(y_changed_0s)
#y_german = y_changed_0s_new.replace(to_replace=2, value=1) # og bit in my code

# TODO: check what values the races / ages are, 0 is disadvantaged and 1 is advantaged

# FICO data
print('FICO dataset, number of individuals:', len(y_fico))
unique_fico, unique_counts_fico = np.unique(y_fico, return_counts=True)
print(np.asarray((unique_fico, unique_counts_fico)).T)


count_b_default_fico = 0
count_b_repay_fico = 0
count_w_default_fico = 0
count_w_repay_fico = 0
index = 0
for row in x_fico:
    print(row)
    if row[1] == 0 and y_fico[index] == 0:  # black defaulter
        count_b_default_fico += 1
    elif row[1]== 0 and y_fico[index]==1:  # black repayer
        count_b_repay_fico += 1
    elif row[1]==1 and y_fico[index] == 0:  # white defaulter
        count_w_default_fico += 1
    elif row[1] ==1 and y_fico[index]== 1:  # white repayer
        count_w_repay_fico += 1
    index += 1
print('FICO Scores dataset. Black group has: ' , count_b_default_fico, ' defaulters, and ', count_b_repay_fico , 'repayers.' )
print('FICO Scores dataset. White group has: ' , count_w_default_fico, ' defaulters, and ', count_w_repay_fico , 'repayers.' )

'''
FICO Scores dataset. Black group has:  7920  defaulters, and  4080 repayers.
FICO Scores dataset. White group has:  21120  defaulters, and  66880 repayers.
'''

# the below will work for pandas
#print('the number of positive classification occurrances is: ', y_fico.value_counts()[1])
#print('the number of negative classification occurrances is: ', y_fico.value_counts()[0])


# German data
print('German dataset, number of individuals:', len(y_german))
unique_german, unique_counts_german = np.unique(y_german, return_counts=True)
print(np.asarray((unique_german, unique_counts_german)).T)
unique_german_str, unique_counts_german_str = np.unique(data_german['credit'], return_counts=True)
print(np.asarray((unique_german_str, unique_counts_german_str)).T)


count_y_default_fico = 0
count_y_repay_fico = 0
count_o_default_fico = 0
count_o_repay_fico = 0
for index, row in x_german.iterrows():
    #print(row['age'])
    if row['age'] == 0 and y_german[index] == 0:  # youth bad
        count_y_default_fico += 1
    elif row['age']== 0 and y_german[index]==1:  # youth good
        count_y_repay_fico += 1
    elif row['age']==1 and y_german[index] == 0:  # old bad
        count_o_default_fico += 1
    elif row['age'] ==1 and y_german[index]== 1:  # old good
        count_o_repay_fico += 1
    index += 1
print('German dataset. Youth group has: ' , count_y_default_fico, ' defaulters, and ', count_y_repay_fico , 'repayers.' )
print('German dataset. Youth group has: ', x_german['age'].value_counts()[0], '# of applicants')

print('German dataset. Old group has: ' , count_o_default_fico, ' defaulters, and ', count_o_repay_fico , 'repayers.' )
print('German dataset. Old group has: ', x_german['age'].value_counts()[1], '# of applicants')
'''
German dataset. Youth group has:  61  defaulters, and  88 repayers.
German dataset. Youth group has:  149 # of applicants
German dataset. Old group has:  239  defaulters, and  612 repayers.
German dataset. Old group has:  851 # of applicants
'''

