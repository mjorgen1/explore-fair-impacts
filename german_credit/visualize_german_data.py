import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


data_path = 'german_data.csv'
data = pd.read_csv(data_path)
fname = 'German Credit'

x = data.drop(['credit'], axis=1)
print(x)

# target label is credit, 1 (Good) remains or 2 (Bad)-->0
y = data['credit']
#print(y)
y = y.replace(to_replace=2, value=0)
#print(y)
y_numpy = y.to_numpy()
print(y_numpy)
print(type(y_numpy))

x_credit_amount = x['credit_amount']
#print(x_credit_amount)

# Adult (advantaged) is 1
# Youth (disadvantaged) is 0
x_age = x['age']
#print(x_age)
#print(type(x_age))

# combine 'credit_amount' and 'age' into a numpy array
combined_amount_by_age = pd.concat([x_credit_amount, x_age], axis=1)
#print(combined_amount_by_age)
#print(type(combined_amount_by_age))
combined_numpy = combined_amount_by_age.to_numpy()
print(combined_numpy)
print(type(combined_numpy))




# code to create repayment label distributions --> so change it to good / bad credit risk for young/old ppl

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

fig.suptitle(f'Histogram of Credit Class Labels from {fname} Data')
plt.xlabel('Credit Risk')

# divide samples into groups
youth_label_dist = y[np.where(combined_numpy[:, -1] == 0)[0]]
old_label_dist = y[np.where(combined_numpy[:, -1] == 1)[0]]

n_bins = 2
# We can set the number of bins with the *bins* keyword argument.
stepsize = 1
axs[0].xaxis.set_ticks(np.arange(0, 2, stepsize))
axs[0].set_xticklabels(['Bad', 'Good'])
axs[0].hist(youth_label_dist, bins=n_bins)
axs[0].set_title('Youth Group')
axs[0].set_ylabel('No. of Individuals')
axs[0].set_xlabel('Credit Risk')

axs[1].xaxis.set_ticks(np.arange(0, 2, stepsize))
axs[1].set_xticklabels(['Bad', 'Good'])
axs[1].hist(old_label_dist, bins=n_bins)
axs[1].set_title('Adult Group')
plt.savefig(f'{fname}_label_distr.png', dpi=300)



# make histogram of credit amts by age
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

fig.suptitle(f'Histogram of Credit Amounts Requested from {fname} Data')
plt.xlabel('Credit Amount')
# plt.ylabel('No. of Individuals')

youth_credit_dist = combined_numpy[np.where(combined_numpy[:, 1] == 0)[0]][:, 0]
adult_credit_dist = combined_numpy[np.where(combined_numpy[:, 1] == 1)[0]][:, 0]

n_bins = 50
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(youth_credit_dist, bins=n_bins)
axs[0].set_title('Youth Group')
axs[0].set_xlabel('Credit Amount')
axs[0].set_ylabel('No. of Individuals')
axs[1].set_title('Adult Group')
axs[1].hist(adult_credit_dist, bins=n_bins)
plt.savefig(f'{fname}_cred_amts_demo_distr.png', dpi=300)