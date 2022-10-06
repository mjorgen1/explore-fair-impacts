import pandas as pd
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':

    # Parameters
    data_path = 'data/raw/german.data'
    results_dir = 'data/testing/'
    file_name = "german_credit_test.csv"

    # col names
    colnames = ["Checking account", "Duration", "Credit History", "Purpose", "Credit amount", "Savings account",
             "Present employment","Installment rate","Sex", "Other debtors", "Present residence", "Propety",
             "Age", "Other installment plans", "Hounsing", "No of credits", "Job", "Dependent People", "Telephone","Foreign worker", "Risk"]
    # load german data
    df = pd.read_csv(data_path, sep=' ', names=colnames, header=None)

    # transform sensitive attribute and label binary (0,1)
    df['Sex'] = df['Sex'].map({'A91':1,'A92':0,'A93':1,'A94':1,'A95':0})
    df['Risk'] = (df['Risk'] == 1).astype(int)

    # transform all categorical cols to numerical ones
    categorical_cols = ["Checking account", "Credit History", "Purpose", "Savings account",
             "Present employment", "Other debtors", "Propety", "Other installment plans", "Hounsing", "Job", "Telephone","Foreign worker",]
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

    #save dataset
    df.to_csv(index=False, path_or_buf=results_dir+file_name)