from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from itertools import zip_longest
import argparse
import warnings
import csv
import pandas as pd
import numpy as np
import os
from scripts.evaluation_utils import evaluating_model
from scripts.classification_utils import load_args,train_test_split, balance_test_set_ratios, get_classifier, get_types, add_constraint_and_evaluate,add_values_in_dict, save_dict_in_csv
from scripts.visualization_utils import visual_label_dist_german




def classify_german(data_path,results_dir,weight_idx,testset_size,balance_bool, models,constraints,save):
    """
    Classification and evaluation function for the German credit set, able to train many models (different classifier or constraint) in one run.
    Args:
        data_path <str>: path to the dataset csv-file
        results_dir <str>: directory to save the results
        weight_idx <int>: weights for samples (1 in our runs)
        testset_size <float>: prportion of testset samples in the dataset (e.g. 0.3)
        balance_bool <bool>: False= default (testset like trainset), True= balanced testset
        models <dict>: classifers used for training
        constraints <dict>: fairness constraints used for training different models
        save <bool>: indicator if the results shoud be saved
    """

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

    # Load and Prepare data
    data = pd.read_csv(data_path)
    x = data.loc[:,["Duration","Checking account",  "Credit History", "Purpose", "Credit amount", "Savings account",
         "Present employment","Installment rate", "Other debtors", "Present residence", "Propety", "Other installment plans",
         "Hounsing", "No of credits", "Job", "Dependent People", "Telephone","Foreign worker", "Age",'Sex']].values
    y = data.loc[:,'Risk'].values

    print(' Whole set:', len(y))

    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = testset_size, random_state=42)

    # balancing testset
    if balance_bool:
        X_test, y_test = balance_test_set_ratios(X_test, y_test, 1000)

    # collect our sensitive,protected attribute
    gender_train = X_train[:, -1]
    gender_test = X_test[:, -1]
    print(' train:', len(gender_train))
    print(' test:', len(gender_test))
    # weight_index: 1 means all equal weights
    if weight_idx:
        #print('Sample weights are all equal.')
        sample_weight_train = np.ones(shape=(len(y_train),))
        sample_weight_test = np.ones(shape=(len(y_test),))
    # weight_index: 0 means use sample weights
    elif weight_idx:
        print('Sample weights are NOT all equal.')

    visual_label_dist_german(results_dir,'all',x,y)
    visual_label_dist_german(results_dir,'train',X_train, y_train)
    visual_label_dist_german(results_dir,'test',X_test,y_test)

    # split up X_test by race
    X_test_b = []
    X_test_w = []
    y_test_b = []
    y_test_w = []

    for index in range(len(X_test)):
        if gender_test[index] == 0:  # black
            X_test_b.append(X_test[index][0])
            y_test_b.append(y_test[index])
        elif gender_test[index] == 1:  # white
            X_test_w.append(X_test[index][0])
            y_test_w.append(y_test[index])

    # for each classifier
    for model_str in models.values():
        print(model_str)
        results_path = results_dir
        results_path += f'{model_str}/'
        os.makedirs(results_path, exist_ok=True)

        models_dict = {}
        overall_results_dict = {}
        black_results_dict = {}
        white_results_dict = {}
        all_types = []
        scores_names = []

        # get the type retio from the true samples
        T_test_b = ['TP' if e==1 else "TN" for e in y_test_b]
        T_test_w = ['TP' if e==1 else "TN" for e in y_test_w]

        all_types.extend([T_test_b,T_test_w])
        scores_names.extend(['testB', 'testW'])

        # train unconstrained model
        classifier = get_classifier(model_str)
        #np.random.seed(0)
        model = classifier.fit(X_train,y_train, sample_weight_train)
        y_predict = model.predict(X_test)

        # Scores on test set
        test_scores = model.predict_proba(X_test)[:, 1]
        models_dict = {"Unmitigated": (y_predict, test_scores)}

        #save scores and types (TP,FP,TN,FN) in list
        T_b, T_w = get_types(X_test, y_predict, y_test, gender_test)
        all_types.extend([T_b,T_w])
        scores_names.extend(['unmitB', 'unmitW'])
        # evaluate model
        constraint_str = 'Un-'
        results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, (0,0),(0,0), sample_weight_test,gender_test)
        # adding results to dict
        run_key = f'{model_str} Unmitigated'
        overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
        black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
        white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # train all constrained model for this classifier type
        for constraint_str in constraints.values():

            # adding contrained, classify and evaluate it
            print(constraint_str)
            mitigator, results_overall, results_black, results_white, y_pred_mitigated = add_constraint_and_evaluate(model, constraint_str, X_train, y_train, gender_train, gender_test, X_test, y_test, y_predict, sample_weight_test)

            #save types in list
            T_b, T_w = get_types(X_test, y_pred_mitigated, y_test, gender_test)
            all_types.extend([T_b,T_w])
            scores_names.extend([f'{constraint_str.lower()}B', f'{constraint_str.lower()}W'])

            run_key = f'{model_str} {constraint_str} Mitigated'
            overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
            black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
            white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # save evaluations
        if save == True:
            overall_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB','DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
            byrace_fieldnames = ['Run', 'Acc', 'ConfMatrix','F1micro', 'F1weighted','F1binary', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
            save_dict_in_csv(overall_results_dict, overall_fieldnames, results_path+model_str+'_overall_results.csv')
            save_dict_in_csv(black_results_dict, byrace_fieldnames, results_path+model_str+'_0_results.csv')
            save_dict_in_csv(white_results_dict, byrace_fieldnames, results_path+model_str+'_1_results.csv')

            # Save overall zype results
            columns_data_types = zip_longest(*all_types)

            with open(results_path+model_str+'_all_types.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_types)
                f.close()



if __name__ == '__main__':
    # load arguments
    parser = argparse.ArgumentParser(description='Specify the path to your config file.')
    parser.add_argument('-config', type=str, help="Path to where your config yaml file is stored.")
    args = parser.parse_args()

    try:
        args = load_args(f'configs/{args.config}.yaml')
    except:
        print(f'File does not exist: configs/{args.config}.yaml')
    # classification (incl. evaluation)
    classify_german(**args)
