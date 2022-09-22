from itertools import zip_longest
import argparse
import warnings
import pandas as pd
import os
import csv
from scripts.evaluation_utils import evaluating_model
from scripts.visualization_utils import visual_label_dist, visual_scores_by_race
from scripts.classification_utils import load_args,prep_data,get_classifier, get_new_scores, add_constraint_and_evaluate,add_values_in_dict, save_dict_in_csv



def classify(data_path,results_dir,weight_idx,testset_size, test_set_variant, test_set_bound, di_means, di_stds, models,constraints,save):
    """
    Classification and evaluation function for the synthetic datasets (based on FICO-data), able to train many models (different classifier or constraint) in one run.
    Args:
        data_path <str>: path to the dataset csv-file
        results_dir <str>: directory to save the results
        weight_idx <int>: weight index for samples (1 in our runs)
        testset_size <float>: prportion of testset samples in the dataset (e.g. 0.3)
        test_set_variant <int>: 0= default (testset like trainset), 1= balanced testset, 2= original,true FICO distribution
        test_set_bound <int>:  upper bound for absolute test_set size
        di_means <list or tuple>: means for delayed impact distributions (rewardTP,penaltyFP)
        di_stds <list or tuple>:  standart deviations for delayed impact distributions (rewardTP,penaltyFP)
        models <dict>: classifers used for training
        constraints <dict>: fairness constraints used for training different models
        save <bool>: indicator if the results shoud be saved
    """

    warnings.filterwarnings('ignore', category=FutureWarning)
    # Load and extract data
    data = pd.read_csv(data_path)
    data[['score', 'race']] = data[['score', 'race']].astype(int)
    x = data[['score', 'race']].values
    y = data['repay_indices'].values

    # preprocess data
    X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test = prep_data(data, testset_size,test_set_variant,test_set_bound, weight_idx)

    # plotting set stats
    visual_scores_by_race(results_dir,'all',x)
    visual_scores_by_race(results_dir,'train',X_train)
    visual_scores_by_race(results_dir,'test',X_test)
    visual_label_dist(results_dir,'all',x,y)
    visual_label_dist(results_dir,'train',X_train, y_train)
    visual_label_dist(results_dir,'test',X_test,y_test)

    # split up X_test by race
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

    for model_str in models.values():
        print(model_str)
        results_path = results_dir
        results_path += f'{model_str}/'
        os.makedirs(results_path, exist_ok=True)

        models_dict = {}
        overall_results_dict = {}
        black_results_dict = {}
        white_results_dict = {}
        all_scores = []
        all_types = []
        scores_names = []

        T_test_b = ['TP' if e==1 else "TN" for e in y_test_b]
        T_test_w = ['TP' if e==1 else "TN" for e in y_test_w]

        all_types.extend([T_test_b,T_test_w])
        all_scores.extend([X_test_b,X_test_w])
        scores_names.extend(['testB', 'testW'])

        # Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
        # train unconstrained model
        classifier = get_classifier(model_str)
        #np.random.seed(0)
        model = classifier.fit(X_train,y_train, sample_weight_train)
        y_predict = model.predict(X_test)

        # Scores on test set
        test_scores = model.predict_proba(X_test)[:, 1]
        models_dict = {"Unmitigated": (y_predict, test_scores)}

        #save scores and types (TP,FP,TN,FN) in list
        X_b, X_w, T_b, T_w = get_new_scores(X_test, y_predict, y_test, di_means, di_stds, race_test)
        all_types.extend([T_b,T_w])
        all_scores.extend([X_b,X_w])
        scores_names.extend(['unmitB', 'unmitW'])

        # evaluate model
        constraint_str = 'Un-'
        results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, di_means,di_stds, sample_weight_test,race_test)

        # adding results to dict
        run_key = f'{model_str} Unmitigated'
        overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
        black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
        white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # train all constrained model for this model type
        for constraint_str in constraints.values():

            print(constraint_str)
            mitigator, results_overall, results_black, results_white, y_pred_mitigated = add_constraint_and_evaluate(model, constraint_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, False, di_means,di_stds,)

            #save scores in list
            X_b, X_w, T_b, T_w = get_new_scores(X_test, y_pred_mitigated, y_test, di_means, di_stds, race_test)
            all_types.extend([T_b,T_w])
            all_scores.extend([X_b,X_w])
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

            # Save overall score results
            columns_data_scores = zip_longest(*all_scores)
            columns_data_types = zip_longest(*all_types)

            with open(results_path+model_str+'_all_scores.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_scores)
                f.close()
            with open(results_path+model_str+'_all_types.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_types)
                f.close()







if __name__ == '__main__':
    #load arguments
    parser = argparse.ArgumentParser(description='Specify the path to your config file.')
    parser.add_argument('-config', type=str, help="Path to where your config yaml file is stored.")
    args = parser.parse_args()

    try:
        args = load_args(f'configs/{args.config}.yaml')
    except:
        print(f'File does not exist: configs/{args.config}.yaml')
    #run classification (incl. evaluation)
    classify(**args)
