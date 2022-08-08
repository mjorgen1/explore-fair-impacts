from classification import *
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)

    models = {'Decision Tree': 'dt', 'Gaussian Naive Bayes':'gnb','Logistic Regression': 'lgr', 'Gradient_Boosted_Trees': 'gbt'}
    constraints = {'unmitigated': 'Un', 'DemograficParity': 'DP'}
    reduction_algorithms = {'Exponential Gradient':'EG','Grid Search':'GS'}

    data_path = 'data/final/simData_oom100.csv'  # ...oom10, ...oom50, ...oom100
    results_path = 'data/results/'
    save = True

    models_dict = {}
    overall_results_dict = {}
    black_results_dict = {}
    white_results_dict = {}
    all_scores = []
    black_scores = []
    white_scores = []
    eg_scores = []
    gs_scores = []
    scores_redu_fieldnames = ['testB', 'testW', 'unmitB', 'unmitW', 'egdpB', 'egdpW', 'egeoB', 'egeoW', 'egeooB', 'egeooW', 'egfprpB', 'egfprpW', 'egerpB', 'egerpW']


    for model in models.values():
        for algo in reduction_algorithms.values():
            for constraint_str in constraints.values():
                print(model,algo,constraint_str)
                results_overall,results_black, results_white, X_b, X_w, X_test_b, X_test_w = classify(data_path,model,constraint_str,algo)


                #save scores in list
                all_scores.append(X_b)
                all_scores.append(X_w)
                black_scores.append(X_b)
                white_scores.append(X_w)
                if algo == 'EG':
                    eg_scores.append(X_b)
                    eg_scores.append(X_w)
                elif algo == 'GS':
                    gs_scores.append(X_b)
                    gs_scores.append(X_w)


                #save evals in dict:
                if constraint_str== 'Un':
                    run_key = f'{model} Unmitigated'
                else:
                    run_key = f'{model} {algo} {constraint_str} Mitigated'
                overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
                black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
                white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)
                print(overall_results_dict)

        # save evaluations:
        if save == True:
            overall_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB/DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
            byrace_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
            save_dict_in_csv(overall_results_dict, overall_fieldnames, results_path+model+'_overall_results.csv')
            save_dict_in_csv(black_results_dict, byrace_fieldnames, results_path+model+'_black_results.csv')
            save_dict_in_csv(white_results_dict, byrace_fieldnames, results_path+model+'_white_results.csv')
