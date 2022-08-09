from classification import *
import warnings

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)

    models = {'Decision Tree': 'dt', 'Gaussian Naive Bayes':'gnb','Logistic Regression': 'lgr', 'Gradient_Boosted_Trees': 'gbt'}
    constraints = {'Demografic Parity': 'DP', 'Equalized Odds': 'EO', 'Equality of Opportunity': 'TPRP', 'False Positive Rate Parity': 'FPRP', 'Error Rate Parity': 'ERP'}
    reduction_algorithms = {'Exponential Gradient':'EG','Grid Search':'GS'}

    data_path = 'data/final/simData_oom100.csv'  # ...oom10, ...oom50, ...oom100
    results_path = 'data/results/'
    save = True



    ###BGL Misiing!!!

    # Load and Prepare data
    data = get_data(data_path)

    X_train, X_test, y_train, y_test, race_train, race_test, sample_weight_train, sample_weight_test = prep_data(data=data, test_size=0.3, weight_index=1)


    # split up X_test by race
    X_test_b = []
    X_test_w = []

    for index in range(len(X_test)):
        if race_test[index] == 0:  # black
            X_test_b.append(X_test[index][0])
        elif race_test[index] == 1:  # white
            X_test_w.append(X_test[index][0])





    for model_str in models.values():
        print(model_str)
        results_path = 'data/results/'
        results_path += f'{model_str}/'

        models_dict = {}
        overall_results_dict = {}
        black_results_dict = {}
        white_results_dict = {}
        all_scores = []
        scores_names = []

        all_scores.extend([X_test_b,X_test_w])
        scores_names.extend(['testB', 'testW'])


        # Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
        # train unconstrained model
        classifier = get_classifier(model_str)
        np.random.seed(0)
        model = classifier.fit(X_train,y_train, sample_weight_train)
        y_predict = model.predict(X_test)

        # Scores on test set
        test_scores = model.predict_proba(X_test)[:, 1]
        models_dict = {"Unmitigated": (y_predict, test_scores)}

        # given predictions+outcomes, I'll need to do the same
        x = data[['score', 'race']].values
        y = data['repay_indices'].values
        scores = cross_val_score(model, x, y, cv=5, scoring='f1_weighted')


        #save scores in list
        X_b, X_w = get_new_scores(X_test, y_predict, y_test, race_test)
        all_scores.extend([X_b,X_w])
        scores_names.extend(['unmitB', 'unmitW'])

        # evaluate model
        constraint_str = 'Un-'
        results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, sample_weight_test,race_test)

        # adding results to dict
        run_key = f'{model_str} Unmitigated'
        overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
        black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
        white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        # train all constrained model for this model type
        for algo_str in reduction_algorithms.values():
            for constraint_str in constraints.values():

                print(algo_str,constraint_str)
                mitigator, results_overall, results_black, results_white, y_pred_mitigated = add_constraint(model, constraint_str, algo_str, X_train, y_train, race_train, race_test, X_test, y_test, y_predict, sample_weight_test, dashboard_bool=False)

                if algo_str ==' GS':
                    pass
                    # We can examine the values of lambda_i chosen for us:
                    #lambda_vecs = mitigator.lambda_vecs_
                    #print(lambda_vecs[0])
                    #models_dict = grid_search_show(mitigator, demographic_parity_difference, y_predict, X_test, y_test, race_test, 'DemParityDifference','GS DPD', models_dict, 0.3)
                    #models_dict.pop('GS DPD')

                #save scores in list
                X_b, X_w = get_new_scores(X_test, y_pred_mitigated, y_test, race_test)
                all_scores.extend([X_b,X_w])
                scores_names.extend([f'{algo_str.lower()}{constraint_str.lower()}B', f'{algo_str.lower()}{constraint_str.lower()}W'])


                run_key = f'{model_str} {algo_str} {constraint_str} Mitigated'
                overall_results_dict = add_values_in_dict(overall_results_dict, run_key, results_overall)
                black_results_dict = add_values_in_dict(black_results_dict, run_key, results_black)
                white_results_dict = add_values_in_dict(white_results_dict, run_key, results_white)

        print(overall_results_dict)

        # save evaluations:
        if save == True:
            overall_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DIB/DIW', 'DP Diff', 'EO Diff', 'TPR Diff', 'FPR Diff', 'ER Diff']
            byrace_fieldnames = ['Run', 'Acc', 'F1micro/F1w/F1bsr', 'SelectionRate', 'TNR rate', 'TPR rate', 'FNER', 'FPER', 'DI']
            save_dict_in_csv(overall_results_dict, overall_fieldnames, results_path+model_str+'_overall_results.csv')
            save_dict_in_csv(black_results_dict, byrace_fieldnames, results_path+model_str+'_black_results.csv')
            save_dict_in_csv(white_results_dict, byrace_fieldnames, results_path+model_str+'_white_results.csv')

            # Save overall score results
            columns_data_scores = zip_longest(*all_scores)

            with open(results_path+model_str+'_overall_scores.csv',mode='w') as f:
                writer = csv.writer(f)
                writer.writerow(scores_names)
                writer.writerows(columns_data_scores)
                f.close()
