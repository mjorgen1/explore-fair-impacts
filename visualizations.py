import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import matplotlib.patches as mpatches

def visualize_data_distribution(samples_A,samples_A_probs,samples_B,samples_B_probs):

    samples_all_A = (samples_A, samples_A_probs)
    samples_all_B = (samples_B, samples_B_probs)

    data = (samples_all_A, samples_all_B)
    colors = ('blue', 'orange')
    groups = ('Group A (Black)', 'Group B (White)')

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for data, color, group in zip(data, colors, groups):
        x, y = data
        #ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        ax.plot(x, y, alpha=0.8, c=color, label=group) #plot instead of scatter

    plt.title('Sample Distributions by Group')
    plt.legend(loc=2)
    plt.xlabel('Credit Score')
    plt.ylabel('Repay Probability')
    plt.show()


def visual_scores_by_race(data):
    # make histogram of credit scores by race
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.suptitle('Histogram Credit Score Distribution')
    plt.xlabel('Credit Score')
    #plt.ylabel('No. of Individuals')

    black_credit_dist = data['score'].loc[data['race']==0]
    white_credit_dist = data['score'].loc[data['race']==1]

    n_bins = 50
    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(black_credit_dist, bins=n_bins)
    axs[0].set_title('Black Group')
    axs[0].set_xlabel('Credit Score')
    axs[0].set_ylabel('No. of Individuals')
    axs[1].set_title('White Group')
    axs[1].hist(white_credit_dist, bins=n_bins)



def visual_repay_dist(data):
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.suptitle('Histogram of Repay Distribution')
    plt.xlabel('Repay Label')

    black_label_dist = data['repay_indices'].loc[data['race']==0]
    white_label_dist = data['repay_indices'].loc[data['race']==1]

    #print(black_label_dist)
    #print(white_label_dist)

    # maybe see if the label is an int or a float

    # default: 0, repay: 1

    n_bins = 2
    # We can set the number of bins with the *bins* keyword argument.
    #start, end = ax.get_xlim()
    stepsize=1
    axs[0].xaxis.set_ticks(np.arange(0, 2, stepsize))
    axs[0].set_xticklabels(['Default','Repay'])
    axs[0].hist(black_label_dist, bins=n_bins)
    axs[0].set_title('Black Group')
    axs[0].set_ylabel('No. of Individuals')
    axs[0].set_xlabel('Repay Label')

    axs[1].xaxis.set_ticks(np.arange(0, 2, stepsize))
    axs[1].set_xticklabels(['Default','Repay'])
    axs[1].hist(white_label_dist, bins=n_bins)
    axs[1].set_title('White Group')


def update_model_perf_dict(sweep, models_dict, sweep_preds, sweep_scores, non_dominated, decimal, y_test, race_test, model_name):
    # Compare GridSearch models with low values of fairness-diff with the previously constructed models
    ##print(model_name)
    grid_search_dict = {model_name.format(i): (sweep_preds[i], sweep_scores[i]) #{'GS_DP'.format(i): (sweep_preds[i], sweep_scores[i])
                        for i in range(len(sweep_preds))
                        if non_dominated[i] and sweep[i] < decimal}
    models_dict.update(grid_search_dict)
    #print(get_metrics_df(models_dict, y_test, race_test))
    return models_dict

def grid_search_show(model, constraint, y_predict, X_test, y_test, race_test, constraint_name, model_name, models_dict, decimal):
    sweep_preds = [predictor.predict(X_test) for predictor in model.predictors_]
    sweep_scores = [predictor.predict_proba(X_test)[:, 1] for predictor in model.predictors_]

    sweep = [constraint(y_test, preds, sensitive_features=race_test)
             for preds in sweep_preds]
    accuracy_sweep = [accuracy_score(y_test, preds) for preds in sweep_preds]
    # auc_sweep = [roc_auc_score(y_test, scores) for scores in sweep_scores]

    # Select only non-dominated models (with respect to accuracy and equalized odds difference)
    all_results = pd.DataFrame(
        {'predictor': model.predictors_, 'accuracy': accuracy_sweep, 'disparity': sweep}
    )
    non_dominated = []
    for row in all_results.itertuples():
        accuracy_for_lower_or_eq_disparity = all_results['accuracy'][all_results['disparity'] <= row.disparity]
        if row.accuracy >= accuracy_for_lower_or_eq_disparity.max():
            non_dominated.append(True)
        else:
            non_dominated.append(False)

    sweep_non_dominated = np.asarray(sweep)[non_dominated]
    accuracy_non_dominated = np.asarray(accuracy_sweep)[non_dominated]
    # auc_non_dominated = np.asarray(auc_sweep)[non_dominated]

    # Plot DP difference vs balanced accuracy
    plt.scatter(accuracy_non_dominated, sweep_non_dominated, label=model_name)
    plt.scatter(accuracy_score(y_test, y_predict),
                constraint(y_test, y_predict, sensitive_features=race_test),
                label='Unmitigated Model')
    plt.xlabel('Accuracy')
    plt.ylabel(constraint_name)
    plt.legend(bbox_to_anchor=(1.55, 1))
    plt.show()
    models_dict = update_model_perf_dict(sweep, models_dict, sweep_preds, sweep_scores, non_dominated, decimal, y_test, race_test, model_name)

    return models_dict


def impact_bar_plots(data_path, b_or_w = 'black',folders= ['dt','lgr','gbt','gnb']):

    dfs = {} # list for pandas dfs
    for i,f in enumerate(folders):
        if b_or_w == 'black':
            path = f'{data_path}{f}/{f}_black_results.csv'
        else:
            path = f'{data_path}{f}/{f}_white_results.csv'
        df = pd.read_csv(path,index_col=0)
        df = df.reset_index()
        dfs[i] = list(df.iloc[:,-1])
        plt.rcParams["figure.figsize"] = [6, 4.5]
        plt.rcParams["figure.autolayout"] = True

    colors=['black','blue','blue','blue','blue','blue','cyan','cyan','cyan','cyan','cyan']
    black_patch = mpatches.Patch(color='black', label ='No Reduction')
    blue_patch = mpatches.Patch(color='blue', label ='Exponentiated Gradient Reduction')
    cyan_patch = mpatches.Patch(color='cyan', label ='Grid Search Reduction')
    idx = ['Unmitigated', 'DP', 'EO', 'TPRP', 'FPRP', 'ERP', ' DP', ' EO', ' TPRP', ' FPRP', ' ERP']
    for i in range(len(folders)):
        print(dfs[i])
        fig, ax = plt.subplots()
        ax.set_title(f'Impact for all Models for Classifier: {folders[i]} and Group {b_or_w}')
        plt.bar(idx,dfs[i],width= 0.8, color=colors)
        ax.set_xticks(idx)
        ax.set_xticklabels(idx, rotation=90)
        #plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        ax.set_xlabel('Models')
        ax.set_ylabel('Impact')
        ax.legend(handles=[black_patch,blue_patch,cyan_patch],loc = 4)
    plt.show()
