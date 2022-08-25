import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import matplotlib.patches as mpatches
from scipy import stats


def visualize_data_distribution(path,samples_A,samples_A_probs,samples_B,samples_B_probs):
    """
    Plot of the Repay distribution (by score) by group.
        Args:
            - path <str>: path for saving the plot
            - samples_A <numpy.ndarray>: score samples of group A (x)
            - samples_A_probs <numpy.ndarray>: repay values of group A (y)
            - samples_B <numpy.ndarray>: score samples of group B (x)
            - samples_B_probs <numpy.ndarray>: repay values of group B (y)
    """

    # (x,y) tuples
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
        ax.scatter(x, y, alpha=0.8, c=color, label=group)

    plt.title('Sample Distributions by Group')
    plt.legend(loc=2)
    plt.xlabel('Credit Score')
    plt.ylabel('Repay Probability')
    plt.savefig(f'{path}repay_by_score.png')
    plt.show()


def visual_scores_by_race(path,fname,x):
    """
    Plots the number of individuals for each score by group (race).
        Args:
            - path <str>: path for saving the plot
            - fname <str>: file name for saving the plot
            - x <numpy.ndarray>: ['score','race'] list of all score samples and race indicator
    """

    # make histogram of credit scores by race
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.suptitle(f'Histogram Credit Score Distribution {fname} data')
    plt.xlabel('Credit Score')
    #plt.ylabel('No. of Individuals')

    black_credit_dist = x[np.where(x[:, 1] == 0)[0]][:,0]
    white_credit_dist = x[np.where(x[:, 1] == 1)[0]][:,0]

    n_bins = 50
    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(black_credit_dist, bins=n_bins)
    axs[0].set_title('Black Group')
    axs[0].set_xlabel('Credit Score')
    axs[0].set_ylabel('No. of Individuals')
    axs[1].set_title('White Group')
    axs[1].hist(white_credit_dist, bins=n_bins)
    plt.savefig(f'{path}{fname}_demo_distr.png')


def visual_repay_dist(path,fname,x,y):
    """
    Plots the number of repay labels by group (race).
        Args:
            - path <str>: path for saving the plot
            - fname <str>: file name for saving the plot
            - x <numpy.ndarray>: ['score','race'] list of all score samples and race indicator
            - y <numpy.ndarray>: ['repay_indices'] list of all lables of the samples
    """
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.suptitle(f'Histogram of Repay Distribution for {fname} data')
    plt.xlabel('Repay Label')

    black_label_dist = y[np.where(x[:, 1] == 0)[0]]
    white_label_dist = y[np.where(x[:, 1] == 1)[0]]

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
    plt.savefig(f'{path}{fname}_label_distr.png')



def impact_bar_plots(data_path, b_or_w = 'Black',folders= ['dt','lgr','gbt','gnb']):
    """
    Bar plots of the Delayed Impact for each model by classifier
        Args:
            - data_path <str>: path for loading the data file (csv)
            - b_or_w <str>: (Black,White) group indicator
            - folders <list<str>>: list with the names of the model folder where csv are stored
    """
    # loading all datafiles and store them in a dict
    dfs = {} # dict for pandas dfs
    for i,f in enumerate(folders):
        if b_or_w == 'Black': # load results of black group
            path = f'{data_path}{f}/{f}_black_results.csv'
        elif b_or_w == 'White': # load results of white group
            path = f'{data_path}{f}/{f}_white_results.csv'
        df = pd.read_csv(path,index_col=0)
        df = df.reset_index()
        dfs[i] = list(df.iloc[:,-1])

    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 11

    # Colors and labels for bars,labels and legend
    colors=['#FFAE49','#024B7A','#024B7A','#024B7A','#024B7A','#024B7A']
    black_patch = mpatches.Patch(color='#FFAE49', label ='Unmitigated')
    blue_patch = mpatches.Patch(color='#024B7A', label ='Mitigated')
    idx = ['-', 'DP', 'EO', 'TPRP', 'FPRP', 'ERP']

    # enumerate through folders and make a bar plot for each classifier
    for i in range(len(folders)):
        fig, ax = plt.subplots()
        ax.set_title(f'Delayed Impact for Classifier: {folders[i]} / Group: {b_or_w}\n\n')
        plt.bar(idx,dfs[i],width= 0.9, color=colors)

        # Add the y value as label on top of each bar
        for j, v in enumerate(dfs[i]):
            ax.text(j, v, v, ha = 'center', color = 'black', fontsize= 10, bbox = dict(facecolor= '#F9F9F9', edgecolor= colors[j], alpha=0.9, pad=0.5))

        # labels
        ax.set_xticks(idx)
        ax.set_xlabel('Fairness Constraint')
        ax.set_ylabel('Impact')
        #legend
        ax.legend(handles=[black_patch,blue_patch], loc='lower right')

        plt.savefig(f'{data_path}{folders[i]}/{b_or_w}_DI.png')
