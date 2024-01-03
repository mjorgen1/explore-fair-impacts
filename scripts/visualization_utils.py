import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def visualize_data_distribution(path,samples_A,samples_A_probs,samples_B,samples_B_probs):
    """
    Scatter-plot of the Repay distribution (by score) by group.
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
    plt.savefig(f'{path}repay_by_score.png', dpi=300)
    plt.show()


def visual_scores_by_race(path,fname,x):
    """
    Histogram-plot of the credit scores by group (race).
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
    plt.savefig(f'{path}{fname}_demo_distr.png', dpi=300)


def visual_label_dist(path,fname,x,y):
    """
    Histogram-plot of the labels labels by group (race).
        Args:
            - path <str>: path for saving the plot
            - fname <str>: file name for saving the plot
            - x <numpy.ndarray>: ['score','race'] list of all score samples and race indicator
            - y <numpy.ndarray>: ['repay_indices'] list of all lables of the samples
    """

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    fig.suptitle(f'Histogram of Repay Distribution for {fname} data')
    plt.xlabel('Repay Label')

    # devide samples into groups
    black_label_dist = y[np.where(x[:, -1] == 0)[0]]
    white_label_dist = y[np.where(x[:, -1] == 1)[0]]

    n_bins = 2
    # We can set the number of bins with the *bins* keyword argument.
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
    plt.savefig(f'{path}{fname}_label_distr.png', dpi=300)


def delayed_impact_bar_plot(data_path, group = 0, classifier= ['DT','GNB','LGR','GBT']):
    """
    Bar plots of the Delayed Impact for each model by classifier
        Args:
            - data_path <str>: path for loading the data file (csv)
            - group <int>: (0,1) group indicator
            - classifier <list<str>>: list with the names of the model folder where csv are stored
    """
    # load data
    path = f'{data_path}{group}_DI.csv'
    df = pd.read_csv(path)
    df = df.set_index('Constraint')

    # extract values for each group
    line = []
    for i,c in enumerate(classifier):
        df2 = df.loc[:,c]
        line.append(df2)

    # make and save plot
    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 11

    idx = ['Unmitigated', 'DP', 'EO', 'EOO', 'FPER', 'ERP']
    df = pd.DataFrame(line, columns=idx)

    ax = df.plot(kind="bar")
    ax.set_title(f'Delayed Impact for Group: {group}')

    # labels and legend
    ax.set_ylabel('Impact')
    ax.set_xlabel('Classifier')
    plt.xticks(rotation=0)
    ax.legend(loc='lower right')

    plt.savefig(f'{data_path}{group}_DI.png', dpi=300)


def immediate_impact_bar_plot(data_path, group = 0, classifier= ['DT','GNB','LGR','GBT']):
    """
    Bar plots of the Immediate Impact for each model by classifier
        Args:
            - data_path <str>: path for loading the data file (csv)
            - group <int>: (0,1) group indicator
            - classifier <list<str>>: list with the names of the model folder where csv are stored
    """

    # load data
    path = f'{data_path}{group}_I.csv'
    df = pd.read_csv(path)
    df = df.set_index('Constraint')

    # extract by classifier
    line = []
    for i,c in enumerate(classifier):
        df2 = df.loc[:,c]
        line.append(df2)

    # make and save plot
    plt.rcParams["figure.figsize"] = [6, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["font.size"] = 11

    idx = ['Unmitigated', 'DP', 'EO', 'EOO', 'FPER', 'ERP']
    df = pd.DataFrame(line, columns=idx)

    ax = df.plot(kind="bar")
    ax.set_title(f'Immediate Impact for Group: {group}')
    # labels
    ax.set_ylabel('TP-FN-Difference')
    ax.set_xlabel('Classifier')
    plt.xticks(rotation=0)

    ax.legend(loc='lower right')

    plt.savefig(f'{data_path}{group}_I.png', dpi=300)
