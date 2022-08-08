def visualize_data(samples_A,samples_A_probs,samples_B,samples_B_probs):

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
