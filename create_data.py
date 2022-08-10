import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import all of our files
import sys
sys.path.append('../')
import Liu_paper_code.fico as fico
import Liu_paper_code.distribution_to_loans_outcomes as dlo

# imports for my own code
import pandas as pd
import random
from random import choices
import argparse

# to populate group distributions
def get_pmf(cdf):
    """
    Calculation of the probability mass function.

    Code from Lydia's delayedimpact repository FICO-figures.ipynb:
    https://github.com/lydiatliu/delayedimpact/tree/master/notebooks
        Args:
            - cdf <numpy.ndarray>: cumulative distribution function (for discrete scores)
        Returns:
            - pis <numpy.ndarray>: array with the probabiliy by for each score (probability mass function)
    """
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size-1):
        pis[score+1] = cdf[score+1] - cdf[score]
    return pis

# Code is primarily from Lydia's FICO-figures.ipynb
def load_and_parse(data_dir):
    """
    Loading the Fico data and applying some parsing steps.

    Code from Lydia's delayedimpact repository FICO-figures.ipynb:
    https://github.com/lydiatliu/delayedimpact/tree/master/notebooks

        Args:
            - data_dir <str>: path to the directorz with the raw fico data
        Returns:
            - scores_list <list>: list with all scores
            - repay_A <pandas.core.series.Series>: performance (repay probability by score) of group A (black)
            - repay_B <pandas.core.series.Series>: performance (repay probabilitz by score) of group B (white)
            - pi_A <numpy.ndarray>: probability mass function for group A
            - pi_B <numpy.ndarray>: probability mass function fro group B
    """

    # set plotting parameters
    sns.set_context('talk')
    sns.set_style('white')
    # this needs to be here so we can edit figures later
    plt.rcParams['pdf.fonttype'] = 42

    all_cdfs, performance, totals = fico.get_FICO_data(data_dir= data_dir);

    cdfs = all_cdfs[['White','Black']]
    # B is White; A is Black
    cdf_B = cdfs['White'].values
    cdf_A = cdfs['Black'].values

    repay_B = performance['White']
    repay_A = performance['Black']

    scores = cdfs.index
    scores_list = scores.tolist()
    scores_repay = cdfs.index

    # basic parameters
    N_scores = cdf_B.size
    N_groups = 2

    # get probability mass functions of each group
    pi_A = get_pmf(cdf_A)
    pi_B = get_pmf(cdf_B)
    pis = np.vstack([pi_A, pi_B])

    # demographic statistics
    group_ratio = np.array((totals['Black'], totals['White']))
    group_size_ratio = group_ratio/group_ratio.sum()

    # to get loan repay probabilities for a given score
    loan_repaid_probs = [lambda i: repay_A[scores[scores.get_loc(i,method='nearest')]],
                         lambda i: repay_B[scores[scores.get_loc(i,method='nearest')]]]

    # unpacking repay probability as a function of score
    loan_repay_fns = [lambda x: loan_repaid_prob(x) for
                          loan_repaid_prob in loan_repaid_probs]

    return scores_list, repay_A, repay_B, pi_A, pi_B


# Round function described below used, round(float_num, num_of_decimals)
# Reference: https://www.guru99.com/round-function-python.html
def get_repay_probabilities(samples, scores_arr, repay_probs, round_num):
    """
    Gets the rounded repay probabilities for all samples.
        Args:
            - samples <numpy.ndarray>: list with score samples
            - repay_probs <numpy.ndarray>: list of repay
            - round_num <int> {0,1,2}:
                0: no rounding (not recommended)
                1: round to the hundreth decimal
                2: round to the nearest integer
        Returns:
            - sample_probs <numpy.ndarray>: rounded repay probability for the samples
    """

    sample_probs = []
    for index, score in enumerate(samples):
        prob_index = np.where(scores_arr == score)
        if round_num == 0:
            repay_prob = repay_probs[prob_index[0][0]]
        elif round_num == 1:
            repay_prob = round(repay_probs[prob_index[0][0]], 2)
        elif round_num == 2:
            repay_prob = round(repay_probs[prob_index[0][0]], 0)
        else:
            raise ValueError('unvalid ound_num value (0,1,2)')
        sample_probs.insert(index, repay_prob)
    return np.asarray(sample_probs)

def get_scores(scores, round_num):  # takes in a list and returns a list
    """
    ---
        Args:
            - scores <list>:
            - round_num <int> {0,1,2}:
                0: no rounding (not recommended)
                1: round to the hundreth decimal
                2: round to the nearest integer
        Returns:
            - updated_scores <list>: (rounded) scores
    """
    if round_num == 0:  # don't change anything
        return scores
    updated_scores = []
    for index, score in enumerate(scores):
        if round_num == 1:  # round to the hundreth decimal
            rounded_score = round(score, 2)
        elif round_num == 2:  # round to the nearest integer
            rounded_score = round(score, 0)
        updated_scores.append(rounded_score)
    return updated_scores




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Specify the path from where the data should be loaded and where the preprocessed datasets should be stored')
    parser.add_argument('--data_dir', type=str, help='Path to data folders',required=True)
    parser.add_argument('--output_path', type=str,help='Path to where the preprocessed datasets should be stored and name of csv',required=True)
    parser.add_argument('--file_name', type=str, help='Path to data folders',required=True)
    parser.add_argument('--round_num', type=int,help='Identifier for rounding', default = 2)
    parser.add_argument('--ord_of_magnitude', type=int,help='Size of the dataset', default = 100)

    args = parser.parse_args()

    data_path = args.data_dir
    result_path = args.output_path + args.file_name
    order_of_magnitude = args.ord_of_magnitude
    round_num = args.round_num

    # Make repay probabilities into percentages from decimals
    # NOTE: A is Black, B is White
    scores_list,repay_A,repay_B,pi_A,pi_B = load_and_parse(data_dir = data_path)
    scores_arr = np.asarray(get_scores(scores=scores_list, round_num=round_num)) # we recommend 1 or 2 for round_num

    repay_A_arr = pd.Series.to_numpy(repay_A)*100
    repay_B_arr = pd.Series.to_numpy(repay_B)*100

    # Sample data according to the pmf
    # Reference: https://www.w3schools.com/python/ref_random_choices.asp

    num_A_samples = 120 * order_of_magnitude
    num_B_samples = 880 * order_of_magnitude

    samples_A = np.asarray(sorted(choices(scores_arr, pi_A, k=num_A_samples)))
    samples_B = np.asarray(sorted(choices(scores_arr, pi_B, k=num_B_samples)))

    # Calculate samples groups' probabilities and make arrays for race

    # A == Black == 0 (later defined as 0.0 when converting to pandas df)
    samples_A_probs = get_repay_probabilities(samples=samples_A,scores_arr=scores_arr, repay_probs=repay_A_arr, round_num=1)
    samples_A_race = np.zeros(num_A_samples, dtype= int)
    # B == White == 1 (later defined as 1.0 when converting to pandas df)
    samples_B_probs = get_repay_probabilities(samples=samples_B,scores_arr=scores_arr, repay_probs=repay_B_arr, round_num=1)
    samples_B_race = np.ones(num_B_samples, dtype= int)

    # Get data in dict form with score and repay prob
    data_A_dict = {'score': samples_A, 'repay_probability': samples_A_probs} #,'race': samples_A_race}
    data_B_dict = {'score': samples_B, 'repay_probability': samples_B_probs} #,'race': samples_B_race}

    # Get data in dict form with score, repay prob, and race
    data_A_dict = {'score': samples_A, 'repay_probability': samples_A_probs ,'race': samples_A_race}
    data_B_dict = {'score': samples_B, 'repay_probability': samples_B_probs,'race': samples_B_race}

    # Convert from dict to df
    data_A_df = pd.DataFrame(data=data_A_dict, dtype=np.float64)
    data_B_df = pd.DataFrame(data=data_B_dict, dtype=np.float64)

    # Combine all of the data together and shuffle
    # NOTE: not currently being used but could be useful at a later time
    data_all_df = pd.concat([data_A_df, data_B_df], ignore_index=True)
    #print(data_all_df)
    data_all_df_shuffled = data_all_df.sample(frac=1).reset_index(drop=True)
    #print(data_all_df_shuffled)

    # Add Final Column to dataframe, repay indices
    # repay: 1.0, default: 0.0

    # Create a random num and then have that decide given a prob if the person gets a loan or not
    # (e.g. If 80% prob, then calculate a random num, then if that is below they will get loan, if above, then they don't)


    probabilities = data_all_df_shuffled['repay_probability']
    repay_indices = []

    for index, prob in enumerate(probabilities):
        rand_num = random.randint(0,1000)/10
        if rand_num > prob:  # default
            repay_indices.append(0)
        else:
            repay_indices.append(1)  # repay

    data_all_df_shuffled['repay_indices'] = np.array(repay_indices)

    #Save pandas Dataframes in CSV
    data_all_df_shuffled.to_csv(index=False, path_or_buf=result_path)

    # To save the data separately by race
    #data_A_df.to_csv(index=False, path_or_buf='simData_2decProbs_0decScores_groupA_black.csv')
    #data_B_df.to_csv(index=False, path_or_buf='simData_2decProbs_0decScores_groupB_white.csv')
