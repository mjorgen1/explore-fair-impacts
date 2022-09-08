import numpy as np
import pandas as pd
import random
from random import choices
import yaml
from yaml.loader import SafeLoader

# import all of our files
import Liu_paper_code.fico as fico
import Liu_paper_code.distribution_to_loans_outcomes as dlo

def load_args(file):
    """
    Load args and run some basic checks.
        Args:
            - file <str>: full path to .yaml config file
        Returns:
            - data <dict>: dictionary with all args from file
    """
    with open(file, "r") as stream:
        try:
            data = yaml.load(stream, Loader=SafeLoader)
            print('Arguments: ',data)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_pmf(cdf):
    """
    Calculation of the probability mass function. (to populate group distributions)
    Code from Lydia's delayedimpact repository FICO-figures.ipynb:
    https://github.com/lydiatliu/delayedimpact/tree/master/notebooks
        Args:
            - cdf <numpy.ndarray>: cumulative distribution function (for discrete scores)
        Returns:
            - pis <numpy.ndarray>: the probabiliy by for each score (probability mass function)
    """
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size-1):
        pis[score+1] = cdf[score+1] - cdf[score]
    return pis


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
    #group_ratio = np.array((totals['Black'], totals['White']))
    #group_size_ratio = group_ratio/group_ratio.sum()

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
            - samples <numpy.ndarray>: all scores
            - repay_probs <numpy.ndarray>: repay probabilities
            - round_num <int> {0,1,2}: rounding decimal indicator
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


def get_scores(scores, round_num):
    """
    Returns rounded scores.
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


def adjust_set_ratios(x_data, y_data, label_ratio, race_ratio, set_size_upper_bound):
    """
    Changes the proportions of samples in the set. Proportion of each group (race) and proportion of labels for the Black (0) group.
        Args:
            - x_data <numpy.ndarray>: ['score','repay_probability','race'] -> array of samples
            - y_data <numpy.ndarray>: ['repay_indices'] -> array of samples
            - label_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of samples with each labels for the black group (False,True)
            - race_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of black to white samples generated (Black,White)
            - set_size_upper_bound <int>: absolute upper bound of the size for the dataset (e.g 100,000)
        Returns:
            subset of x_data and y_data
    """
    # Black = 0; White = 1
    # limits the absolute test_size if necessary
    if len(y_data) > set_size_upper_bound:
        set_size = set_size_upper_bound
    else:
        set_size = len(y_data)

    num_0 = int(round(set_size * race_ratio[0]))
    num_1 = int(round(set_size * race_ratio[1]))

    # number of samples for the Black group, according to the label ratio
    num_0P = int(round(num_0 * label_ratio[1]))
    num_0N = int(round(num_0 * label_ratio[0]))
    num_1P = int(round(num_1 * 0.76))
    num_1N = int(round(num_1 * 0.24))

    # getting the indices of each samples for each group
    idx_0N = np.where((x_data[:, 2] == 0) & (y_data == 0))[0]
    idx_0P = np.where((x_data[:, 2] == 0) & (y_data == 1))[0]

    idx_1N = np.where((x_data[:, 2] == 1) & (y_data == 0))[0]
    idx_1P = np.where((x_data[:, 2] == 1) & (y_data == 1))[0]

    # if group size numbers are larger than the available samples for that group adjust it
    if len(idx_0P) < num_0P:
        num_0P = len(idx_0P)
        num_0N = int(round(num_0P/label_ratio[1] * label_ratio[0]))
        num_1P =  int(round((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.76))
        num_1N =  int(round((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.24))
    if len(idx_0N) < num_0N:
        num_0N = len(idx_0N)
        num_0P = int(round(num_0N/label_ratio[0] * label_ratio[1]))
        num_1P =  int(round((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.76))
        num_1N =  int(round((num_0N + num_0P)/race_ratio[0] * race_ratio[1] * 0.24))
    if len(idx_1P) < num_1P:
        num_1P = len(idx_1P)
        num_1N = int(round(num_1P/0.76 * 0.24))
        num_0P =  int(round((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * label_ratio[1]))
        num_0N =  int(round((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * label_ratio[0]))
    if len(idx_1N) < num_1N:
        num_1N = len(idx_1N)
        num_1P = int(round(num_1N/0.24 * 0.76))
        num_0P =  int(round((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * label_ratio[1]))
        num_0N =  int(round((num_1N + num_1P)/race_ratio[1] * race_ratio[0] * label_ratio[0]))

    # take the amount of samples, by getting the amount of indices
    idx_0N = idx_0N[:num_0N]
    idx_0P = idx_0P[:num_0P]
    idx_1N = idx_1N[:num_1N]
    idx_1P = idx_1P[:num_1P]
    # concatenate indices
    idx = sorted(np.concatenate((idx_0N,idx_0P,idx_1N,idx_1P)))

    return x_data[idx,:], y_data[idx]


def sample(group_size_ratio, order_of_magnitude, shuffle_seed,scores_arr, pi_A, pi_B, repay_A_arr, repay_B_arr):
    """
    Samples data according to the pmfs and scores from the Fico dataset.
        Args:
            - group_size_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of black to white samples generated (Black,White)
            - order_of_magnitude <int>: total size of the datase
            - shuffle_seed <int>: Seed to control randomness inthe shuffeling of the dataset
            - scores_arr <list>: list with all avalible scores
            - pi_A <numpy.ndarray>: pmf of group A
            - pi_B <numpy.ndarray>: pmf of group B
            - repay_A_arr <numpy.ndarray>: repay probabilities group A
            - repay_B_arr <numpy.ndarray>: repay probabilities group B
        Returns:
            - data_all_df_shuffled <pd.DataFrame>: shuffled dataFrame
    """
    num_A_samples = int(group_size_ratio[0] * order_of_magnitude)
    num_B_samples = int(group_size_ratio[1] * order_of_magnitude)

    # Sample data according to the pmf
    # Reference: https://www.w3schools.com/python/ref_random_choices.asp
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
    np.random.seed(shuffle_seed)
    data_all_df_shuffled = data_all_df.sample(frac=1).reset_index(drop=True)
    #print(data_all_df_shuffled)

    # Add Final Column to dataframe, repay indices
    # repay: 1.0, default: 0.0
    probabilities = data_all_df_shuffled['repay_probability']
    repay_indices = []
    # Create a random num and then have that decide given a prob if the person gets a loan or not
    # (e.g. If 80% prob, then calculate a random num, then if that is below they will get loan, if above, then they don't)
    for index, prob in enumerate(probabilities):
        rand_num = random.randint(0,1000)/10
        if rand_num > prob:  # default
            repay_indices.append(0)
        else:
            repay_indices.append(1)  # repay

    data_all_df_shuffled['repay_indices'] = np.array(repay_indices)

    return data_all_df_shuffled


def load_sample_and_save(data_dir, result_path, order_of_magnitude, group_size_ratio, black_label_ratio, set_size, round_num_scores, shuffle_seed = None):
    """
    Complete pipeline of loading, parsing,sampling and saving of the created synthetic dataset.
        Args:
            - data_dir <str>: Path to the directorz of the raw data
            - results_path <str>: Path for the data file, including its file_name
            - order_of_magnitude <int>: total size of the dataset
            - group_size_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of black to white samples generated (Black,White)
            - black_label_ratio <list<float>>: contains two 2 floats between 0 and 1 (sum = 1), representing the ratio of samples with each labels for the black group (False,True)
            - set_size <int>: absolute size for the dataset (e.g 100,000)
            - round_num_scores <list> {0,1,2}: look at def:get_scores
            - shuffle_seed <int>: Seed to control randomness in the shuffeling of the dataset
    """
    # Make repay probabilities into percentages from decimals
    # NOTE: A is Black, B is White
    scores_list,repay_A,repay_B,pi_A,pi_B = load_and_parse(data_dir)
    scores_arr = np.asarray(get_scores(scores=scores_list, round_num=round_num_scores)) # we recommend 1 or 2 for round_num

    repay_A_arr = pd.Series.to_numpy(repay_A)*100
    repay_B_arr = pd.Series.to_numpy(repay_B)*100

    # generate first batch of samples:
    data = sample([0.12,0.88], order_of_magnitude,shuffle_seed, scores_arr, pi_A, pi_B, repay_A_arr, repay_B_arr)

    # split the data cols (x,y)
    x = data[['score','repay_probability', 'race']].values
    y = data['repay_indices'].values

    # adjust the set according to the ratios specified
    x,y = adjust_set_ratios(x, y, black_label_ratio, group_size_ratio, set_size)

    i = 1
    # as long as the set is to small, generate additional samples
    while len(y) < set_size:
        i += 1

        # Generate new samples
        data_add = sample([0.12,0.88], order_of_magnitude,i, scores_arr, pi_A, pi_B, repay_A_arr, repay_B_arr)
        data = pd.concat([data,data_add])

        # split the data cols (x,y)
        x = data[['score','repay_probability', 'race']].values
        y = data['repay_indices'].values

        # adjust the set according to the ratios specified
        x,y = adjust_set_ratios(x,y, black_label_ratio, group_size_ratio, set_size)


    # merge x,y back into a DataFrame
    df = {'score':x[:,0],'repay_probability': x[:,1],'race':x[:,2],'repay_indices': y}
    data = pd.DataFrame(df)

    # print proportions of dataset
    idx_An = np.where((x[:, 2] == 0) & (y == 0))[0]
    idx_Ap = np.where((x[:, 2] == 0) & (y == 1))[0]
    idx_B = np.where((x[:, 2] == 1))[0]
    print(i,'Black N/P:',len(idx_An),'/',len(idx_Ap),'White:',len(idx_B))

    #Save pandas Dataframes in CSV
    data.to_csv(index=False, path_or_buf=result_path)
