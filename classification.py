from impt_functions import *
from evaluation import *

import csv
from itertools import zip_longest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity, BoundedGroupLoss
from fairlearn.metrics import *
from raiwidgets import FairnessDashboard



def classify(data_path,model_name,constraint_str = None,reduction_alg = None):

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

    # given predictions+outcomes, I'll need to do the same

    x = data[['score', 'race']].values
    y = data['repay_indices'].values

    classifier = get_classifier(model_name)
    np.random.seed(0)
    model = classifier.fit(X_train,y_train, sample_weight_train)
    # Reference: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
    if constraint_str == 'Un':
    # Train the classifier:

        # Make predictions with the classifier:
        y_predict = model.predict(X_test)
        y_predict_un = y_predict
        # Scores on test set
        test_scores = model.predict_proba(X_test)[:, 1] #?????????????????????
        scores = cross_val_score(model, x, y, cv=5, scoring='f1_weighted')#??????????????????????
    else:
        np.random.seed(0)
        constraint = get_constraint(constraint_str)
        mitigator = get_reduction_algo(model,constraint, reduction_alg)
        mitigator.fit(X_train, y_train, sensitive_features=race_train)
        y_predict = mitigator.predict(X_test) #y_pred_mitigated

        if reduction_alg ==' GS':
            pass
            # We can examine the values of lambda_i chosen for us:
            lambda_vecs = mitigator.lambda_vecs_
            #print(lambda_vecs[0])
            #grid_search_show(mitigator, demographic_parity_difference, y_predict, X_test, y_test, race_test, 'DemParityDifference','GS DPD', models_dict, 0.3)

    #if dashboard_bool:
    #    pass
        #FairnessDashboard(sensitive_features=race_test,y_true=y_test,
                         # y_pred={'initial model': y_predict_un, 'mitigated model': y_predict})

    results_overall, results_black, results_white = evaluating_model(constraint_str,X_test,y_test, y_predict, sample_weight_test,race_test)
    # get new scores bz race
    X_b, X_w = get_new_scores(X_test, y_predict, y_test, race_test)


    return results_overall, results_black, results_white, X_b, X_w, X_test_b, X_test_w
