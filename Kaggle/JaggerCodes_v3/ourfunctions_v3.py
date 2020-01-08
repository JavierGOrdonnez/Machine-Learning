# In this file, the different functions that we have developed will be included, for easier use and maintenance

import numpy as np
import pandas as pd
import zipfile
import _pickle
import pickle
from sklearn.model_selection import GridSearchCV
import time
import os

# if there are any NaN values, we should remove those samples
def clean_nan_samples(spectrum, targets, c, cat):
    if (targets[cat].isnull().sum() > 0).all():
        merged = pd.concat([spectrum, targets], axis=1, copy=True)
        clean = merged.dropna(subset=[cat])
        Y = clean.iloc[:, -9+c].to_numpy().reshape(-1,)
        X = clean.iloc[:, :-9]

    else:
        Y = targets.iloc[:, c].to_numpy().reshape(-1,)
        X = spectrum.copy(deep=True)
    return X, Y


def try_clf(clf, params, spectrum_train, targets_train, n_cv=5, njobs=5,
            FEATURE_SELECTION=False, feature_vector_list=None):  # new version! (after Sevilla)
    # new version --> Incorporates feature selection
    t1 = time.time()

    best_classifiers = []
    grid_list = []
    AUC_train = []; AUC_valid = []

    categories = targets_train.columns[:]
    for c, cat in enumerate(categories):

        print([cat])  # indicate in which antibiotic we are

        # Selection of train and test data (depending on whether there are NaN target values)
        X_train, Y_train = clean_nan_samples(spectrum_train, targets_train, c, cat)

        if FEATURE_SELECTION:  # a boolean that decides whether to apply feature selection
            # (feature list has to be already defined, and input to the function)
            X_train = apply_feature_selection(X_train, feature_vector_list[c])

        # perform a GridSearchCV in order to train a classifier for this antibiotic
        grid = GridSearchCV(clf, param_grid=params, scoring='roc_auc', n_jobs=njobs, pre_dispatch='2*n_jobs', cv=n_cv,
                            iid=False, return_train_score=True)
        grid.fit(X_train, Y_train)

        # print the best parameters (to detect edge values), and save that classifier
        print('The best parameters are: ', grid.best_params_)
        best_clf = grid.best_estimator_
        best_classifiers.append(best_clf)
        grid_list.append(grid)

        best_clf = np.where(grid.cv_results_['rank_test_score'] == 1)[0][0]
        AUC_train.append(grid.cv_results_['mean_train_score'][best_clf])
        AUC_valid.append(grid.cv_results_['mean_test_score'][best_clf])

        print('Train AUC: ', np.round(AUC_train[-1], 4), ' and validation AUC: ', np.round(AUC_valid[-1], 4))

    avg_AUC_train = np.mean(AUC_train)
    avg_AUC_valid = np.mean(AUC_valid)
    print('\n\nThe average train AUC is', np.round(avg_AUC_train, 4), 'and the avg validation AUC is',
          np.round(avg_AUC_valid, 4))

    t2 = time.time()
    print('\nFull execution took ', np.round(t2 - t1, 1), 'seconds')
    print('\nDONE!')
    return best_classifiers, grid_list, AUC_train, AUC_valid


def apply_feature_selection(spectrum_train, vect):  # vect is feat_list[c]
    new_spectrum = spectrum_train.copy(deep=True).iloc[:, vect]
    return new_spectrum