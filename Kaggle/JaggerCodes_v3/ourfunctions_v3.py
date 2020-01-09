# In this file, the different functions that we have developed will be included, for easier use and maintenance

import numpy as np
import pandas as pd
import zipfile
import _pickle
import pickle
from sklearn.model_selection import GridSearchCV
import time
import os
import peakutils

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


def remove_noise(df):
    N = len(df)  # number of samples
    idx_list = []
    for idx in range(N):
        intensity = df[['intensity']].iloc[idx].values[0]
        if np.var(intensity) < 100:
            idx_list.append(idx)
            print('Training sample', idx, ' eliminated')
    new_df = df.drop(index = idx_list)
    return new_df


def interpolate_spectra(df, m, M, step_size):
    # step_size is the size of each step; 1 interpolates very well.
    mz_range = np.arange(m, M + 1, step_size)

    N = len(df)  # number of samples
    L = len(mz_range)  # length of new spectrum (number of bins)
    all_data = np.zeros((N, L))

    for idx in range(N):
        intensity = df[['intensity']].iloc[idx].values[0]
        mzcoord = df[['coord_mz']].iloc[idx].values[0]
        interpolated_spectrum = np.interp(x=mz_range, xp=mzcoord, fp=intensity)
        interpolated_spectrum = interpolated_spectrum / np.max(interpolated_spectrum)
        all_data[idx, :] = interpolated_spectrum
    new_df = pd.DataFrame(data=all_data, columns=mz_range, index=df.index)
    return new_df


def spectrum_in_bins_2(df, m, M, bin_size): # new version
    # Now, let's define the mz ranges, and the label associated to each of them (the mean of the limiting values of each bin)
    range_min = []; range_max = []; range_mean = []
    for mz in range(m,M,bin_size):
        range_min.append(mz)
        range_max.append(mz+bin_size)
        range_mean.append(np.mean([range_min[-1],range_max[-1]]).astype(int))
    N = len(df)  # number of samples
    L = len(range_min)  # length of new spectrum (number of bins)
    all_data = np.zeros((N,L))
    for idx in range(N):
        intensity = df[['intensity']].iloc[idx].values[0]
        mzcoord   = df[['coord_mz']].iloc[idx].values[0]
        idx_data_in_bins = np.zeros((1,L))
        for i,mz in enumerate(range_min):
            intensity_range = intensity[(mzcoord > mz) & (mzcoord < (mz+bin_size))]
            if len(intensity_range) > 0 :
                # as we are interested in peak values, let's keep the maximum value in the interval
                idx_data_in_bins[0,i] = np.max(intensity_range)
            else: # if those mz coordinates are not in that spectrum we interpolate
                idx_data_in_bins[0,i] = np.interp(x=range_mean[i],xp=mzcoord,fp=intensity)

        # Normalize the amplitude of the spectrum
        idx_data_in_bins[0,:] = idx_data_in_bins[0,:] / np.max(idx_data_in_bins[0,:])
        all_data[idx,:] = idx_data_in_bins
    new_df = pd.DataFrame(data=all_data, columns = range_mean, index = df.index)
    return new_df


def spectrum_in_bins_3(df, m, M, bin_size): # incorporates baseline removal
    # Now, let's define the mz ranges, and the label associated to each of them (the mean of the limiting values of each bin)
    range_min = []; range_max = []; range_mean = []
    for mz in range(m,M,bin_size):
        range_min.append(mz)
        range_max.append(mz+bin_size)
        range_mean.append(np.mean([range_min[-1],range_max[-1]]).astype(int))
    N = len(df)  # number of samples
    L = len(range_min)  # length of new spectrum (number of bins)
    all_data = np.zeros((N,L))
    for idx in range(N):
        intensity = df[['intensity']].iloc[idx].values[0]
        mzcoord   = df[['coord_mz']].iloc[idx].values[0]
        idx_data_in_bins = np.zeros((1,L))
        for i,mz in enumerate(range_min):
            intensity_range = intensity[(mzcoord > mz) & (mzcoord < (mz+bin_size))]
            if len(intensity_range) > 0 :
                # as we are interested in peak values, let's keep the maximum value in the interval
                idx_data_in_bins[0,i] = np.max(intensity_range)
            else: # if those mz coordinates are not in that spectrum we interpolate
                idx_data_in_bins[0,i] = np.interp(x=range_mean[i],xp=mzcoord,fp=intensity)

        # Normalize the amplitude of the spectrum
        idx_data_in_bins[0,:] = idx_data_in_bins[0,:] / np.max(idx_data_in_bins[0,:])
        # Remove baseline
        idx_data_in_bins[0,:] -= peakutils.baseline(idx_data_in_bins[0,:],deg=4)
        # Store in matrix
        all_data[idx,:] = idx_data_in_bins
    new_df = pd.DataFrame(data=all_data, columns = range_mean, index = df.index)
    return new_df






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