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


def spectrum_in_bins(df, m, M, bin_size):
    # Now, let's define the mz ranges, and the label associated to each of them (the mean of the limiting values of each bin)
    range_min = []; range_max = []; range_label = []
    for mz in range(m, M, bin_size):
        range_min.append(mz)
        range_max.append(mz+bin_size)
        range_label.append(np.mean([range_min[-1], range_max[-1]]).astype(int))
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
                idx_data_in_bins[0, i] = np.max(intensity_range)
            else: # if those mz coordinates are not in that spectrum
                idx_data_in_bins[0, i] = 0

        # Normalize the amplitude of the spectrum
        idx_data_in_bins[0,:] = idx_data_in_bins[0,:] / np.max(idx_data_in_bins[0,:])
        all_data[idx,:] = idx_data_in_bins
    new_df = pd.DataFrame(data=all_data, columns = range_label, index = df.index)
    return new_df


# save and load models
def save_clf(savepath, filename, clf_list):
    # filename must be without extension
    if (savepath[-1] != '/'): savepath = savepath + '/'
    with open(savepath+filename+'.data', 'wb') as filehandle:
        pickle.dump(clf_list, filehandle)


def load_clf(savepath, filename):
    if (savepath[-1] != '/'): savepath = savepath + '/'
    if os.path.isfile(savepath+filename+'.data'):
        with open(savepath+filename+'.data', 'rb') as filehandle:
            new_list = pickle.load(filehandle)
        print('Loaded!')
    else:
        print('File not found')
        new_list = []
    return new_list


def try_clf(clf, params, n_cv=5):  # new version! (after Sevilla)
    t1 = time.time()

    best_classifiers = [];
    grid_list = [];
    AUC_train = [];
    AUC_test_train = [];

    categories = targets_train.columns[:]
    for c, cat in enumerate(categories):

        print([cat])  # indicate in which antibiotic we are

        # Selection of train and test data (depending on whether there are NaN target values)
        X_train, Y_train = clean_nan_samples(spectrum_train, targets_train, c, cat)
        X_test_train, Y_test_train = clean_nan_samples(spectrum_test_train, targets_test_train, c, cat)

        # perform a GridSearchCV in order to train a classifier for this antibiotic
        grid = GridSearchCV(clf, param_grid=params, scoring='roc_auc', n_jobs=njobs, pre_dispatch='2*n_jobs', cv=n_cv,
                            iid=False, return_train_score=True)
        grid.fit(X_train, Y_train)

        # print the best parameters (to detect edge values), and save that classifier
        print('The best parameters are: ', grid.best_params_)
        best_clf = grid.best_estimator_
        best_classifiers.append(best_clf)
        grid_list.append(grid)

        # compute the AUC of the classifier
        if callable(getattr(best_clf, "predict_proba", None)):
            pred_train = best_clf.predict_proba(X_train)[:, -1]  # only take last column, the prob of Y = +1
            pred_test = best_clf.predict_proba(X_test_train)[:, -1]
        else:
            print('Using decision_function instead of predict_proba')
            pred_train = best_clf.decision_function(X_train)
            pred_test = best_clf.decision_function(X_test_train)
        auc_score_train = roc_auc_score(Y_train, pred_train)
        auc_score_test = roc_auc_score(Y_test_train, pred_test)
        print('Train AUC: ', np.round(auc_score_train, 4), ' and test_train AUC: ', np.round(auc_score_test, 4))
        AUC_train.append(auc_score_train)
        AUC_test_train.append(auc_score_test)

    avg_AUC_train = np.mean(AUC_train)
    avg_AUC_test_train = np.mean(AUC_test_train)
    print('\n\nThe average train AUC is', np.round(avg_AUC_train, 4), 'and the avg test_train AUC is',
          np.round(avg_AUC_test_train, 4))

    t2 = time.time()
    print('\nFull execution took ', np.round(t2 - t1, 1), 'seconds')
    print('\nDONE!')
    return best_classifiers, grid_list, AUC_train, AUC_test_train


def get_l1_clfs():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1e6, class_weight='balanced')
    params = {'C': 10. ** np.arange(0, 3)}  # only up to 100, so that enough little features remain
    l1_best_clfs, _, _, _ = try_clf(clf, params)
    return l1_best_clfs


def obtain_l1_vects(l1_best_clfs, spectrum_train, targets_train):
    l1_feat_list = []
    categories = targets_train.columns[:]

    for c, cat in enumerate(categories):
        n = np.sum(np.abs(l1_best_clfs[c].coef_) > 0)
        print('Number of features:', n)
        while n == 0:
            clf = l1_best_clfs[c]
            c_value = clf.get_params()['C']
            new_c = c_value * 10
            clf.set_params(C=new_c)
            X_train, Y_train = clean_nan_samples(spectrum_train, targets_train, c, cat)
            clf.fit(X_train, Y_train)  # refit with higher C
            l1_best_clfs[c] = clf
            n = np.sum(np.abs(clf.coef_) > 0)
            print(n)

        # once we know we have at least one non-zero feature
        vect = (np.abs(l1_best_clfs[c].coef_) > 0).reshape(-1, )
        l1_feat_list.append(vect)
    return l1_feat_list


def try_clf_feat_selection(clf, params, feature_vector_list, n_cv=5):  # after Sevilla
    t1 = time.time()

    best_classifiers = [];
    grid_list = [];
    AUC_train = [];
    AUC_test_train = [];

    categories = targets_train.columns[:]
    for c, cat in enumerate(categories):

        print([cat])  # indicate in which antibiotic we are

        # Selection of train and test data (depending on whether there are NaN target values)
        X_train, Y_train = clean_nan_samples(spectrum_train, targets_train, c, cat)
        X_test_train, Y_test_train = clean_nan_samples(spectrum_test_train, targets_test_train, c, cat)

        X_train = apply_l1_feature_selection(X_train, feature_vector_list[c])
        X_test_train = apply_l1_feature_selection(X_test_train, feature_vector_list[c])

        # perform a GridSearchCV in order to train a classifier for this antibiotic
        grid = GridSearchCV(clf, param_grid=params, scoring='roc_auc', n_jobs=njobs, pre_dispatch='2*n_jobs', cv=n_cv,
                            iid=False, return_train_score=True)
        grid.fit(X_train, Y_train)

        # print the best parameters (to detect edge values), and save that classifier
        print('The best parameters are: ', grid.best_params_)
        best_clf = grid.best_estimator_
        best_classifiers.append(best_clf)
        grid_list.append(grid)

        # compute the AUC of the classifier
        if callable(getattr(best_clf, "predict_proba", None)):
            pred_train = best_clf.predict_proba(X_train)[:, -1]  # only take last column, the prob of Y = +1
            pred_test = best_clf.predict_proba(X_test_train)[:, -1]
        else:
            print('Using decision_function instead of predict_proba')
            pred_train = best_clf.decision_function(X_train)
            pred_test = best_clf.decision_function(X_test_train)
        auc_score_train = roc_auc_score(Y_train, pred_train)
        auc_score_test = roc_auc_score(Y_test_train, pred_test)
        print('Train AUC: ', np.round(auc_score_train, 4), ' and test_train AUC: ', np.round(auc_score_test, 4))
        AUC_train.append(auc_score_train)
        AUC_test_train.append(auc_score_test)

    avg_AUC_train = np.mean(AUC_train)
    avg_AUC_test_train = np.mean(AUC_test_train)
    print('\n\nThe average train AUC is', np.round(avg_AUC_train, 4), 'and the avg test_train AUC is',
          np.round(avg_AUC_test_train, 4))

    t2 = time.time()
    print('\nFull execution took ', np.round(t2 - t1, 1), 'seconds')
    print('\nDONE!')
    return best_classifiers, grid_list, AUC_train, AUC_test_train


def apply_feature_selection(spectrum_train, vect):  # vect is feat_list[c]
    new_spectrum = spectrum_train.copy(deep=True).iloc[:, vect]
    return new_spectrum


def get_unique_spectre(spectre, df):
    # Include the ID_sample column for the group_by
    spectre['ID_sample'] = df.ID_sample
    # MEAN OF THE SPECTRE
    spectre = spectre.groupby('ID_sample').mean().reset_index()
    return spectre


def generate_csv_from_clf(clf_list, path, path_results, file_name):
    # classifiers must be provided with parameters, and in a list [clf_antibiotic0, clf_antibiotic1, ...]
    # spectrum and targets full train (containing all training points) will be used for training the clfs in clf list
    # df_test must be provided as loaded from the file

    # read all data from files
    zf = zipfile.ZipFile(path+'zipped_TrainData.zip', 'r')
    df_full_train = _pickle.loads(zf.open('TrainData.pkl').read());   zf.close()

    zf = zipfile.ZipFile(path+'zipped_TestDataUnlabeled.zip', 'r')
    df_test = _pickle.loads(zf.open('TestDataUnlabeled.pkl').read());   zf.close()

    # Process test df to get UNIQUE samples and convert to spectrum

    # df_unique_test = df_test.drop_duplicates(subset='ID_sample')

    spectrum_test_forcsv = spectrum_in_bins(df_test, m, M, bin_size)
    spectrum_test_forcsv = get_unique_spectre(spectrum_test_forcsv, df_test)

    # Process train set to later train the clfs
    spectrum_full_train = spectrum_in_bins(df_full_train.iloc[:, -2:], m, M, bin_size)
    targets_full_train = df_full_train.iloc[:, 1:-2]

    # read the submission example file
    df_submission = pd.read_csv(path+'SubmissionSample.csv')
    categories = df_submission.columns[1:]
    df_submission['ID'] = spectrum_test_forcsv['ID_sample'].values
    # To eliminate the ID_sample from the spectrum
    spectrum_test_forcsv = spectrum_test_forcsv.drop(columns=['ID_sample'])
    for c, cat in enumerate(categories):
        # clean NaN values
        X_train, Y_train = clean_nan_samples(spectrum_full_train, targets_full_train, c, cat)

        # fit the classifier
        clf_base = clf_list[c].fit(X_train, Y_train)
        # Compute its test prediction and save this output
        o_test = clf_base.predict_proba(spectrum_test_forcsv)[:, 1]
        df_submission[cat] = o_test

    # Save the data frame with the predicted outputs
    df_submission = df_submission.set_index('ID')
    df_submission.to_csv(path_results + file_name + '.csv')
    print('DONE!')
    return df_submission
