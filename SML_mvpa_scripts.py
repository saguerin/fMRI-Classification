import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, rankdata, zscore
from nilearn.input_data import NiftiMasker
import nibabel as nib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def CV_LogReg_Permutation(X, y, groups, onsets_df, tot_num_vox = 500,
                          n_permutations=0, onset_col='onset'):

    onsets_vec = np.array(onsets_df.loc[:, onset_col])

    y_str = y.copy()
    y = get_binary_y(y)

    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, y, groups) # funct ignores X, y

    auc_df = pd.DataFrame()
    prob_df = pd.DataFrame()

    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_groups = groups[train_index]
        test_group = float(np.unique(groups[test_index]))

        auc_df_fold, prob_df_fold = LogReg_Permutation(X_train, X_test, \
            y_train, y_test,train_groups, tot_num_vox, n_permutations)

        auc_df_fold.Run = test_group

        prob_df_fold.loc[:, 'Run'] = test_group
        prob_df_fold.loc[:, 'Onset'] = onsets_vec[test_index]
        prob_df_fold.loc[:, 'Category'] = np.array(y_str[test_index])

        auc_df = auc_df.append(auc_df_fold, ignore_index=True)

        prob_df = prob_df.append(prob_df_fold, ignore_index=True)

    return auc_df, prob_df


def LogReg_Permutation(X_train, X_test, y_train, y_test, train_groups,
                                   tot_num_vox = 500, n_permutations=0):

    y_train, y_test = get_binary_y(y_train), get_binary_y(y_test)

    auc_df = pd.DataFrame(columns = ['Run'] + \
        ['AUC Permutation {}'.format(x) for x in xrange(n_permutations+1)])

    prob_df = pd.DataFrame(index=range(len(y_test)), columns = \
        ['Run', 'Onset', 'Category'] + ['Prob Permutation {}'.format(x) \
        for x in xrange(n_permutations+1)])

    for perm in xrange(n_permutations+1):
        # randomize training labels within runs for permutation testing
        if perm > 0:
            y_train_shuf = shuffle_y(y_train, train_groups)

        else:
            y_train_shuf = y_train.copy()

        # Feature selection
        voxels2use = balanced_feat_sel(X_train, y_train_shuf, tot_num_vox)[0]
        X_train_fs = X_train[:, voxels2use]
        X_test_fs = X_test[:, voxels2use]

        # Fit classifier
        classifier = LogisticRegression(penalty = 'l2', C = 1.0)
        classifier.fit(X_train_fs, y_train_shuf)

        # Apply classifier to test data & write AUC to auc_df
        dec_funct = classifier.decision_function(X_test_fs)

        auc_df.loc[0, 'AUC Permutation {}'.format(perm)] = \
            roc_auc_score(y_test, dec_funct)

        # Get probabilities for each trial
        prob_df.loc[:, 'Prob Permutation {}'.format(perm)] = \
            classifier.predict_proba(X_test_fs)[:,0]

    return auc_df, prob_df


def load_data(bold_files, roi_file, check_mask=True):
    # Loop through runs, get bold data, standardize within run
    num_runs = len(bold_files)
    list_of_bold_matrices = list()
    for this_bold_file in bold_files:
        print('Loading file {}'.format(this_bold_file))

        # Apply mask to bold data and standardize (z-score individually at each
        # voxel, separatley for each run)
        func_masker = NiftiMasker(mask_img=roi_file, smoothing_fwhm=None,
                                  standardize=True)
        bold_2D = func_masker.fit_transform(this_bold_file)
        list_of_bold_matrices.append(bold_2D)

    big_X = np.concatenate(list_of_bold_matrices, axis=0)

    if check_mask:
        # Check for out of brain voxels from any run. Z-scores from out of brain
        # voxels will be equal to nan. Take sum across time and excluded any
        # voxels where sum is nan.
        vox_sum_vec = big_X.sum(axis=0)
        exclude_vox_cols = np.where(np.isnan(vox_sum_vec))
        original_mask_cols = np.where(np.logical_not(np.isnan(vox_sum_vec)))
        big_X = np.delete(big_X, exclude_vox_cols, axis=1)

        return big_X, original_mask_cols

    else:
        return big_X


def concat_onsets(onsets_file, bold_files, TR=2, cond_col='condition', \
                  onset_col='onset'):

    onsets_df = pd.read_csv(onsets_file)
    onsets_df[onset_col] = np.round(onsets_df[onset_col]) / TR

    # Create a new column that records the concatenated onsets (across runs)
    max_TR_last_run = 0
    num_runs = np.max(onsets_df.run)
    for run in xrange(1, num_runs+1):
        if run > 1:
            # Get # of TRs in last run
            img = nib.load(bold_files[run-1])
            max_TR_last_run += img.shape[3]

        onsets_df.loc[onsets_df.run==run, 'concat_onset'] = \
            onsets_df.loc[onsets_df.run==run, onset_col] + max_TR_last_run

    return onsets_df


def avg_peak(big_X, onsets_df, peak_TRs=[2, 3, 4]):
    # Note: peak_TRs uses 0-based indexing

    X = np.zeros((len(onsets_df.index), big_X.shape[1]))

    # Loop through trials and average time points
    X_row = 0
    for index, row in onsets_df.iterrows():
        onset = row.concat_onset
        onsets2avg = (np.array(peak_TRs) + onset).astype(int)

        X[X_row, :] = np.mean(big_X[onsets2avg, :], axis = 0)

        X_row += 1

    return X


def get_binary_y(y_str):
    # Labels are sorted in alphabetic order. The first label is assigned 0 and
    # the second label is assgined 1
    y_str = np.array(y_str)
    labels = np.unique(y_str)

    if len(labels) > 2:
        raise ValueError('More than 2 category labels found in y_str.')

    y = np.zeros(y_str.shape)
    y[y_str==labels[0]], y[y_str==labels[1]] = 0, 1

    return y


def balanced_feat_sel(X_train, y_train, tot_num_vox):
    # Univariate feature selection with an equal # of voxels active for Category
    # 1 and Category 2. y_train must be binary.

    y_train = get_binary_y(y_train)

    if tot_num_vox % 2 != 0:
        raise ValueError('tot_num_vox must be even.')

    num_sel_vox_per_cat = tot_num_vox / 2

    sample0 = X_train[y_train==0, :]
    sample1 = X_train[y_train==1, :]
    t, p = ttest_ind(sample0, sample1, axis=0, equal_var=True)

    # Important - set nan t stats to 0 so that they are not assigned
    # highest rank. As the number of selected voxels approahces the
    # size of the ROI, some out of brain voxels could be included
    # (should not occur in practice)
    t[np.isnan(t)] = 0

    t_ranks = rankdata(t, method = 'ordinal')
    num_ROI_vox = X_train.shape[1]
    cat0_voxels = t_ranks >= (num_ROI_vox-num_sel_vox_per_cat+1)
    cat1_voxels = t_ranks <= num_sel_vox_per_cat
    voxels2use = np.logical_or(cat0_voxels, cat1_voxels)

    return voxels2use, cat0_voxels, cat1_voxels


def shuffle_y(y_train, train_groups):
    y_train_shuf = y_train.copy()
    for run in np.unique(train_groups):
        rows2shuffle = np.array(train_groups) == run
        y_train_shuf[rows2shuffle] = \
            np.random.permutation(y_train_shuf[rows2shuffle])

    return y_train_shuf
