import pandas as pd
import numpy as np

from itertools import product
from operator import itemgetter
from run_recall_peak_port import manage_files, split_data, get_onsets
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from SML_mvpa_scripts import *

import pdb

def main():
    # Initialize df
    grandD_recall_df = pd.DataFrame(columns=['Subject', 'AUC'])

    subs2run = list(pd.read_csv('subs2run_SFN_with_beh.csv').iloc[:,1])

    for subject in subs2run:
        d = preliminary(subject)

        # 2D encoding model
        twoD_enc_coef_df, twoD_enc_df = twoD_enc(d)
        twoD_enc_coef_df.to_csv('twoD_enc_coef_df_{}.csv'.format(subject))
        twoD_enc_df.to_csv('twoD_enc_df_{}.csv'.format(subject))

        # 2D recall model
        twoD_recall_coef_df, twoD_recall_df = twoD_recall(d)
        twoD_recall_coef_df.to_csv('twoD_recall_coef_df_{}.csv'.format(subject))
        twoD_recall_df.to_csv('twoD_recall_df_{}.csv'.format(subject))

        # 1000D recall model
        auc = grandD_recall(d)
        new_data = pd.DataFrame.from_dict({'Subject' : subject, 'AUC' : auc},
                                          orient='index').T
        grandD_recall_df = grandD_recall_df.append(new_data, ignore_index=True)
        grandD_recall_df.to_csv('grandD_recall_df.csv')


def preliminary(subject):
    # Some preliminary work loading files, handing onsets, getting features.
    files = manage_files(subject)

    # Load MRI data
    print('Loading bold data for subject {}'.format(subject))
    big_X, big_X_groups, original_mask_cols = load_data(files['bold_files'],
                                                        files['roi_file'])

    # Estimate regression modeling effects of head motion and artifacts on
    # fMRI time series and get residuals
    # --------------------------------------------------------------------------
    DM = get_dm(big_X_groups, motion_files=files['motion_files'],
            artifact_files=files['artifact_files'])
    betas, big_X_resid = glm(DM, big_X)

    # Split up the data into study and test components
    big_X_study, big_X_test, big_X_groups_study, big_X_groups_test = \
        split_data(big_X_resid, big_X_groups, files)

    # Get category labels for machine learning classification (drops trials
    # with artifacts)
    # --------------------------------------------------------------------------
    study_onsets_df, test_onsets_df = get_onsets(files)
    y_study = study_onsets_df.condition
    groups_study = study_onsets_df.run

    # Average peak for each trial (includes z-scoring)
    # --------------------------------------------------------------------------
    X_study = avg_peak(big_X_study, study_onsets_df, peak_TRs=[2, 3, 4])
    X_test_all_trials = avg_peak(big_X_test, test_onsets_df,
                                 peak_TRs=[3, 4, 5])

    # Remove 'other' items from test onsets and X_test. These trials
    # contribute to z-scoring but will not be included in the final test.
    X_test = X_test_all_trials[
        np.array(test_onsets_df['condition']!='OTHER'), :]

    y_test = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',
        'condition']
    groups_test = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',
        'run']

    study_onsets_df = study_onsets_df.loc[study_onsets_df['condition']!='OTHER',:]
    test_onsets_df = test_onsets_df.loc[test_onsets_df['condition']!='OTHER',:]

    # Write output in a dictionary
    d = {'y_study': y_study,
         'X_study' : X_study,
         'groups_study' : groups_study,
         'y_test' : y_test,
         'X_test' : X_test,
         'groups_test' : groups_test,
         'study_onsets_df' : study_onsets_df,
         'test_onsets_df' : test_onsets_df}

    return d


def twoD_enc(d):
    # Data for Figure X
    # Scatter plot of 2D decoding (one face activity feature, one scene activity
    # feature). Encoding/study only.
    # --------------------------------------------------------------------------
    y_study, X_study, groups_study = itemgetter('y_study', 'X_study',
                                                'groups_study')(d)

    y_study_str = y_study.copy()
    y_study = np.array(get_binary_y(y_study))

    # Initialize df to write results in
    twoD_enc_coef_df = pd.DataFrame(columns=['Test Run', 'B1', 'B2', 'AUC'])
    twoD_enc_df_columns = ['Test Run', 'Picture Type', 'Voxel Type', 'Trial #',
                           'Mean Signal']
    twoD_enc_df = pd.DataFrame(columns=twoD_enc_df_columns)

    # Leave one run out cross-validation to define face voxels and scene voxels
    # and get logistic regression coefficients
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X_study, y_study, groups_study) # funct ignores X, y

    for train_index, test_index in logo.split(X_study, y_study, groups_study):
        X_train_cross, X_test_cross = X_study[train_index], X_study[test_index]
        y_train_cross, y_test_cross = y_study[train_index], y_study[test_index]

        # Feature selection and regularized logistic regression
        new_data, voxels2use, cat0_voxels, cat1_voxels = \
            twoD_logistic(X_train_cross, y_train_cross, X_test_cross, y_test_cross)

        new_data.loc[0, 'Test Run'] = np.unique(groups_study[test_index])
        twoD_enc_coef_df = twoD_enc_coef_df.append(new_data, ignore_index=True)

        # Loop through voxel types and get mean signal across voxels
        for voxel_type, voxels2avg in zip(['Face Voxels', 'Scene Voxels'],
                                          [cat0_voxels, cat1_voxels]):

            # Calculate Mean MRI Signal
            mean_sig = np.mean(X_test_cross[:, voxels2avg], axis=1)

            # Write result to df
            new_data = pd.DataFrame(columns=twoD_enc_df_columns)
            new_data.loc[:, 'Test Run'] = groups_study[test_index]
            new_data.loc[:, 'Picture Type'] = y_study_str[test_index]
            new_data.loc[:, 'Voxel Type'] = voxel_type
            new_data.loc[:, 'Trial #'] = range(len(test_index))
            new_data.loc[:, 'Mean Signal'] = mean_sig

            twoD_enc_df = twoD_enc_df.append(new_data, ignore_index=True)

    return twoD_enc_coef_df, twoD_enc_df


def twoD_recall(d):
    # Data for Figure X
    # Scatter plot of 2D decoding (one face activity feature, one scene activity
    # feature). Train encoding, test recall.
    # --------------------------------------------------------------------------
    y_study, y_test, X_study, X_test, groups_study, groups_test = itemgetter(
        'y_study', 'y_test', 'X_study', 'X_test', 'groups_study',
        'groups_test')(d)

    y_study_str, y_test_str = y_study.copy(), y_test.copy()
    y_study = np.array(get_binary_y(y_study))
    y_test = np.array(get_binary_y(y_test))

    # Initialize df to write results in
    twoD_recall_coef_df = pd.DataFrame(columns=['Test Run', 'B1', 'B2', 'AUC'])
    twoD_recall_df_columns = ['Test Run', 'Picture Type', 'Voxel Type',
                              'Trial #', 'Mean Signal']
    twoD_recall_df = pd.DataFrame(columns=twoD_recall_df_columns)

    # Feature selection and regularized logistic regression
    new_data, voxels2use, cat0_voxels, cat1_voxels = \
        twoD_logistic(X_study, y_study, X_test, y_test)
    twoD_recall_coef_df = twoD_recall_coef_df.append(new_data, ignore_index=True)

    # Loop through voxel types and get mean signal across voxels
    for voxel_type, voxels2avg in zip(['Face Voxels', 'Scene Voxels'],
                                      [cat0_voxels, cat1_voxels]):

        # Calculate Mean MRI Signal
        mean_sig = np.mean(X_test[:, voxels2avg], axis=1)

        # Write result to df
        new_data = pd.DataFrame(columns=twoD_recall_df_columns)
        new_data.loc[:, 'Test Run'] = groups_test
        new_data.loc[:, 'Picture Type'] = y_test_str
        new_data.loc[:, 'Voxel Type'] = voxel_type
        new_data.loc[:, 'Trial #'] = range(len(y_test_str))
        new_data.loc[:, 'Mean Signal'] = mean_sig

        twoD_recall_df = twoD_recall_df.append(new_data, ignore_index=True)

    return twoD_recall_coef_df, twoD_recall_df

def grandD_recall(d):
    # Data for Figure X
    # Scatter plot of 2D decoding (one face activity feature, one scene activity
    # feature). Train encoding, test recall.
    # --------------------------------------------------------------------------
    y_study, y_test, X_study, X_test, groups_study, groups_test = itemgetter(
        'y_study', 'y_test', 'X_study', 'X_test', 'groups_study',
        'groups_test')(d)

    y_study_str, y_test_str = y_study.copy(), y_test.copy()
    y_study = np.array(get_binary_y(y_study))
    y_test = np.array(get_binary_y(y_test))

    # Feature selection and regularized logistic regression
    auc_df, prob_df = LogReg_Permutation(X_study, X_test, y_study, y_test,
                                         groups_study,tot_num_vox = 500)
    auc = auc_df.loc[0, 'AUC Permutation 0']

    return auc


def twoD_logistic(X_train, y_train, X_test, y_test):
    # Feature selection
    voxels2use, cat0_voxels, cat1_voxels = \
        balanced_feat_sel(X_train, y_train, tot_num_vox=500)

    X_means_train = np.concatenate([
        X_train[:, cat0_voxels].mean(axis=1, keepdims=True),
        X_train[:, cat1_voxels].mean(axis=1, keepdims=True)], axis=1)

    X_means_test = np.concatenate([
        X_test[:, cat0_voxels].mean(axis=1, keepdims=True),
        X_test[:, cat1_voxels].mean(axis=1, keepdims=True)], axis=1)

    # Get coefficients from regularized logistic regression, write to df
    classifier = LogisticRegression(penalty = 'l2', C = 1.0)
    classifier.fit(X_means_train, y_train)
    coef = classifier.coef_.squeeze()
    dec_funct = classifier.decision_function(X_means_test)
    auc = roc_auc_score(y_test, dec_funct)

    new_data = pd.DataFrame(columns=['Test Run', 'B1', 'B2', 'AUC'])
    new_data.loc[0, 'B1'] = coef[0]
    new_data.loc[0, 'B2'] = coef[1]
    new_data.loc[0, 'AUC'] = auc

    return new_data, voxels2use, cat0_voxels, cat1_voxels


if __name__ == "__main__":
    main()
