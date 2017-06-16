from create_freesurfer_masks import write_masks
from glob import glob
from os.path import exists
from SML_mvpa_scripts import *

import pdb

pdb.set_trace()

def main():
    subs2run = list(pd.read_csv('subs2run_SFN_with_beh.csv').iloc[:,1])

    for subject in subs2run:

        # Handle files and load BOLD data
        # ----------------------------------------------------------------------
        files = manage_files(subject)

        # We want to load the study and test data together in one matrix (and
        # then split) to ensure that masking is equivalent for the two (out of
        # brain voxels are excluded by load_data by default)
        print('Loading bold data for subject {}'.format(subject))
        big_X, big_X_groups, original_mask_cols = load_data(files['bold_files'],
                                                            files['roi_file'])

        # Estimate regression modeling effects of head motion and artifacts on
        # fMRI time series and get residuals
        # ----------------------------------------------------------------------
        DM = get_dm(big_X_groups, motion_files=files['motion_files'],
                artifact_files=files['artifact_files'])

        betas, big_X_resid = glm(DM, big_X)

        # Split up the data into study and test components
        big_X_study, big_X_test, big_X_groups_study, big_X_groups_test = \
            split_data(big_X_resid, big_X_groups, files)

        # Get category labels for machine learning classification (drops trials
        # with artifacts)
        # ----------------------------------------------------------------------
        study_onsets_df, test_onsets_df = get_onsets(files)
        y_study = study_onsets_df.condition
        groups_study = study_onsets_df.run

        # Average peak for each trial (includes z-scoring)
        # ----------------------------------------------------------------------
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

        # L2 Logistic Regression Classifier (train study, test recall)
        # ----------------------------------------------------------------------
        print('Train encoding, test recall for subject {}'.format(subject))
        auc_df, prob_df = LogReg_Permutation(X_train=X_study, X_test=X_test,
            y_train=y_study, y_test=y_test, train_groups=groups_study,
            tot_num_vox = 500, n_permutations=500)

        # Add fields normally added by CV_LogReg_Permutation:
        prob_df.loc[:, 'Run'] = np.array(groups_test)
        prob_df.loc[:, 'Category'] = np.array(y_test)
        prob_df.loc[:, 'Onset'] = np.array(test_onsets_df.loc[
            test_onsets_df['condition']!='OTHER', 'concat_onset'])

        # Save output files
        auc_df.to_csv('./port_data/train_enc_test_rec_port{}_VOTC_auc.csv'.\
            format(subject))
        prob_df.to_csv('./port_data/train_enc_test_rec_port{}_VOTC_prob.csv'.\
            format(subject))

    return None


def manage_files(subject):
    labels2include = ['fusiform', 'parahippocampal', 'inferiortemporal']
    roi_mask_name = 'VOTC_fusi_parahipp_inftemp.nii.gz'
    SUBJECTS_DIR = '/share/awagner/AM/data'

    # Get combined mask if file does not already exist
    roi_file = '/share/awagner/AM/data/{}/masks/{}'.format(subject,
        roi_mask_name)

    if not exists(roi_file):
        print('Creating freesurfer mask for subject {}'.format(subject))
        write_masks(subject, labels2include, roi_mask_name, SUBJECTS_DIR,
            exp='study')

    # Get paths to files we need
    bold_files = sorted(glob(('/share/awagner/AM/analysis/*/{}/reg/epi/'
        + 'unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    study_bold_files = sorted(glob(('/share/awagner/AM/analysis/study/{}/reg/'
        + 'epi/unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    test_bold_files = sorted(glob(('/share/awagner/AM/analysis/test/{}/reg/epi/'
        + 'unsmoothed/run_*/timeseries_xfm.nii.gz').format(subject)))

    study_onsets_file = \
        '/share/awagner/AM/data/{}/behav/study.csv'.format(subject)
    test_onsets_file = \
        '/share/awagner/AM/data/{}/behav/test.csv'.format(subject)

    # Get motion & artifact files for all study and test runs (glob)
    motion_files = sorted(glob(('/share/awagner/AM/analysis/*/{}/preproc/run_*/'
        + 'realignment_params.csv').format(subject)))

    artifact_files = sorted(glob(('/share/awagner/AM/analysis/*/{}/preproc/'
        + 'run_*/artifacts.csv').format(subject)))

    study_artifact_files = sorted(glob(('/share/awagner/AM/analysis/study/{}/'
        + 'preproc/run_*/artifacts.csv').format(subject)))

    test_artifact_files = sorted(glob(('/share/awagner/AM/analysis/test/{}/'
        + 'preproc/run_*/artifacts.csv').format(subject)))

    n_study_runs = len(study_bold_files)
    n_test_runs = len(test_bold_files)

    files = {'roi_file' : roi_file,
             'bold_files' : bold_files,
             'study_bold_files' : study_bold_files,
             'test_bold_files' : test_bold_files,
             'study_onsets_file' : study_onsets_file,
             'test_onsets_file' : test_onsets_file,
             'motion_files' : motion_files,
             'artifact_files' : artifact_files,
             'study_artifact_files' : study_artifact_files,
             'test_artifact_files' : test_artifact_files,
             'n_study_runs' : n_study_runs,
             'n_test_runs' : n_test_runs}

    return files


def split_data(big_X_resid, big_X_groups, files):
    n_study_runs = files['n_study_runs']
    n_test_runs = files['n_test_runs']

    # Split data into study and test, use residuals from regression model
    big_X_study = big_X_resid[big_X_groups <= n_study_runs, :]
    big_X_test = big_X_resid[big_X_groups > n_study_runs, :]

    big_X_groups_study = big_X_groups[big_X_groups <= n_study_runs]
    big_X_groups_test = big_X_groups[big_X_groups > n_study_runs]

    return big_X_study, big_X_test, big_X_groups_study, big_X_groups_test


def get_onsets(files):
    # Handle onsets
    study_onsets_ALL_df = concat_onsets(files['study_onsets_file'], \
        files['study_bold_files'])
    study_onsets_df = remove_artifact_trials(study_onsets_ALL_df, \
        files['study_artifact_files'])

    test_onsets_ALL_df = concat_onsets(files['test_onsets_file'], \
        files['test_bold_files'])
    test_onsets_df = remove_artifact_trials(test_onsets_ALL_df, \
        files['test_artifact_files'])

    # Recode condition labels for test_onsets_df to match study
    # Note that there is considerable flexibility here in the combinations of
    # trial types you can include
    test_labels = {'WP' : ['SHP'],
                   'WF' : ['SHF']}

    test_onsets_df['orig_cond'] =  test_onsets_df['condition']
    for index, row in test_onsets_df.iterrows():
        if row['condition'] in test_labels['WP']:
            test_onsets_df.loc[index, 'condition'] = 'WP'
        elif row['condition'] in test_labels['WF']:
            test_onsets_df.loc[index, 'condition']  = 'WF'
        else:
            # new items & source miss or error
            test_onsets_df.loc[index, 'condition']  = 'OTHER'

    return study_onsets_df, test_onsets_df


if __name__ == "__main__":
    main()
