from nilearn.input_data import NiftiMasker
import nibabel as nib
import numpy as np
from numpy.linalg import inv
import pandas as pd
from scipy.stats import ttest_ind, rankdata, zscore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from scipy.linalg import block_diag

def get_dm(big_X_groups, onsets_df=None, motion_files=None, artifact_files=None,
           onset_col='concat_onset', cond_col='condition',
           duration_col='duration'):
    """Generate design matrix for linear model of fMRI time series.
    Flexible specification of a linear model that can include any of the
    following predictors: (1) Neural activity associated with specific events
    (i.e., trials); (2) artifacts due to minor head motion; (3) artifacts due to
    discrete, transient events (e.g., a spike. Similar to Lyman, a separate
    regressor is added for each scan that contains an artifact). Dummy regressors
    to code differences in mean activity across runs/sessions are included
    automatically.

    Parameters
    ----------
    big_X_groups : numpy array (int)
        Vector specifying the run # for each scan. Length must equal the total #
        of scans (i.e., TRs). Returned by load_data.

    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial. Returned by
        concat_onsets and remove_artifact_trials.

    motion_files : list
        List of file paths to files containing motion parameters (produced by
        Lyman). Each element corresponds to a run. Length must equal # of runs.

    artifact_files : list
        List of file paths to files specifying scans with artifacts (produced by
        Lyman). Each element corresponds to a run. Length must equal # of runs.

    onset_col : str
        Name of column in onsets_df to use for onsets (specified in scan units,
        not seconds). Default is 'concat_onset', which is added to onsets_df by
        the concat_onsets function.

    cond_col : str
        Name of column in onsets_df to use for condition labels. Default is
        'condition'.

    duration_col : str
        Name of column in onsets_df to use for durations (specified in scan
        units, not seconds). Default is 'condition'.

    Returns
    ----------
    DM : numpy array (float)
        Design matrix (X) for a linear model. nrows = # of scans (i.e., TRs).
        ncols = number of parameters (i.e. betas). Number of parameters will
        vary depending on predictors included, number of runs, and number of
        artifacts that are modeled."""

    numTRs = len(big_X_groups)

    # Store each subset of DM columns in a list and then concatenate horz.
    DM_list = list()

    # Stim hemodynamic response
    if onsets_df is not None:
        DM_list = onsets_dm(DM_list, onsets_df, numTRs)

    # Motion regressors
    if motion_files is not None:
        DM_list = motion_dm(DM_list, motion_files, numTRs)

    # Artifacts
    if artifact_files is not None:
        DM_list = artifact_dm(DM_list, artifact_files, numTRs)

    # Run regressors & intercept
    DM_list = run_intercept_dm(DM_list, big_X_groups, numTRs)

    # Join columns of design matrix (concatenate horz.)
    DM = np.concatenate(DM_list, axis=1)

    return DM


def onsets_dm(DM_list, onsets_df, hrf, numTRs, onset_col='concat_onset',
    cond_col='condition', duration_col='duration'):
    """Generates components of design matrix that model neural activity
    associated with specific events (i.e., trials).

    Creates a box car specified by onsets and duration and then convolves
    the box car with the specified hemodynamic response function (HRF)

    Parameters
    ----------
    DM_list : list
        A list of design matrix components that will be concatenated
        horizontally. The function will append motion components to this list.

    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial. Returned by
        concat_onsets and remove_artifact_trials.

    hrf : numpy array (float)
        Vector of canonical hrf model (e.g., as returned by spm_hrf.m). Each
        datapoint in the vector corresponds to the canonical hemodynamic
        response at each time point (in TR-length bins) following trial onset.

    numTRs : int
        Total number of scans (TRs) in the dataset collapsing across all runs.

    onset_col : str
        Name of column in onsets_df to use for onsets (specified in scan units,
        not seconds). Default is 'concat_onset', which is added to onsets_df by
        the concat_onsets function.

    cond_col : str
        Name of column in onsets_df to use for condition labels. Default is
        'condition'.

    duration_col : str
        Name of column in onsets_df to use for durations (specified in scan
        units, not seconds). Default is 'condition'.

    Returns
    ----------
    DM_list : list
        Same as input but with motion components appended."""

    all_conditions = np.unique(onsets_df[cond_col])
    DM_trial_hrf = np.zeros((numTRs, num_cond_this_run))
    column = 0
    for condition in all_conditions:
        box_car = np.zeros((numTRs, 1))
        condition_onsets = onsets_this_run.ix[onsets_this_run[cond_col]
            == condition, onset_col]

        durations = onsets_this_run.ix[onsets_this_run[cond_col]
            == condition, duration_col]

        for start_row, this_duration in zip(condition_onsets, durations):
            rows2fill = start_row + range(int(this_duration))
            rows2fill = rows2fill.astype(int)
            box_car[rows2fill] = 1

        conv_w_hrf = np.convolve(box_car[:, 0], hrf[:, 0])
        DM_trial_hrf[:, column] = conv_w_hrf[0:numTRs]
        column += 1

    DM_list.append(DM_trial_hrf)

    return DM_list


def motion_dm(DM_list, motion_files, numTRs):
    """Generates component of design matrix that models motion artifacts.

    Called by get_dm.

    Parameters
    ----------
    DM_list : list
        A list of design matrix components that will be concatenated
        horizontally. The function will append motion components to this list.

    motion_files : list
        List of file paths to files containing motion parameters (produced by
        Lyman). Each element corresponds to a run. Length must equal # of runs.

    numTRs : int
        Total number of scans (TRs) in the dataset collapsing across all runs.

    Returns
    ----------
    DM_list : list
        Same as input but with motion components appended."""

    DM_motion_list = list()
    for file in motion_files:
        motion = pd.read_csv(file, index_col = 0)
        DM_motion_list.append(zscore(motion.iloc[:, 0:5], axis=0))

    DM_list.append(np.concatenate(DM_motion_list, axis=0))

    return DM_list


def artifact_dm(DM_list, artifact_files, numTRs):
    """Generates component of design matrix that models transient artifacts.

    Called by get_dm.

    Parameters
    ----------
    DM_list : list
        A list of design matrix components that will be concatenated
        horizontally. The function will append artifact components to this list.

    artifact_files : list
        List of file paths to files specifying scans with artifacts (produced by
        Lyman). Each element corresponds to a run. Length must equal # of runs.

    numTRs : int
        Total number of scans (TRs) in the dataset collapsing across all runs.

    Returns
    ----------
    DM_list : list
        Same as input but with artifact components appended."""

    # Concatenate artifact files across runs
    artifact_df = pd.DataFrame()
    for file in artifact_files:
        artifact_df = artifact_df.append(pd.read_csv(file), ignore_index=True)

    artifact_array = np.array(artifact_df)

    # Check that we have the correct number of TRs
    if artifact_array.shape[0] != numTRs:
        raise ValueError('Length of artifact files does not equal total'
             + ' number of TRs')

    # Sum across all types of artifacts (motion, spike, global intensity)
    artifact_vector = np.sum(artifact_array, axis=1)
    TRs_with_artifacts = np.where(artifact_vector > 0)[0]

    # Form matrix of artifacts covariates.
    if len(TRs_with_artifacts)>0:
        DM_artifacts = np.zeros((numTRs, len(TRs_with_artifacts)))
        DM_artifacts[TRs_with_artifacts, range(len(TRs_with_artifacts))] = 1

        DM_list.append(DM_artifacts)

    return DM_list


def run_intercept_dm(DM_list, big_X_groups, numTRs):
    """Generates component of design matrix that models mean differences across
    runs.

    Called by get_dm. Run effects will be modeled by N-1 dummy regressors and an
    intercept term.

    Parameters
    ----------
    DM_list : list
        A list of design matrix components that will be concatenated
        horizontally. The function will append run components.

    big_X_groups : numpy array (int)
        Vector specifying the run # for each scan. Length must equal the total #
        of scans (i.e., TRs). Returned by load_data.

    numTRs : int
        Total number of scans (TRs) in the dataset collapsing across all runs.

    Returns
    ----------
    DM_list : list
        Same as input but with run components appended."""

    runs = np.unique(big_X_groups)
    DM_run_intercept = np.zeros((numTRs, len(runs)))

    # Fill up to runs n-1. Last run will go into intercept.
    # Keep track of cols independently of run # in case user skips a run
    col = 0
    for this_run in runs[0:-1]:
        DM_run_intercept[big_X_groups==this_run, col] = 1
        col += 1

    # Last column/run
    DM_run_intercept[:, col] = 1

    DM_list.append(DM_run_intercept)

    return DM_list


def glm(DM, big_X, get_residuals=True):
    """Estimates a linear model of the fMRI time series independently at each
    voxel using OLS.

    Parameters
    ----------
    DM : numpy array (float)
        Design matrix for a linear model. nrows = # of scans (i.e., TRs).
        ncols = number of parameters (i.e. betas). Number of parameters will
        vary depending on predictors included, number of runs, and number of
        artifacts that are modeled. Returned by get_dm.

    big_X : numpy array (float)
        Data matrix of BOLD signal in same format as scikit-learn. nrows = # of
        scans (i.e., TRs). ncols = # of voxels.

    get_residuals : bool
        Will return residuals if True.

    Returns
    ----------
    betas : numpy array (float)
        Matrix of beta estimates. Each row corresponds to a parameter. (Number
        of rows in betas is the same as the number of columns in DM). Each
        column corresponds to a voxel.

    residuals : numpy array (float)
        Residuals of the model in same format as bix_X. Set to None if
        get_residuals is False."""

    pX = np.dot(inv(np.dot(DM.T, DM)), DM.T)
    num_vox = big_X.shape[1]
    num_betas = DM.shape[1]
    betas = np.zeros((num_betas, num_vox))

    if get_residuals:
        residuals = np.zeros(big_X.shape)
    else:
        residuals = None

    # Loop through voxels (cols of big_X)
    vox_num = 0
    for y_vec in big_X.T:
        beta_vec = np.dot(pX, y_vec)
        betas[:, vox_num] = beta_vec

        if get_residuals:
            residuals[:, vox_num] = y_vec - np.dot(DM, beta_vec)

        vox_num += 1

    return betas, residuals


def CV_LogReg_Permutation(X, y, groups, onsets_df, tot_num_vox = 500,
                          n_permutations=0, onset_col='onset'):
    """ Leave-one-run-out cross validation for logistic regression
    classification with permutation testing.

    Parameters
    ----------
    X : numpy array (float)
        Data matrix. Same format as X returned by avg_peak. Each row
        corresponds to a sample (i.e., a trial) and each column corresponds
        to a voxel. You can also use betas after removing parameters
        corresponding to artifacts of no interest.

    y : numpy array (int)
        Vector of binary class labels as returned by get_binary_y.

    groups : numpy array (int)
        Vector specifying the run # for each sample. Length must equal the total
        # of samples (i.e., trials).

    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial. Returned by
        concat_onsets and remove_artifact_trials

    tot_num_vox : int
        Number of voxels to select during univariate feature selection. Half
        will be the most active voxels for Category 1, the other half will be
        the most active voxels for category two (see balanced_feat_sel). Default
        = 500.

    n_permutations : int
        Number of permutations to run. For each permutation, the class labels
        are randomized within runs and the model is re-estimated. The first
        iteration is always the true (non-shuffled) data. Default is 0 (true
        data only).

    onset_col : str
        Name of column in onsets_df to use for onsets (specified in scan units,
        not seconds). Default is 'concat_onset', which is added to onsets_df by
        the concat_onsets function.

    Returns
    ----------
    auc_df : pandas dataframe
        Dataframe reporting area under the receiver operating characteristic
        curve (AUC) for each permutation. Reports each cross-validation fold
        in a separate row. Column labeled 'Run' records the left out test run
        for each fold.

    prob_df : pandas dataframe
        Dataframe reporting the probability output of the logistic regression
        model for each trial/sample (i.e., the estimated probability that the
        trial is an instance of the class labeled "1" - the first category when
        sorted in alphanumeric order if using get_binary_y). 'Run' records the
        run in which the trial occurred. 'Onset' records the onset for that
        trial specified in onsets_df (in scan units, not seconds). 'Category'
        reports the original string labels for the trial (i.e., the trial's true
        category membership)."""

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

        auc_df_fold, prob_df_fold = LogReg_Permutation(X_train, X_test,
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
    """ Logistic regression classification with permutation testing.

    Parameters
    ----------
    X_train : numpy array (float)
        Data matrix for training. Same format as X returned by avg_peak. Each
        row corresponds to a sample (i.e., a trial) and each column corresponds
        to a voxel. You can also use betas after removing parameters
        corresponding to artifacts of no interest.

    X_test : numpy array (float)
        Data matrix for testing; same format as
        X_train.

    y_train : numpy array (int)
        Vector of binary class labels as returned by
        get_binary_y for the training data.

    y_test : numpy array (int)
        Vector of binary class labels as returned by
        get_binary_y for the testing data.

    train_groups : numpy array (int)
        Vector specifying the run # for each sample in the training data. Length
        must equal the total # of samples (i.e., trials) within the training
        data.

    tot_num_vox : int
        Number of voxels to select during univariate feature selection. Half
        will be the most active voxels for Category 1, the other half will be
        the most active voxels for category two (see balanced_feat_sel). Default
        = 500.

    n_permutations : int
        Number of permutations to run. For each permutation, the class labels
        are randomized within runs and the model is re-estimated. The first
        iteration is always the true (non-shuffled) data. Default is 0 (true
        data only).

    onset_col : str
        Name of column in onsets_df to use for onsets (specified in scan units,
        not seconds). Default is 'concat_onset', which is added to onsets_df by
        the concat_onsets function.

    Returns
    ----------
    auc_df : pandas dataframe
        Dataframe reporting area under the receiver operating characteristic
        curve (AUC) for each permutation.

    prob_df : pandas dataframe
        Dataframe reporting the probability output of the logistic regression
        model for each trial/sample (i.e., the estimated probability that the
        trial is an instance of the class labeled "1" - the first category when
        sorted in alphanumeric order if using get_binary_y)."""

    y_train, y_test = get_binary_y(y_train), get_binary_y(y_test)

    auc_df = pd.DataFrame(columns = ['Run'] + \
        ['AUC Permutation {}'.format(x) for x in xrange(n_permutations+1)])

    prob_df = pd.DataFrame(index=range(len(y_test)),
        columns = ['Run', 'Onset', 'Category'] + \
        ['Prob Permutation {}'.format(x) for x in xrange(n_permutations+1)])

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
    """Loads fMRI data from NIFTI images.

    Uses a specified ROI file and and creates a data matrix for use in logistic
    regression classification. Z-scores individually at each voxel, separately
    for each run.

    Parameters
    ----------
    bold_files : list
        List of file paths to bold files. Assumes each bold file is a 4D brick
        corresponding to a single run.

    roi_file : str
        File path to ROI mask image. Used by NiftiMasker to load the data.

    check_mask : bool
        Whether to check for out of brain voxels within the functional data.
        Default = True. If true, any voxels with an NaN at any time point will
        be excluded.

    Returns
    ----------
    big_X : numpy array (float)
        Data matrix of the continuous fMRI time series. Each row corresponds to
        a time point (TR) and each column corresponds to a voxel.

    big_X_groups : numpy array (int)
        Vector specifying the run # for each scan (TR) Length must equal the
        total # of scans (i.e., TRs).

    original_mask_cols : numpy array (int)
        For each column in big_X, specifies the column index in the original
        data matrix before out of brain voxels were excluded. Provides a
        reference to the original ROI mask:

        big_X = big_X_original[:, original_mask_cols]

        Returned only if check_mask = True."""

    # Loop through runs, get bold data, standardize within run
    num_runs = len(bold_files)
    list_of_bold_matrices = list()
    list_of_groups = list()
    run = 1
    for this_bold_file in bold_files:
        print('Loading file {}'.format(this_bold_file))

        # Apply mask to bold data and standardize (z-score individually at each
        # voxel, separatley for each run)
        func_masker = NiftiMasker(mask_img=roi_file, smoothing_fwhm=None,
                                  standardize=True)
        bold_2D = func_masker.fit_transform(this_bold_file)
        numTRs = bold_2D.shape[0]
        list_of_bold_matrices.append(bold_2D)

        list_of_groups.append(np.ones((numTRs))*run)

        run += 1

    big_X = np.concatenate(list_of_bold_matrices, axis=0)
    big_X_groups = np.concatenate(list_of_groups, axis=0)

    if check_mask:
        # Check for out of brain voxels from any run. Z-scores from out of brain
        # voxels will be equal to nan. Take sum across time and excluded any
        # voxels where sum is nan.
        vox_sum_vec = big_X.sum(axis=0)
        exclude_vox_cols = np.where(np.isnan(vox_sum_vec))
        original_mask_cols = np.where(np.logical_not(np.isnan(vox_sum_vec)))
        big_X = np.delete(big_X, exclude_vox_cols, axis=1)

        return big_X, big_X_groups, original_mask_cols

    else:
        return big_X, big_X_groups


def concat_onsets(onsets_file, bold_files, TR=2, cond_col='condition',
                  onset_col='onset', duration_col='duration'):
    """Concatenate onsets across runs.

    Creates a new column in in the onsets dataframe named "concat_onsets" where
    all onsets are referenced to the first scan of the first run. These new
    onsets are correct indices for a data matrix that is concatenated across
    runs (e.g., big_X returned by load_data). Converts both onsets columns to
    scan (TR) units rather than seconds. (Note that this assumes each onset is
    synched with the start of a scan).

    Parameters
    ----------
    onsets_file : str
        File path to csv file containing table of onsets (same format as Lyman)

    bold_files : list
        List of file paths to bold files. Assumes each bold file is a 4D brick
        corresponding to a single run.

    TR : float
        TR (in seconds). Used to convert onsets specified in seconds to scan
        (TR) units. Default = 2.

    cond_col : str
        Label for the column in the onsets dataframe coding the condition
        labels. Default = 'condition'.

    onset_col : str
        Label for the column in the onsets dataframe coding the onsets (in
        seconds). Default = 'onset'.

    duration_col : str
        Label for the column in the onsets dataframe coding the durations (in
        seconds). Default = 'duration'.

    Returns
    ----------
    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial."""

    onsets_df = pd.read_csv(onsets_file)
    onsets_df[onset_col] = np.round(onsets_df[onset_col]) / TR
    onsets_df[duration_col] = np.round(onsets_df[duration_col]) / TR

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

def remove_artifact_trials(onsets_df, artifact_files, window=6):
    """Removes trials flagged as having artifacts from the onsets dataframe.

    Parameters
    ----------
    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial. Returned by
        concat_onsets.

    artifact_files : list
        List of file paths to csv files containing data on which scans have
        been flagged for artifacts (same format as Lyman). Length should equal
        the number of runs.

    window : int
        Length of the window for modeling each trial (in TR units). If a scan
        falling within the window is flagged as having an artifact, the trial
        will be dropped. As a general rule, the window should include the
        duration of the trial plus any active baseline or inter-trial interval.
        Default = 6.

    Returns
    ----------
    onsets_no_art_df : pandas dataframe
        Same format as onsets_df but with rows coding trials with artifacts
        removed."""

    # Concatenate artifact files across runs
    artifact_df = pd.DataFrame()
    for file in artifact_files:
        artifact_df = artifact_df.append(pd.read_csv(file), ignore_index=True)

    artifact_array = np.array(artifact_df)

    # Sum across all types of artifacts (motion, spike, global intensity)
    artifact_vector = np.sum(artifact_array, axis=1)
    TRs_with_artifacts = np.where(artifact_vector > 0)[0]

    # Loop through trials, exclude any trials with an artifact occuring during
    # its window.
    onsets_no_art_df = pd.DataFrame()
    for index, row in onsets_df.iterrows():
        onset = row['concat_onset']
        trial_TRs = onset + np.arange(window)

        # Check for intersection
        if any(set(trial_TRs) & set(TRs_with_artifacts)):
            continue # drop this trial
        else:
            onsets_no_art_df = onsets_no_art_df.append(row, ignore_index=True)

    return onsets_no_art_df

def avg_peak(big_X, onsets_df, peak_TRs=[2, 3, 4], standardize=True):
    """Reduces the BOLD signal for a given trial to a scalar value by averaging
    across a specified window.

    This is a potential alternative to using a canonical hemodynamic response
    function.

    Parameters
    ----------
    big_X : numpy array (float)
        Data matrix of the continuous fMRI time series. Each row corresponds to
        a time point (TR) and each column corresponds to a voxel.

    onsets_df : pandas dataframe
        Specifies onsets and condition labels for each trial. Returned by
        concat_onsets and remove_artifact_trials.

    peak_TRs : list
        Scans following trial onset to include in the average. Uses 0-based
        indexing with respect to the start of the trial. Default = [2, 3, 4].

    standardize : bool
        If true, will z-score the output data matrix across trials individually
        at each voxel (collapsing across all runs). Default = True.

    Returns
    ----------
    X : numpy array (float)
        Data matrix. Each row corresponds to a sample (i.e., a trial) and each
        column corresponds to a voxel."""

    X = np.zeros((len(onsets_df.index), big_X.shape[1]))

    # Loop through trials and average time points
    X_row = 0
    for index, row in onsets_df.iterrows():
        onset = row.concat_onset
        onsets2avg = (np.array(peak_TRs) + onset).astype(int)

        X[X_row, :] = np.mean(big_X[onsets2avg, :], axis = 0)

        X_row += 1

    if standardize:
        X = zscore(X, axis=0)

    return X


def get_binary_y(y_str):
    """Converts string category labels to binary category labels needed by
    scikit-learn.

    String labels are sorted in alphanumeric order. The first label is assigned
    0 and the second label is assigned 1.

    Parameters
    ----------
    y_str : numpy array or list
        Category labels for each sample (i.e., trial). Length should equal #
        of trials.

    Returns
    ----------
    y : numpy array
        A binary version of y_str."""

    y_str = np.array(y_str)
    labels = np.unique(y_str)

    if len(labels) > 2:
        raise ValueError('More than 2 category labels found in y_str.')

    y = np.zeros(y_str.shape)
    y[y_str==labels[0]], y[y_str==labels[1]] = 0, 1

    return y


def balanced_feat_sel(X_train, y_train, tot_num_vox):
    """ Univariate feature selection with an equal # of selected voxels for
    Category 1 and Category 2.

    A t-test is performed independently at each voxel. Given a specified total
    number of voxels (N), the top N/2 voxels for Category 1 and the top N/2
    voxels for Category 2 are selected.

    Parameters
    ----------
    X_train : numpy array (float)
        Data matrix for training. Same format as X returned by avg_peak. Each
        row corresponds to a sample (i.e., a trial) and each column corresponds
        to a voxel. You can also use betas after removing parameters
        corresponding to artifacts of no interest.

    y_train : numpy array (int)
        Vector of binary class labels as returned by get_binary_y for the
        training data.

    tot_num_vox : int
        Number of voxels to select during univariate feature selection. Half
        will be the most active voxels for Category 1, the other half will be
        the most active voxels for category 2. Default = 500.

    Returns
    ----------
    voxels2use : numpy array (bool)
        Boolean index specifying which voxels (columns of X_train) were
        selected.

    cat0_voxels : numpy array (bool)
        Boolean index specifying which voxels (columns of X_train) were
        selected for category 0.

    cat1_voxels : numpy array (bool)
        Boolean index specifying which voxels (columns of X_train) were
        selected for category 1."""

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
    """Shuffle category labels randomly separately for each run.

    Parameters
    ----------
    y_train : numpy array (int)
        Vector of binary class labels as returned by get_binary_y for the
        training data.

    train_groups : numpy array (int)
        Vector specifying the run # for each sample in the training data. Length
        must equal the total # of samples (i.e., trials) within the training
        data.

    Returns
    ----------
    y_train_shuf : numpy array (int)
        Same as y_train but shuffled randomly separately for each run."""

    y_train_shuf = y_train.copy()
    for run in np.unique(train_groups):
        rows2shuffle = np.array(train_groups) == run
        y_train_shuf[rows2shuffle] = \
            np.random.permutation(y_train_shuf[rows2shuffle])

    return y_train_shuf
