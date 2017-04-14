from SML_mvpa_scripts import *
from glob import glob

subject = 'am45'

bold_files = \
    glob('/share/awagner/AM/analysis/study/{}/reg/epi/unsmoothed/run_*/timeseries_xfm.nii.gz'.format(subject))
bold_files.sort()

roi_file = \
    '/share/awagner/AM/data/{}/masks/parahippocampal.nii.gz'.format(subject)

onsets_file = '/share/awagner/AM/data/{}/behav/study.csv'.format(subject)

big_X, original_mask_cols = load_data(bold_files, roi_file)

onsets_df = concat_onsets(onsets_file, bold_files)
X = avg_peak(big_X, onsets_df, peak_TRs=[2, 3, 4])
y = onsets_df.condition
groups = onsets_df.run

auc_df, prob_df = CV_LogReg_Permutation(X, y, groups, onsets_df, \
    n_permutations=500)
    
auc_df.to_csv('am45_auc.csv')
prob_df.to_csv('am45_prob.csv')
