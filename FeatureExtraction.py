from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# an example of some feature extraction performed

def append_col_names(df: pd.DataFrame, append_with: str):
    # conviance function to rename columns in a pd dataframe
    col_dict = {}
    for col in df.columns:
        col_dict[col] = "{}_{}".format(col, append_with)

    df = df.rename(columns=col_dict)
    return df


def min_max_scale(df, cols_to_duplicate=[]):
    """
    performs min max scaling across a df with an option to duplicate and preserve the raw values of provided columns for later analysis
    :param df: df of signals normalize
    :param cols_to_duplicate: columns to duplicate and retain raw versions of
    :return: normalized df with additional raw duplicated columns
    """
    cols = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled, columns=cols)

    if cols_to_duplicate != []:
        df_duplicated = df[cols_to_duplicate]
        df_duplicated = append_col_names(df_duplicated, "raw")

        df_normalized = pd.concat(
            [
                df_normalized.reset_index(drop=True),
                df_duplicated.reset_index(drop=True)
            ], axis=1)

    return df_normalized

# set up the columns that I want to generate these features for
# in this case the normalized vectors representing the gaze direction
cols_mean_median_max_min_std = [
        'gazeForward_x',
        'gazeForward_y',
        'gazeForward_z']

# a full df is already loaded that has all of the features
# the tmp_df will only what the ones that we want to get features for
tmp_df = df[cols_mean_median_max_min_std]

# based on the file being used I need to find how many rows should be grouped together
group_size = samples_per_second * segment_length_in_seconds

# using pandas group by we group the data by index into groups of group_size
tmp_df = tmp_df.groupby(tmp_df.index // group_size)

# using pandas to compute group level measures
df_mean = tmp_df.mean()
df_std = tmp_df.std()
df_min = tmp_df.min()
df_max = tmp_df.max()
df_median = tmp_df.median()

# append the column names with the feature that they now represent
df_mean = append_col_names(df_mean, "mean")
df_std = append_col_names(df_std, "std")
df_min = append_col_names(df_min, "min")
df_max = append_col_names(df_max, "max")
df_median = append_col_names(df_median, "median")

# merge the extracted features into a single dataframe
df_summary_stats = pd.concat([
    df_mean.reset_index(drop=True),
    df_std.reset_index(drop=True),
    df_min.reset_index(drop=True),
    df_max.reset_index(drop=True),
    df_median.reset_index(drop=True)
],axis=1)

# free up some memory
del (cols_mean_median_max_min_std, tmp_df, df_mean, df_std, df_min, df_max, df_median)

# creating a feature for principle compentent analysis generated feature for the total rotation of the eye
motion_list = list(zip(signal_gaze['gaze_raw_pitch'], signal_gaze['gaze_raw_roll'], signal_gaze['gaze_raw_yaw']))  # conversion of x,y,z to one list
pca = PCA(1)
pca.fit(motion_list)
motion_list_tmp = np.reshape(pca.transform(motion_list), -1)
signal_gaze['PCA_motion_gaze'] = motion_list_tmp.tolist()

signal_gaze = min_max_scale(signal_gaze, cols_to_duplicate=['gaze_pitch', 'gaze_yaw']) # gaze pitch and yaw are needed later in the pipeline