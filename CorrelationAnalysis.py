import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

# when features were extracted they were done over n second windows
# this code would make a labeled heat map for the correlation between features and labels (and each other)
# these correlations helped to
# 1: indicate errors in features (i.e., when two unrelated features were perfectly correlated)
# 2: see which features had the strongest linear relationships to the features

for segment_size in [2, 5, 15]:  # for features computed on 2, 5, and 10 second windows

    # load the helper class from a file that was generated when features were extracted for a n second window
    details = MLFormattedDetails.load_file(segment_size)
    # load a pandas dataframe (df) for the participants that had good data for both types of labeling tasks
    df = details.return_all_participants_data(details.good_both)

    # retrieve the label columns
    labels = [item for item in df.columns if "label" in item]

    # controller movements were collected but were not a good predictor of affect so they are excluded
    # retrieve feature columns without controller features (including label columns)
    features = [item for item in df.columns if "controller" not in item]

    # trim the df to only have intended features and labels
    df = df[features]

    # convert label column data into categorical data
    for label in labels:
        df[label] = df[label].astype('category').cat.codes

    # there are different ways to look at correlation, because our data is non-parametric we are using non-parametric correlation analysis
    for method in ['kendall', 'spearman']: # ‘pearson’  is parametric. kendall and spearman are non-parmetric, kendall examines dependance not corr

        # compute the correlation matrix and round the results using pandas
        corr = df.corr(method=method).round(2)

        # labels included valance for 2 and 3 class, arousal for 2 and 3 class, and quadrant labels for hi/low arousal/valance
        # make a list of the labels we are looking at
        labels = [item for item in df.columns if "valance" in item or 'arousal' in item or 'quadrant' in item and 'label' in item]  # labels for main targets

        all_labels = [item for item in df.columns if "label" in item]

        # to simplify analysis I only want to see features that had above a given threshold of correlation
        # get the parts of the correlation df where values were correlated above the threshold
        thresh = .099
        tmp = (corr.where(np.abs(corr.values) > thresh))

        # This turns the n by n matrix into a feature by label matrix to make a smaller figure
        tmp = tmp[labels]
        tmp = tmp.drop(all_labels)
        tmp = tmp.dropna(how='all')
        tmp = tmp.sort_index()

        # seaborn is built on matplotlib
        # setting up the matplotlib attributes to make for a nice figure
        plt.rcParams['savefig.dpi'] = 150
        # not all feature by label matrices were the same
        # after trial and error a figure size of these dimensions was consistently readable
        plt.figure(figsize=(len(labels) * 1, int(tmp.shape[0] * .3)))

        # use seaborn to generate a heatmap
        sb.heatmap(tmp, cmap="Blues", annot=True)
        plt.tight_layout()

        # save the heatmap
        plt.savefig("{}_{}_{}_{}_segment_correlation".format(method, 'affect', segment_size, str(thresh).replace(".", "p")))
        plt.close()

        # indicate that it was saved
        print("Saved: {}_{}_{}_{}_segment_correlation".format('affect', segment_size, method, str(thresh).replace(".", "p")))
