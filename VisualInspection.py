import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# responsed_df = participant data

for i in responses_df.pid.unique():

    # get the data for the single participant from the data from all participants
    tmp_df = responses_df[responses_df['pid'] == i]

    # loop for each type of label
    for score_type in ["arousal", "valance"]:

        # make a plot and set its size
        fig, ax = plt.subplots()
        fig.set_size_inches(18.5, 7.5)

        # generate a label
        response_name = "user_" + score_type

        # group the data by the expected label
        groups = tmp_df.groupby('video_quadrant_' + score_type)

        ax.margins(0.05)

        # add a line to represent neutral
        plt.axhline(y=4, color='r', linestyle='-')

        # for each group plot the points in that group as a different color
        for name, group in groups:
            ax.plot(group.video, group[response_name], marker='o', linestyle='', ms=12, label=name)

        # collect summery info for how often that person labeled a stimuli as high, low, or medium
        # the intended classes were equally distributed
        count_low = (len(tmp_df[response_name][tmp_df[response_name] < 4]))
        count_mid = (len(tmp_df[response_name][tmp_df[response_name] == 4]))
        count_high = (len(tmp_df[response_name][tmp_df[response_name] > 4]))

        # add information to the chart
        ax.legend(numpoints=1, loc='lower left')
        plt.title("{} SAM reported {}\nLow: {}, Med: {}, High: {}".format(i, response_name.replace("user_", ""), count_low, count_mid, count_high), size=30)
        plt.xlabel("Video", size=25)
        plt.ylabel("{}".format(response_name.replace("user_", "")), size=25)

        # make the ticks more visible
        plt.xticks(size=22, rotation=45)

        # rename the ticks
        plt.yticks(range(0, 9), size=22, rotation=0)
        plt.show()