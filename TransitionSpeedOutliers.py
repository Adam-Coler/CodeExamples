from scipy.signal import savgol_filter
import numpy as np

# transition speed outliers were removed for eye movements and pupil dilation speeds
# points with abnormal transition speeds could represent sensor errors


def smooth_and_remove_transition_speed_mad_outliers(_signal, _times, mad_deviations=3.5):
    """
    Remove Mean Absolute Deviation (MAD) outliers from a series based on a savgol smoothed version of that series
    :param _signal: the time series data
    :param _times: the time data
    :param mad_deviations: mean absolute deviations away to remove points at
    :return: the adjusted series and times
    """
    # check that input is appropriate
    if type(_signal) != np.ndarray:
        _signal = _signal.replace(0, np.nan)

    # set up a window size for the filter, this was set based on experimentation with different window sizes
    window = int(len(_times) / (max(_times) * 3))

    polynomial_order = 2

    # error checking to make sure the window is larger than the polynomial_order and that the window was odd
    if window < polynomial_order:
        window = polynomial_order + 1

    if window % 2 == 0:  # or window += not (window & 1)
        window += 1

    # get an array representing the moving average done by:
    signal_moving_average = np.convolve(_signal, np.ones(window), 'valid')  # moving average

    # use scipy's savgol filer
    signal_savgol = savgol_filter(signal_moving_average, window, polynomial_order)

    # the new time stamps = old time stamps less half the window size from the front and back
    times_savgol = np.array(_times[window // 2:len(_times) - window // 2])

    # same for the signal
    _signal = _signal[window // 2:len(_signal) - window // 2]

    # get the absolute differences between the signal and the filtered signal
    differences = (np.abs(_signal - signal_savgol))

    # set up to get the transition differences forwards and backwards along the series
    left_diff, right_diff = differences[:-1], differences[1:]

    # take the maximum of the two
    max_diff = np.maximum(left_diff, right_diff)

    # subtract the mean
    max_diff -= np.nanmean(max_diff)
    max_diff = np.array(max_diff)
    # compute the threshold based on the deviation limits placed (3.5)
    threshold = np.nanmean(max_diff) + mad_deviations * np.nanmean(np.abs(max_diff))

    # find where the values are above the threshold
    indices = np.argwhere(max_diff > threshold)
    indices = indices.flatten()

    # remove those indices
    out_series = np.delete(signal_savgol, indices)
    out_times = np.delete(times_savgol, indices)

    return out_series, out_times