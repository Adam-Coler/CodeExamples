import numpy as np

class InterpolationStruct:
    def __init__(self, index_last, index_next, transform, interpolated_time_stamp):
        """
        This class holds the data needed to apply a liner transform to a single point
        :param index_last: The last points index in the signal
        :param index_next: The next points index in the signal
        :param transform: The transform to apply
        :param interpolated_time_stamp:  The new time stamp
        """
        self.index_last = index_last
        self.index_next = index_next
        self.transform = transform
        self.interpolated_time_stamp = interpolated_time_stamp

    def __str__(self) -> str:
        return "Transform: {} to reach time {}, Last index: {}, Next index: {}, duration: {}.".format(self.transform, self.interpolated_time_stamp, self.index_last, self.index_next)


# This was in a separate class that has a number of time conversions
def second_to_nanosecond(seconds: float) -> int:
    return int(seconds * 1000000000)


class ResampleTimeseries:
    def __init__(self, desired_start_time, desired_end_time, desired_time_step, input_signal, input_time_stamps):
        """
        This class takes a given signal and sets it up to be linearly interpolated
        :param desired_start_time: when the resampled signal should start
        :param desired_end_time: when the resampled signal should stop
        :param desired_time_step: the new sample rate
        :param input_signal: signal to be resamples
        :param input_time_stamps: time stamps for the original signal
        """

        # convert times to nanoseconds
        desired_start_time = second_to_nanosecond(desired_start_time)
        desired_end_time = second_to_nanosecond(desired_end_time)
        desired_time_step = second_to_nanosecond(desired_time_step)

        self.start_time = desired_start_time
        self.end_time = desired_end_time
        self.time_step = desired_time_step

        # convert inputs into np arrays
        self.input_signal = np.array(input_signal)
        self.input_time_stamps = np.array(input_time_stamps)

        # generate the new time stamps between start and end using step time_step
        self.resampled_timestamps = np.array(range(self.start_time, self.end_time, self.time_step))

        # make an empty object array to hold the interpolation structures
        self.interpolation_structs = np.empty(len(self.resampled_timestamps), dtype=object)

        # place holder
        self.resampled_signal = None

        # conduct the resampling
        self._up_sample_prep()
        self._apply_transforms()

    def _up_sample_prep(self):

        # the current actual time stamp
        ts_index = 1

        # for each new time stamp
        for i in range(1, len(self.resampled_timestamps) - 1):
            time_goal = self.resampled_timestamps[i]

            # initialize to a lower index
            time_actual_last = self.input_time_stamps[ts_index - 1]
            time_actual_next = self.input_time_stamps[ts_index]

            # move to the appropriate actual time stamp
            while time_actual_next < time_goal + 1 and ts_index < len(self.input_time_stamps) - 1:
                ts_index += 1
                time_actual_next = self.input_time_stamps[ts_index]
                time_actual_last = self.input_time_stamps[ts_index - 1]

            # compute the transform required to get from the old time stamp to the new time stamp
            if time_actual_last == time_goal:
                transform = 1
            else:
                units = abs(time_actual_next - time_actual_last)  # unit differance
                distance = abs(time_goal - time_actual_last)  # traveled distance between original and new times
                transform = (distance / units)  # a ratio representing the transform between the original point and the new point given the distance and units

            # add information to the array of interpolation structs
            self.interpolation_structs[i] = InterpolationStruct(ts_index - 1, ts_index, transform, time_goal)

        # the first value is estimated based on the next, at 200 Hz this is a small amount of error
        self.interpolation_structs[0] = self.interpolation_structs[1]

        # start at the last index
        start_i = len(self.interpolation_structs) - 1

        # and work backwards until the last valid data point is found
        # this covers the case where the original time stamps cover less ground than the new time stamps
        while self.interpolation_structs[start_i] is None:
            start_i -= 1

        # estimate the units based on the last two valid time stamps
        units = abs(self.input_time_stamps[-2] - self.input_time_stamps[-1])

        # compute the rest of the interpolation structs
        # this could lead to spikes at the end of the transformed signal
        # in practice the amount of missing data was limited so spikes did not occur
        for i in range(start_i + 1, len(self.interpolation_structs)):

            distance = (self.resampled_timestamps[i] - self.input_time_stamps[-1])
            transform = (distance / units)

            self.interpolation_structs[i] = InterpolationStruct(-2, -1, transform, self.resampled_timestamps[i])

    def _apply_transforms(self):
        self.resampled_signal = []
        # run through each struct and resample based on the two points nearest the new time stamp and the transform
        for i in range(0, len(self.interpolation_structs)):
            struct = self.interpolation_structs[i]

            value_lower = self.input_signal[struct.index_last]
            value_upper = self.input_signal[struct.index_next]
            transform = struct.transform
            self.resampled_signal.append((value_upper - value_lower) * transform + value_lower)

        return self.resampled_signal


# given more time I would love to optimize this to use less for loops and more matrix operations
# that is a low priority as the current run time is manageable and there are more pressing
# tasks in the project 
