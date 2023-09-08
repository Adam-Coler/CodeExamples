# when collecting the dataset some data was lost when there was an incorrectly placed cast from a 3-tuple to a 2-tuple
# as the 3-tuples represented unit vectors this code was able to generate the missing data points
# this cast was corrected for most subjects

import numpy as np

def pythagorean_theorem_find_missing(x, y):
    x = np.abs(x)
    y = np.abs(y)
    z = np.sqrt(1 - np.power(x, 2) - np.power(y, 2))
    return np.round(z, 6)

# a df was loaded with the gaze data for a subject
# this code applied the function to all the row values for the known data points down the column (axis 1)
# the result was a new column populated with the computed values
df['rightGazeForward_z'] = df.apply(lambda row: pythagorean_theorem_find_missing(row['rightGazeForward_x'], row['rightGazeForward_y']), axis=1)