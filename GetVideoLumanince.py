import cv2
import pandas as pd
import os

# this code was used to grab the luminance value for each frame in a video from it's hsv color space

def save_frame_df(file, file_path, out_path):
    # log the start for each file, this could be a long process depending on file size
    print("starting: {}".format(file_path))

    # use open cv to load the video
    cap = cv2.VideoCapture(file_path)

    # get the video fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_no = 0
    success, frame = cap.read()

    hues = []
    saturations = []
    values = []
    frames = []
    time_stamps = []

    # get the hsv color space for the first frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # average it
    hsv = cv2.mean(hsv)[:3]
    # append the values
    hues.append(hsv[0])
    saturations.append(hsv[1])
    values.append(hsv[2])
    frames.append(frame_no)
    time_stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    # while there are additional frames
    while success:
        # read in the frame
        success, frame = cap.read()
        if success:
            frame_no += 1

            # get the hsv color space for the frame and average it
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.mean(hsv)[:3]

            # record the values
            hues.append(hsv[0])
            saturations.append(hsv[1])
            values.append(hsv[2])
            frames.append(frame_no)
            time_stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))

    # make a dictionary that will later be saved as a dataframe
    video_dict = {
        'video_name': [file_path] * len(values),  # duplicating cells as an easy way to have a df contain all needed info
        'fps': [fps] * len(values),
        'was_replay': [was_replay] * len(values),
        'time_stamp': time_stamps,
        'hue': hues,
        'saturation': saturations,
        'value': values
    }

    # save the information with the same name as the video, but as a csv
    save_path = os.path.join(out_path, file.replace('.mp4', ".csv"))

    df = pd.DataFrame.from_dict(video_dict)
    df.to_csv(save_path)

    print("\tsaving: {}\n\t{}".format(save_path, file_path))


video_path = './VideosRaw/'
file_names = os.listdir('./VideosRaw/')
out_path = './video_luminance'

# make an output directory if there is not one
if not os.path.exists(out_path):
    os.mkdir(out_path)

# for each video file in the folder get the color information
# during the dataset collection each video was shown twice, once as a normal video, once as a blurred video
# this records which version the values are for
for file in file_names:
    was_replay = False
    if 'Blurred' in file:
        was_replay = True

    file_path = os.path.join(video_path, file)
    save_frame_df(file, file_path, out_path)
