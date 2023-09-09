# Credit for the ordiginal implementation of these models goes to https://github.com/cwi-dis/CEAP-360VR-Dataset

from AffectML import FileDetails
import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras import optimizers
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os


class Results:
    """
    This class was used to record results from different model combinations
    In the full version of this class there are methods for loading json files into results class instances
    """
    def __init__(self):
        self.number = 0
        self.model_type = ""
        self.label_type = ""
        self.results = {}
        self.feature_segment_length = 0
        self.segments_fed = 0
        self.features = []
        self.mean_test_acc = 0
        self.mean_train_acc = 0
        self.mean_f1 = 0

    def to_json(self):
        # convert self to a json friendly format to save to a file
        dic = {'number': self.number,
               'model_type': self.model_type,
               'label_type': self.label_type,
               'feature_segment_length': self.feature_segment_length,
               'segments_fed': self.segments_fed,
               'features': self.features,
                'mean_test_acc': self.mean_test_acc,
                'mean_train_acc': self.mean_train_acc,
                'mean_f1': self.mean_f1,
               'results': self.results
               }
        dic = json.dumps(dic)
        return dic

    def save(self):
        file_name = "./results_10fold_all/{}_{}_{}_{}.json".format(self.model_type, self.label_type, self.number, self.feature_segment_length)
        with open(file_name, "w") as outfile:
            outfile.write(self.to_json())


def init_tf_gpus():
    # initialize the GPU for running the models
    # print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def create_mode_deep_learning(model_type, num_classes, num_features):

    window_size = 1 # set to one because this script focuses on testing the extracted features for n second windows
    input_signals = Input(shape=(window_size, num_features))  # input(shape(window by number of features))

    if num_classes == 2:
        activation = "sigmoid"
    else:
        activation = "softmax"

    # make a model
    # using TensorFlow models
    if model_type == "LSTM":
        x = LSTM(window_size)(input_signals)
        x = Dense(num_classes, activation=activation)(x)  # if two class use a sigmoid activation
    elif model_type == "1DCNN":
        # if we are using a 1D cnn set up layers
        x = Conv1D(4, 256, activation='relu', input_shape=(window_size, num_features), padding="same")(input_signals)
        x = Conv1D(8, 128, activation='relu', padding="same")(x)
        x = Conv1D(32, 64, activation='relu', padding="same")(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(num_classes, activation=activation)(x)  # if two class use a sigmoid activation
    else:
        print("Not a supported model")
        exit(0)
    model = Model(input_signals, x)
    return model


def create_model_ML(model_type):
    # set up models using TensorFlow models
    if model_type == "SVM":
        model = svm.SVC(decision_function_shape="ovr")
    elif model_type == "RF":
        model = RandomForestClassifier(max_depth=4) # random_state=80, if a seed is set tests are consistent
    elif model_type == "NB":
        model = GaussianNB()
    elif model_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_type == "LR":
        model = LogisticRegression(max_iter=3000)
    else:
        print("Not a supported model")
        exit(0)
    return model


def train_model_deep_learning(train_x, train_y, model_type):
    init_tf_gpus()

    numberOfClasses = max(train_y) + 1

    # print(train_x.shape)
    model = create_mode_deep_learning(model_type, numberOfClasses, train_x.shape[-1])

    train_y = to_categorical(train_y, int(max(train_y) + 1))

    if numberOfClasses == 2:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"

    # set learning rate for gradient decent
    RMSprop = optimizers.RMSprop(learning_rate=0.001)

    # set up model parameters
    model.compile(loss=loss,
                  optimizer=RMSprop,
                  metrics=["acc"])

    # fit model with set batch size and epochs
    model.fit(train_x, train_y,
              batch_size=256,
              # callbacks=callbacks,
              epochs=50,
              verbose=0)
    return model


def train_model_ML(train_x, train_y, model_type):
    train_y = train_y[:, 0]
    model = create_model_ML(model_type)
    model.fit(train_x, train_y)
    return model

def load_data(model_type, subject_id, details):
    # using the helper class we load the df for one participant
    # this df only has the included features and a single label column
    data = details.return_df_with_included_signals(subject_id)

    # for reshaping the df we store the first and last feature columns
    range_index = [0, len(data.columns) - 2]

    # the label column is always the last one
    label_column_number = len(data.columns) - 1

    # set up the test features and labels using the subject that was left out
    test_x = np.array(data)[:, range_index[0]:range_index[1]]
    test_y = np.array(data)[:, label_column_number].reshape([-1, 1])

    # set up train and test multidimensional arrays
    train_x = np.zeros([0, range_index[1] - range_index[0]])
    train_y = np.zeros([0, 1])

    # iterate through each subject that had good data for both types of labels
    for pid in details.good_both:
        # if it's not the subject we are leaving out
        if pid != subject_id:
            # load their data
            df = details.return_df_with_included_signals(i)
            # format their data
            d_x = np.array(df)[:, range_index[0]:range_index[1]]
            d_y = np.array(df)[:, label_column_number].reshape([-1, 1])
            # add it to the total train test set
            train_x = np.concatenate((train_x, d_x), axis=0)
            train_y = np.concatenate((train_y, d_y), axis=0)

    if model_type in ["LSTM", "1DCNN"]: # if we are not using features, reshape to feed more time
        segments_to_feed = 1 # These tests all used computed features over n second windows so this is fixed at one
        test_x = test_x.reshape([-1, segments_to_feed, range_index[1] - range_index[0]])
        train_x = train_x.reshape([-1, segments_to_feed, range_index[1] - range_index[0]])

        train_y = train_y.reshape([-1, segments_to_feed])
        train_y = np.round(np.mean(train_y, axis=1)).reshape([-1, 1])

        test_y = test_y.reshape([-1, segments_to_feed])
        test_y = np.round(np.mean(test_y, axis=1)).reshape([-1, 1])

    return train_x, np.array(train_y, dtype='int'), test_x, np.array(test_y, dtype='int')

def subject_independent_run_model(pid, model_type, details):

    train_x, train_y, test_x, test_y = load_data(model_type, pid, details)

    train_y, result_train, test_y, result_test = to_result(model_type, train_x, train_y, test_x, test_y)

    return train_y, result_train, test_y, result_test


def to_result(model_type, train_x, train_y, test_x, test_y):
    if model_type in ["SVM", "RF", "NB", "KNN", "LR"]:
        model = train_model_ML(train_x, train_y, model_type)
        # print(model.feature_importances_)
        result_train = model.predict(train_x)
        result_test = model.predict(test_x)
    elif model_type in ["1DCNN", "LSTM"]:
        model = train_model_deep_learning(train_x, train_y, model_type)
        result_train = np.argmax(model.predict(train_x), axis=1)
        result_test = np.argmax(model.predict(test_x), axis=1)

    return train_y, result_train, test_y, result_test


def leave_one_subject_out(number, segment_length_or_segements_to_feed, model_type, details):
    # subject independent model,leave-one-subject-out cross validation
    # these models are predicting user affect, leave one subject out cross validation is less accurate than leaving out certain stimuli but
    # it provides a nice image of how generalizable the model's performance is

    ACC_train = []
    ACC_test = []
    F1_test = []

    all_test_y = []
    all_train_y = []
    all_test_result = []
    all_train_result = []

    # results are being saved to a results class that is then saved to a json file
    # with this set up I could run many hours of comparisons overnight and skim the results the next day
    my_results = Results()
    my_results.number = number
    my_results.model_type = model_type
    my_results.segments_fed = segment_length_or_segements_to_feed
    my_results.label_type = details.desired_label
    my_results.features = details.included_features
    my_results.feature_segment_length = details.samples_per_second

    # some participants have bad data or their labels are not usable
    # in another script each participant's labels were graphed with ranges compared.
    # Participants that only ever labeled one class (i.e., happy) for all stimuli would be considered bad
    # this says for each participant ID (pid) in the list of participants with good labels:
    for pid in details.good_both:

        train_y, result_train, test_y, result_test = subject_independent_run_model(pid, model_type, details)

        # compute the train/test accuracy and the test f1 score
        acc_train = accuracy_score(train_y, result_train)
        acc_test = accuracy_score(test_y, result_test)
        f1_test = f1_score(test_y, result_test, average='weighted')

        # add the scores to the running total
        ACC_train.append(acc_train)
        ACC_test.append(acc_test)
        F1_test.append(f1_test)

        # also add the results to the totals across all participants for later analysis
        all_test_y.append(test_y)
        all_train_y.append(train_y)
        all_test_result.append(result_test)
        all_train_result.append(result_train)

        # add the results for this pid to a dictionary in the results for individual inspection
        my_results.results[pid] = {"test_y": [int(s) for s in list(np.asarray(test_y).flatten())],
                                   "test_pred": [int(s) for s in list(np.asarray(result_test).flatten())]}

        # outside this script the results objects can be loaded and used to generate confusion matrices
        # these lines of code helped during early tests to make sure that early on in running the model
        # I could see if the model was only predicting one class or presenting oddly
        # print('predicted')
        # print(tf.math.confusion_matrix(labels=test_y, predictions=result_test,
        #                                num_classes=int(max(test_y) + 1)))
        # each class may have different counts of actual labels
        # the presented stimuli had balanced labels (i.e., 2 joy, 2 fear, 2 sad)
        # people responded differently to the stimuli and the collected labels were less balanced
        # this code gives a quick visual of the actual class distribution to compare to the predicted classes
        # print('actual')
        # print(tf.math.confusion_matrix(labels=test_y, predictions=test_y.flatten(),
        #                                num_classes=int(max(test_y) + 1)))


    print("===%s all, train_acc = %.2f%%, test_acc = %.2f%%, f1_test = %.4f===" % (model_type,
                                                                                  np.mean(ACC_train) * 100,
                                                                                  np.mean(ACC_test) * 100,
                                                                                  np.mean(F1_test)))

    # add the mean score to the results, when running ten times this would give the mean of the ten when the results are viewed
    my_results.mean_test_acc = np.mean(ACC_test) * 100
    my_results.mean_train_acc = np.mean(ACC_train) * 100
    my_results.mean_f1 = np.mean(F1_test) * 100

    return my_results, np.mean(ACC_train), np.mean(ACC_test), np.mean(F1_test)


model = 'RF' # use a random forest
# other models are available
# SVM, RF, ND, KNN, LR, 1DCNN, LSTM

for seg_length in [2]: # the number of seconds that were used when deriving features. Features were extracted across n second windows.
    # this example is using features extracted from two second windows of time
    for label in ['label_user_valance_2']:  # to run models for different labels, reduced to one here
        # label_user_valance_2 is the 2 class high or low user valance

        # details is a helper class that is generated with features are extracted
        details = FileDetails.MLFormattedDetails.load_file(FileDetails.get_feature_file_and_path_name(seg_length))

        # later in the process the label to predict will be requested of the details class passed into the model
        details.desired_label = label

        # requests all the columns that are recorded in the files and lists any that are not labels
        cols_to_include = [item for item in details.columns if "label" not in item]

        # because many combinations of features are tested it was easiest to use the helper class to handle passing data into the models
        # when data is passed by the details class we can specify which columns we are loading using the included features attribute
        details.included_features = cols_to_include

        # logging for visual notes on where in the process this tests are
        print("{} {} {}".format(model, label, seg_length))

        # for reporting models are compared using average scores across 10 runs, this would run each model combination i times
        for i in range(0, 1): # set to run each combination once here
            _, _, _, _ = leave_one_subject_out(i, seg_length, model, details) # all the results are being saved