import face_landmarks as face_landmarks
import math
import numpy as np

def get_img_data_without_val():
    # Extract the features with labels (gender and smile)
    X, label_gender, label_emo = face_landmarks.extract_features_labels()
    # Split the data into train set(80%) and test set(20%)
    train_split = math.floor(label_gender.shape[0] * 0.8)

    # For Task A1 Gender Detection
    Y_label_gender = np.array([label_gender, (-label_gender-1)]).T
    X_train_gender, Y_train_gender = X[:train_split], Y_label_gender[:train_split]
    X_test_gender, Y_test_gender = X[train_split:], Y_label_gender[train_split:]

    # For Task A2 Emotion Detection
    X_feature_emotion = X[:, 48:, :]
    Y_label_emotion = np.array([label_emo,(-label_emo-1)]).T
    X_train_emotion, Y_train_emotion = X_feature_emotion[:train_split], Y_label_emotion[:train_split]
    X_test_emotion, Y_test_emotion = X_feature_emotion[train_split:], Y_label_emotion[train_split:]


    # return X_train, Y_train, X_test, Y_test
    return X_train_gender, Y_train_gender, X_train_emotion, Y_train_emotion, X_test_gender, Y_test_gender, X_test_emotion, Y_test_emotion
