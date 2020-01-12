import os
import face_landmarks as face_landmarks
import numpy as np
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self):
        print("=====Images Datasets Preprocessing=====")

    def preprocess_celeba(self, test):
        print("Processing Datasets: celeba")

        # Extract the features with labels (gender and smile)
        # X, label_gender, label_emo = face_landmarks.extract_features_labels()
        X, Y = face_landmarks.extract_features_labels(test)

        Y_gender = np.array([Y[:,0], -(Y[:,0]-1)]).T
        X_feature_emo = X[:, 48:, :]
        Y_emo = np.array([Y[:,1],-(Y[:,1]-1)]).T

        if test is True:
            scaler_gender_test = StandardScaler()
            X = scaler_gender_test.fit_transform(X.reshape((X.shape[0], 68*2)))
            scaler_emo_test = StandardScaler()
            X_feature_emo = scaler_emo_test.fit_transform(X_feature_emo.reshape((X_feature_emo.shape[0], 20*2)))
            return X, Y_gender, X_feature_emo, Y_emo

        # Split the data into train set(80%) and validation set(20%)
        split_train = np.int(Y.shape[0] * 0.8)

        # For Task A1 Gender Detection
        X_train_gender, Y_train_gender = X[:split_train], Y_gender[:split_train]
        X_val_gender, Y_val_gender = X[split_train:], Y_gender[split_train:]

        # For Task A2 Emotion Detection
        X_train_emo, Y_train_emo = X_feature_emo[:split_train], Y_emo[:split_train]
        X_val_emo, Y_val_emo = X_feature_emo[split_train:], Y_emo[split_train:]

        scaler_gender = StandardScaler()
        X_train_gender = scaler_gender.fit_transform(X_train_gender.reshape((X_train_gender.shape[0], 68*2)))
        X_val_gender = scaler_gender.transform(X_val_gender.reshape((X_val_gender.shape[0], 68*2)))

        scaler_emo = StandardScaler()
        X_train_emo = scaler_emo.fit_transform(X_train_emo.reshape((X_train_emo.shape[0], 20*2)))
        X_val_emo = scaler_emo.transform(X_val_emo.reshape((X_val_emo.shape[0], 20*2)))

        return X_train_gender, Y_train_gender, X_train_emo, Y_train_emo, X_val_gender, Y_val_gender, X_val_emo, Y_val_emo

    def preprocess_cartoon(self, test):
        print("Processing Datasets: cartoon_set")

        temp_imgs, temp_labels = [], []

        images_dir_cartoon="./Datasets/cartoon_set/img"
        images_dir_cartoon_test = "./Datasets/dataset_test_AMLS_19-20/cartoon_set_test/img"
        basedir_cartoon = "./Datasets/cartoon_set"
        basedir_cartoon_test = "./Datasets/dataset_test_AMLS_19-20/cartoon_set_test"
        labels_filename = "labels.csv"

        select_dir_cartoon, select_basedir_cartoon = None, None
        if test is False:
            select_dir_cartoon, select_basedir_cartoon = images_dir_cartoon, basedir_cartoon
        else:
            select_dir_cartoon, select_basedir_cartoon = images_dir_cartoon_test, basedir_cartoon_test

        image_paths_cartoon = [os.path.join(select_dir_cartoon, l) for l in os.listdir(select_dir_cartoon)]
        labels_file = open(os.path.join(select_basedir_cartoon, labels_filename), 'r')
        lines = labels_file.readlines()

        eye_color_labels = {line.split('\t')[-1].split('\n')[0] : np.array([int(line.split('\t')[2]), int(line.split('\t')[1])]) for line in lines[1:]}

        if os.path.isdir(select_dir_cartoon):

            for img_path in image_paths_cartoon:

                file_name = img_path.split('.')[1].split('/')[-1]

                img = image.img_to_array(image.load_img(img_path,
                                                        target_size=(100,100),
                                                        interpolation='bicubic'))
                temp_imgs.append(img/255.)
                temp_labels.append(eye_color_labels[file_name+'.png'])

        all_imgs, all_labels = np.asarray(temp_imgs, np.float32), np.asarray(temp_labels, np.int32)

        if test is True:
            return all_imgs, all_labels

        ratio_train = 0.8
        split_train = np.int(all_imgs.shape[0] * ratio_train)

        x_train, y_train = all_imgs[:split_train], all_labels[:split_train]
        x_val, y_val = all_imgs[split_train:], all_labels[split_train:]

        return x_train, y_train, x_val, y_val
