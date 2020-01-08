import os
import face_landmarks as face_landmarks
import numpy as np
from keras.preprocessing import image

class Preprocess:
    def __init__(self):
        print("=====Images Datasets Preprocessing=====")

    def preprocess_celeba(self):
        print("Processing Datasets: celeba")

        # Extract the features with labels (gender and smile)
        # X, label_gender, label_emo = face_landmarks.extract_features_labels()
        X, Y = face_landmarks.extract_features_labels()

        # Split the data into train set(80%) and test set(20%)
        split_train = np.int(Y.shape[0] * 0.8)

        # For Task A1 Gender Detection
        Y_gender = np.array([Y[:,0], -(Y[:,0]-1)]).T
        X_train_gender, Y_train_gender = X[:split_train], Y_gender[:split_train]
        X_test_gender, Y_test_gender = X[split_train:], Y_gender[split_train:]

        # For Task A2 Emotion Detection
        X_feature_emo = X[:, 48:, :]
        Y_emo = np.array([Y[:,1],-(Y[:,1]-1)]).T
        X_train_emo, Y_train_emo = X_feature_emo[:split_train], Y_emo[:split_train]
        X_test_emo, Y_test_emo = X_feature_emo[split_train:], Y_emo[split_train:]

        return X_train_gender, Y_train_gender, X_train_emo, Y_train_emo, X_test_gender, Y_test_gender, X_test_emo, Y_test_emo

    def preprocess_cartoon(self):
        print("Processing Datasets: cartoon_set")

        temp_imgs, temp_labels = [], []

        images_dir_cartoon="./Datasets/cartoon_set/img"
        basedir_cartoon = "./Datasets/cartoon_set"
        labels_filename = "labels.csv"

        image_paths_cartoon = [os.path.join(images_dir_cartoon, l) for l in os.listdir(images_dir_cartoon)]
        labels_file = open(os.path.join(basedir_cartoon, labels_filename), 'r')
        lines = labels_file.readlines()

        eye_color_labels = {line.split('\t')[-1].split('\n')[0] : np.array([int(line.split('\t')[2]), int(line.split('\t')[1])]) for line in lines[1:]}

        if os.path.isdir(images_dir_cartoon):

            for img_path in image_paths_cartoon:

                file_name = img_path.split('.')[1].split('/')[-1]

                img = image.img_to_array(image.load_img(img_path,
                                                        target_size=(100,100),
                                                        interpolation='bicubic'))
                temp_imgs.append(img/255.)
                temp_labels.append(eye_color_labels[file_name+'.png'])

        all_imgs, all_labels = np.asarray(temp_imgs, np.float32), np.asarray(temp_labels, np.int32)

        ratio_train, ratio_val = 0.6, 0.2
        split_train, split_val = np.int(all_imgs.shape[0] * ratio_train), np.int(all_imgs.shape[0] * (ratio_train + ratio_val))

        x_train, y_train = all_imgs[:split_train], all_labels[:split_train]
        x_val, y_val = all_imgs[split_train:split_val], all_labels[split_train:split_val]
        x_test, y_test = all_imgs[split_val:], all_labels[split_val:]

        return x_train, y_train, x_val, y_val, x_test, y_test
