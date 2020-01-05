import face_landmarks as face_landmarks

import math
import numpy as np

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def get_img_data():
    # Extract the features
    X, y = face_landmarks.extract_features_labels()

    # Split the data into train set (60%), validation set(20%) and test set (20%)
    train_split = math.floor(y.shape[0] * 0.6)
    val_split = math.floor(y.shape[0] * 0.8)
    Y = np.array([y, -(y-1)]).T
    X_train = X[:train_split]
    Y_train = Y[:train_split]
    X_val = X[train_split:val_split]
    Y_val = Y[train_split:val_split]
    X_test = X[val_split:]
    Y_test = Y[val_split:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def sklearn_svm_model(train_img, train_label, test_img, test_label, c_para, val_flag):
    svm = SVC(kernel='linear', probability=True, C=c_para)
    svm.fit(train_img, train_label)
    # pred_gender = svm.predict(test_img)
    # acc_gender = accuracy_score(test_label, pred_gender)
    acc_gender = svm.score(test_img, test_label)
    if val_flag is True:
        print("sklearn_svm Gender Classification Validation Accuracy:", acc_gender)
    else:
        print("sklearn_svm Gender Classification Test Accuracy:", acc_gender)

    return acc_gender

# def sklearn_mlp_model(train_img, train_label, test_img, test_label):
#     mlp = MLPClassifier(solver='sgd',
#                         activation='relu',
#                         alpha=1e-4,
#                         hidden_layer_sizes=(68,68),
#                         max_iter=200,
#                         learning_rate_init=0.001)
#     mlp.fit(train_img, train_label)
#     pred_gender = mlp.predict(test_img)
#     print("MLP Model Gender Classification Accuracy:", accuracy_score(test_label, pred_gender))
#
#     return pred_gender

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_img_data()

# pred=sklearn_mlp_model(X_train.reshape((X_train.shape[0], 68*2)), list(zip(*Y_train))[0], X_test.reshape((X_test.shape[0], 68*2)), list(zip(*Y_test))[0])
#pred = sklearn_svm_model(X_train.reshape((X_train.shape[0], 68*2)), list(zip(*Y_train))[0], X_test.reshape((X_test.shape[0], 68*2)), list(zip(*Y_test))[0])

tuning_dict = []
svm_c_para = [0.0001, 0.001, 0.01, 0.1, 1.0]
X_train_re = X_train.reshape((X_train.shape[0], 68*2))
X_val_re = X_val.reshape((X_val.shape[0], 68*2))
X_test_re = X_test.reshape((X_test.shape[0], 68*2))
Y_train_re = list(zip(*Y_train))
Y_val_re = list(zip(*Y_val))
Y_test_re = list(zip(*Y_test))
for c_para in svm_c_para:
    acc = sklearn_svm_model(X_train_re, Y_train_re[0], X_val_re, Y_val_re[0], c_para, True)
    tuning_dict.append(acc)

pred_final = sklearn_svm_model(X_train_re, Y_train_re[0], X_test_re, Y_test_re[0],svm_c_para[tuning_dict.index(max(tuning_dict))], False)
