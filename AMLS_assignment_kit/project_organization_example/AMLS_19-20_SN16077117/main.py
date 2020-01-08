import img_preprocessing as img_preprocessing
import warnings
from sklearn.preprocessing import StandardScaler

from A1 import train_A1 as task_A1
from A2 import train_A2 as task_A2
from B1 import train_B1 as task_B1

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
warnings.filterwarnings('ignore','Solver terminated early.*')

# Start 20200106
img_train_gender, label_train_gender, img_train_emotion, label_train_emotion, img_test_gender, label_test_gender,img_test_emotion, label_test_emotion = img_preprocessing.get_img_data_celeba()

scaler_gender = StandardScaler()
img_train_gender = scaler_gender.fit_transform(img_train_gender.reshape((img_train_gender.shape[0], 68*2)))
img_test_gender = scaler_gender.transform(img_test_gender.reshape((img_test_gender.shape[0], 68*2)))

scaler_emotion = StandardScaler()
img_train_emotion = scaler_emotion.fit_transform(img_train_emotion.reshape((img_train_emotion.shape[0], 20*2)))
img_test_emotion = scaler_emotion.transform(img_test_emotion.reshape((img_test_emotion.shape[0], 20*2)))

img_train_face_shape, label_train_face_shape, img_test_face_shape, label_test_face_shape = img_preprocessing.get_img_data_cartoon()

scaler_face_shape = StandardScaler()
img_train_face_shape = scaler_face_shape.fit_transform(img_train_face_shape.reshape((img_train_face_shape.shape[0], 68*2)))
img_test_face_shape = scaler_face_shape.transform(img_test_face_shape.reshape((img_test_face_shape.shape[0], 68*2)))
# End 20200106

# ======================================================================================================================
# Task A1
# Build model object.
model_A1 = task_A1.train_A1()
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_train, clf_gender = model_A1.svm_train_with_parameter_tuning(img_train_gender, label_train_gender)
# Test model based on the test set.
acc_A1_test = model_A1.svm_test(clf_gender, img_test_gender, label_test_gender)

# # ======================================================================================================================
# # Task A2
# Build model object.
model_A2 = task_A2.train_A2()
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A2_train, clf_emotion = model_A2.svm_train_with_parameter_tuning(img_train_emotion, label_train_emotion)
# Test model based on the test set.
acc_A2_test = model_A2.svm_test(clf_emotion, img_test_emotion, label_test_emotion)
#
# # ======================================================================================================================
# # Task B1
model_B1 = task_B1.train_B1()
acc_B1_train, clf_face_shape = model_B1.svm_train(img_train_face_shape, label_train_face_shape)
acc_B1_test = model_B1.svm_test(clf_face_shape, img_test_face_shape, label_test_face_shape)
#
#
# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))
print('TA1: {:.3f} , {:.3f}; TA2: {:.3f} , {:.3f}; TA3: {:.3f} , {:.3f};'.format(acc_A1_train, acc_A1_test,
                                                                                 acc_A2_train, acc_A2_test,
                                                                                 acc_B1_train, acc_B1_test))
# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
