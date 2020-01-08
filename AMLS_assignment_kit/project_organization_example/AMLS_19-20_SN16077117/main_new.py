import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings

from A1 import model_svm_a1 as svm_a1
from A2 import model_svm_a2 as svm_a2
from B1 import model_cnn_b1 as cnn_b1
from B2 import model_cnn_b2 as cnn_b2

warnings.filterwarnings('ignore','Solver terminated early.*')
# ======================================================================================================================
# Data preprocessing
pre = preprocessing.Preprocess()
# Preprocess celeba dataset, 68 landmark features extraction
img_train_gender, label_train_gender, img_train_emo, label_train_emo, img_test_gender, label_test_gender, img_test_emo, label_test_emo = pre.preprocess_celeba()

scaler_gender = StandardScaler()
img_train_gender = scaler_gender.fit_transform(img_train_gender.reshape((img_train_gender.shape[0], 68*2)))
img_test_gender = scaler_gender.transform(img_test_gender.reshape((img_test_gender.shape[0], 68*2)))

scaler_emo = StandardScaler()
img_train_emo = scaler_emo.fit_transform(img_train_emo.reshape((img_train_emo.shape[0], 20*2)))
img_test_emo = scaler_emo.transform(img_test_emo.reshape((img_test_emo.shape[0], 20*2)))

# Preprocess cartoon dataset, image preprocess
train_imgs_cartoon, train_labels_cartoon, val_imgs_cartoon, val_labels_cartoon, test_imgs_cartoon, test_labels_cartoon = pre.preprocess_cartoon()

# ======================================================================================================================
# Task A1
model_A1 = svm_a1.Utils_A1()
acc_A1_train, clf_gender = model_A1.train(img_train_gender, label_train_gender)
acc_A1_test = model_A1.test(clf_gender, img_test_gender, label_test_gender)

# ======================================================================================================================
# Task A2
model_A2 = svm_a2.Utils_A2()
acc_A2_train, clf_emo = model_A2.train(img_train_emo, label_train_emo)
acc_A2_test = model_A2.test(clf_emo, img_test_emo, label_test_emo)

# ======================================================================================================================
# Task B1
model_B1 = cnn_b1.Utils_B1()
acc_B1_train, loss_B1_train = model_B1.train(train_imgs_cartoon, train_labels_cartoon[:,0],
                                             val_imgs_cartoon, val_labels_cartoon[:,0])
acc_B1_test, loss_B1_test = model_B1.test(test_imgs_cartoon, test_labels_cartoon[:,0])

# ======================================================================================================================
# Task B2
model_B2 = cnn_b2.Utils_B2()
acc_B2_train, loss_B2_train = model_B2.train(train_imgs_cartoon, train_labels_cartoon[:,1],
                                             val_imgs_cartoon, val_labels_cartoon[:,1])
acc_B2_test, loss_B2_test = model_B2.test(test_imgs_cartoon, test_labels_cartoon[:,1])

# ======================================================================================================================
## Print out results
print('TA1:{:.4f},{:.4f};\nTA2:{:.4f},{};\nTB1:{:.4f},{:.4f};\nTB2:{:.4f},{:.4f};'.format(acc_A1_train, acc_A1_test,
                                                                                    acc_A2_train, acc_A2_test,
                                                                                    acc_B1_train, acc_B1_test,
                                                                                    acc_B2_train, acc_B2_test))
