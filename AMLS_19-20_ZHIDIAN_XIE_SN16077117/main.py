import preprocessing

from A1 import model_svm_a1 as svm_a1
from A2 import model_svm_a2 as svm_a2
from B1 import model_cnn_b1 as cnn_b1
from B2 import model_cnn_b2 as cnn_b2

# ======================================================================================================================
# Data preprocessing
pre = preprocessing.Preprocess()
# Preprocess celeba dataset, 68 landmark features extraction
img_train_gender, label_train_gender, img_train_emo, label_train_emo, img_val_gender, label_val_gender, img_val_emo, label_val_emo = pre.preprocess_celeba(False)

# Preprocess cartoon dataset, image preprocess
train_imgs_cartoon, train_labels_cartoon, val_imgs_cartoon, val_labels_cartoon = pre.preprocess_cartoon(False)

# Additional test dataset
img_test_gender, label_test_gender, img_test_emo, label_test_emo = pre.preprocess_celeba(True)
test_imgs_cartoon, test_labels_cartoon = pre.preprocess_cartoon(True)
# ======================================================================================================================
# Task A1
model_A1 = svm_a1.Utils_A1()
acc_A1_train, clf_gender = model_A1.train(img_train_gender, label_train_gender)
acc_A1_val = model_A1.test(clf_gender, img_val_gender, label_val_gender)

#Additional test dataset
acc_A1_test = model_A1.test(clf_gender, img_test_gender, label_test_gender)
#
# # ======================================================================================================================
# # # Task A2
model_A2 = svm_a2.Utils_A2()
acc_A2_train, clf_emo = model_A2.train(img_train_emo, label_train_emo)
acc_A2_val = model_A2.test(clf_emo, img_val_emo, label_val_emo)

#Additional test dataset
acc_A2_test = model_A2.test(clf_emo, img_test_emo, label_test_emo)
# #
# # ======================================================================================================================
# # Task B1
model_B1 = cnn_b1.Utils_B1()
acc_B1_train, loss_B1_train = model_B1.train(train_imgs_cartoon, train_labels_cartoon[:,0])
acc_B1_val, loss_B1_val = model_B1.test(val_imgs_cartoon, val_labels_cartoon[:,0])

#Additional test dataset
acc_B1_test, loss_B1_test = model_B1.test(test_imgs_cartoon, test_labels_cartoon[:,0])
# ======================================================================================================================
# # Task B2
model_B2 = cnn_b2.Utils_B2()
acc_B2_train, loss_B2_train = model_B2.train(train_imgs_cartoon, train_labels_cartoon[:,1])
acc_B2_val, loss_B2_val = model_B2.test(val_imgs_cartoon, val_labels_cartoon[:,1])

#Additional test dataset
acc_B2_test, loss_B2_test = model_B2.test(test_imgs_cartoon, test_labels_cartoon[:,1])
#
# # ======================================================================================================================
# ## Print out results
print("==========Results: Training Accuracy, Validation Accuracy==========")
print('TA1:{},{};\nTA2:{},{};\nTB1:{},{};\nTB2:{},{};'.format(acc_A1_train, acc_A1_val,
                                                              acc_A2_train, acc_A2_val,
                                                              acc_B1_train, acc_B1_val,
                                                              acc_B2_train, acc_B2_val))
print("==========Results: Testing Accuracy on Additional Test Dataset==========")
print('TA1:{:.4f};\nTA2:{:.4f};\nTB1:{:.4f};\nTB2:{:.4f};'.format(acc_A1_test, acc_A2_test, acc_B1_test, acc_B2_test))
