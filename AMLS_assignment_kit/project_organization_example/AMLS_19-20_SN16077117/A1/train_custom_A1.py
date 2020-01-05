import face_landmarks as face_landmarks

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# Graph Training Operation
learning_rate = 1.
training_epochs = 100

sess = tf.Session()

train_img, train_label, val_img, val_label, test_img, test_label = get_img_data()

batch_size = 100

X = tf.placeholder("float", [None, 68, 2])
Y = tf.placeholder("float", [None, 2])

W = tf.Variable(tf.random_normal([68 * 2, 2]))
b = tf.Variable(tf.random_normal([2]))

images_flat = tf.contrib.layers.flatten(X)

model_output = tf.add(tf.matmul(images_flat, W), b)
l2_norm = tf.reduce_sum(tf.square(W))
alpha = tf.constant([0.1])
classification_term = tf.reduce_mean(tf.maximum(0., 1.-model_output*Y))
loss = classification_term + alpha * l2_norm

prediction = tf.sign(model_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, Y),tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess.run(tf.global_variables_initializer())

loss_vec = []
train_acc = []
test_acc = []

for epoch in range(training_epochs):
    rand_index = np.random.choice(train_img.shape[0], size=batch_size)
    rand_x = train_img[rand_index]
    #rand_y = np.transpose([train_label[rand_index]])
    rand_y = train_label[rand_index]

    sess.run(train_step, feed_dict={X: rand_x, Y:rand_y})

    temp_loss = sess.run(loss, feed_dict={X: rand_x, Y:rand_y})
    loss_vec.append(temp_loss)
    train_acc_temp = sess.run(accuracy, feed_dict={X: train_img, Y: train_label})
    train_acc.append(train_acc_temp)
    test_acc_temp = sess.run(accuracy, feed_dict={X: test_img, Y: test_label})
    test_acc.append(test_acc_temp)

    print('Step #' + str(epoch+1))
    print(temp_loss)
    print(train_acc_temp)
    print(test_acc_temp)

plt.plot(loss_vec)
plt.plot(train_acc)
plt.plot(test_acc)
plt.legend(['a','b','c'])
plt.ylim(0.,1.)
