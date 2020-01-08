import os
import tensorflow as tf
from tensorflow.keras import layers, models

class CNN_B2(object):
    def __init__(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(100, 100, 3) ))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.2))

        model.add(layers.Conv2D(64, (5,5), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64,(3,3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))

        model.summary()

        self.model = model

class Utils_B2:
    def __init__(self):
        print("Processing task B2")
        self.cnn = CNN_B2()

    def train(self, train_imgs, train_labels, val_imgs, val_labels):

        # check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.cnn.model.fit(train_imgs, train_labels, epochs=5)

        val_loss, val_acc = self.cnn.model.evaluate(val_imgs, val_labels, verbose=0)

        print("Validation Accuracy: %.4f, validate on %d images" % (val_acc, len(val_labels)))

        return val_acc, val_loss

    def test(self, test_imgs, test_labels):

        test_loss, test_acc = self.cnn.model.evaluate(test_imgs, test_labels, verbose=0)

        print("Test Accuracy: %.4f, test on %d images" % (test_acc, len(test_labels)))

        return test_acc, test_loss
