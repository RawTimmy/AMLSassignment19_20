import os
import tensorflow as tf
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
import numpy as np

class CNN_B1(object):
    def __init__(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(100, 100, 3) ))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.1))

        model.add(layers.Conv2D(64, (5,5), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64,(3,3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))

        model.summary()

        self.model = model

class Utils_B1:
    def __init__(self):
        print("Processing task B1")
        self.cnn = CNN_B1()

    # def train(self, train_imgs, train_labels, val_imgs, val_labels):
    #
    #     self.cnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #
    #     self.cnn.model.fit(train_imgs, train_labels, epochs=5)
    #
    #     val_loss, val_acc = self.cnn.model.evaluate(val_imgs, val_labels, verbose=0)
    #
    #     print("Validation Accuracy: %.4f, validate on %d images" % (val_acc, len(val_labels)))
    #
    #     return val_acc, val_loss

    def train(self, train_imgs, train_labels):

        # Optional: Function to plot learning curves
        def plot_curve(title, train_val, train_loss, val_val, val_loss, num_epoch, axes=None, ylim=None):
            num = np.linspace(1, num_epoch, num_epoch)
            axes[0].set_title(title)
            if ylim is not None:
                axes[0].set_ylim(*ylim)
            axes[0].set_xticks(num)
            axes[0].set_xlabel("Number of epochs")
            axes[0].set_ylabel("Score")

            axes[0].grid()
            axes[0].plot(num, train_val, 'o-', color="r",
                         label="Training score")
            axes[0].plot(num, val_val, 'o-', color="g",
                         label="Cross-validation score")
            axes[0].legend(loc="best")

            axes[1].grid()
            axes[1].plot(num, train_loss, 'o-', color="r",
                         label="Training loss")
            axes[1].plot(num, val_loss, 'o-', color="g",
                         label="Cross-validation loss")
            axes[1].legend(loc="best")
            axes[1].set_title("Training Loss v.s. Validation Loss")
            axes[1].set_xticks(num)
            axes[1].set_xlabel("Number of epochs")
            axes[1].set_ylabel("Loss")

            plt.show()
            return plt


        num_epochs, val_split = 8, 0.2
        self.cnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        hist = self.cnn.model.fit(train_imgs, train_labels, epochs=num_epochs, validation_split=val_split, shuffle=True)

        # Optional: Plot learning curves
        # fig, axes = plt.subplots(1,2, figsize=(20,5))
        # plot_curve("Learning Curve(CNN)",
        #            hist.history['accuracy'],
        #            hist.history['loss'],
        #            hist.history['val_accuracy'],
        #            hist.history['val_loss'],
        #            num_epochs,
        #            axes, ylim=(0.0, 1.01))

        return hist.history['val_accuracy'][-1], hist.history['val_loss'][-1]

    def test(self, test_imgs, test_labels):

        test_loss, test_acc = self.cnn.model.evaluate(test_imgs, test_labels, verbose=0)

        print("Test Accuracy: %.4f, test on %d images" % (test_acc, len(test_labels)))

        return test_acc, test_loss
