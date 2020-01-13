import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from keras.preprocessing import image

class CNN(object):
    def __init__(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(100, 100, 3) ))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64, (5,5), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64,(3,3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))

        model.summary()

        self.model = model

class DataSource(object):
    def __init__(self):
        all_imgs = []
        all_labels = []

        images_dir_cartoon="../Datasets/cartoon_set/img"
        basedir_cartoon = "../Datasets/cartoon_set"
        labels_filename = "labels.csv"

        image_paths_cartoon = [os.path.join(images_dir_cartoon, l) for l in os.listdir(images_dir_cartoon)]
        labels_file = open(os.path.join(basedir_cartoon, labels_filename), 'r')
        lines = labels_file.readlines()

        eye_color_labels = {line.split('\t')[-1].split('\n')[0] : int(line.split('\t')[1]) for line in lines[1:]}

        if os.path.isdir(images_dir_cartoon):

            for img_path in image_paths_cartoon:

                file_name = img_path.split('.')[2].split('/')[-1]

                img = image.img_to_array(image.load_img(img_path,
                                                        target_size=(100,100),
                                                        interpolation='bicubic'))
                all_imgs.append(img/255.)
                all_labels.append(eye_color_labels[file_name+'.png'])

        whole_imgs, whole_labels = np.asarray(all_imgs, np.float32), np.asarray(all_labels, np.int32)

        ratio = 0.8
        s = np.int(whole_imgs.shape[0] * ratio)
        x_train = whole_imgs[:s]
        y_train = whole_labels[:s]
        x_val = whole_imgs[s:]
        y_val = whole_labels[s:]

        self.train_imgs, self.train_labels = x_train, y_train
        self.test_imgs, self.test_labels = x_val, y_val

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'

        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.cnn.model.fit(self.data.train_imgs, self.data.train_labels, epochs=5, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_imgs, self.data.test_labels)

        print("Accuracy: %.4f, test %d images" % (test_acc, len(self.data.test_labels)))

if __name__ == "__main__":
    app = Train()
    app.train()
