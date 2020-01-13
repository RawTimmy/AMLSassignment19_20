import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

# PATH TO ALL IMAGES
# global basedir_celeba, basedir_cartoon, image_paths_celeba, image_paths_cartoon, target_size
global basedir_celeba, basedir_celeba_test, image_paths_celeba, image_paths_celeba_test, target_size

basedir_celeba = './Datasets/celeba'
basedir_celeba_test = './Datasets/dataset_test_AMLS_19-20/celeba_test'
images_dir_celeba = os.path.join(basedir_celeba,'img')
images_dir_celeba_test = os.path.join(basedir_celeba_test,'img')
# basedir_cartoon = './Datasets/cartoon_set'
# images_dir_cartoon = os.path.join(basedir_cartoon,'img')

labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(test):
    """
    This funtion extracts the landmarks features for all images in the folder 'Datasets/celeba/img'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    select_dir_celeba, select_basedir_celeba = None, None
    if test is False:
        select_dir_celeba, select_basedir_celeba = images_dir_celeba, basedir_celeba
    else:
        select_dir_celeba, select_basedir_celeba = images_dir_celeba_test, basedir_celeba_test

    image_paths_celeba = [os.path.join(select_dir_celeba, l) for l in os.listdir(select_dir_celeba)]
    target_size = None
    labels_file = open(os.path.join(select_basedir_celeba, labels_filename), 'r')
    lines = labels_file.readlines()

    gender_emo_labels = {line.split('\t')[1] : np.array([int(line.split('\t')[2]), int(line.split('\t')[3])]) for line in lines[1:]}

    if os.path.isdir(select_dir_celeba):
        all_features = []
        all_labels = []

        for img_path in image_paths_celeba:
            file_name= img_path.split('.')[1].split('/')[-1]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_emo_labels[file_name+'.jpg'])

    output_features = np.array(all_features)
    output_labels = (np.array(all_labels) + 1) / 2

    return output_features, output_labels

# def extract_features_labels_cartoon():
#     image_paths_cartoon = [os.path.join(images_dir_cartoon, l) for l in os.listdir(images_dir_cartoon)]
#     target_size = None
#     labels_file = open(os.path.join(basedir_cartoon, labels_filename), 'r')
#     lines = labels_file.readlines()
#     face_shape_labels = {line.split('\t')[-1].split('\n')[0] : int(line.split('\t')[2]) for line in lines[1:]}
#
#     eye_color_labels = {line.split('\t')[-1].split('\n')[0] : int(line.split('\t')[2]) for line in lines[1:]}
#
#     if os.path.isdir(images_dir_cartoon):
#         all_features = []
#         all_labels_face_shape = []
#         all_labels_eye_color = []
#         for img_path in image_paths_cartoon:
#             file_name = img_path.split('.')[1].split('/')[-1]
#
#             img = image.img_to_array(image.load_img(img_path,
#                                                     target_size=(128,128),
#                                                     interpolation='bicubic'))
#             features, _ = run_dlib_shape(img)
#             if features is not None:
#                 all_features.append(features)
#                 all_labels_face_shape.append(face_shape_labels[file_name+'.png'])
#                 all_labels_eye_color.append(eye_color_labels[file_name+'.png'])
#
#     landmark_features = np.array(all_features)
#     face_shape_labels = np.array(all_labels_face_shape)
#     eye_color_labels = np.array(all_labels_eye_color)
#     return landmark_features, face_shape_labels, eye_color_labels
