# AMLSassignment19_20

## Introduction
This repo includes the codes for Applied Machine Learning Assignment 19-20.  
Binary classification tasks: Gender Detection(Task A1), Emotion Detection(Task A2)  
Multi-class classification tasks: Face Shape Recognition(Task B1), Eye Color Recognition(Task B2)

## Requirement
* tensorflow == 2.0.0
* keras == 2.3.1
* dlib == 19.19.0
* numpy == 1.18.1
* openCV == 4.1.2
* scikit-learn == 0.22.1

## Note
* The size of this repository is about 800 Mb, downloading the whole repo could take some minutes.
* The time taken for the data preprocessing and model training is long (More than 10 minutes).

## Known Issue  
* \#1: While training with SVM on task A1 and A2, the terminal will print out "ConvergenceWarning" continuously and it cannot be suppressed. Please ignore. It won't affect the final results.

## Structure

    ├── main.py                                 # Main
    ├── face_landmarks.py                       # Used for landmark features extraction for task A1 A2
    ├── preprocessing.py                        # Image Data Preprocessing
    ├── shape_predictor_68_face_landmarks.dat   
    ├── A1/
    │   ├── model_svm_a1.py                     # Engineering of SVM for task A1
    ├── A2/
    │   ├── model_svm_a2.py                     # Engineering of SVM for task A2
    ├── B1/
    │   ├── model_cnn_b1.py                     # Engineering of CNN for task B1
    ├── B2/
    │   ├── model_cnn_b2.py                     # Engineering of CNN for task B1
    ├── Datasets/
    │   ├── cartoon_set/
    │   │   ├── img/
    │   │   ├── labels.csv
    │   ├── celeba/
    │   │   ├── img/
    │   │   ├── labels.csv
    │   ├── dataset_test_AMLS_19-20/
    │   │   ├── cartoon_set_test/
    │   │   │   ├── img/
    │   │   │   ├── labels.csv
    │   │   ├── celeba_test/
    │   │   │   ├── img/
    │   │   │   ├── labels.csv
