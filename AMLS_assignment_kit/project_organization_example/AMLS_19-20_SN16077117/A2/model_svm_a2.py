from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve, ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np

class Utils_A2:

    def __init__(self):
        print("Processing task A2")

    def train(self, img_train, label_train):

        # Optional: Function to plot learning curves
        def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
            if axes is None:
                _, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].set_title(title)
            if ylim is not None:
                axes[0].set_ylim(*ylim)
            axes[0].set_xlabel("Training examples")
            axes[0].set_ylabel("Score")

            train_sizes, train_scores, test_scores, fit_times, _ = \
                learning_curve(estimator, X, list(zip(*y))[0], cv=cv, n_jobs=n_jobs,
                               train_sizes=train_sizes,
                               return_times=True)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            # Plot learning curve
            axes[0].grid()
            axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
            axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1,
                                 color="g")
            axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
            axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")
            axes[0].legend(loc="best")

            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].plot(train_sizes, fit_times_mean, 'o-')
            axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                                 fit_times_mean + fit_times_std, alpha=0.1)
            axes[1].set_xlabel("Training examples")
            axes[1].set_ylabel("fit_times")
            axes[1].set_title("Scalability of the model")

            # Plot fit_time vs score
            axes[2].grid()
            axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
            axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1)
            axes[2].set_xlabel("fit_times")
            axes[2].set_ylabel("Score")
            axes[2].set_title("Performance of the model")

            return plt

        svm = SVC(max_iter=500)
        parameter_space ={
            'C': [0.00001, 0.0001, 0.01, 0.1, 1.0],
            'kernel': ['linear', 'rbf'],
            'probability': [True, False],
        }

        # Classifier with optimized parameters
        clf = GridSearchCV(svm, parameter_space, n_jobs=-1, cv=3, return_train_score=True)

        # Optional: Plotting learning curves
        # fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # title = "Learning Curves (SVM)"
        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # plot_learning_curve(clf, title, img_train, label_train, axes=axes, ylim=(0.7, 1.01),
        #                     cv=cv, n_jobs=4)
        # plt.show()


        clf_fit = clf.fit(img_train, list(zip(*label_train))[0])
        print('Best parameters found: ', clf_fit.best_params_)

        # means_test = clf_fit.cv_results_['mean_test_score']
        # stds_test = clf_fit.cv_results_['std_test_score']
        # means_train = clf_fit.cv_results_['mean_train_score']
        # stds_train = clf_fit.cv_results_['std_train_score']
        #
        # for mean_test, std_test, mean_train, std_train, params in zip(means_test, stds_test, means_train, stds_train, clf.cv_results_['params']):
        #     print("Train Acc: %0.3f(+/-%0.03f) Test Acc: %0.3f(+/-%0.03f) for %r" % (mean_train, std_train*2, mean_test, std_test*2, params))

        return clf_fit.best_score_, clf_fit

    def test(self, svm_clf, img_val, label_val):
        y_true, y_pred = list(zip(*label_val))[0], svm_clf.predict(img_val)
        print("Results on the validation set: ")
        print(classification_report(y_true, y_pred))
        return accuracy_score(y_true, y_pred)
