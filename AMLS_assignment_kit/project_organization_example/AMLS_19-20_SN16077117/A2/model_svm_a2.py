from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from sklearn.model_selection import GridSearchCV

class Utils_A2:

    def __init__(self):
        print("Processing task A2")

    def train(self, img_train, label_train):
        svm = SVC(max_iter=500)
        parameter_space ={
            'C': [0.00001, 0.0001, 0.01, 0.1, 1.0],
            'kernel': ['linear', 'rbf'],
            'probability': [True, False],
        }

        clf = GridSearchCV(svm, parameter_space, n_jobs=-1, cv=3, return_train_score=True)
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

    def test(self, svm_clf, img_test, label_test):
        y_true, y_pred = list(zip(*label_test))[0], svm_clf.predict(img_test)
        print("Results on the test set: ")
        print(classification_report(y_true, y_pred))
        return accuracy_score(y_true, y_pred)
