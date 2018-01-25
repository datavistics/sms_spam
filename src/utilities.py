import re

import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from matplotlib import pyplot as plt
import numpy as np

tokenization_regex = r'[^\s][-a-zA-Z0-9]*[^\s]?'
mcc_scorer = make_scorer(matthews_corrcoef)


def scrm114_tokenizer(in_string):
    return re.findall(tokenization_regex, in_string)


def eager_split_tokenizer(in_string):
    return re.split(r'[\s\.,:-]', in_string)


def grid_search_analysis(pipeline, parameters, x_train, y_train, x_test, y_test):
    """
    Discovers best parameters for model and returns them
    :param pipeline:
    :param parameters:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    gs_clf = GridSearchCV(pipeline, parameters,
                          verbose=1, scoring=mcc_scorer, refit=True)
    gs_clf = gs_clf.fit(x_train, y_train)

    print('Best score: ', gs_clf.best_score_)
    print('Best params: ', gs_clf.best_params_)

    y_true, y_pred = y_test, gs_clf.predict(x_test)
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))

    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))

    print('Matthews Correlation Coefficient: ', matthews_corrcoef(y_true, y_pred))
    return gs_clf


def metric_by_training_size(X, y, classifier_list, training_set, metric, as_percentage=True):
    """
    This is a refactoriation of code to repeat metrics for best fitted models by training set percentage size.
    i.e.: Find accuracy rating for
    :param X:
    :param y:
    :param classifier_list:
    :param training_set:
    :param metric:
    :param as_percentage:
    :return:
    """
    metric_array = np.zeros((len(training_set), len(classifier_list)))
    for row_num, training_size in enumerate(training_set):
        X_train_iter, X_test_iter, y_train_iter, y_test_iter = train_test_split(X, y,
                                                                                test_size=1 - training_size,
                                                                                random_state=0)
        metric_list = []
        for classifier in classifier_list:
            y_pred = classifier.fit(X_train_iter, y_train_iter).predict(X_test_iter)
            metric_list.append(metric(y_test_iter, y_pred))

        metric_array[row_num] = metric_list
    metric_array = metric_array.transpose()
    return 100 * metric_array if as_percentage else metric_array


def plot_by_training_size(x: np.ndarray, y: np.array, metric: sklearn.metrics, classifier_names: list):
    """
    Matplotlib is ugly by default, this helps it be more pretty
    :param x:
    :param y:
    :param metric:
    :param classifier_names:
    :return:
    """
    plt.figure(figsize=(12, 9), dpi=1200)
    ax = plt.subplot(111)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.yticks(range(0, 101, 10), [str(x) + "%" for x in range(0, 101, 10)], fontsize=14)
    plt.xticks(fontsize=14)

    plt.plot(x, y[0], lw=2.5, label=classifier_names[0])
    plt.plot(x, y[1], lw=2.5, label=classifier_names[1])
    plt.plot(x, y[2], lw=2.5, label=classifier_names[2])
    plt.legend()

    plt.xlabel('Training size percentage', fontsize=16)
    plt.ylabel(f'{metric} value by model', fontsize=16)

    plt.title(f"{metric} by training size percentage", fontsize=17, ha="center")
    plt.show()