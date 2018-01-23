import re

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.model_selection import GridSearchCV

tokenization_regex = r'[^\s][-a-zA-Z0-9]*[^\s]?'
mcc_scorer = make_scorer(matthews_corrcoef)


def scrm114_tokenizer(in_string):
    return re.findall(tokenization_regex, in_string)


def eager_split_tokenizer(in_string):
    return re.split(r'[\s\.,:-]', in_string)


def grid_search_analysis(pipeline, parameters, x_train, y_train, x_test, y_test):
    gs_clf = GridSearchCV(pipeline, parameters,
                          verbose=1, scoring=mcc_scorer)
    gs_clf = gs_clf.fit(x_train, y_train)

    print('Best score: ', gs_clf.best_score_)
    print('Best params: ', gs_clf.best_params_)

    y_true, y_pred = y_test, gs_clf.predict(x_test)
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))

    print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))

    print('Matthews Correlation Coefficient: ', matthews_corrcoef(y_true, y_pred))
