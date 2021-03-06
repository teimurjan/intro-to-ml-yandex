from datetime import datetime

import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

import utils
import gb_utils


def make_kaggle_prediction(X_train, y_train, X_test):
  clf = GradientBoostingClassifier(n_estimators=500, random_state=241)
  clf.fit(X_train, y_train)
  clf.predict_proba(X_test)
  pred = clf.predict_proba(X_test)[:, 1]
  result = pandas.DataFrame({'radiant_win': pred}, index=X_test.index)
  result.to_csv('result.csv')


def make_coursera_testing(X_train, y_train):
  estimators = [10, 20, 30, 100, 250]
  scores = []
  cross_val_times = []
  cv = KFold(n_splits=5, shuffle=True, random_state=241)
  for n_estimators in estimators:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=241)
    start_time = datetime.now()
    score = cross_val_score(estimator=clf, cv=cv, X=X_train, y=y_train, scoring='roc_auc').mean()
    cross_val_times.append(datetime.now() - start_time)
    scores.append(score)
  gb_utils.plot(estimators, scores)


def main():
  train_data = utils.get_data()
  test_data = utils.get_data(test=True, sample=False)
  y = train_data['radiant_win']
  X_train = train_data.loc[:, train_data.columns != 'radiant_win']
  X_train = utils.replace_with_bag_of_words(utils.prepare_data(X_train))
  X_test = utils.replace_with_bag_of_words(utils.prepare_data(test_data))
  make_coursera_testing(X_train, y)


if __name__ == '__main__':
  main()
