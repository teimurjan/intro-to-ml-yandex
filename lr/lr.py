import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import utils
import lr_utils


def train(X, y, scale=True):
  cv = KFold(n_splits=5, shuffle=True, random_state=241)
  X_ = X.copy()
  if scale:
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X_)
  scores = []
  for C in C_range:
    clf = LogisticRegression(C=C, random_state=241)
    score = cross_val_score(estimator=clf, cv=cv, X=X_, y=y, scoring='roc_auc').mean()
    scores.append(score)
  return scores


def train_raw_data(X, y):
  scores = train(X, y)
  lr_utils.plot(C_pows, scores)


def train_data_without_categorical(X, y):
  r_hero_cols, d_hero_cols = utils.get_personal_columns('hero')
  scores = train(X.drop([*r_hero_cols, *d_hero_cols, 'lobby_type'], axis=1), y)
  lr_utils.plot(C_pows, scores)


def train_with_bag_of_words(X, y):
  X = utils.replace_with_bag_of_words(utils.prepare_data(X))
  scores = train(X, y)
  lr_utils.plot(C_pows, scores)


def make_coursera_testing(X, y):
  train_raw_data(X, y)
  train_data_without_categorical(X, y)
  lr_utils.count_heroes_ids()
  train_with_bag_of_words(X, y)


def make_kaggle_prediction(X_test, X_train, y):
  X_train_ = utils.replace_with_bag_of_words(utils.prepare_data(X_train))
  X_test_ = utils.replace_with_bag_of_words(utils.prepare_data(X_test))
  clf = LogisticRegression(C=0.01, random_state=241)
  clf.fit(X_train_, y)
  clf.predict_proba(X_test_)
  pred = clf.predict_proba(X_test_)[:, 1]
  result = pandas.DataFrame({'radiant_win': pred}, index=X_test_.index)
  result.to_csv('result.csv')


def main():
  train_data = utils.get_data(sample=False)
  test_data = utils.get_data(test=True, sample=False)
  y = train_data['radiant_win']
  X_train = train_data.loc[:, train_data.columns != 'radiant_win']
  make_coursera_testing(X_train, y)


if __name__ == '__main__':
  C_pows = range(-5, 1)
  C_range = [10.0 ** i for i in C_pows]
  main()
