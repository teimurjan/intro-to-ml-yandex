import numpy as np
import pandas
from matplotlib import pyplot as plt


def get_personal_columns(column_name):
  r_columns = ['r{}_{}'.format(i, column_name) for i in range(1, 6)]
  d_columns = ['d{}_{}'.format(i, column_name) for i in range(1, 6)]
  return r_columns, d_columns


def replace_with_mean(columns, X):
  X_copy = X.copy()
  for column in columns:
    r_columns, d_columns = get_personal_columns(column)
    X_copy['r_mean_{}'.format(column)] = X_copy.loc[:, r_columns].mean(axis=1)
    X_copy['d_mean_{}'.format(column)] = X_copy.loc[:, d_columns].mean(axis=1)
    X_copy = X_copy.drop([*r_columns, *d_columns], axis=1)
  return X_copy


def replace_with_bag_of_words(X, column='hero', file='heroes.csv'):
  words_df = pandas.read_csv('data/dictionaries/{}'.format(file))
  X_words = np.zeros((X.shape[0], len(words_df)))
  for i, match_id in enumerate(X.index):
    for p in range(1, 6):
      X_words[i, X.loc[match_id, 'r{}_{}'.format(p, column)] - 1] = 1
      X_words[i, X.loc[match_id, 'd{}_{}'.format(p, column)] - 1] = -1
  r_columns, d_columns = get_personal_columns(column)
  X_cleaned = X.drop([*r_columns, *d_columns], axis=1)
  words_columns = ['{}_{}'.format(column, i) for i in range(len(words_df))]
  X_words = pandas.DataFrame(X_words, index=X.index, columns=words_columns)
  return pandas.concat([X_cleaned, X_words], axis=1)


def prepare_data(X):
  X_prepared = X.drop(['start_time', 'lobby_type', 'first_blood_time', 'first_blood_player1', 'first_blood_player2'], axis=1)
  X_prepared = X_prepared.fillna(0)
  return replace_with_mean(['gold', 'xp', 'lh', 'kills', 'deaths', 'level', 'items'], X_prepared)


def plot(estimators, scores):
  plt.plot(estimators, scores)
  plt.xlabel('n_estimators')
  estimators30_score = scores[estimators.index(30)]
  plt.scatter(30, estimators30_score, marker='o', c='r')
  plt.annotate('Score with 30 estimators is {:0.3f}'.format(estimators30_score), (30, estimators30_score))
  plt.ylabel('score')
  plt.show()


def get_data(test=False, sample=True, frac=0.5):
  data = pandas.read_csv('data/features{}.csv'.format('_test' if test else ''), index_col='match_id')
  return data if not sample else data.sample(frac=frac)


def get_skips_df(X):
  cols_with_skips = X.loc[:, X.isnull().any()]
  return cols_with_skips.apply(lambda c: c.isnull().sum()).to_frame("skips_count")
