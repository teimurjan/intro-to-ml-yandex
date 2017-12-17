import os
import pandas
from matplotlib import pyplot as plt

from settings import BASE_DIR


def plot(C_range, scores):
  plt.plot(C_range, scores)
  plt.xlabel('C pows')
  best_score = max(scores)
  best_C = C_range[scores.index(best_score)]
  plt.scatter(best_C, best_score, marker='o', c='r')
  plt.annotate('Best score is {:0.3f}'.format(best_score), (best_C, best_score))
  plt.ylabel('score')
  plt.show()


def count_heroes_ids():
  heroes_df = pandas.read_csv(os.path.join(BASE_DIR, 'data/dictionaries/heroes.csv'))
  return len(heroes_df)
