from matplotlib import pyplot as plt


def plot(estimators, scores):
  plt.plot(estimators, scores)
  plt.xlabel('n_estimators')
  estimators30_score = scores[estimators.index(30)]
  plt.scatter(30, estimators30_score, marker='o', c='r')
  plt.annotate('Score with 30 estimators is {:0.3f}'.format(estimators30_score), (30, estimators30_score))
  plt.ylabel('score')
  plt.show()


def get_skips_df(X):
  cols_with_skips = X.loc[:, X.isnull().any()]
  return cols_with_skips.apply(lambda c: c.isnull().sum()).to_frame("skips_count")
