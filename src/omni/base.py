from sklearn.base import BaseEstimator, TransformerMixin


class MakeSklearn(BaseEstimator, TransformerMixin):
    """ Sklearnifies an object provided __call__ can correspond to transform. """
    def __init__(self, obj):
        self.obj = obj

    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        return self.obj(data)


class NullTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X