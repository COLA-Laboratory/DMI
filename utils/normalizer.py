from abc import abstractmethod
import numpy as np


class Normalizer:
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def do(self, X):
        pass

    @abstractmethod
    def undo(self, X):
        pass


class BoundedNormalizer(Normalizer):
    """
    Normalizing data to [0, 1] according to bounds
    """
    def __init__(self, bounds):
        super(BoundedNormalizer, self).__init__()
        self.bounds = bounds

    def fit(self, X):
        pass

    def do(self, X):
        return np.clip((X.copy() - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0]), 0, 1)

    def undo(self, X):
        return np.clip(X.copy(), 0, 1) * (self.bounds[:,1] - self.bounds[:,0]) + self.bounds[:,0]


class StandardNormalizer(Normalizer):
    """
    Normalizing data to N(0,1)
    """
    def __init__(self):
        super(StandardNormalizer, self).__init__()
        self._mean = None
        self._std = None

    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)

    def do(self, X):
        if np.any(self._std == 0):
            return X.copy() - self._mean
        else:
            return (X.copy() - self._mean) / self._std

    def undo(self, X):
        if np.any(self._std == 0):
            return X.copy() + self._mean
        else:
            return X.copy() * self._std + self._mean
