from abc import ABC, abstractmethod
from inspect import _void
import numpy as np


class BaseModel(ABC):
    """
    Base class for our models. All methods inherit this class to provide a unified interface.
    """
    @abstractmethod
    def train(self, train_X: np.array, train_y: np.array, val_X: np.array, val_y: np.array) -> None:
        """
        Train the classifier using training data
        @train_X: Numpy array of shape (N, d,) consisting of the training features
        @train_y: Numpy array of shape (N,) consisting of the training target labels
        @val_X: Numpy array of shape (N, d,) consisting of the validation features
        @val_y: Numpy array of shape (N,) consisting of the validation target labels
        """
        pass

    @abstractmethod
    def predict(self, test_X: np.array) -> np.array:
        """
        Predict using the learnt classifier
        @test_X: Numpy array of shape (N, d,) consisting of the test features
        returns: Numpy array of shape (N,) consisting of the predicted labels
        """
        pass
