import os
from typing import Optional
import sys
import warnings

import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class Forecaster:
    """XGBoost Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """
    MODEL_NAME = "XGBoost_Timeseries_Forecaster"

    def __init__(
            self,
            encode_len:int,
            decode_len:int,
            n_estimators: Optional[int] = 50,
            max_depth: Optional[int] = 5,
            eta: Optional[float] = 0.3,
            gamma: Optional[float] = 0.0,
            **kwargs
        ):
        """
        Construct a new XGBoost Forecaster.        

        Args:
            encode_len (int): Encoding (history) length.
            decode_len (int): Decoding (forecast window) length.
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 50.
            max_depth (int, optional): The maximum depth of the tree. 
                If None, then nodes are expanded until all leaves are pure or until
                all leaves contain less than min_samples_split samples
                Defaults to 10.
            eta (int, optional): Step size shrinkage used in update to prevents
                overfitting - alias learning rate.
                Defaults to 0.3.
            gamma (int, optional): Minimum loss reduction required to make a further
                partition on a leaf node of the tree.
                Defaults to 0.0. 
        """
        self.encode_len = int(encode_len)
        self.decode_len = int(decode_len)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.eta = eta
        self.gamma = gamma
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> XGBRegressor:
        """Build a new XGBoost regressor."""
        model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            eta=self.eta,
            gamma=self.gamma,
            objective ='reg:squarederror',
            n_jobs=n_jobs,
            tree_method="gpu_hist"
        )
        return model

    def _get_X_and_y(self, data: np.ndarray, is_train:bool=True) -> np.ndarray:
        """Extract X (historical target series), y (forecast window target) 
            When is_train is True, data contains both history and forecast windows.
            When False, only history is contained.
        """
        N, T, D = data.shape
        if is_train:
            if T != self.encode_len + self.decode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len + self.decode_len}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, :self.encode_len, :].reshape(N, -1) # shape = [N, T*D]
            y = data[:, self.encode_len:, 0] # shape = [N, T]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.encode_len:, :].reshape(N, -1)
            y = None
        return X, y

    def fit(self, train_data):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, data):
        X = self._get_X_and_y(data, is_train=False)[0]
        preds = self.model.predict(X)
        return np.expand_dims(preds, axis=-1)

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.model is not None:
            return self.model.score(x_test, y_test)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the XGBoost forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the XGBoost forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded XGBoost forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        return (
            f"Model name: {self.MODEL_NAME} ("
            f"max_depth: {self.max_depth}, "
            f"eta: {self.eta}, "
            f"gamma: {self.gamma}, "
            f"n_estimators: {self.n_estimators})"
        )


def train_predictor_model(
    train_data: np.ndarray,
    forecast_length: int,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        train_data (np.ndarray): The train split from training data.
        valid_data (np.ndarray): The valid split of training data.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        encode_len=train_data.shape[1] - forecast_length,
        decode_len=forecast_length,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(
    model: Forecaster, test_data: np.ndarray
) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, test_split: np.ndarray
) -> float:
    """
    Evaluate the Forecaster model and return the r-squared value.

    Args:
        model (Forecaster): The Forecaster model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The r-squared value of the Forecaster model.
    """
    return model.evaluate(test_split)
