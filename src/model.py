from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a logistic regression model."""
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate model accuracy."""
    accuracy = model.score(X_test, y_test)
    return accuracy