import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def predict(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Make predictions using the trained model."""
    return model.predict(X)


def predict_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Get prediction probabilities."""
    return model.predict_proba(X)


def decode_predictions(predictions: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    """Decode numeric predictions back to species names."""
    return label_encoder.inverse_transform(predictions)


def predict_single(model: LogisticRegression, scaler: StandardScaler, label_encoder: LabelEncoder,
                   bill_length: float, bill_depth: float, flipper_length: float, body_mass: float) -> str:
    """Predict species for a single penguin."""
    features = np.array([[bill_length, bill_depth, flipper_length, body_mass]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    species = label_encoder.inverse_transform(prediction)
    return species[0]