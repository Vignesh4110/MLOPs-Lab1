import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .data_loader import get_feature_columns, get_target_column


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values."""
    return df.dropna()


def encode_target(y: pd.Series) -> tuple:
    """Encode target labels to integers."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def scale_features(X: pd.DataFrame) -> tuple:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def preprocess_data(df: pd.DataFrame) -> dict:
    """Full preprocessing pipeline."""
    df_clean = remove_missing_values(df)
    
    feature_cols = get_feature_columns()
    target_col = get_target_column()
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    y_encoded, label_encoder = encode_target(y)
    X_scaled, scaler = scale_features(X)
    
    return {
        "X": X_scaled,
        "y": y_encoded,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "feature_names": feature_cols,
    }