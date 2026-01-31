import pytest
import pandas as pd
import numpy as np
from src.data_loader import load_penguins, get_feature_columns, get_target_column
from src.preprocessor import remove_missing_values, encode_target, scale_features, preprocess_data
from src.model import split_data, train_model, evaluate_model
from src.predictor import predict, decode_predictions, predict_single


class TestDataLoader:
    def test_load_penguins_returns_dataframe(self):
        df = load_penguins()
        assert isinstance(df, pd.DataFrame)

    def test_load_penguins_has_rows(self):
        df = load_penguins()
        assert len(df) > 0

    def test_feature_columns_exist(self):
        df = load_penguins()
        feature_cols = get_feature_columns()
        for col in feature_cols:
            assert col in df.columns

    def test_target_column_exists(self):
        df = load_penguins()
        target_col = get_target_column()
        assert target_col in df.columns


class TestPreprocessor:
    @pytest.fixture
    def sample_df(self):
        return load_penguins()

    def test_remove_missing_values(self, sample_df):
        df_clean = remove_missing_values(sample_df)
        assert df_clean.isnull().sum().sum() == 0

    def test_encode_target_returns_integers(self, sample_df):
        df_clean = remove_missing_values(sample_df)
        y_encoded, encoder = encode_target(df_clean["species"])
        assert y_encoded.dtype in [np.int32, np.int64]

    def test_encode_target_has_three_classes(self, sample_df):
        df_clean = remove_missing_values(sample_df)
        y_encoded, encoder = encode_target(df_clean["species"])
        assert len(np.unique(y_encoded)) == 3

    def test_scale_features_zero_mean(self, sample_df):
        df_clean = remove_missing_values(sample_df)
        X = df_clean[get_feature_columns()]
        X_scaled, scaler = scale_features(X)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)

    def test_preprocess_data_returns_dict(self, sample_df):
        result = preprocess_data(sample_df)
        assert isinstance(result, dict)
        assert "X" in result
        assert "y" in result


class TestModel:
    @pytest.fixture
    def preprocessed_data(self):
        df = load_penguins()
        return preprocess_data(df)

    def test_split_data_shapes(self, preprocessed_data):
        X, y = preprocessed_data["X"], preprocessed_data["y"]
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

    def test_train_model_returns_model(self, preprocessed_data):
        X, y = preprocessed_data["X"], preprocessed_data["y"]
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_model(X_train, y_train)
        assert hasattr(model, "predict")

    def test_model_accuracy_above_threshold(self, preprocessed_data):
        X, y = preprocessed_data["X"], preprocessed_data["y"]
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        assert accuracy > 0.8


class TestPredictor:
    @pytest.fixture
    def trained_model(self):
        df = load_penguins()
        data = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(data["X"], data["y"])
        model = train_model(X_train, y_train)
        return model, data, X_test

    def test_predict_returns_array(self, trained_model):
        model, data, X_test = trained_model
        predictions = predict(model, X_test)
        assert isinstance(predictions, np.ndarray)

    def test_predict_correct_length(self, trained_model):
        model, data, X_test = trained_model
        predictions = predict(model, X_test)
        assert len(predictions) == len(X_test)

    def test_decode_predictions_returns_species(self, trained_model):
        model, data, X_test = trained_model
        predictions = predict(model, X_test)
        species = decode_predictions(predictions, data["label_encoder"])
        valid_species = ["Adelie", "Chinstrap", "Gentoo"]
        assert all(s in valid_species for s in species)

    def test_predict_single_returns_string(self, trained_model):
        model, data, _ = trained_model
        species = predict_single(
            model, data["scaler"], data["label_encoder"],
            bill_length=39.1, bill_depth=18.7, flipper_length=181.0, body_mass=3750.0
        )
        assert isinstance(species, str)