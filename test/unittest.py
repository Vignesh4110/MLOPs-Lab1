import unittest
import pandas as pd
import numpy as np
from src.data_loader import load_penguins, get_feature_columns, get_target_column
from src.preprocessor import remove_missing_values, encode_target, scale_features, preprocess_data
from src.model import split_data, train_model, evaluate_model
from src.predictor import predict, decode_predictions, predict_single


class TestDataLoader(unittest.TestCase):
    def test_load_penguins_returns_dataframe(self):
        df = load_penguins()
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_penguins_has_rows(self):
        df = load_penguins()
        self.assertGreater(len(df), 0)

    def test_feature_columns_exist(self):
        df = load_penguins()
        feature_cols = get_feature_columns()
        for col in feature_cols:
            self.assertIn(col, df.columns)

    def test_target_column_exists(self):
        df = load_penguins()
        target_col = get_target_column()
        self.assertIn(target_col, df.columns)


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = load_penguins()

    def test_remove_missing_values(self):
        df_clean = remove_missing_values(self.df)
        self.assertEqual(df_clean.isnull().sum().sum(), 0)

    def test_encode_target_returns_integers(self):
        df_clean = remove_missing_values(self.df)
        y_encoded, encoder = encode_target(df_clean["species"])
        self.assertIn(y_encoded.dtype, [np.int32, np.int64])

    def test_encode_target_has_three_classes(self):
        df_clean = remove_missing_values(self.df)
        y_encoded, encoder = encode_target(df_clean["species"])
        self.assertEqual(len(np.unique(y_encoded)), 3)

    def test_scale_features_zero_mean(self):
        df_clean = remove_missing_values(self.df)
        X = df_clean[get_feature_columns()]
        X_scaled, scaler = scale_features(X)
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10))

    def test_preprocess_data_returns_dict(self):
        result = preprocess_data(self.df)
        self.assertIsInstance(result, dict)
        self.assertIn("X", result)
        self.assertIn("y", result)


class TestModel(unittest.TestCase):
    def setUp(self):
        df = load_penguins()
        self.data = preprocess_data(df)
        self.X = self.data["X"]
        self.y = self.data["y"]

    def test_split_data_shapes(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(y_train) + len(y_test), len(self.y))

    def test_train_model_returns_model(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        model = train_model(X_train, y_train)
        self.assertTrue(hasattr(model, "predict"))

    def test_model_accuracy_above_threshold(self):
        X_train, X_test, y_train, y_test = split_data(self.X, self.y)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertGreater(accuracy, 0.8)


class TestPredictor(unittest.TestCase):
    def setUp(self):
        df = load_penguins()
        self.data = preprocess_data(df)
        X_train, self.X_test, y_train, y_test = split_data(self.data["X"], self.data["y"])
        self.model = train_model(X_train, y_train)

    def test_predict_returns_array(self):
        predictions = predict(self.model, self.X_test)
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_correct_length(self):
        predictions = predict(self.model, self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_decode_predictions_returns_species(self):
        predictions = predict(self.model, self.X_test)
        species = decode_predictions(predictions, self.data["label_encoder"])
        valid_species = ["Adelie", "Chinstrap", "Gentoo"]
        for s in species:
            self.assertIn(s, valid_species)

    def test_predict_single_returns_string(self):
        species = predict_single(
            self.model, self.data["scaler"], self.data["label_encoder"],
            bill_length=39.1, bill_depth=18.7, flipper_length=181.0, body_mass=3750.0
        )
        self.assertIsInstance(species, str)


if __name__ == "__main__":
    unittest.main()