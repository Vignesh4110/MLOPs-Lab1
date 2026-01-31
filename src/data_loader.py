import seaborn as sns
import pandas as pd


def load_penguins() -> pd.DataFrame:
    """Load the penguins dataset from seaborn."""
    df = sns.load_dataset("penguins")
    return df


def get_feature_columns() -> list:
    """Return list of feature column names."""
    return ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def get_target_column() -> str:
    """Return target column name."""
    return "species"