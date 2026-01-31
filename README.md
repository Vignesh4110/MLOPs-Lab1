# Penguin Classifier

Classifies penguin species (Adelie, Chinstrap, Gentoo) using the Palmer Penguins dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
# Pytest
pytest tests/test_pytest.py -v

# Unittest
python -m unittest tests/test_unittest.py -v
```

## Quick Usage

```python
from src import load_penguins, preprocess_data, train_model
from src.model import split_data

df = load_penguins()
data = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(data["X"], data["y"])
model = train_model(X_train, y_train)
```

## CI/CD

GitHub Actions runs pytest and unittest on push/PR to main.