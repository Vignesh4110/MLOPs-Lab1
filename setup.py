from setuptools import setup, find_packages

setup(
    name="penguin-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "numpy>=1.23.0",
    ],
    python_requires=">=3.8",
)