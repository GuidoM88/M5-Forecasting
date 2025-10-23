from setuptools import setup, find_packages

setup(
    name="m5-hierarchical-forecasting",
    version="1.0.0",
    description="Hierarchical LightGBM forecasting for M5 competition",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "lightgbm>=4.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
)