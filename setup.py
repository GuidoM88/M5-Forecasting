from setuptools import find_packages, setup

setup(
    name='m5-hierarchical-forecasting',
    version='2.0.0',
    description='Reproducible M5 fixed-origin forecasting backtest and artifact API',
    author='Guido Morgante',
    packages=find_packages(include=['src', 'src.*', 'api', 'api.*', 'scripts']),
    python_requires='>=3.10',
    install_requires=['numpy>=1.26,<3', 'pandas>=2.1,<3', 'lightgbm>=4.6,<5',
                      'PyYAML>=6,<7'],
    extras_require={
        'api': ['fastapi>=0.115,<1', 'uvicorn>=0.30,<1'],
        'dev': ['pytest>=8,<10', 'httpx>=0.27,<1', 'nbformat>=5,<6'],
        'tracking': ['mlflow>=2.20,<4'],
        'notebooks': ['jupyterlab>=4,<5', 'matplotlib>=3.8,<4', 'scikit-learn>=1.4,<2'],
        'download': ['kaggle>=1.6,<3'],
    },
)
