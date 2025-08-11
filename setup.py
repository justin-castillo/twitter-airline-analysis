from setuptools import find_packages, setup

setup(
    name="twitter_airline_analysis",
    version="0.1.0",
    description="Airline sentiment analysis (TF-IDF + Logistic Regression) with API + CLI.",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.100",
        "uvicorn>=0.22",
        "scikit-learn>=1.2",
        "joblib>=1.3",
        "numpy>=1.24",
        "pandas>=1.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-cov>=4",
            "ruff>=0.3",
            "mypy>=1.5",
            "httpx>=0.24",  # handy for API tests
        ],
    },
    entry_points={
        "console_scripts": [
            # NOTE: because of package_dir={"": "src"}, your import path does NOT include "src".
            "twitter-predict=twitter_airline_analysis.predict:main",
        ],
    },
    include_package_data=True,
)
