# Volatility Prediction: Traditional vs. Machine Learning Methods

This project compares traditional econometric volatility forecasting models against machine learning approaches for short-, medium-, and longer-horizon volatility prediction across major U.S. equity index ETFs. The analysis constructs realized volatility from high-frequency intraday data, engineers HAR-style and intraday risk features, trains multiple forecasting models under a strict temporal split, and evaluates model performance across market regimes.

## Project Overview

The goal of this project is to evaluate whether machine learning models can improve volatility forecasts relative to standard benchmark models such as GARCH and HAR. The notebook focuses on three broad equity market proxies:

- **S&P 500** represented by SPY
- **NASDAQ** represented by QQQQ
- **Russell 2000** represented by IWM

Forecasts are generated for multiple horizons:

- **1-day realized volatility**
- **5-day realized volatility**
- **22-day realized volatility**

The project is designed as an end-to-end volatility forecasting pipeline, covering data extraction, realized volatility construction, feature engineering, model training, statistical testing, interpretability, and regime analysis.

## Repository Contents

```text
.
├── code_final_v1.ipynb        # Main project notebook
├── cache/                     # Cached model predictions, created during execution
├── data_cache/                # Cached WRDS intraday data, created during execution
└── README.md                  # Project documentation
```

## Data Sources

The primary dataset is intraday market data downloaded from **WRDS**. The notebook uses 5-minute intraday prices from 2005 to 2025 to construct daily realized volatility measures.

The notebook also downloads **VIX** data using `yfinance` for regime-based analysis.

### Assets Used

| Index Proxy | ETF Symbol | Description |
|---|---|---|
| SP500 | SPY | S&P 500 ETF proxy |
| NASDAQ | QQQQ | NASDAQ ETF proxy |
| Russell | IWM | Russell 2000 ETF proxy |

## Methodology

### 1. Intraday Data Collection

The notebook connects to WRDS, downloads 5-minute intraday price data, and caches the raw data locally to avoid repeated WRDS queries.

Key configuration:

```python
START = "2005-01-01"
END = "2025-12-31"
BAR_FREQ = "5min"
```

### 2. Realized Volatility Construction

Using intraday log returns, the notebook constructs several realized volatility and jump-related measures:

- Realized variance / realized volatility
- Bipower variation
- Jump variation
- Relative jump contribution
- Realized quarticity

These are computed over the following windows:

- 1 day
- 5 days
- 22 days

The realized volatility framework is designed to separate continuous volatility from jump-driven volatility and to provide richer inputs for forecasting models.

### 3. Feature Engineering

The feature set extends the standard HAR volatility model with additional lagged and intraday-derived predictors.

Main feature groups include:

- HAR core features: daily, weekly, and monthly realized volatility lags
- Extended realized volatility lags
- Jump features: jump variation, bipower variation, and relative jump contribution
- Realized quarticity
- Leverage proxies from daily returns
- Volatility-of-volatility regime features

All features are lagged to avoid look-ahead bias.

### 4. Train, Validation, and Test Split

The project uses a strict chronological split:

| Split | Period |
|---|---|
| Training | 2005–2017 |
| Validation | 2018–2020 |
| Test | 2021 onward |

No random shuffling is used. This is important because volatility forecasting is a time-series problem, and random splits would introduce leakage.

## Models Implemented

The notebook compares both traditional and machine learning models.

### Traditional Models

- **GARCH(1,1)** using expanding-window out-of-sample forecasts
- **HAR** using OLS
- **HAR-J** with jump variation
- **HAR-Leverage** with return-based leverage proxies
- **HAR-CJX** with continuous, jump, and quarticity extensions

### Machine Learning Models

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor
- XGBoost Linear Booster
- Multilayer Perceptron
- LSTM neural network, when PyTorch is available

## Evaluation Metrics

Models are evaluated using standard forecast accuracy and volatility-loss metrics:

- RMSE
- MAE
- R²
- QLIKE loss

The notebook also performs **Diebold-Mariano tests** to compare forecast errors statistically across models.

## Interpretability and Diagnostics

The notebook includes several interpretation and diagnostic sections:

### Return Distribution Analysis

Plots empirical return distributions against normal distributions to highlight heavy tails and non-normality in equity index returns.

### ACF and PACF Analysis

Plots autocorrelation and partial autocorrelation of realized volatility to motivate HAR-style lag structures.

### Feature Importance

Compares feature impact across models using:

- XGBoost gain-based importance
- Random Forest impurity-based importance
- Lasso standardized coefficients
- LSTM correlation-based proxy importance

## Regime Analysis

The project evaluates model performance under different market conditions using three approaches.

### 1. Event-Based Regimes

Hard-coded historical market regimes include:

- Pre-crisis period
- Global Financial Crisis
- Post-crisis expansion
- COVID period
- 2022 rate-hike shock
- 2023–2024 low-volatility expansion

### 2. VIX-Based Regimes

The notebook classifies periods based on VIX levels:

| Regime | VIX Level |
|---|---|
| Calm | VIX < 15 |
| Normal | 15 ≤ VIX < 25 |
| Elevated | 25 ≤ VIX < 35 |
| Crisis | VIX ≥ 35 |

### 3. Rolling R² Analysis

A rolling 60-day R² is computed to show how model performance changes over time, especially during crisis and high-volatility periods.

## Key Outputs

The notebook produces:

- Realized volatility datasets for each index and horizon
- Out-of-sample predictions for traditional and ML models
- Model comparison tables
- Diebold-Mariano test results
- Feature importance charts
- Regime-specific performance tables
- Rolling R² visualizations
- VIX regime plots

## How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn wrds arch statsmodels scikit-learn xgboost scipy yfinance torch
```

`torch` is optional. If PyTorch is not installed, the notebook skips the LSTM model.

### 2. Configure WRDS Credentials

Update the WRDS username in the notebook:

```python
WRDS_USERNAME = "your_wrds_username"
```

When the notebook runs, it prompts for the WRDS password and downloads the intraday data.

### 3. Run the Notebook

Open and run:

```text
code_final_v1.ipynb
```

The notebook automatically creates local cache folders for data and model predictions.

### 4. Use Caching for Faster Re-runs

The notebook saves downloaded data and model predictions locally. To force all models to retrain, set:

```python
FORCE_RETRAIN = True
```

Otherwise, cached results are loaded where available.

## Technical Stack

- Python
- NumPy
- pandas
- WRDS
- arch
- statsmodels
- scikit-learn
- XGBoost
- PyTorch
- yfinance
- matplotlib
- seaborn

## Project Strengths

This project demonstrates:

- High-frequency data handling
- Realized volatility construction
- Traditional volatility modeling
- Machine learning model comparison
- Time-series validation discipline
- Forecast evaluation and statistical testing
- Feature importance analysis
- Market regime analysis

## Limitations and Future Improvements

Potential extensions include:

- Adding more asset classes such as commodities, FX, rates, or crypto
- Testing longer forecast horizons beyond 22 trading days
- Using additional realized volatility estimators
- Adding option-implied volatility features
- Incorporating macroeconomic variables
- Testing transformer-based or temporal convolutional models
- Building a fully automated forecasting dashboard

## Suggested Resume Bullet

> Built an end-to-end volatility forecasting pipeline using 5-minute WRDS intraday data, comparing GARCH/HAR benchmarks against ML models across S&P 500, NASDAQ, and Russell 2000 proxies with regime-based performance analysis.
