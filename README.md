# ames-housing-price-prediction
A machine learning pipeline that predicts home sale prices in Ames, Iowa by leveraging detailed property features with thorough preprocessing and model tuning for real estate valuation.

## Context

As part of a data science role at a fictional real estate aggregator (similar to Zillow), the objective is to estimate home prices using historical sales data. Accurate price predictions can support users—whether buyers or sellers—in making more informed decisions.

## Data

The dataset contains housing records from Ames, Iowa, collected between 2006 and 2010. It includes a wide range of features such as:

- Lot size, frontage, and shape
- Neighborhood and zoning
- Year built and renovation history
- Interior features (square footage, bathrooms, fireplaces)
- Garage and basement details
- Sale date and condition

Train and test sets were provided in `.parquet` format.

## Methods

The following steps were taken to build and tune the model:

1. **Feature-Engineering**
   - Numerical features were imputed with the median and scaled.
   - Categorical features were imputed with the most frequent category and one-hot encoded.
   - Data types were optimized by converting object columns to pandas `CategoricalDtype`.

2. **Modeling**
   - A `HistGradientBoostingRegressor` was used, which is efficient for mixed-type tabular data and robust to missing values.
   - A pipeline was built using `make_pipeline` and `make_column_transformer`.
   - Hyperparameter tuning was performed using `GridSearchCV`.
   - Model performance was evaluated using `mean_absolute_percentage_error`.

3. **Export**
   - The best model was serialized using `joblib` for use in production scenarios.


## Results

The tuned model achieved a low mean absolute percentage error (MAPE) on the test set, indicating solid predictive performance and practical applicability for real-world real estate pricing tasks.




