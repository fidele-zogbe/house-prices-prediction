# House Prices Prediction

Predicting house sale prices with machine learning, built for the Kaggle competition
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The goal is to predict `SalePrice` for houses in Ames, Iowa, using about 80 features that describe each house (size, quality, neighborhood, year built, and more).

## Result

| Model | Kaggle public score |
|-------|---------------------|
| Random Forest | **0.14750** |

The competition metric is the RMSE between the logarithm of the predicted price and the logarithm of the real price. Lower is better.

## Approach

1. **Exploratory data analysis**
   - Studied the distribution of `SalePrice` (right-skewed).
   - Explored how `SalePrice` relates to `Alley`, `OverallQual` and `GrLivArea` using boxplots, scatter plots and correlation tables.
2. **Missing values**
   - Treated columns where a missing value means "feature absent" (for example `Alley`) as a real category instead of filling them with the most frequent value.
   - Filled the remaining gaps using values computed from the training data.
3. **Outliers**
   - Removed two houses with a very large living area (`GrLivArea` > 4000) and top quality but a very low price, because they broke the trend shown by the other houses.
   - Kept the expensive houses that were consistent with their size and quality.
4. **Feature selection**
   - Computed correlations between all numeric features.
   - Found pairs of features with a correlation above 0.8 and dropped the one less correlated with `SalePrice`, to avoid duplicated information.
5. **Encoding**
   - Used `ColumnTransformer` with `OneHotEncoder` for categorical columns. The encoder is fitted on the training data only and then applied to the validation and test data.
6. **Model**
   - Trained a `RandomForestRegressor` on `log1p(SalePrice)` and converted the predictions back with `expm1`.
   - Evaluated on a validation split before submitting to Kaggle.

## Getting started

1. Clone this repository:
   ```bash
   git clone https://github.com/fidele-zogbe/house-prices-prediction.git
   cd house-prices-prediction
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download `train.csv`, `test.csv` and `sample_submission.csv` from the
   [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
   and place them next to the notebook. The data is not included in this repository.
4. Open the notebook with Jupyter and run the cells from top to bottom.

## Repository contents

- `*.ipynb`: the full analysis and model, from data exploration to the submission file
- `requirements.txt`: Python dependencies

## Next steps

- Try gradient boosting models (`GradientBoostingRegressor`, XGBoost, LightGBM).
- Try regularized linear models (Ridge, Lasso).
- Add engineered features, such as total living area and house age.
- Reduce the skew of numeric features with a log transform.
- Average several models to lower the error.

## Author

Fidèle Togbédji ZOGBE, student in Computer Engineering and Telecommunications at EPAC (University of Abomey-Calavi, Benin).
GitHub: [fidele-zogbe](https://github.com/fidele-zogbe)
