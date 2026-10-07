# House Price Prediction

A machine learning regression project that predicts house prices from property attributes.

The project covers data exploration, feature preparation, model comparison and Ridge Regression tuning.

## Repository Files

| File | Description |
| --- | --- |
| [House Price Prediction.ipynb](House%20Price%20Prediction.ipynb) | Notebook containing the analysis, training code and saved results. |
| [House Price Prediction.pptx](House%20Price%20Prediction.pptx) | Presentation summarising the workflow and findings. |
| [housing_price_data.csv](housing_price_data.csv) | Dataset used for training and evaluation. |

## Dataset

The included dataset contains **545 records and 8 columns**, with no missing values.

- **Target:** `Price ($)`
- **Features:** city, house area, number of bedrooms, number of toilets, stories and renovation status.
- **Excluded identifier:** `House ID`

The dataset includes houses from Boston, Chicago, Denver, New York and Seattle.

## Project Workflow

1. Explore price distributions, property attributes and correlations.
2. Remove `House ID` from the predictors.
3. Encode renovation status:
   - Unfurnished: `0`
   - Semi-furnished: `1`
   - Furnished: `2`
4. One-hot encode city.
5. Use an **80/20 train/test split**, with `random_state=42`:
   - Training: **436 records**
   - Test: **109 records**
6. Apply **RobustScaler** to house area, fitted on the training set.
7. Compare regression models against a mean-prediction baseline.
8. Remove city dummy variables and tune Ridge Regression using **GridSearchCV**.

## Models Compared

| Model | Test MAE ($) | Test R² |
| --- | ---: | ---: |
| Dummy Regressor | 174,862 | -0.018 |
| K-Nearest Neighbours | 121,182 | 0.436 |
| Linear Regression | 118,061 | 0.520 |
| Decision Tree | 125,770 | 0.484 |
| Support Vector Regression | 174,914 | -0.084 |
| Ridge Regression (`alpha=1`) | 117,944 | 0.520 |
| Lasso Regression | 118,060 | 0.520 |

These are the initial model results recorded in the notebook.

## Hyperparameter Tuning

Ridge Regression was tuned using **5-fold cross-validation**, with R² as the scoring metric.

Alpha values tested:

```python
[0.01, 0.1, 1, 10, 100, 1000]
```

The best alpha was **10** for both feature sets.

| Feature Set | Best Cross-validation R² |
| --- | ---: |
| With city variables | 0.5198 |
| Without city variables | 0.5286 |

## Final Model Results

The selected model is **Ridge Regression with `alpha=10`, without city variables**.

It uses five predictors:
- House area
- Number of bedrooms
- Number of toilets
- Stories
- Renovation status

| Metric | Test Result |
| --- | ---: |
| Mean Absolute Error (MAE) | $115,884.44 |
| Mean Squared Error (MSE) | 23,992,023,309.72 |
| R² | 0.52534 |

The model's average absolute prediction error is approximately **$116,000**. It explains about **52.5% of the variation in test-set prices**.

MAE is approximately **33.7% lower than the dummy baseline**.

These results come from the notebook's saved outputs.

## Tools Used

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## How to Run

1. Download or clone the repository.
2. Install the required packages:

   ```bash
   python -m pip install pandas numpy scikit-learn matplotlib seaborn notebook
   ```

3. Keep `housing_price_data.csv` beside the notebook.
4. Replace the absolute Windows path in the data-loading cell with:

   ```python
   df = pd.read_csv("housing_price_data.csv")
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `House Price Prediction.ipynb` and run the cells in order.

## Limitations and Future Improvements

- The dataset is small, and results are based on one random train/test split.
- Several models were compared on the same test set. A fresh holdout set would support final evaluation.
- Scaling occurs before cross-validation. A pipeline would allow preprocessing to be fitted within each fold.
- Only house area is scaled, so raw coefficient sizes are not directly comparable as feature importance.
- Future experiments could test full feature scaling, log-price transformation and residual analysis.

## Author

**Yan Myoe Naing**  
Applied AI and Analytics, Singapore Polytechnic
