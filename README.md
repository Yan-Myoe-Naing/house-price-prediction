# Machine Failure Detection

A machine learning project that predicts machine failures from operating data. It covers data exploration, preprocessing, model comparison and hyperparameter tuning.

Since failures represent only **3.39%** of the dataset, the project focuses on **recall and F1 score** to assess how well models detect failures.

## Repository Files

| File | Description |
| --- | --- |
| [Machine Failure Detection(1).ipynb](Machine%20Failure%20Detection%281%29.ipynb) | Notebook containing the analysis, model training and saved results. |
| [Machine Failure Detection(1).pptx](Machine%20Failure%20Detection%281%29.pptx) | Slides summarising the workflow, results and limitations. |
| [factory_data.csv](factory_data.csv) | Dataset used in the project. |

## Dataset

The included dataset contains **20,000 records and 9 columns**.

| Class | Records | Proportion |
| --- | ---: | ---: |
| Normal operation (`0`) | 19,322 | 96.61% |
| Machine failure (`1`) | 678 | 3.39% |

The target column is `Machine Status`.

The input features are:
- Quality
- Ambient temperature
- Process temperature
- Rotation speed
- Torque
- Tool wear

`Unique ID` and `Product ID` are excluded from model training.

## Workflow

1. Explore missing values, class distribution, feature distributions and correlations.
2. Fill missing quality values with the mode, process temperature with forward-fill and rotation speed with the mean.
3. Encode quality as `L = 0`, `M = 1` and `H = 2`.
4. Split the data into **16,000 training records** and **4,000 test records** using a stratified 80/20 split.
5. Explore scaling and two-component PCA for baseline models.
6. Compare classification models.
7. Tune Gradient Boosting without quality using **50 Optuna trials** and **5-fold stratified cross-validation**, maximising F1.

Ensemble models use the original features or a subset without quality, rather than PCA inputs.

## Models Compared

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Gaussian Naive Bayes
- Random Forest
- Gradient Boosting

## Selected Model Results

The selected model is a **tuned Gradient Boosting Classifier without the quality feature**.

The results below come from the notebook's saved test outputs.

| Metric | Result |
| --- | ---: |
| Accuracy | 99.60% |
| Precision | 96.88% |
| Recall | 91.18% |
| F1 score | 0.9394 |

### Confusion Matrix

| Actual / Predicted | Normal | Failure |
| --- | ---: | ---: |
| Normal | 3,860 | 4 |
| Failure | 12 | 124 |

The selected model:
- Detects **124 of 136 failures**
- Misses **12 failures**
- Produces **4 false alarms**

The majority-class baseline achieves **96.60% test accuracy** but detects no failures. This demonstrates why accuracy alone is insufficient for this dataset.

The best Optuna cross-validation F1 is **0.88435**, which is separate from the final test F1.

## Tools Used

Python, pandas, NumPy, scikit-learn, Optuna, Matplotlib, Seaborn and Jupyter Notebook.

## How to Run

1. Download or clone the repository.
2. Install the required packages:

   ```bash
   python -m pip install pandas numpy scikit-learn optuna matplotlib seaborn notebook
   ```

3. Keep `factory_data.csv` beside the notebook.
4. Replace the absolute Windows path in the data-loading cell with:

   ```python
   df = pd.read_csv("factory_data.csv")
   ```

5. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `Machine Failure Detection(1).ipynb` and run the cells in order.

The Optuna search may take some time. Tuning results may vary between runs.

## Limitations and Future Improvements

- Fit missing-value imputation within the training set and cross-validation folds. The current notebook performs imputation before splitting.
- Check whether the dataset's row order supports forward-fill.
- Reserve a fresh holdout set for final evaluation.
- Check performance across different training seeds.

## Author

**Yan Myoe Naing**  
Applied AI and Analytics, Singapore Polytechnic**
