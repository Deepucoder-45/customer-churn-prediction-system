# Customer Churn Prediction and Analysis System

A beginner-friendly classification project that explores customer churn in a real telecom dataset, trains a Random Forest classifier, evaluates it on a held-out test set, and provides a Streamlit interface for analysis and predictions.

## Dataset

This project uses the **Iranian Churn** dataset from the UCI Machine Learning Repository. UCI reports 3,150 rows, 13 numeric features, a binary `Churn` target, and no missing values.

- Dataset page: https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset
- DOI: `10.24432/C5JW3Z`
- Download URL used by the project: https://archive.ics.uci.edu/static/public/563/data.csv

The UCI metadata does not specify a license. The raw CSV is therefore downloaded locally and ignored by Git. Review the dataset page terms before redistributing it.

The source includes a binary `Status` field, but does not explain what it means. It has a strong association with `Churn`, so this project excludes it from the model and prediction form rather than risk relying on an unclear, outcome-adjacent field.

## Run Locally

Use Python 3.10 or later. In PowerShell, from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.train_model
streamlit run app.py
```

On first launch, the app downloads the UCI dataset and trains the model automatically if either is missing. The explicit download and training commands above are useful when following the machine learning steps separately.

Run the focused data-validation tests with:

```powershell
python -m unittest discover -s tests
```

## What the Project Does

1. **Loads and validates data.** Column spacing is normalized, numeric fields are checked, and the target is verified as binary.
2. **Preprocesses features.** `Status` and the target are not model inputs. A median imputer is fitted inside the training pipeline, which also makes the model resilient to a missing numeric value.
3. **Splits data.** A stratified 80/20 split keeps the churn proportion similar in training and test sets. A fixed random seed makes the result reproducible.
4. **Trains a classifier.** A class-weighted Random Forest is used because churners are a minority in this dataset. The forest combines many decision trees.
5. **Evaluates the model.** Accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a per-class report are saved under `reports/`.
6. **Explores patterns.** The training step saves charts for churn distribution, churn by complaints/age group/tariff plan, customer value by outcome, and test-set permutation importance.
7. **Makes predictions.** The web form accepts the 12 selected customer features and shows the predicted class and estimated churn probability.

The charts describe associations in this historical dataset; they do not show that a feature causes churn. The model is for a college assignment and should not be used to make real customer decisions.

## Project Structure

```text
customer-churn-prediction/
├── app.py                    # Streamlit analysis, evaluation, and prediction pages
├── data/
│   ├── raw/                   # Downloaded UCI CSV (not committed)
│   └── processed/             # Reserved for later preprocessing outputs
├── models/                    # Trained model artifact
├── reports/
│   ├── figures/               # Churn and feature-importance charts
│   └── metrics.json           # Held-out test results
├── src/
│   ├── data_utils.py          # Downloading, schema checks, and data loading
│   ├── download_data.py       # Dataset download command
│   └── train_model.py         # Preprocessing, training, evaluation, charts
├── tests/
│   └── test_data_utils.py     # Focused dataset validation tests
├── requirements.txt
└── .gitignore
```

## GitHub and Deployment Notes

The raw CSV is excluded from Git; the app can download it from UCI when it starts. A trained model and generated reports are produced by `python -m src.train_model`. Before publishing or deploying, review UCI's current dataset terms and confirm the hosting environment can access the UCI download URL.