import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_utils import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_dataset,
)


MODEL_FILE = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_FILE = PROJECT_ROOT / "reports" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def save_analysis_charts(
    data: pd.DataFrame, importance: pd.Series, figures_dir: Path = FIGURES_DIR
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    churn_counts = data[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["Stayed", "Churned"], churn_counts.reindex([0, 1], fill_value=0))
    axis.set(title="Customer churn labels", ylabel="Number of customers")
    figure.tight_layout()
    figure.savefig(figures_dir / "churn_distribution.png", dpi=150)
    plt.close(figure)

    for feature, title in [
        ("Complains", "Churn rate by complaints"),
        ("Age Group", "Churn rate by age group"),
        ("Tariff Plan", "Churn rate by tariff plan"),
    ]:
        churn_rates = data.groupby(feature, sort=True)[TARGET_COLUMN].mean()
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(churn_rates.index.astype(str), churn_rates.values)
        axis.set(title=title, xlabel=feature, ylabel="Fraction who churned", ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(figures_dir / f"churn_by_{feature.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    value_groups = [
        data.loc[data[TARGET_COLUMN] == label, "Customer Value"] for label in [0, 1]
    ]
    axis.boxplot(value_groups, tick_labels=["Stayed", "Churned"], showfliers=False)
    axis.set(title="Customer value by churn outcome", ylabel="Customer value")
    figure.tight_layout()
    figure.savefig(figures_dir / "customer_value_by_churn.png", dpi=150)
    plt.close(figure)

    top_importance = importance.sort_values().tail(10)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(top_importance.index, top_importance.values)
    axis.set(title="Most useful features in the test set", xlabel="F1 decrease when shuffled")
    figure.tight_layout()
    figure.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close(figure)


def train_model() -> dict[str, object]:
    data = load_dataset()
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].astype(int)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    churn_probabilities = model.predict_proba(test_features)[:, 1]

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(test_target, predictions)),
        "precision": float(precision_score(test_target, predictions, zero_division=0)),
        "recall": float(recall_score(test_target, predictions, zero_division=0)),
        "f1": float(f1_score(test_target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_target, churn_probabilities)),
        "confusion_matrix": confusion_matrix(test_target, predictions).tolist(),
        "classification_report": classification_report(
            test_target, predictions, output_dict=True, zero_division=0
        ),
        "train_rows": int(len(train_target)),
        "test_rows": int(len(test_target)),
        "churn_rate": float(target.mean()),
        "excluded_features": ["Status"],
    }

    importance_result = permutation_importance(
        model,
        test_features,
        test_target,
        scoring="f1",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.Series(importance_result.importances_mean, index=FEATURE_COLUMNS)
    metrics["permutation_importance"] = {
        feature: float(value) for feature, value in importance.sort_values(ascending=False).items()
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_columns": FEATURE_COLUMNS},
        MODEL_FILE,
    )
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_analysis_charts(data, importance)
    return metrics


if __name__ == "__main__":
    results = train_model()
    print("Model evaluation on the held-out test set:")
    for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{metric_name}: {results[metric_name]:.3f}")
    print(f"Saved model to {MODEL_FILE}")