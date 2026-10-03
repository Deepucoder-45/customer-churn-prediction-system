import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_utils import (
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    download_dataset,
    load_dataset,
)
from src.train_model import MODEL_FILE, train_model


METRICS_FILE = PROJECT_ROOT / "reports" / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
INTEGER_FEATURES = {
    "Call Failure",
    "Complains",
    "Subscription Length",
    "Charge Amount",
    "Seconds of Use",
    "Frequency of use",
    "Frequency of SMS",
    "Distinct Called Numbers",
    "Age Group",
    "Tariff Plan",
    "Age",
}
CODED_FEATURES = {"Complains", "Age Group", "Tariff Plan"}


@st.cache_data
def get_dataset() -> pd.DataFrame:
    download_dataset()
    return load_dataset()


@st.cache_resource
def get_model_bundle() -> dict[str, object]:
    if not MODEL_FILE.exists():
        train_model()
    return joblib.load(MODEL_FILE)


@st.cache_data
def get_metrics() -> dict[str, object]:
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def show_dashboard(data: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    churn_count = int(data[TARGET_COLUMN].sum())
    columns = st.columns(3)
    columns[0].metric("Customers", f"{len(data):,}")
    columns[1].metric("Churned", f"{churn_count:,}")
    columns[2].metric("Churn rate", f"{data[TARGET_COLUMN].mean():.1%}")

    chart_rows = [
        ["churn_distribution.png", "churn_by_complains.png"],
        ["churn_by_age_group.png", "churn_by_tariff_plan.png"],
        ["customer_value_by_churn.png", "feature_importance.png"],
    ]
    for chart_row in chart_rows:
        chart_columns = st.columns(2)
        for column, filename in zip(chart_columns, chart_row):
            image_path = FIGURES_DIR / filename
            if image_path.exists():
                column.image(str(image_path), width="stretch")

    complaint_rates = data.groupby("Complains")[TARGET_COLUMN].mean()
    if 0 in complaint_rates and 1 in complaint_rates:
        st.info(
            "In this dataset, the churn rate is "
            f"{complaint_rates[1]:.1%} for customers with complaints and "
            f"{complaint_rates[0]:.1%} for customers without complaints."
        )
    st.caption(
        "The charts describe this historical dataset. They show associations, "
        "not proof that a feature causes churn."
    )


def make_prediction_form(data: pd.DataFrame, model: object) -> None:
    st.subheader("Enter customer information")
    st.caption(
        "Use the coded values shown for age group and tariff plan. "
        "Inputs are limited to the ranges observed in the training dataset."
    )
    values: dict[str, int | float] = {}
    with st.form("churn_prediction_form"):
        input_columns = st.columns(3)
        for index, feature in enumerate(FEATURE_COLUMNS):
            column = input_columns[index % len(input_columns)]
            observed_values = data[feature]
            if feature in CODED_FEATURES:
                options = sorted(int(value) for value in observed_values.unique())
                values[feature] = column.selectbox(
                    f"{feature} code", options, key=f"input_{feature}"
                )
            elif feature in INTEGER_FEATURES:
                minimum = int(observed_values.min())
                maximum = int(observed_values.max())
                default = int(round(observed_values.median()))
                values[feature] = column.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=1,
                    key=f"input_{feature}",
                )
            else:
                minimum = float(observed_values.min())
                maximum = float(observed_values.max())
                default = float(observed_values.median())
                values[feature] = column.number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=10.0,
                    format="%.2f",
                    key=f"input_{feature}",
                )
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        customer = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0, 1])
        if prediction == 1:
            st.warning(f"Likely to churn. Estimated probability: {probability:.1%}")
        else:
            st.success(f"Less likely to churn. Estimated probability: {probability:.1%}")
        st.caption(
            "This is an educational prediction from a historical dataset, "
            "not a guarantee about an individual customer."
        )


def show_model_report(metrics: dict[str, object]) -> None:
    st.subheader("Held-out test results")
    st.caption(
        f"Evaluated on {metrics['test_rows']} customers not used to fit the model."
    )
    metric_columns = st.columns(5)
    for column, metric_name in zip(
        metric_columns, ("accuracy", "precision", "recall", "f1", "roc_auc")
    ):
        column.metric(metric_name.replace("_", " ").title(), f"{metrics[metric_name]:.3f}")

    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual stayed", "Actual churned"],
        columns=["Predicted stayed", "Predicted churned"],
    )
    st.write("Confusion matrix")
    st.dataframe(matrix, width="stretch")
    st.write("Classification report")
    report = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report.round(3), width="stretch")
    st.caption(
        "Recall measures how many churners the model found. Precision measures "
        "how many predicted churners actually churned. F1 balances the two."
    )


def main() -> None:
    st.set_page_config(page_title="Customer Churn Analysis", layout="wide")
    st.title("Customer Churn Prediction and Analysis")
    st.write("Explore churn patterns, review model performance, and test a prediction.")

    try:
        with st.spinner("Preparing the dataset and model on first run..."):
            data = get_dataset()
            model_bundle = get_model_bundle()
            metrics = get_metrics()
    except Exception as error:
        st.error(f"The project could not prepare its data or model: {error}")
        st.stop()

    dashboard_tab, prediction_tab, report_tab = st.tabs(
        ["Analysis", "Predict", "Model evaluation"]
    )
    with dashboard_tab:
        show_dashboard(data)
    with prediction_tab:
        make_prediction_form(data, model_bundle["model"])
    with report_tab:
        show_model_report(metrics)

    with st.expander("Dataset source and limitations"):
        st.write(
            "Dataset: Iranian Churn, UCI Machine Learning Repository, "
            "DOI 10.24432/C5JW3Z. The UCI metadata does not specify a license, "
            "so review the dataset page terms before redistributing the raw file."
        )
        st.write(
            "The binary Status column is excluded from prediction because its "
            "meaning is not explained in the UCI metadata. This project is for "
            "education and is not a production customer-retention system."
        )


if __name__ == "__main__":
    main()