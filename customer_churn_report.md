# Customer Churn Prediction Report

## 1. Introduction
Customer churn prediction is a common problem in telecom and subscription-based businesses. It focuses on identifying customers who are likely to stop using a service so that interventions can be targeted before the customer leaves. This project applies a machine learning approach to predict churn using a structured customer dataset and evaluates the model on a held-out test set.

The problem is important because retaining existing customers is usually more cost-effective than acquiring new ones. In industries such as telecommunications, even a modest reduction in churn can lead to noticeable improvements in revenue, service efficiency, and customer satisfaction.

## 2. Objective
The main objective of this project is to build and evaluate a predictive model that can distinguish between customers who stay and customers who churn. The model should help uncover the main behavioral and usage patterns associated with churn while remaining interpretable enough to support decision-making.

## 3. Dataset Description
This project uses the Iranian Churn dataset from the UCI Machine Learning Repository. The dataset contains customer-level information and a binary target variable called Churn, where:

- 0 = customer stayed
- 1 = customer churned

The dataset includes several numerical features related to customer behavior, including usage activity, complaint history, age group, tariff plan, customer value, and call-related indicators. The original dataset also contains a field named Status, but its meaning is not clearly described in the metadata, so it was excluded from the prediction model to avoid relying on an ambiguous or outcome-adjacent variable.

The final model uses 12 selected feature columns and excludes the target variable and the unclear Status field.

## 4. Methodology
The workflow followed in this project is as follows:

1. Data loading and validation
2. Feature selection and preprocessing
3. Train-test splitting with stratification
4. Model training using a class-weighted Random Forest Classifier
5. Performance evaluation on the held-out test set
6. Visualization of churn patterns and feature importance
7. Development of a Streamlit interface for interactive prediction

### 4.1 Data Preparation
The dataset was checked for consistency and schema validity. Numeric columns were validated, and the target variable was confirmed to be binary. Missing numeric values were handled using a median imputer inside a scikit-learn pipeline, allowing the model to be robust to small missing-value issues.

### 4.2 Train-Test Split
A stratified 80:20 split was used to keep the churn proportion similar in both training and test sets. This is important because churn is a minority class and a random split could distort the evaluation.

### 4.3 Modeling Approach
A Random Forest Classifier was selected due to its ability to handle tabular data, capture nonlinear relationships, and provide interpretable feature importance. To address class imbalance, a balanced subsampling strategy was used.

The model also includes median imputation in the pipeline, ensuring that input values are processed consistently before prediction.

## 5. Evaluation Results
The trained model was evaluated on the held-out test set containing 630 customers.

### Summary Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.963 |
| Precision | 0.873 |
| Recall | 0.899 |
| F1-score | 0.886 |
| ROC-AUC | 0.987 |

### Confusion Matrix

| Actual \ Predicted | Stayed | Churned |
| --- | ---: | ---: |
| Stayed | 518 | 13 |
| Churned | 10 | 89 |

The model correctly identified 89 churned customers and misclassified only 10 churners as non-churners. It also performed well in identifying non-churn customers, with 518 correct predictions for the stayed class.

### Classification Report
For the churn class:

- Precision: 0.873
- Recall: 0.899
- F1-score: 0.886

This indicates the model is effective at identifying likely churners while maintaining a strong balance between false positives and false negatives.

## 6. Key Findings and Feature Importance
The project also analyzes how different features relate to churn. The most influential features according to permutation importance are:

1. Frequency of use
2. Seconds of Use
3. Complains
4. Customer Value
5. Distinct Called Numbers
6. Call Failure
7. Subscription Length
8. Frequency of SMS
9. Age Group
10. Age
11. Charge Amount
12. Tariff Plan

The strongest signals appear to come from usage behavior and customer complaint indicators. This is consistent with business intuition: customers who use the service less often, generate lower value, or report complaints may be at greater risk of churn.

The project also includes visual charts showing:

- overall churn distribution
- churn rate by complaints
- churn rate by age group
- churn rate by tariff plan
- customer value by churn outcome
- feature importance ranking

These visualizations help explain the patterns found in the historical dataset.

## 7. Business Interpretation
From a business perspective, the model suggests that churn is not random. It is strongly associated with customer engagement and service experience. In practice, this means telecom operators can focus retention strategies on customers who:

- have low usage frequency
- show reduced call or service activity
- have more complaints
- have lower customer value or weaker engagement

Retention efforts such as proactive customer support, personalized offers, and service recovery programs would likely be more effective when targeted at these at-risk segments.

## 8. Limitations
This project is useful for educational and analytical purposes, but it has important limits:

- The model is trained on historical data and reflects associations, not causal relationships.
- The project excludes the ambiguous Status feature because its meaning is unclear.
- The dataset is from a specific telecom context and may not generalize to all markets.
- The model should not be treated as a production decision system without additional validation and business review.

It is therefore best regarded as a prototype analytical tool rather than a fully operational customer retention system.

## 9. Conclusion
This project successfully demonstrates how a machine learning-based churn prediction system can be built and evaluated using a realistic telecom dataset. The Random Forest model achieved strong performance, with an accuracy of 96.3% and a ROC-AUC of 98.7%, indicating high predictive power on the held-out test set.

The most important churn indicators are usage behavior and complaint-related factors, which provides valuable actionable insight for customer retention strategies. The project also includes a user-friendly Streamlit interface that allows users to explore patterns and test predictions interactively.

Overall, the project shows that predictive analytics can be an effective tool for identifying customers at risk of churn and supporting data-driven retention decisions.
