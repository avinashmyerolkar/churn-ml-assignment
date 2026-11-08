# Telco Customer Churn Prediction – ML Assignment 2

## a. Problem statement

A telecom company loses revenue every time a customer cancels their service ("churns"). The goal of this project is to predict, from a customer's account and service details, whether that customer is likely to churn. This is a **binary classification** problem (Churn = Yes / No).

Five classification models are trained on the same dataset, evaluated with six metrics, and compared. A Streamlit web app lets anyone upload test data, choose a model, and view its evaluation metrics, confusion matrix, and classification report.

## b. Dataset description

- **Source:** Telco Customer Churn dataset (IBM sample data, published on Kaggle)
- **Instances:** 7,043 customers
- **Features:** 19 input features (after dropping the `customerID` identifier) + 1 target column
- **Target:** `Churn` (Yes / No). The classes are imbalanced: 26.5% of customers churned and 73.5% stayed.

| Feature group | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Phone services | PhoneService, MultipleLines |
| Internet services | InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

**Preprocessing:**
- `customerID` was dropped (an identifier with no predictive value).
- `TotalCharges` is stored as text and has 11 blank values (brand-new customers with tenure 0). It was converted to numeric and blanks were filled with the median.
- The 3 numeric features (tenure, MonthlyCharges, TotalCharges) were standardised with `StandardScaler`.
- The 16 categorical features were one-hot encoded.
- Data was split 80% train / 20% test (1,409 test rows), stratified on the target so both sets keep the 26.5% churn rate. Random seed: 5068.
- Preprocessing and model are saved together as one scikit-learn `Pipeline`, so the Streamlit app can take raw CSV data directly.

## c. GitHub repository link

https://github.com/avinashmyerolkar/churn-ml-assignment

**Live Streamlit app:** _(link added after deployment)_

### Repository structure

```
churn-ml-assignment/
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── requirements.txt        # pinned library versions
├── README.md
├── test_data.csv           # 20% hold-out test set used in the app
├── data/
│   └── telco_customer_churn.csv
└── model/
    ├── train_models.py     # preprocessing, training, and evaluation script
    ├── metrics.csv         # metrics of all 5 models on the test set
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

**To reproduce:** `pip install -r requirements.txt`, then `python model/train_models.py`, then `streamlit run app.py`.

## d. Models used

All five models were trained on the same training split and evaluated on the same 1,409-row test set.

1. **Logistic Regression** – `max_iter=2000`
2. **Decision Tree Classifier** – `max_depth=6`, `min_samples_leaf=20` (limited depth to avoid overfitting)
3. **K-Nearest Neighbor Classifier** – `n_neighbors=15`, distance-weighted
4. **Naive Bayes Classifier** – Gaussian Naive Bayes
5. **Support Vector Machine** – RBF kernel, `C=1.0`

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8133 | 0.8349 | 0.6961 | 0.5267 | 0.5997 | 0.4890 |
| Decision Tree | 0.7906 | 0.8184 | 0.6468 | 0.4652 | 0.5412 | 0.4196 |
| kNN | 0.7665 | 0.7805 | 0.5676 | 0.5053 | 0.5347 | 0.3806 |
| Naive Bayes | 0.6927 | 0.8137 | 0.4568 | 0.8342 | 0.5904 | 0.4204 |
| SVM | 0.8119 | 0.7942 | 0.7171 | 0.4813 | 0.5760 | 0.4762 |

Precision, Recall, and F1 are reported for the positive class (Churn = Yes).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | The best model overall: highest Accuracy (0.8133), AUC (0.8349), F1 (0.5997), and MCC (0.4890). After one-hot encoding, churn is largely explained by additive effects (month-to-month contract, short tenure, fibre-optic internet, electronic-check payment), which suits a linear model well. Its main weakness is Recall (0.5267): at the default 0.5 threshold it misses about half of the churners, so lowering the threshold would help if catching churners matters more. |
| Decision Tree | Solid but behind Logistic Regression (Accuracy 0.7906, MCC 0.4196). Limiting depth to 6 kept it from overfitting, but a single tree splits the data into coarse groups, which gives a lower AUC (0.8184) and the lowest Recall of all models (0.4652). Its advantage is interpretability: the splits on contract type and tenure are easy to explain to business users. |
| kNN | The weakest model: lowest AUC (0.7805) and MCC (0.3806). With 19 features, most of them one-hot encoded, distances between customers become less meaningful, and because 73.5% of customers stay, a customer's nearest neighbours are mostly non-churners. This pulls predictions toward "No" and lowers its ability to separate the classes. |
| Naive Bayes | Lowest Accuracy (0.6927) and Precision (0.4568), but by far the highest Recall (0.8342): it catches 83% of churners. Its independence assumption is clearly broken here (for example, TotalCharges is roughly tenure × MonthlyCharges, and the internet add-on services are correlated), which makes its probabilities overconfident and leads it to flag many loyal customers as churners. Its AUC (0.8137) is still decent, so it ranks customers reasonably well. Useful when missing a churner is much costlier than a false alarm. |
| SVM | Accuracy (0.8119) and MCC (0.4762) close to Logistic Regression, and the highest Precision (0.7171): when it predicts churn, it is usually right. However, it is conservative, with low Recall (0.4813), and the lowest AUC among the strong models (0.7942), because its RBF decision scores rank customers less smoothly than logistic probabilities. It is also the slowest model to train. |

**Overall:** Because the classes are imbalanced, Accuracy alone is misleading (predicting "No" for everyone would already score about 73.5%). MCC and F1 give a fairer picture, and on both, **Logistic Regression** is the best choice for this dataset. If the business goal is to catch as many churners as possible, **Naive Bayes** (or Logistic Regression with a lower threshold) is the better pick.

## Streamlit app features

- Upload a test CSV (must contain the 19 features and the `Churn` column)
- Model selection dropdown (all 5 models)
- Display of all 6 evaluation metrics
- Confusion matrix, ROC curve, and classification report
- "Compare all models" tab with a metrics table and bar chart
- Download a sample test file and the model's predictions
