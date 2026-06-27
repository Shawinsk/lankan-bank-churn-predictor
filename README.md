# 🇱🇰 Bank of Ceylon (BOC) - Customer Churn Prediction System

<div align="center">

[![Language](https://img.shields.io/badge/Python-3.10-darkblue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Framework](https://img.shields.io/badge/LightGBM-green.svg?style=flat&logo=scikit-learn&logoColor=white)](https://lightgbm.readthedocs.io/)
[![API Security](https://img.shields.io/badge/API_Auth-Enabled-brightgreen.svg?style=flat)](#api-documentation)
[![Target Bank](https://img.shields.io/badge/Bank-Bank_of_Ceylon-gold.svg?style=flat)](https://www.boc.lk)

**Developer:** SHAWN KAWYANATH

</div>

An end-to-end Machine Learning pipeline and secure REST API developed specifically for **Bank of Ceylon (BOC)** to predict customer churn risk using localized Sri Lankan customer attributes, CRIB credit statuses, and LKR balance profiles.

---

## 📋 Table of Contents
1. [Overview & Sri Lankan Context](#overview)
2. [Technology Stack](#tech-stack)
3. [Model Attributes & Features](#features)
4. [REST API Documentation & Security](#api-documentation)
5. [Model & Algorithm Details (LightGBM)](#algorithm-details)
6. [Model Validation & Test Results](#test-results)
7. [Running the System Locally](#running-locally)
8. [Model Training & Pipelines](#pipelines)

---

## 1. Overview & Sri Lankan Context <a id="overview"></a>
Predicting customer churn (identifying customers likely to leave the bank) allows BOC to take proactive retention steps. This system is tailored specifically for the Sri Lankan banking landscape:
* **Salary Profiles:** Tailored using DCS Labour Force Survey statistics (median salary of LKR 35,000 - 55,000).
* **CRIB Credit Risks:** Integrates Credit Information Bureau (CRIB) default rates (peaked at 13.9% in Q3 2023).
* **Digital Banking Users:** Calibrated according to digital banking adoption metrics.
* **Geographical Scope:** Covers all 25 Sri Lankan districts (e.g., Colombo, Gampaha, Kandy, Galle, Jaffna, Vavuniya, Matara, etc.) rather than European regions.

---

## 2. Technology Stack <a id="tech-stack"></a>
This project leverages the following modern technologies, frameworks, and libraries:

### 🐍 Core Programming & Logic
* **Python (v3.10+)**: Core programming language.

### 🤖 Machine Learning & Data Processing
* **LightGBM**: High-performance gradient boosting framework used to train the churn classifier.
* **Scikit-Learn**: Used for feature transformation pipelines (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) and evaluation metrics.
* **Pandas & NumPy**: For efficient tabular data manipulation, cleaning, and array math.
* **Joblib**: For serializing and saving the trained model and preprocessing pipeline.

### 🔄 MLOps & Experiment Tracking
* **MLflow**: Used for logging experiment parameters, performance metrics, and model versioning/registry.
* **DVC (Data Version Control)**: Orchestrates pipeline steps, version-controls data files, and manages stage dependency graphs.
* **Dagshub**: Serves as the remote hosting platform for the MLflow tracking server.

### 🌐 Web & API Development
* **Flask**: Micro web framework in Python to build the user interface and secure REST API endpoints.
* **SQLite**: Lightweight database storage to save prediction history and user records programmatically.
* **HTML5 & CSS3**: Custom Dark-Gold themed user interface matched to the Bank of Ceylon (BOC) brand styling.

---

## 3. Model Attributes & Features <a id="features"></a>
The LightGBM classification model is trained on Sri Lankan customer records using the following features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **Age** | `int` | Customer age in years |
| **Tenure_Years** | `int` | Number of years active with BOC |
| **Account_Balance_LKR** | `float` | Customer's total account balance in Sri Lankan Rupees (LKR) |
| **District** | `string` | One of the 25 Sri Lankan districts where the account is held |
| **CRIB_Status** | `string` | Credit Bureau Status (`Clean` or `Defaulted`) |
| **Recent_Loan_Rejected** | `int` | Binary indicator if a loan request was recently rejected (`1` = Yes, `0` = No) |
| **Digital_Banking_User**| `int` | Binary indicator if the user is active on BOC digital portals (`1` = Yes, `0` = No) |
| **Churn** (Target) | `int` | Binary target classification (`1` = Left the bank, `0` = Active) |

---

## 4. REST API Documentation & Security <a id="api-documentation"></a>
The system exposes secure endpoints for programmatic integration with BOC core banking portals or CRM systems.

### 🔒 Authentication
All requests must present a valid API Key in the HTTP Request Headers.
* **Header Name:** `X-API-Key`
* **Default Value:** `BOC-CHURN-SECRET-KEY-2026` *(can be overridden in production using the `BOC_API_KEY` environment variable)*
* Alternative header: `Authorization: Bearer <key>`

---

### 📥 1. Single Predict Endpoint
Predicts churn risk for a single customer.

* **URL:** `/api/predict`
* **Method:** `POST`
* **Headers:**
  ```http
  X-API-Key: BOC-CHURN-SECRET-KEY-2026
  Content-Type: application/json
  ```
* **Request JSON Payload:**
  ```json
  {
    "CustomerId": 15620007,
    "Surname": "Senanayake",
    "Age": 42,
    "Tenure_Years": 10,
    "Account_Balance_LKR": 1200000.0,
    "District": "Colombo",
    "CRIB_Status": "Clean",
    "Recent_Loan_Rejected": 0,
    "Digital_Banking_User": 1
  }
  ```
* **Response JSON:**
  ```json
  {
    "CustomerId": 15620007,
    "Surname": "Senanayake",
    "Churn_Prediction": "No"
  }
  ```

---

### 📥 2. Batch Predict Endpoint
Predicts churn risk for a list of customer objects.

* **URL:** `/api/predict_batch`
* **Method:** `POST`
* **Headers:**
  ```http
  X-API-Key: BOC-CHURN-SECRET-KEY-2026
  Content-Type: application/json
  ```
* **Request JSON Payload:**
  ```json
  [
    {
      "CustomerId": 15620008,
      "Surname": "Perera",
      "Age": 29,
      "Tenure_Years": 1,
      "Account_Balance_LKR": 15000.0,
      "District": "Kandy",
      "CRIB_Status": "Defaulted",
      "Recent_Loan_Rejected": 1,
      "Digital_Banking_User": 0
    },
    {
      "CustomerId": 15620009,
      "Surname": "Silva",
      "Age": 45,
      "Tenure_Years": 10,
      "Account_Balance_LKR": 1500000.0,
      "District": "Colombo",
      "CRIB_Status": "Clean",
      "Recent_Loan_Rejected": 0,
      "Digital_Banking_User": 1
    }
  ]
  ```
* **Response JSON:**
  ```json
  [
    {
      "CustomerId": 15620008,
      "Surname": "Perera",
      "Churn_Prediction": "Yes"
    },
    {
      "CustomerId": 15620009,
      "Surname": "Silva",
      "Churn_Prediction": "No"
    }
  ]
  ```

---

## 5. Model & Algorithm Details (LightGBM) <a id="algorithm-details"></a>

### What is LightGBM?
The system utilizes **LightGBM (Light Gradient Boosting Machine)**, a high-performance, distributed gradient boosting framework developed by Microsoft. It is designed for tree-based learning algorithms and is optimized for speed, low memory usage, and high predictive accuracy.

### Why LightGBM was selected for BOC Customer Churn:
1. **Handling Categorical Data:** Handles high-cardinality categorical attributes (like the 25 Sri Lankan `District` values) highly efficiently.
2. **Speed and Efficiency:** Uses Histogram-based algorithms to bucket continuous values, resulting in significantly faster training speeds.
3. **Class Imbalance Support:** In customer churn prediction, the active class (`1` - Churn) is typically the minority class. LightGBM provides `scale_pos_weight` parameter support to automatically balance class weights during training.
4. **Regularization:** Built-in L1 (`reg_alpha`) and L2 (`reg_lambda`) regularization configurations prevent overfitting to local noise.

### Preprocessing & Feature Engineering:
Before training or making predictions, raw customer attributes undergo transformation inside a structured Scikit-Learn `ColumnTransformer` pipeline:
* **Numerical Scaling (`StandardScaler`)**: Applied to `Age`, `Tenure_Years`, and `Account_Balance_LKR`. It standardizes features by removing the mean and scaling to unit variance, ensuring large values (like account balance in LKR) do not dominate the gradient updates.
* **Categorical Encoding (`OneHotEncoder`)**: Applied to high-cardinality columns `District` (25 values) and `CRIB_Status` (`Clean` / `Defaulted`). Converts categorical strings into numeric binary vectors without asserting any natural ordering.
* **Binary Passthrough**: Passes columns `Recent_Loan_Rejected` and `Digital_Banking_User` directly as boolean indicators.

### Model Training Rationale:
1. **Stratified Train-Test Split (80/20)**: Splitting is stratified based on the target column (`Churn`) to preserve minority class ratios in both splits, ensuring robust model validation.
2. **Class Weighting (`scale_pos_weight`)**: Set dynamically to the exact ratio of non-churned to churned customers (approx. `3.0`). This forces the boosting loss function to penalize errors on minority churn predictions more heavily.
3. **Early Stopping Callback**: Configured with `early_stopping(stopping_rounds=50)` using the validation set to stop tree iterations once loss convergence halts, successfully avoiding overfitting.

### Hyperparameter Configurations:
The model is optimized using the following parameters:
* `n_estimators`: `500` (Number of boosted trees to fit)
* `learning_rate`: `0.05` (Step size shrinkage to prevent overfitting)
* `max_depth`: `6` (Limits tree depth to control model complexity)
* `num_leaves`: `40` (Maximum tree leaves for base learners)
* `min_child_samples`: `20` (Minimum data points in a leaf to prevent overfitting)
* `scale_pos_weight`: Balanced dynamically based on the ratio of active customers to churned customers.

### Model Evaluation Metrics:
* **Accuracy:** `75.09%`
* **Precision:** `65.07%`
* **Recall:** `69.51%`
* **F1-Score:** `67.21%`
* **ROC-AUC:** `0.7781`

---

## 6. Model Validation & Test Results <a id="test-results"></a>
The prediction pipeline and the secure REST API endpoints have been rigorously tested against localized edge-cases. The test results verify both prediction logic accuracy and API authentication security.

### 🧪 1. Functional Customer Scenarios (Model Logic)
The model was evaluated against 4 real-world banking profiles:
1. **Loyal Colombo Customer** (High balance, long tenure, clean CRIB):
   * *Status:* `PASS`
   * *Expected:* `No Churn` | *Got:* `No Churn`
2. **High Risk defaulted customer** (Low balance, defaulted CRIB, recent loan rejected, Kandy branch):
   * *Status:* `PASS`
   * *Expected:* `Churn Risk` | *Got:* `Churn Risk`
3. **Remote district zero balance non-digital customer** (Mullaitivu, inactive, 0 tenure):
   * *Status:* `PASS`
   * *Expected:* `Churn Risk` | *Got:* `Churn Risk`
4. **Average Gampaha middle-class digital user** (Clean CRIB, active member):
   * *Status:* `PASS`
   * *Expected:* `No Churn` | *Got:* `No Churn`

---

### 🛡️ 2. API Security & Key Authorization Tests
The endpoint security was tested against request credential headers:
1. **No API Key Header provided:**
   * *Status:* `PASS`
   * *Result:* Rejected with **`401 Unauthorized`** (Access Blocked).
2. **Invalid API Key Header provided:**
   * *Status:* `PASS`
   * *Result:* Rejected with **`401 Unauthorized`** (Access Blocked).
3. **Valid API Key via `X-API-Key` Header:**
   * *Status:* `PASS`
   * *Result:* Accepted with **`200 OK`** (Successful Churn Prediction).
4. **Valid API Key via `Authorization: Bearer <key>` Header:**
   * *Status:* `PASS`
   * *Result:* Accepted with **`200 OK`** (Successful Churn Prediction).

---

## 7. Running the System Locally <a id="running-locally"></a>

### Step 1: Create a virtual environment
```bash
python -m venv venv
```

### Step 2: Activate the environment
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask Web Application
```bash
python app.py
```
Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser. The application includes a premium Dark-Gold BOC branded web interface, a single prediction form, a CSV batch upload system, and a database prediction dashboard (`/db_dashboard`).

### Step 5: Test the API Security
Run the integrated API verification suite:
```bash
python test_model.py
```

---

## 8. Model Training & Pipelines <a id="pipelines"></a>

To retrain the model or regenerate simulated customer data:
1. **Regenerate simulated Sri Lankan data:**
   ```bash
   python generate_sl_data.py
   ```
2. **Train the LightGBM classifier:**
   ```bash
   python train_sl_model.py
   ```
3. **Execute MLflow Evaluation pipeline stage:**
   ```bash
   python src/mlFlowProject/pipeline/stage_05_model_evaluation.py
   ```
   Evaluation results will log metrics (accuracy, precision, recall, f1) and register the resulting model artifacts inside MLflow/Dagshub.

---

## 👨‍💻 Developer Information
* **Lead ML Engineer:** SHAWN KAWYANATH
* **Project Scope:** Bank of Ceylon (BOC) Churn Prediction Analytics
