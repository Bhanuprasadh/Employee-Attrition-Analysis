# Employee Attrition Analysis & Prediction System

An end-to-end Machine Learning and HR Analytics project built to predict employee turnover risk and identify key drivers influencing employee retention.

Developed as part of an **M.Tech academic curriculum** using clean, modular Python and an interactive **Streamlit** dashboard.

---

## 📌 Project Overview & Objectives

Employee attrition (turnover) poses substantial direct and indirect costs to organizations, including recruitment expenses, training overhead, and loss of institutional knowledge. The goals of this project are:
1. **Analyze Key Turnover Drivers**: Identify organizational and demographic factors that trigger employee attrition (e.g., overtime, compensation, commute distance, work satisfaction).
2. **Predict Attrition Risk**: Build and evaluate machine learning models capable of identifying employees at high risk of resigning.
3. **Deploy Proactive Decision Support**: Provide an intuitive web application via Streamlit enabling HR managers to input employee profiles, evaluate turnover probabilities, and view targeted retention recommendations.

---

## 📊 Dataset Description

The project utilizes the **IBM HR Employee Attrition** dataset:
- **Total Records**: 1,470 employees
- **Features**: 31 columns (after dropping 4 zero-variance / identifier columns: `EmployeeCount`, `StandardHours`, `Over18`, and `EmployeeNumber`)
- **Target Variable**: `Attrition` (`Yes` = 237 [16.1%], `No` = 1,233 [83.9%])
- **Class Imbalance**: ~16% positive rate (handled using stratified sampling and class-weighted learning).

---

## 📁 Project Architecture & Directory Structure

```text
Employee Attrition Analysis/
├── data/
│   ├── raw/                  # Original IBM HR dataset
│   └── processed/            # Cleaned data and preprocessed train/test matrices
├── notebooks/                # Step-by-step academic analysis workflow
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_exploratory_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_evaluation.ipynb
│   └── 08_model_interpretation.ipynb
├── models/
│   ├── attrition_model.pkl   # Serialized model artifact & preprocessor
│   ├── preprocessor.pkl      # ColumnTransformer pipeline
│   └── model_comparison.csv # Performance comparison across algorithms
├── src/
│   └── train.py              # Standalone clean model training script
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation & viva guide
```

---

## 🔬 Methodology & Workflow

The analysis follows standard Data Science lifecycle phases across 8 organized notebooks:

1. **`01_data_understanding.ipynb`**: Dataset schema inspection, variable taxonomy, null check, summary statistics.
2. **`02_data_cleaning.ipynb`**: Duplicate verification, removal of constant/uninformative columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`), data integrity validation.
3. **`03_exploratory_analysis.ipynb`**: Univariate and bivariate exploratory data analysis across departmental roles, tenure, salary, and satisfaction scores.
4. **`04_statistical_analysis.ipynb`**: Hypothesis testing (Chi-Square test of independence with Cramér's V, Welch's t-test, and Mann–Whitney U test) to establish statistically significant associations with attrition.
5. **`05_feature_engineering.ipynb`**: Binary target mapping, 80/20 stratified splitting, numerical standardization with `StandardScaler`, and nominal categorical encoding with `OneHotEncoder(drop='first')`.
6. **`06_model_training.ipynb`**: Training candidate classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) with balanced class weights to address minority class rarity.
7. **`07_model_evaluation.ipynb`**: Evaluation via Confusion Matrix, Classification Reports, and ROC-AUC curves.
8. **`08_model_interpretation.ipynb`**: Feature importance analysis and directional log-odds extraction for actionable business takeaways.

---

## 📈 Model Performance & Comparison

Evaluated on an unseen test set (294 samples, 20% holdout):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.755** | **0.356** | **0.660** | **0.463** | **0.804** |
| **Decision Tree (Balanced)** | 0.772 | 0.357 | 0.532 | 0.427 | 0.652 |
| **Random Forest (Balanced)** | 0.837 | 0.444 | 0.085 | 0.143 | 0.759 |
| **Gradient Boosting** | 0.850 | 0.588 | 0.213 | 0.312 | 0.794 |

### Why Logistic Regression is the Primary Model:
In employee turnover modeling, **Recall** and **ROC-AUC** are the paramount metrics:
- **False Negative Cost**: Failing to identify an employee who leaves results in costly talent loss and replacement disruption (50–100% of annual salary).
- **False Positive Cost**: Identifying a satisfied employee as at-risk merely prompts proactive engagement and recognition.
- **Logistic Regression with Balanced Weights** correctly flags **66% of all attrited employees** while maintaining a strong ROC-AUC of **0.804**.

---

## 🔑 Key Findings & Business Insights

1. **OverTime is the Strongest Predictor**: Employees working overtime resign at nearly **3x the rate** of employees who do not work overtime (30.5% vs 10.4%).
2. **Compensation & Stock Options**: Lower salary bands and level-0 stock options correlate strongly with turnover.
3. **Tenure & Early Career Mobility**: Employees within their first 1–2 years at the company or in their current role are significantly more likely to leave.
4. **Commute Distance**: Longer commute distances (>15 km) steadily increase attrition probability.
5. **Satisfaction Indicators**: Low job involvement, low environment satisfaction, and poor work-life balance serve as clear leading indicators of resignation.

---

## 🚀 How to Run the Project

### 1. Activate Environment
Ensure your Python virtual environment is active:
```bash
source eaa-venv/bin/activate
```

### 2. (Optional) Retrain the Model
To re-run the training pipeline from the terminal:
```bash
python src/train.py
```

### 3. Launch the Streamlit Web Application
Run the dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

The application provides 3 modules:
- **🔮 Predict Attrition**: Interactive calculator to input employee parameters and compute resignation risk.
- **📊 EDA & Insights**: Interactive charts showing high-level employee attrition patterns.
- **📈 Model Performance**: Evaluation metrics, ROC curve, and feature importances.

---

## 🎓 Academic Viva & Presentation Cheat Sheet

**Q1: How did you handle the class imbalance in the target variable?**
> *Answer*: The dataset has an 84:16 class split. We addressed this by:
> 1. Using stratified sampling (`stratify=y`) during train-test splitting so both subsets preserve the 16% ratio.
> 2. Setting `class_weight='balanced'` in the estimators, which penalizes errors on the minority class inversely proportional to class frequencies.

**Q2: Why did you drop `EmployeeCount`, `StandardHours`, `Over18`, and `EmployeeNumber`?**
> *Answer*: `EmployeeCount` (always 1), `StandardHours` (always 80), and `Over18` (always 'Y') have zero variance and provide zero predictive signal. `EmployeeNumber` is an arbitrary sequential ID that would lead to overfitting.

**Q3: Why did you choose Recall and ROC-AUC over Accuracy?**
> *Answer*: With 84% majority class, a naive model predicting "No" for everyone would achieve 84% accuracy while catching zero leaving employees. Recall measures our ability to detect actual resignations, and ROC-AUC evaluates discrimination power across all probability thresholds.

**Q4: How does the model provide actionable recommendations?**
> *Answer*: The model calculates turnover probability while an inference engine inspects dominant risk flags (e.g., overtime, low satisfaction, low compensation) to offer targeted retention guidance.
