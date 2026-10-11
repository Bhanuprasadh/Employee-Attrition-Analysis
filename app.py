import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load Model & Dataset (Cached for Speed)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_artifact():
    model_path = "models/attrition_model.pkl"
    if not os.path.exists(model_path):
        from src.train import train_pipeline
        return train_pipeline()
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    data_path = "data/processed/employee_attrition_cleaned.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    return pd.read_csv(data_path)


artifact = load_model_artifact()
df = load_dataset()

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("👥 HR Analytics Hub")
st.sidebar.markdown("**Employee Attrition Prediction System**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    ["🔮 Predict Attrition", "📊 EDA & Insights", "📈 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.caption("M.Tech Academic Project | IBM HR Attrition Dataset")

# =============================================================================
# PAGE 1: PREDICT ATTRITION
# =============================================================================
if page == "🔮 Predict Attrition":
    st.title("🔮 Employee Attrition Risk Predictor")
    st.markdown(
        "Enter employee profile details below to predict the likelihood of turnover "
        "and receive proactive retention recommendations."
    )

    st.subheader("Employee Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 👤 Personal & Commute")
        age = st.slider("Age", 18, 60, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
        distance = st.slider("Distance From Home (km)", 1, 30, 8)
        education = st.selectbox(
            "Education Level",
            [1, 2, 3, 4, 5],
            index=2,
            format_func=lambda x: {
                1: "1 - Below College",
                2: "2 - College",
                3: "3 - Bachelor",
                4: "4 - Master",
                5: "5 - Doctor"
            }[x]
        )

    with col2:
        st.markdown("##### 💼 Job & Department")
        dept = st.selectbox(
            "Department",
            ["Sales", "Research & Development", "Human Resources"],
            index=1
        )
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources"
            ],
            index=1
        )
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years at Company", 0, 30, 4)
        years_with_manager = st.slider("Years With Current Manager", 0, 20, 2)

    with col3:
        st.markdown("##### 💰 Compensation & Workload")
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
        overtime = st.selectbox("OverTime", ["Yes", "No"], index=1)
        job_satisfaction = st.select_slider(
            "Job Satisfaction",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x]
        )
        env_satisfaction = st.select_slider(
            "Environment Satisfaction",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x]
        )
        work_life_balance = st.select_slider(
            "Work-Life Balance",
            options=[1, 2, 3, 4],
            value=3,
            format_func=lambda x: {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}[x]
        )

    st.markdown("---")

    # Predict Button
    if st.button("🎯 Predict Attrition Risk", type="primary"):
        # 1. Start with dataset median/mode defaults to avoid arbitrary zeros
        user_input = artifact["defaults"].copy()

        # 2. Overwrite with user-specified values
        user_input.update({
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital_status,
            "DistanceFromHome": distance,
            "Education": education,
            "Department": dept,
            "JobRole": job_role,
            "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company,
            "YearsWithCurrManager": years_with_manager,
            "MonthlyIncome": monthly_income,
            "OverTime": overtime,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": env_satisfaction,
            "WorkLifeBalance": work_life_balance,
        })

        input_df = pd.DataFrame([user_input])
        preprocessor = artifact["preprocessor"]
        processed_input = preprocessor.transform(input_df)

        # 3. Model prediction (Using balanced Logistic Regression for high Recall and calibration)
        lr_model = artifact["lr_model"]
        prob = lr_model.predict_proba(processed_input)[0, 1]
        risk_pct = round(prob * 100, 1)

        # 4. Display Results
        st.subheader("Prediction Result")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if prob >= 0.50:
                st.error("⚠️ **HIGH ATTRITION RISK**")
            elif prob >= 0.35:
                st.warning("⚠️ **MODERATE ATTRITION RISK**")
            else:
                st.success("✅ **LOW ATTRITION RISK**")

            st.metric(label="Predicted Resignation Probability", value=f"{risk_pct}%")

        with res_col2:
            st.markdown("##### 🔍 Identified Contributing Factors:")
            factors = []
            if overtime == "Yes":
                factors.append("• **Working OverTime**: Overtime is the single strongest factor triggering attrition.")
            if monthly_income < 3500:
                factors.append("• **Below-Average Compensation**: Monthly income is low for this industry segment.")
            if job_satisfaction <= 2:
                factors.append("• **Low Job Satisfaction**: Subdued engagement and satisfaction ratings.")
            if env_satisfaction <= 2:
                factors.append("• **Low Environment Satisfaction**: Friction with the immediate workplace atmosphere.")
            if distance >= 15:
                factors.append("• **Long Commute**: Commute distance exceeds 15 km.")
            if years_with_manager <= 1:
                factors.append("• **Short Manager Relationship**: Under 1 year under the current supervisor.")
            if not factors:
                factors.append("• **Positive Indicators**: Strong satisfaction ratings, no overtime, and reasonable tenure.")

            for factor in factors:
                st.markdown(factor)

        # 5. HR Recommendation
        st.markdown("---")
        st.markdown("##### 💡 Recommended HR Action Plan:")
        if prob >= 0.35:
            st.info(
                "**Retention Strategies:**\n"
                "1. **Workload Assessment**: Rebalance assignments or authorize compensatory off to minimize overtime burnout.\n"
                "2. **1-on-1 Feedback**: Schedule an informal pulse check with the employee to discuss career development.\n"
                "3. **Compensation & Progression**: Review whether recent contributions warrant a market salary adjustment."
            )
        else:
            st.info(
                "**Maintenance Strategy:**\n"
                "The employee exhibits stable retention signals. Continue regular recognition and clear growth pathways."
            )

# =============================================================================
# PAGE 2: EXPLORATORY DATA INSIGHTS
# =============================================================================
elif page == "📊 EDA & Insights":
    st.title("📊 Exploratory Data Insights")
    st.markdown("High-level patterns and distributions from the 1,470 employee records.")

    # KPI Metrics
    total_emp = len(df)
    attrition_cnt = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_cnt / total_emp) * 100
    avg_income = df["MonthlyIncome"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Employees", f"{total_emp:,}")
    m2.metric("Attrited Employees", f"{attrition_cnt}")
    m3.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    m4.metric("Avg Monthly Income", f"${avg_income:,.0f}")

    st.markdown("---")

    # Charts Row 1
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 1. Overtime vs Attrition Rate")
        ot_pct = df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        ot_pct.columns = ["OverTime", "AttritionRate"]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=ot_pct, x="OverTime", y="AttritionRate", hue="OverTime", palette=["#2ecc71", "#e74c3c"], legend=False, ax=ax)
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_ylim(0, 40)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + 0.3, p.get_height() + 1))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Employees working OverTime leave at nearly 3x the rate of non-overtime peers.")

    with c2:
        st.markdown("##### 2. Attrition by Department")
        dept_pct = df.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        dept_pct.columns = ["Department", "AttritionRate"]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=dept_pct, x="Department", y="AttritionRate", hue="Department", palette="Blues_r", legend=False, ax=ax)
        ax.set_ylabel("Attrition Rate (%)")
        ax.set_ylim(0, 30)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + 0.3, p.get_height() + 1))
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Sales exhibits the highest departmental turnover (~20.6%).")

    st.markdown("---")

    # Charts Row 2
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("##### 3. Monthly Income Distribution by Attrition")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", palette=["#2ecc71", "#e74c3c"], legend=False, ax=ax)
        ax.set_ylabel("Monthly Income ($)")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Employees leaving earn substantially lower median monthly salaries.")

    with c4:
        st.markdown("##### 4. Attrition by Job Role")
        role_pct = df.groupby("JobRole")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        role_pct = role_pct.sort_values(by="Attrition", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=role_pct, x="Attrition", y="JobRole", hue="JobRole", palette="viridis", legend=False, ax=ax)
        ax.set_xlabel("Attrition Rate (%)")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Sales Representatives and Laboratory Technicians face the highest turnover.")

# =============================================================================
# PAGE 3: MODEL PERFORMANCE & EVALUATION
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation & Academic Summary")
    st.markdown(
        "Comparison of 4 machine learning classifiers trained on the IBM HR Attrition dataset "
        "using stratified 80-20 train-test splits."
    )

    # Model comparison table
    st.subheader("Model Comparison on Test Set")
    comparison_df = artifact.get("comparison_df")
    if comparison_df is not None:
        st.dataframe(comparison_df)

    st.markdown(
        "> **Why Recall and ROC-AUC Matter for HR:**\n"
        "> In employee attrition, missing an employee who is about to quit (**False Negative**) "
        "> is much more costly than checking in with someone who intends to stay (**False Positive**). "
        "> **Logistic Regression with balanced class weights** achieves the best Recall (~66%) and "
        "> highest ROC-AUC (~0.804)."
    )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Top 10 Drivers of Attrition (Random Forest)")
        rf = artifact["rf_model"]
        feature_names = artifact["feature_names"]

        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=imp_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax)
        ax.set_xlabel("Relative Importance")
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("##### Directional Impact of Key Features (Logistic Regression)")
        lr = artifact["lr_model"]
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": lr.coef_[0]
        })
        top_positive = coef_df.sort_values(by="Coefficient", ascending=False).head(5)
        top_negative = coef_df.sort_values(by="Coefficient").head(5)
        comb_coef = pd.concat([top_positive, top_negative])

        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in comb_coef["Coefficient"]]
        ax.barh(comb_coef["Feature"], comb_coef["Coefficient"], color=colors)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("Log-Odds Impact (Red = Increases Risk, Green = Decreases Risk)")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("##### 🎓 M.Tech Viva Quick Reference")
    st.markdown(
        """
        - **Handling Class Imbalance**: Dataset contains ~16% positive labels (imbalanced). Handled via `stratify=y` and `class_weight='balanced'`.
        - **Preprocessing**: `StandardScaler` for numeric columns, `OneHotEncoder(drop='first')` for categorical columns wrapped in `ColumnTransformer`.
        - **Primary Evaluation Metric**: Recall and ROC-AUC (prioritizing detection of leaving employees).
        - **Top Predictors**: OverTime, MonthlyIncome, Age, TotalWorkingYears, DistanceFromHome.
        """
    )
