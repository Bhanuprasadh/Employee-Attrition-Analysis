"""
train.py
--------
Simple, clean training script for the Employee Attrition project.
Trains Logistic Regression and Random Forest models and saves the artifacts.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_pipeline(data_path="data/processed/employee_attrition_cleaned.csv", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load cleaned data
    df = pd.read_csv(data_path)
    X = df.drop("Attrition", axis=1)
    y = (df["Attrition"] == "Yes").astype(int)

    # 2. Identify column types
    num_cols = X.select_dtypes(exclude=["object", "str"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    # 3. Stratified split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Preprocessing: StandardScaler + OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_encoded_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_encoded_names

    # 5. Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # 6. Evaluate models
    results = []
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1-Score": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 3),
        })

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(os.path.join(model_dir, "model_comparison.csv"), index=False)
    print("\n=== Model Evaluation Results ===")
    print(comparison_df.to_string(index=False))

    # 7. Compute default feature values for inference
    defaults = {}
    for col in X.columns:
        if col in cat_cols:
            defaults[col] = X[col].mode()[0]
        else:
            defaults[col] = X[col].median()

    # 8. Save artifact
    artifact = {
        "lr_model": models["Logistic Regression"],
        "rf_model": models["Random Forest"],
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "defaults": defaults,
        "comparison_df": comparison_df,
    }
    
    artifact_path = os.path.join(model_dir, "attrition_model.pkl")
    joblib.dump(artifact, artifact_path)
    print(f"\nModel bundle saved to: {artifact_path}")
    return artifact


if __name__ == "__main__":
    train_pipeline()
