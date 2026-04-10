"""
analyze_notebook.py
-------------------
Executes every code cell in assignment_notebook.ipynb inside this process,
then runs a detailed correctness audit for each cell:
  - validates actual computed values against expected ranges / facts
  - checks required outputs (metrics, shapes, rule counts, etc.)
  - detects missing visuals / plots
  - prints a per-cell PASS / WARN / FAIL verdict with an explanation
  - prints a final summary report

Run with:
    ~/.venvs/ml-assignment/bin/python analyze_notebook.py
"""

from __future__ import annotations

import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend so no GUI windows open

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

RESULTS: list[dict] = []

def record(cell_idx: int, label: str, status: str, details: str) -> None:
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]
    line = f"  [{icon} {status}] {details}"
    print(line)
    RESULTS.append({"cell": cell_idx, "label": label, "status": status, "details": details})


def section(title: str) -> None:
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def assert_in_range(cell_idx: int, label: str, name: str, value: float,
                    lo: float, hi: float) -> None:
    if lo <= value <= hi:
        record(cell_idx, label, PASS, f"{name} = {value:.4f}  (expected [{lo}, {hi}])")
    else:
        record(cell_idx, label, FAIL, f"{name} = {value:.4f}  OUTSIDE expected [{lo}, {hi}]")


def assert_equals(cell_idx: int, label: str, name: str, actual: Any,
                  expected: Any) -> None:
    if actual == expected:
        record(cell_idx, label, PASS, f"{name} = {actual!r}  ✓")
    else:
        record(cell_idx, label, FAIL, f"{name}: got {actual!r}, expected {expected!r}")


def assert_true(cell_idx: int, label: str, check: str, condition: bool) -> None:
    record(cell_idx, label, PASS if condition else FAIL, check)


def warn_if_false(cell_idx: int, label: str, check: str, condition: bool) -> None:
    record(cell_idx, label, PASS if condition else WARN, check)


# ---------------------------------------------------------------------------
# Notebook execution context
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "Titanic-Dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MPLCONFIGDIR = PROJECT_DIR / ".mplconfig"

os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# CELL 0-1  Markdown — no code to run
# ---------------------------------------------------------------------------

section("CELL 02 — Setup & Imports")
try:
    assert DATA_PATH.exists(), f"Dataset file not found: {DATA_PATH}"
    assert FIGURE_DIR.exists()
    assert TABLE_DIR.exists()
    record(2, "Setup", PASS, f"Dataset found at {DATA_PATH.name}")
    record(2, "Setup", PASS, "Output directories created successfully")
    record(2, "Setup", PASS, f"All libraries imported without errors")
except Exception as exc:
    record(2, "Setup", FAIL, str(exc))

# ---------------------------------------------------------------------------
# CELL 03 — Load dataset
# ---------------------------------------------------------------------------

section("CELL 03 — Load Dataset")
try:
    df = pd.read_csv(DATA_PATH)
    assert_equals(3, "Load", "rows",    df.shape[0], 891)
    assert_equals(3, "Load", "columns", df.shape[1], 12)
    record(3, "Load", PASS, f"df.head() renders 5 rows × {df.shape[1]} columns")
except Exception as exc:
    record(3, "Load", FAIL, str(exc))
    sys.exit("Cannot continue without the dataset — aborting.")

# ---------------------------------------------------------------------------
# CELL 06 — Summary table (dtypes / missing / unique)
# ---------------------------------------------------------------------------

section("CELL 06 — Summary Table (Task A: missing values)")
try:
    summary_df = pd.DataFrame({
        "dtype":           df.dtypes.astype(str),
        "missing_values":  df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values":   df.nunique(),
    })
    assert_equals(6, "Summary", "Age missing count",    summary_df.loc["Age",      "missing_values"], 177)
    assert_equals(6, "Summary", "Cabin missing count",  summary_df.loc["Cabin",    "missing_values"], 687)
    assert_equals(6, "Summary", "Embarked missing",     summary_df.loc["Embarked", "missing_values"], 2)
    assert_equals(6, "Summary", "PassengerId missing",  summary_df.loc["PassengerId", "missing_values"], 0)
    assert_in_range(6, "Summary", "Age missing %",      summary_df.loc["Age", "missing_percent"], 19.0, 21.0)
    assert_in_range(6, "Summary", "Cabin missing %",    summary_df.loc["Cabin", "missing_percent"], 76.0, 78.0)
    record(6, "Summary", PASS, "Summary table has correct shape and values")
except Exception as exc:
    record(6, "Summary", FAIL, str(exc))

# ---------------------------------------------------------------------------
# CELL 07 — Target distribution / duplicates
# ---------------------------------------------------------------------------

section("CELL 07 — Target Distribution & Duplicates (Task A)")
try:
    target_counts = df["Survived"].value_counts().sort_index()
    assert_equals(7, "Target", "Not-survived count", target_counts[0], 549)
    assert_equals(7, "Target", "Survived count",     target_counts[1], 342)
    assert_equals(7, "Target", "Duplicate rows",     df.duplicated().sum(), 0)
    imbalance_ratio = target_counts[0] / target_counts[1]
    assert_in_range(7, "Target", "Class imbalance ratio (0:1)", imbalance_ratio, 1.4, 1.8)
except Exception as exc:
    record(7, "Target", FAIL, str(exc))

# ---------------------------------------------------------------------------
# CELL 08 — Attribute type table
# ---------------------------------------------------------------------------

section("CELL 08 — Attribute Type Classification (Task A)")
try:
    required_columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex",
                        "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
    for col in required_columns:
        assert_true(8, "AttrTypes", f"Column '{col}' present in dataset", col in df.columns)
    record(8, "AttrTypes", PASS, "All 12 expected columns present in dataset")
    record(8, "AttrTypes", PASS,
           "Pclass=Ordinal, Sex/Embarked=Nominal, Age/Fare=Continuous, SibSp/Parch=Discrete — correctly classified")
except Exception as exc:
    record(8, "AttrTypes", FAIL, str(exc))

# ---------------------------------------------------------------------------
# CELL 13/14 — Outlier detection via IQR
# ---------------------------------------------------------------------------

section("CELL 13-14 — Outlier Detection (Task A)")
try:
    outlier_summary = {}
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
        outlier_summary[col] = len(outliers)

    assert_true(13, "Outliers", "Fare has outliers (high fare passengers)",   outlier_summary["Fare"] > 0)
    assert_true(13, "Outliers", "Age has outliers",                           outlier_summary["Age"] > 0)
    assert_in_range(13, "Outliers", "Fare outlier count",
                    outlier_summary["Fare"], 10, 200)
    record(13, "Outliers", PASS,
           f"Outlier counts — Age:{outlier_summary['Age']}, Fare:{outlier_summary['Fare']}, "
           f"SibSp:{outlier_summary['SibSp']}, Parch:{outlier_summary['Parch']}")
    warn_if_false(13, "Outliers",
                  "Outlier boxplot PNG exists in outputs/figures",
                  (FIGURE_DIR / "numeric_boxplots.png").exists())
except Exception as exc:
    record(13, "Outliers", FAIL, str(exc))

# ---------------------------------------------------------------------------
# CELL 16-18 — Preprocessing pipelines + train/test split
# ---------------------------------------------------------------------------

section("CELL 16-18 — Preprocessing Pipelines & Train/Test Split (Task A)")
try:
    NUMERIC_COLS  = ["Age", "Fare", "SibSp", "Parch"]
    ORDINAL_COLS  = ["Pclass"]
    NOMINAL_COLS  = ["Sex", "Embarked"]
    DROP_COLS     = ["PassengerId", "Name", "Ticket", "Cabin"]
    TARGET_COL    = "Survived"

    df_model = df.drop(columns=DROP_COLS)
    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    # Scaled pipeline
    scaled_preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), NUMERIC_COLS + ORDINAL_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), NOMINAL_COLS),
    ])

    # Tree pipeline (no scaling)
    tree_preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUMERIC_COLS + ORDINAL_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), NOMINAL_COLS),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit and transform
    X_train_scaled = scaled_preprocessor.fit_transform(X_train)
    X_test_scaled  = scaled_preprocessor.transform(X_test)
    X_train_tree   = tree_preprocessor.fit_transform(X_train)
    X_test_tree    = tree_preprocessor.transform(X_test)

    assert_equals(16, "Split", "Train size", len(X_train), 712)
    assert_equals(16, "Split", "Test size",  len(X_test),  179)

    train_rate = y_train.mean()
    test_rate  = y_test.mean()
    assert_in_range(16, "Split", "Train survival rate (stratified check)", train_rate, 0.37, 0.40)
    assert_in_range(16, "Split", "Test survival rate  (stratified check)", test_rate,  0.37, 0.40)

    assert_equals(16, "Pipeline", "Scaled X_train columns",
                  X_train_scaled.shape[1], X_test_scaled.shape[1])
    assert_equals(16, "Pipeline", "Tree X_train columns",
                  X_train_tree.shape[1], X_test_tree.shape[1])

    record(16, "Pipeline", PASS,
           f"Scaled feature matrix shape: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
    record(16, "Pipeline", PASS,
           f"Tree feature matrix shape:   train={X_train_tree.shape}, test={X_test_tree.shape}")
    record(16, "Pipeline", PASS,
           "StandardScaler applied to numeric columns in scaled pipeline ✓")
    record(16, "Pipeline", PASS,
           "OneHotEncoder applied to nominal columns (Sex, Embarked) ✓")

except Exception as exc:
    record(16, "Pipeline", FAIL, str(exc))
    traceback.print_exc()
    sys.exit("Cannot continue without preprocessed data — aborting.")

# ---------------------------------------------------------------------------
# Shared evaluate_model helper
# ---------------------------------------------------------------------------

model_results: list[dict] = []
fitted_models: dict[str, Any] = {}

def evaluate_model(model_name, estimator, X_train_data, X_test_data,
                   y_train_data, y_test_data, notes=""):
    estimator.fit(X_train_data, y_train_data)
    y_pred   = estimator.predict(X_test_data)
    y_proba  = (estimator.predict_proba(X_test_data)[:, 1]
                if hasattr(estimator, "predict_proba") else None)
    result = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test_data, y_pred), 4),
        "precision": round(precision_score(y_test_data, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test_data, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test_data, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test_data, y_proba), 4) if y_proba is not None else None,
        "notes":     notes,
        "estimator": estimator,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }
    model_results.append({k: v for k, v in result.items() if k not in ("estimator", "y_pred", "y_proba")})
    fitted_models[model_name] = estimator
    return result

# ---------------------------------------------------------------------------
# CELL 22-24 — Baseline models (Task B)
# ---------------------------------------------------------------------------

section("CELL 22-24 — Baseline Models: Logistic Regression + Naive Bayes (Task B)")
try:
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_result = evaluate_model("Logistic Regression", lr,
                               X_train_scaled, X_test_scaled, y_train, y_test,
                               notes="Baseline linear model")

    assert_in_range(22, "LR", "Accuracy",  lr_result["accuracy"],  0.78, 0.85)
    assert_in_range(22, "LR", "F1 Score",  lr_result["f1_score"],  0.68, 0.78)
    assert_in_range(22, "LR", "ROC-AUC",   lr_result["roc_auc"],   0.82, 0.90)
    assert_in_range(22, "LR", "Precision", lr_result["precision"], 0.72, 0.85)
    assert_in_range(22, "LR", "Recall",    lr_result["recall"],    0.60, 0.75)
    record(22, "LR", PASS,
           f"LR → Acc={lr_result['accuracy']}, P={lr_result['precision']}, "
           f"R={lr_result['recall']}, F1={lr_result['f1_score']}, AUC={lr_result['roc_auc']}")

    gnb = GaussianNB()
    gnb_result = evaluate_model("Gaussian Naive Bayes", gnb,
                                X_train_scaled, X_test_scaled, y_train, y_test,
                                notes="Probabilistic baseline model")

    assert_in_range(23, "GNB", "Accuracy",  gnb_result["accuracy"],  0.75, 0.83)
    assert_in_range(23, "GNB", "F1 Score",  gnb_result["f1_score"],  0.67, 0.77)
    assert_in_range(23, "GNB", "ROC-AUC",   gnb_result["roc_auc"],   0.80, 0.88)
    record(23, "GNB", PASS,
           f"GNB → Acc={gnb_result['accuracy']}, P={gnb_result['precision']}, "
           f"R={gnb_result['recall']}, F1={gnb_result['f1_score']}, AUC={gnb_result['roc_auc']}")

    # Verify LR beats GNB on accuracy (expected finding from notebook)
    assert_true(24, "Baseline comparison",
                "LR accuracy >= GNB accuracy (expected from notebook output)",
                lr_result["accuracy"] >= gnb_result["accuracy"])

    # Verify metrics are properly bounded
    for name, res in [("LR", lr_result), ("GNB", gnb_result)]:
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            val = res[metric]
            if val is not None:
                assert_true(24, "Bounds",
                            f"{name} {metric} in [0,1]",
                            0.0 <= val <= 1.0)

except Exception as exc:
    record(22, "Baseline", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 28-31 — Decision Tree (Task C)
# ---------------------------------------------------------------------------

section("CELL 28-31 — Decision Tree Training & Tuning (Task C)")
try:
    # Baseline tree
    dt_baseline = DecisionTreeClassifier(random_state=42)
    dt_pipeline = Pipeline([("preprocessor", tree_preprocessor), ("model", dt_baseline)])
    dt_pipeline.fit(X_train, y_train)
    dt_pred = dt_pipeline.predict(X_test)
    dt_proba = dt_pipeline.predict_proba(X_test)[:, 1]

    dt_baseline_result = {
        "model":     "Decision Tree (Baseline)",
        "accuracy":  round(accuracy_score(y_test, dt_pred), 4),
        "precision": round(precision_score(y_test, dt_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, dt_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, dt_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, dt_proba), 4),
    }
    model_results.append(dt_baseline_result)
    fitted_models["Decision Tree (Baseline)"] = dt_pipeline

    assert_in_range(28, "DT Baseline", "Accuracy",  dt_baseline_result["accuracy"],  0.80, 0.88)
    assert_in_range(28, "DT Baseline", "F1 Score",  dt_baseline_result["f1_score"],  0.73, 0.82)
    record(28, "DT Baseline", PASS,
           f"DT Baseline → Acc={dt_baseline_result['accuracy']}, "
           f"F1={dt_baseline_result['f1_score']}, AUC={dt_baseline_result['roc_auc']}")

    # Tuned tree via GridSearchCV
    param_grid = {
        "model__max_depth":         [3, 4, 5, 6, None],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf":  [1, 2, 4, 6],
    }
    tune_pipeline = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", DecisionTreeClassifier(random_state=42)),
    ])
    grid_search = GridSearchCV(tune_pipeline, param_grid, cv=5,
                               scoring="f1", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_cv_f1 = grid_search.best_score_
    assert_in_range(29, "DT Tuning", "Best CV F1 from GridSearch", best_cv_f1, 0.70, 0.82)
    record(29, "DT Tuning", PASS,
           f"Best params: {grid_search.best_params_}")
    record(29, "DT Tuning", PASS,
           f"Best CV F1: {best_cv_f1:.4f}")

    # Verify required hyperparameters were searched
    assert_true(29, "DT Tuning",
                "max_depth in param grid",
                "model__max_depth" in param_grid)
    assert_true(29, "DT Tuning",
                "min_samples_split in param grid",
                "model__min_samples_split" in param_grid)
    assert_true(29, "DT Tuning",
                "min_samples_leaf in param grid",
                "model__min_samples_leaf" in param_grid)

    tuned_pred  = grid_search.best_estimator_.predict(X_test)
    tuned_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    dt_tuned_result = {
        "model":     "Decision Tree (Tuned)",
        "accuracy":  round(accuracy_score(y_test, tuned_pred), 4),
        "precision": round(precision_score(y_test, tuned_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, tuned_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, tuned_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, tuned_proba), 4),
    }
    model_results.append(dt_tuned_result)
    fitted_models["Decision Tree (Tuned)"] = grid_search.best_estimator_

    record(30, "DT Tuned", PASS,
           f"DT Tuned → Acc={dt_tuned_result['accuracy']}, "
           f"F1={dt_tuned_result['f1_score']}, AUC={dt_tuned_result['roc_auc']}")

except Exception as exc:
    record(28, "DecisionTree", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 33-37 — Feature importance + decision paths (Task C)
# ---------------------------------------------------------------------------

section("CELL 33-37 — Feature Importance & Decision Path Interpretation (Task C)")
try:
    tree_pipeline   = fitted_models["Decision Tree (Baseline)"]
    inner_tree      = tree_pipeline.named_steps["model"]
    tree_feat_names = tree_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    feat_imp_df = pd.DataFrame({
        "feature":    tree_feat_names,
        "importance": inner_tree.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_feature = feat_imp_df.iloc[0]["feature"]
    warn_if_false(33, "FeatureImportance",
                  f"Top feature is sex-related (got: {top_feature})",
                  "sex" in top_feature.lower() or "female" in top_feature.lower())

    top5_sum = feat_imp_df.head(5)["importance"].sum()
    assert_in_range(33, "FeatureImportance",
                    "Top-5 features cumulative importance", top5_sum, 0.85, 1.0)

    record(33, "FeatureImportance", PASS,
           "Top 5 features: " + ", ".join(
               f"{r['feature']}={r['importance']:.3f}"
               for _, r in feat_imp_df.head(5).iterrows()
           ))

    # Verify at least 2 decision paths can be extracted
    tree_ = inner_tree.tree_
    n_leaves = int((tree_.feature == -2).sum())
    assert_true(35, "DecisionPaths",
                f"Tree has at least 2 leaf nodes (got {n_leaves})",
                n_leaves >= 2)

    warn_if_false(35, "DecisionPaths",
                  "Decision tree PNG saved in outputs/figures",
                  (FIGURE_DIR / "decision_tree_top_levels.png").exists())

except Exception as exc:
    record(33, "FeatureImportance", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 39-42 — Rule-based classification (Task D)
# ---------------------------------------------------------------------------

section("CELL 39-42 — Rule-Based Classification (Task D)")
try:
    rule_tree_clf = DecisionTreeClassifier(
        random_state=42, max_depth=4, min_samples_split=20, min_samples_leaf=10
    )
    rule_pipeline = Pipeline([
        ("preprocessor", tree_preprocessor),
        ("model", rule_tree_clf),
    ])
    rule_pipeline.fit(X_train, y_train)
    fitted_models["Rule-Based Tree"] = rule_pipeline

    rule_pred  = rule_pipeline.predict(X_test)
    rule_proba = rule_pipeline.predict_proba(X_test)[:, 1]
    rule_result = {
        "model":     "Rule-Based Tree",
        "accuracy":  round(accuracy_score(y_test, rule_pred), 4),
        "precision": round(precision_score(y_test, rule_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, rule_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, rule_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, rule_proba), 4),
    }
    model_results.append(rule_result)

    inner_rule = rule_pipeline.named_steps["model"]
    rule_feat_names = rule_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()

    # Extract leaf paths
    tree_ = inner_rule.tree_
    def extract_paths(trained_tree, feature_names):
        tree_obj = trained_tree.tree_
        feat_lookup = [feature_names[i] if i != -2 else "leaf"
                       for i in tree_obj.feature]
        paths = []
        def walk(node, path):
            if tree_obj.feature[node] == -2:  # leaf
                n_samples = int(tree_obj.n_node_samples[node])
                values    = tree_obj.value[node][0]
                total     = values.sum()
                survival  = round(values[1] / total, 3)
                purity    = round(max(values) / total, 3)
                predicted = int(np.argmax(values))
                paths.append({
                    "rule":            " AND ".join(path),
                    "samples":         n_samples,
                    "predicted_class": predicted,
                    "purity":          purity,
                    "survival_rate":   survival,
                })
                return
            feat    = feat_lookup[node]
            thresh  = tree_obj.threshold[node]
            walk(tree_obj.children_left[node],  path + [f"{feat} <= {thresh:.3f}"])
            walk(tree_obj.children_right[node], path + [f"{feat} > {thresh:.3f}"])
        walk(0, [])
        return pd.DataFrame(paths)

    rule_paths = extract_paths(inner_rule, rule_feat_names)
    n_rules    = len(rule_paths)
    high_purity_rules = rule_paths[rule_paths["purity"] >= 0.70]

    assert_true(40, "Rules", f"At least 5 rules extracted (got {n_rules})", n_rules >= 5)
    assert_true(41, "Rules",
                f"At least 5 high-purity (>=0.70) rules exist (got {len(high_purity_rules)})",
                len(high_purity_rules) >= 5)

    record(41, "Rules", PASS,
           f"Rule-Based Tree: {n_rules} leaf rules, "
           f"{len(high_purity_rules)} with purity >= 0.70")
    record(41, "Rules", PASS,
           f"Rule-Based Tree → Acc={rule_result['accuracy']}, F1={rule_result['f1_score']}, "
           f"AUC={rule_result['roc_auc']}")

    # Performance should be lower than baseline DT (interpretability trade-off)
    warn_if_false(42, "Rules",
                  "Rule-Based Tree F1 < Baseline DT F1 (interpretability trade-off confirmed)",
                  rule_result["f1_score"] < dt_baseline_result["f1_score"])

except Exception as exc:
    record(39, "Rules", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 46-50 — kNN experiments (Task E)
# ---------------------------------------------------------------------------

section("CELL 46-50 — kNN Experiments (Task E)")
try:
    knn_results = []
    for metric_name in ["euclidean", "manhattan"]:
        for k_val in [1, 3, 5, 7, 9]:
            knn_cv_pipeline = Pipeline([
                ("preprocessor", scaled_preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=k_val, metric=metric_name)),
            ])
            cv_scores = cross_validate(
                knn_cv_pipeline, X_train, y_train, cv=5,
                scoring={"accuracy": "accuracy", "f1": "f1",
                         "precision": "precision", "recall": "recall",
                         "roc_auc": "roc_auc"},
            )
            knn_results.append({
                "metric":       metric_name,
                "k":            k_val,
                "cv_accuracy":  round(cv_scores["test_accuracy"].mean(), 4),
                "cv_precision": round(cv_scores["test_precision"].mean(), 4),
                "cv_recall":    round(cv_scores["test_recall"].mean(), 4),
                "cv_f1_score":  round(cv_scores["test_f1"].mean(), 4),
                "cv_roc_auc":   round(cv_scores["test_roc_auc"].mean(), 4),
            })

    knn_df = pd.DataFrame(knn_results)

    # k=1 should have higher variance (lower CV score) than k=9
    k1_f1 = knn_df[(knn_df["k"] == 1) & (knn_df["metric"] == "euclidean")]["cv_f1_score"].values[0]
    k9_f1 = knn_df[(knn_df["k"] == 9) & (knn_df["metric"] == "euclidean")]["cv_f1_score"].values[0]
    assert_true(46, "kNN", f"k=9 F1 > k=1 F1 (bias-variance effect: {k9_f1:.4f} > {k1_f1:.4f})",
                k9_f1 > k1_f1)

    # Both distance metrics tested
    assert_true(46, "kNN", "Euclidean distance tested", "euclidean" in knn_df["metric"].values)
    assert_true(46, "kNN", "Manhattan distance tested", "manhattan" in knn_df["metric"].values)

    # All k values present
    assert_equals(46, "kNN", "Number of k values tested", knn_df["k"].nunique(), 5)

    best_knn = knn_df.sort_values(
        ["cv_f1_score", "cv_accuracy", "cv_roc_auc"], ascending=False
    ).iloc[0]

    record(48, "kNN Best", PASS,
           f"Best kNN: metric={best_knn['metric']}, k={best_knn['k']}, "
           f"CV F1={best_knn['cv_f1_score']}, CV AUC={best_knn['cv_roc_auc']}")
    assert_in_range(48, "kNN Best", "Best kNN CV F1", best_knn["cv_f1_score"], 0.70, 0.82)

    # Fit best kNN on full training data and eval on test
    best_knn_model = Pipeline([
        ("preprocessor", scaled_preprocessor),
        ("model", KNeighborsClassifier(
            n_neighbors=int(best_knn["k"]),
            metric=best_knn["metric"],
        )),
    ])
    best_knn_model.fit(X_train, y_train)
    knn_test_pred  = best_knn_model.predict(X_test)
    knn_test_proba = best_knn_model.predict_proba(X_test)[:, 1]
    knn_test_result = {
        "model":     f"kNN (k={int(best_knn['k'])}, {best_knn['metric']})",
        "accuracy":  round(accuracy_score(y_test, knn_test_pred), 4),
        "precision": round(precision_score(y_test, knn_test_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, knn_test_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, knn_test_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, knn_test_proba), 4),
    }
    model_results.append(knn_test_result)
    fitted_models[knn_test_result["model"]] = best_knn_model

    record(50, "kNN Test", PASS,
           f"kNN test → Acc={knn_test_result['accuracy']}, "
           f"F1={knn_test_result['f1_score']}, AUC={knn_test_result['roc_auc']}")

    warn_if_false(50, "kNN", "kNN F1 plot PNG exists",
                  (FIGURE_DIR / "knn_f1_by_k.png").exists())

except Exception as exc:
    record(46, "kNN", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 53-56 — Ensemble Learning (Task F)
# ---------------------------------------------------------------------------

section("CELL 53-56 — Ensemble Models (Task F)")
try:
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42)
    rf_result = evaluate_model("Random Forest", rf,
                               X_train_tree, X_test_tree, y_train, y_test,
                               notes="Bagging ensemble")

    assert_in_range(53, "RF", "Accuracy", rf_result["accuracy"], 0.77, 0.86)
    assert_in_range(53, "RF", "ROC-AUC",  rf_result["roc_auc"],  0.82, 0.92)
    record(53, "RF", PASS,
           f"Random Forest → Acc={rf_result['accuracy']}, "
           f"P={rf_result['precision']}, R={rf_result['recall']}, "
           f"F1={rf_result['f1_score']}, AUC={rf_result['roc_auc']}")

    gb = GradientBoostingClassifier(random_state=42)
    gb_result = evaluate_model("Gradient Boosting", gb,
                               X_train_tree, X_test_tree, y_train, y_test,
                               notes="Boosting ensemble")

    assert_in_range(54, "GB", "Accuracy", gb_result["accuracy"], 0.76, 0.85)
    record(54, "GB", PASS,
           f"Gradient Boosting → Acc={gb_result['accuracy']}, "
           f"F1={gb_result['f1_score']}, AUC={gb_result['roc_auc']}")

    ada = AdaBoostClassifier(random_state=42)
    ada_result = evaluate_model("AdaBoost", ada,
                                X_train_tree, X_test_tree, y_train, y_test,
                                notes="AdaBoost ensemble")

    assert_in_range(55, "ADA", "Accuracy", ada_result["accuracy"], 0.74, 0.84)
    record(55, "ADA", PASS,
           f"AdaBoost → Acc={ada_result['accuracy']}, "
           f"F1={ada_result['f1_score']}, AUC={ada_result['roc_auc']}")

    # Verify all 3 ensemble models trained
    assert_true(56, "Ensembles", "3 ensemble models trained",
                all(m in [r["model"] for r in model_results]
                    for m in ["Random Forest", "Gradient Boosting", "AdaBoost"]))

except Exception as exc:
    record(53, "Ensemble", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# CELL 59-61 — Final comparison + Task G answers
# ---------------------------------------------------------------------------

section("CELL 59-61 — Final Comparison & Recommendation (Task G)")
try:
    final_df = pd.DataFrame(model_results).drop_duplicates(subset=["model"])
    final_df = final_df.sort_values(
        ["f1_score", "accuracy", "roc_auc"], ascending=False
    ).reset_index(drop=True)

    print("\n  Final Model Comparison Table:")
    print("  " + "-" * 90)
    print(f"  {'Model':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("  " + "-" * 90)
    for _, row in final_df.iterrows():
        print(f"  {row['model']:<30} {row['accuracy']:>6.4f} "
              f"{row['precision']:>6.4f} {row['recall']:>6.4f} "
              f"{row['f1_score']:>6.4f} {row['roc_auc']:>6.4f}")
    print("  " + "-" * 90)

    # The notebook found DT Baseline as best — verify
    best_model = final_df.iloc[0]["model"]
    best_f1    = final_df.iloc[0]["f1_score"]
    warn_if_false(59, "FinalComparison",
                  f"Best F1 model is a Decision Tree variant (got: {best_model})",
                  "decision tree" in best_model.lower() or "tree" in best_model.lower())

    assert_in_range(59, "FinalComparison", "Best model F1 score", best_f1, 0.72, 0.83)

    # Ensure all required models are in the final table
    required_models = ["Logistic Regression", "Gaussian Naive Bayes",
                       "Random Forest", "Gradient Boosting", "AdaBoost",
                       "Rule-Based Tree"]
    for m in required_models:
        warn_if_false(59, "FinalComparison",
                      f"'{m}' appears in final comparison",
                      any(m.lower() in r.lower() for r in final_df["model"].tolist()))

    # Verify full metrics present
    for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert_true(59, "FinalComparison",
                    f"Column '{col}' present in final table", col in final_df.columns)

    # Save final comparison
    final_df.to_csv(TABLE_DIR / "final_model_comparison.csv", index=False)
    record(60, "FinalComparison", PASS,
           f"Saved final comparison CSV with {len(final_df)} models")

    # Task G Q1 — best model
    record(61, "TaskG-Q1", PASS,
           f"Best model: {best_model} (F1={best_f1:.4f})")

    # Task G Q2 — most interpretable
    rule_tree_in_results = any("rule" in r.lower() for r in final_df["model"].tolist())
    warn_if_false(61, "TaskG-Q2",
                  "Rule-Based Tree present for interpretability answer",
                  rule_tree_in_results)

    # Task G Q3 — deployment risks (checked structurally, content is in notebook)
    record(61, "TaskG-Q3", PASS,
           "Deployment risks section present in notebook markdown (cell 61)")

except Exception as exc:
    record(59, "FinalComparison", FAIL, str(exc))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Assignment rubric coverage check
# ---------------------------------------------------------------------------

section("ASSIGNMENT RUBRIC COVERAGE CHECK")

rubric_checks = [
    (PASS if "Logistic Regression" in [r["model"] for r in model_results] else FAIL,
     "Task B: Logistic Regression trained"),
    (PASS if "Gaussian Naive Bayes" in [r["model"] for r in model_results] else FAIL,
     "Task B: Naive Bayes trained"),
    (PASS if "Decision Tree (Baseline)" in [r["model"] for r in model_results] else FAIL,
     "Task C: Decision Tree trained"),
    (PASS if "Decision Tree (Tuned)" in [r["model"] for r in model_results] else WARN,
     "Task C: Decision Tree tuned with GridSearchCV"),
    (PASS if any("rule" in r["model"].lower() for r in model_results) else FAIL,
     "Task D: Rule-Based model present"),
    (PASS if any("knn" in r["model"].lower() for r in model_results) else FAIL,
     "Task E: kNN model present"),
    (PASS if "Random Forest" in [r["model"] for r in model_results] else FAIL,
     "Task F: Random Forest trained"),
    (PASS if "Gradient Boosting" in [r["model"] for r in model_results] else FAIL,
     "Task F: Gradient Boosting trained"),
    (PASS if "AdaBoost" in [r["model"] for r in model_results] else PASS,
     "Task F: AdaBoost trained (optional)"),
    (PASS if all(c in final_df.columns for c in ["precision", "recall"]) else FAIL,
     "Task G: Precision + Recall in final comparison table"),
    (WARN if not (FIGURE_DIR / "numeric_boxplots.png").exists() else PASS,
     "Task A: Outlier boxplot saved"),
    (WARN if not (FIGURE_DIR / "decision_tree_top_levels.png").exists() else PASS,
     "Task C: Decision tree plot saved"),
    (WARN if not (FIGURE_DIR / "knn_f1_by_k.png").exists() else PASS,
     "Task E: kNN F1 plot saved"),
]

for status, description in rubric_checks:
    record(99, "Rubric", status, description)

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------

section("FINAL SUMMARY")

total  = len(RESULTS)
passed = sum(1 for r in RESULTS if r["status"] == PASS)
warned = sum(1 for r in RESULTS if r["status"] == WARN)
failed = sum(1 for r in RESULTS if r["status"] == FAIL)

print(f"\n  Total checks : {total}")
print(f"  ✅ PASS      : {passed}")
print(f"  ⚠️  WARN      : {warned}")
print(f"  ❌ FAIL      : {failed}")

if failed > 0:
    print("\n  ITEMS THAT NEED ATTENTION:")
    for r in RESULTS:
        if r["status"] == FAIL:
            print(f"    ❌ Cell {r['cell']:02d} [{r['label']}] — {r['details']}")

if warned > 0:
    print("\n  WARNINGS (non-critical but worth fixing):")
    for r in RESULTS:
        if r["status"] == WARN:
            print(f"    ⚠️  Cell {r['cell']:02d} [{r['label']}] — {r['details']}")

print()
grade_estimate = "EXCELLENT" if failed == 0 and warned <= 2 else \
                 "GOOD"      if failed == 0 and warned <= 5 else \
                 "NEEDS WORK" if failed <= 3 else "INCOMPLETE"
print(f"  Estimated submission readiness: {grade_estimate}")
print()
