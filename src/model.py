"""Entraînement, tuning, scoring et interprétabilité du modèle de churn.

Le dataset Telco ne contient pas de date d'inscription: le script conserve donc
un split stratifié reproductible et bascule automatiquement vers un split
temporel si une colonne de date exploitable est ajoutée plus tard.

Usage:
    python src/model.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover - l'environnement GitHub installera shap.
    shap = None

from src.etl import FEATURE_PATH, PROJECT_ROOT


NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "tenure_years",
    "charge_per_tenure_month",
    "num_services",
    "is_new_customer",
    "is_long_tenure",
    "is_month_to_month",
    "is_electronic_check",
    "auto_payment",
    "has_premium_support",
    "has_streaming",
]

CAT_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_LABELS = {
    "tenure": "Ancienneté courte",
    "is_new_customer": "Nouveau client",
    "Contract_Month-to-month": "Contrat mensuel",
    "is_month_to_month": "Contrat mensuel",
    "PaymentMethod_Electronic check": "Paiement par chèque électronique",
    "is_electronic_check": "Paiement par chèque électronique",
    "MonthlyCharges": "Facture mensuelle élevée",
    "TotalCharges": "Dépenses cumulées",
    "InternetService_Fiber optic": "Internet fibre",
    "OnlineSecurity_No": "Sans sécurité en ligne",
    "TechSupport_No": "Sans support technique",
    "OnlineBackup_No": "Sans sauvegarde en ligne",
    "DeviceProtection_No": "Sans protection appareil",
    "StreamingTV_Yes": "Streaming TV",
    "StreamingMovies_Yes": "Streaming films",
    "PaperlessBilling_Yes": "Facturation sans papier",
    "num_services": "Nombre de services",
    "has_premium_support": "Sécurité ou support premium",
    "has_streaming": "Services de streaming",
    "SeniorCitizen": "Client senior",
    "Partner_No": "Sans partenaire",
    "Dependents_No": "Sans personne à charge",
}

MODEL_PATH = PROJECT_ROOT / "src" / "model_final.joblib"
COMPAT_MODEL_PATH = PROJECT_ROOT / "src" / "model_xgb.joblib"
FEATURES_PATH = PROJECT_ROOT / "src" / "model_features.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"


def prepare_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Encode les variables catégorielles avec un ordre de colonnes stable."""
    missing = [col for col in NUMERIC_FEATURES + CAT_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Variables nécessaires absentes du feature store: {missing}")
    encoded = pd.get_dummies(
        df[NUMERIC_FEATURES + CAT_FEATURES],
        columns=CAT_FEATURES,
        drop_first=False,
        dtype=float,
    )
    if feature_cols is not None:
        encoded = encoded.reindex(columns=feature_cols, fill_value=0)
    return encoded.astype(float)


def _temporal_or_stratified_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    date_candidates = ["signup_date", "start_date", "registration_date", "tenure_start_date"]
    date_column = next((col for col in date_candidates if col in df.columns), None)
    if date_column:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        if dates.notna().mean() >= 0.95 and dates.nunique() > 10:
            ordered = dates.sort_values().index
            cutoff = max(1, int(len(ordered) * (1 - test_size)))
            train_idx, test_idx = ordered[:cutoff], ordered[cutoff:]
            return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], "temporal"

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    return x_train, x_test, y_train, y_test, "stratified_random_no_signup_date"


def _model_searches(scale_pos_weight: float) -> dict[str, GridSearchCV]:
    """Construit des recherches compactes et reproductibles."""
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    lgbm = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    return {
        "XGBoost tuned": GridSearchCV(
            xgb,
            {
                "n_estimators": [150, 300],
                "max_depth": [3, 5],
                "learning_rate": [0.03, 0.08],
                "subsample": [0.85],
                "colsample_bytree": [0.85],
            },
            scoring="average_precision",
            cv=3,
            n_jobs=1,
            refit=True,
        ),
        "LightGBM tuned": GridSearchCV(
            lgbm,
            {
                "n_estimators": [150, 300],
                "num_leaves": [15, 31],
                "learning_rate": [0.03, 0.08],
                "min_child_samples": [20],
            },
            scoring="average_precision",
            cv=3,
            n_jobs=1,
            refit=True,
        ),
    }


def _safe_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Retourne les contributions SHAP; fallback natif si le module n'est pas disponible."""
    if shap is not None:
        try:
            values = shap.TreeExplainer(model)(X).values
            if values.ndim == 3:
                values = values[:, :, 1]
            return np.asarray(values)
        except Exception:
            pass

    if isinstance(model, LGBMClassifier):
        values = model.booster_.predict(X, pred_contrib=True)
        return np.asarray(values[:, :-1])
    if isinstance(model, XGBClassifier):
        import xgboost as xgb

        values = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return np.asarray(values[:, :-1])
    raise RuntimeError("Impossible de calculer les contributions SHAP pour ce modèle.")


def top_risk_reasons(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    n_reasons: int = 3,
) -> pd.DataFrame:
    """Transforme les contributions positives en raisons lisibles par un analyste."""
    rows: list[dict[str, str]] = []
    for row_values, (_, row) in zip(shap_values, X.iterrows()):
        positive = np.where(row_values > 0, row_values, -np.inf)
        indices = np.argsort(positive)[::-1][:n_reasons]
        indices = [idx for idx in indices if np.isfinite(positive[idx])]
        if not indices:
            indices = np.argsort(np.abs(row_values))[::-1][:n_reasons]
        reasons = []
        for idx in indices:
            feature = X.columns[idx]
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").replace("-", " "))
            contribution = float(row_values[idx])
            reasons.append(f"{label} (impact SHAP {contribution:+.2f})")
        rows.append(
            {
                "top_risk_reason": reasons[0] if reasons else "Profil global",
                "top_risk_reasons": " • ".join(reasons) if reasons else "Profil global",
            }
        )
    return pd.DataFrame(rows, index=X.index)


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metrics, default=lambda value: float(value)))


def train_and_save() -> dict[str, Any]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURE_PATH)
    X = prepare_features(df)
    y = df["Churn"].astype(int)
    X_train, X_test, y_train, y_test, split_strategy = _temporal_or_stratified_split(df, X, y)
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    results: dict[str, dict[str, Any]] = {}
    scaler = StandardScaler()
    baseline = LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42)
    baseline.fit(scaler.fit_transform(X_train), y_train)
    baseline_proba = baseline.predict_proba(scaler.transform(X_test))[:, 1]
    results["Logistic Regression"] = {
        "pr_auc": float(average_precision_score(y_test, baseline_proba)),
        "roc_auc": float(roc_auc_score(y_test, baseline_proba)),
        "best_params": {},
    }

    searches = _model_searches(scale_pos_weight)
    fitted_models: dict[str, Any] = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        proba = search.predict_proba(X_test)[:, 1]
        fitted_models[name] = search.best_estimator_
        results[name] = {
            "pr_auc": float(average_precision_score(y_test, proba)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "cv_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
        }

    selected_name = max(
        fitted_models,
        key=lambda name: results[name]["pr_auc"],
    )
    selected_test_model = fitted_models[selected_name]
    selected_test_proba = selected_test_model.predict_proba(X_test)[:, 1]
    selected_report = classification_report(
        y_test, selected_test_proba >= 0.5, output_dict=True, zero_division=0
    )
    y_test_frame = df.loc[X_test.index, ["CustomerID", "Churn", "tenure", "Contract", "MonthlyCharges"]].copy()
    y_test_frame["churn_proba"] = selected_test_proba
    y_test_frame.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # Refit le meilleur modèle sur tout le dataset pour alimenter le dashboard.
    final_model = clone(selected_test_model)
    final_model.fit(X, y)
    all_proba = final_model.predict_proba(X)[:, 1]
    all_shap = _safe_shap_values(final_model, X)
    reason_frame = top_risk_reasons(X, all_shap)
    scored = df[["CustomerID", "Churn", "tenure", "Contract", "PaymentMethod", "MonthlyCharges"]].copy()
    scored["churn_proba"] = all_proba
    scored = pd.concat([scored, reason_frame], axis=1)
    scored = scored.sort_values("churn_proba", ascending=False)
    scored.to_csv(REPORTS_DIR / "client_risk_scores.csv", index=False)

    global_shap = pd.DataFrame(
        {
            "feature": X.columns,
            "mean_abs_shap": np.abs(all_shap).mean(axis=0),
            "mean_shap": all_shap.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_shap["label"] = global_shap["feature"].map(
        lambda feature: FEATURE_LABELS.get(feature, feature.replace("_", " ").replace("-", " "))
    )
    global_shap.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)

    bundle = {
        "model": final_model,
        "model_name": selected_name,
        "feature_cols": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CAT_FEATURES,
        "feature_labels": FEATURE_LABELS,
        "shap_available_at_training": shap is not None,
    }
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(final_model, COMPAT_MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    metrics = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "churn_rate": float(y.mean()),
        "split_strategy": split_strategy,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_model": selected_name,
        "models": results,
        "selected_test_classification_report": selected_report,
        "shap_available_at_training": shap is not None,
        "shap_fallback": "LightGBM native pred_contrib" if shap is None else None,
    }
    metrics = _jsonable_metrics(metrics)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        [{"model": name, **values} for name, values in results.items()]
    ).drop(columns=["best_params"], errors="ignore").to_csv(
        REPORTS_DIR / "model_comparison.csv", index=False
    )

    print("=" * 68)
    print("Comparaison des modèles — PR-AUC (plus haut = mieux)")
    print("=" * 68)
    for name, values in results.items():
        print(f"{name:24s}: {values['pr_auc']:.3f}")
    print(f"\nModèle retenu : {selected_name}")
    print(f"Split : {split_strategy}")
    print("Artefacts écrits dans src/ et reports/")
    return metrics


if __name__ == "__main__":
    train_and_save()