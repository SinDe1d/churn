"""ETL du vrai dataset Telco Customer Churn.

Usage depuis la racine du projet:
    python src/etl.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "churn_features.parquet"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned" / "telco_churn_clean.csv"

REQUIRED_COLUMNS = {
    "customerid",
    "seniorcitizen",
    "tenure",
    "internetservice",
    "contract",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
    "churn",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes tout en gardant le schéma Telco lisible."""
    aliases = {
        "customerid": "CustomerID",
        "seniorcitizen": "SeniorCitizen",
        "tenure": "tenure",
        "monthlycharges": "MonthlyCharges",
        "totalcharges": "TotalCharges",
        "churn": "Churn",
    }
    renamed = {col: aliases.get(str(col).strip().lower(), str(col).strip()) for col in df.columns}
    return df.rename(columns=renamed)


def extract(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Lit le CSV source et vérifie la présence du schéma minimum."""
    frame = _canonicalize_columns(pd.read_csv(path))
    normalized = {str(col).lower() for col in frame.columns}
    missing = REQUIRED_COLUMNS - normalized
    if missing:
        raise ValueError(f"Colonnes obligatoires absentes du dataset Telco: {sorted(missing)}")
    return frame


def _normalise_yes_no(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"Yes": 1, "No": 0})


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tenure_years"] = df["tenure"] / 12
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["charge_per_tenure_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype("int8")
    df["is_long_tenure"] = (df["tenure"] >= 48).astype("int8")
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype("int8")
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype("int8")
    df["auto_payment"] = df["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype("int8")

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present_service_cols = [col for col in service_cols if col in df.columns]
    df["num_services"] = (
        df[present_service_cols]
        .apply(lambda row: sum(value in {"Yes", 1} for value in row), axis=1)
        .astype("int8")
    )
    df["has_premium_support"] = (
        df[["OnlineSecurity", "TechSupport"]].eq("Yes").any(axis=1).astype("int8")
    )
    df["has_streaming"] = (
        df[["StreamingTV", "StreamingMovies"]].eq("Yes").any(axis=1).astype("int8")
    )
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le CSV réel et produit une feature store sans valeur manquante."""
    frame = _canonicalize_columns(df.copy())
    frame.columns = [str(col).strip() for col in frame.columns]

    for col in frame.select_dtypes(include=["object", "string"]).columns:
        frame[col] = frame[col].astype("string").str.strip()
        frame[col] = frame[col].replace({"": pd.NA, " ": pd.NA})

    frame = frame.drop_duplicates(subset="CustomerID").reset_index(drop=True)
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    # Dans le Telco original, les lignes tenure=0 ont TotalCharges vide.
    frame["TotalCharges"] = frame["TotalCharges"].fillna(
        (frame["MonthlyCharges"] * frame["tenure"].fillna(0)).fillna(0)
    )
    frame["Churn"] = frame["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce").fillna(0).astype("int8")

    frame = _add_engineered_features(frame)

    numeric_cols = frame.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        frame[col] = frame[col].fillna(frame[col].median()).fillna(0)
    categorical_cols = frame.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical_cols:
        frame[col] = frame[col].fillna("Unknown").astype(str)

    if frame["Churn"].isna().any():
        raise ValueError("La cible Churn contient des valeurs inconnues après transformation.")
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Valeurs manquantes restantes après ETL: {missing}")
    return frame


def load(
    df: pd.DataFrame,
    feature_path: str | Path = FEATURE_PATH,
    clean_path: str | Path = CLEAN_PATH,
) -> None:
    feature_path = Path(feature_path)
    clean_path = Path(clean_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feature_path, index=False)
    df.to_csv(clean_path, index=False)
    print(f"Feature store sauvegardé : {feature_path} ({len(df)} lignes)")
    print(f"Dataset nettoyé sauvegardé : {clean_path}")


def run() -> pd.DataFrame:
    transformed = transform(extract())
    load(transformed)
    return transformed


if __name__ == "__main__":
    run()