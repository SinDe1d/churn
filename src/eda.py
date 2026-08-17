"""EDA descriptive reproductible sur le feature store."""

from __future__ import annotations

import pandas as pd

from src.etl import FEATURE_PATH, PROJECT_ROOT


def run_eda() -> None:
    df = pd.read_parquet(FEATURE_PATH)
    print("=" * 68)
    print("TELCO CUSTOMER CHURN — EDA")
    print("=" * 68)
    print(f"Lignes: {len(df):,} | Colonnes: {len(df.columns)}")
    print(f"Taux de churn: {df['Churn'].mean():.1%}")
    print(f"Valeurs manquantes: {int(df.isna().sum().sum())}")

    for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]:
        print(f"\n--- Churn par {column} ---")
        print(df.groupby(column)["Churn"].agg(["count", "mean"]).sort_values("mean", ascending=False).round(3))

    bins = [0, 6, 12, 24, 48, 72]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, include_lowest=True)
    print("\n--- Churn par tranche d'ancienneté ---")
    print(df.groupby("tenure_bucket", observed=True)["Churn"].agg(["count", "mean"]).round(3))
    print("\nNote: le Telco Customer Churn public ne contient pas de date d'inscription.")
    print("Le modèle utilise donc un split stratifié; une colonne de date déclenchera automatiquement un split temporel.")


if __name__ == "__main__":
    run_eda()