"""Dashboard Streamlit du projet Telco Churn.

Usage:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import FEATURE_LABELS, prepare_features, top_risk_reasons  # noqa: E402


st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📈", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data" / "features" / "churn_features.parquet")


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(PROJECT_ROOT / "src" / "model_final.joblib")


@st.cache_data
def score_clients(frame: pd.DataFrame, feature_cols: tuple[str, ...], model) -> pd.DataFrame:
    encoded = prepare_features(frame, list(feature_cols))
    scored = frame.copy()
    scored["churn_proba"] = model.predict_proba(encoded)[:, 1]
    reasons = top_risk_reasons(encoded, _shap_values_for_dashboard(model, encoded))
    return pd.concat([scored, reasons], axis=1)


def _shap_values_for_dashboard(model, encoded: pd.DataFrame):
    try:
        import shap

        values = shap.TreeExplainer(model)(encoded).values
        return values[:, :, 1] if values.ndim == 3 else values
    except ImportError:
        if hasattr(model, "booster_"):
            return model.booster_.predict(encoded, pred_contrib=True)[:, :-1]
        import xgboost as xgb

        return model.get_booster().predict(xgb.DMatrix(encoded), pred_contribs=True)[:, :-1]


try:
    df = load_data()
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error("Les artefacts ne sont pas encore générés. Exécutez `python src/etl.py`, puis `python src/model.py`.")
    st.stop()

model = bundle["model"]
scored = score_clients(df, tuple(bundle["feature_cols"]), model)

st.title("Telco Churn Intelligence")
st.caption(
    f"Analyse de {len(scored):,} clients — modèle actif : {bundle['model_name']} — "
    "les raisons sont issues des contributions SHAP."
)

left, middle, right, fourth = st.columns(4)
left.metric("Clients actifs", f"{(scored['Churn'] == 0).sum():,}")
middle.metric("Taux de churn historique", f"{scored['Churn'].mean():.1%}")
high_risk = scored["churn_proba"] >= 0.5
right.metric("Clients à risque élevé", f"{high_risk.sum():,}")
fourth.metric("MRR à risque", f"${scored.loc[high_risk, 'MonthlyCharges'].sum():,.0f}")

st.divider()

tab_risk, tab_segments, tab_simulator = st.tabs(
    ["Portefeuille à risque", "Segments et drivers", "Simulateur what-if"]
)

with tab_risk:
    st.subheader("Clients les plus à risque")
    risk_threshold = st.slider("Seuil de probabilité", 0.0, 1.0, 0.5, 0.05)
    top_n = st.slider("Nombre de clients à afficher", 5, 100, 20)
    contract_filter = st.multiselect(
        "Filtrer les contrats",
        options=sorted(scored["Contract"].unique()),
        default=sorted(scored["Contract"].unique()),
    )
    risky = scored[
        (scored["churn_proba"] >= risk_threshold)
        & scored["Contract"].isin(contract_filter)
    ].sort_values("churn_proba", ascending=False).head(top_n)
    display_cols = [
        "CustomerID",
        "churn_proba",
        "top_risk_reason",
        "tenure",
        "Contract",
        "PaymentMethod",
        "MonthlyCharges",
        "InternetService",
    ]
    st.dataframe(
        risky[display_cols].rename(
            columns={
                "CustomerID": "Client",
                "churn_proba": "Probabilité de churn",
                "top_risk_reason": "Top raison du risque",
                "tenure": "Ancienneté (mois)",
                "Contract": "Contrat",
                "PaymentMethod": "Paiement",
                "MonthlyCharges": "Facture mensuelle",
                "InternetService": "Internet",
            }
        ).style.format({"Probabilité de churn": "{:.1%}", "Facture mensuelle": "${:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_segments:
    st.subheader("Rétention par ancienneté")
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
    segment_df = scored.copy()
    segment_df["tenure_bucket"] = pd.cut(
        segment_df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
    retention = (
        segment_df.groupby("tenure_bucket", observed=False)["Churn"]
        .agg(["count", "mean"])
        .rename(columns={"count": "Clients", "mean": "Taux de churn"})
    )
    st.bar_chart((1 - retention["Taux de churn"]).rename("Taux de rétention"))
    st.dataframe(
        retention.style.format({"Taux de churn": "{:.1%}"}),
        use_container_width=True,
        hide_index=False,
    )

    st.subheader("Drivers globaux du risque")
    shap_path = PROJECT_ROOT / "reports" / "shap_global_importance.csv"
    if shap_path.exists():
        drivers = pd.read_csv(shap_path).head(12).sort_values("mean_abs_shap")
        drivers["label"] = drivers["label"].fillna(drivers["feature"])
        st.bar_chart(drivers.set_index("label")["mean_abs_shap"].rename("Impact SHAP moyen"))
        st.caption("Plus l'impact SHAP moyen est élevé, plus le driver explique les prédictions.")

with tab_simulator:
    st.subheader("Simulateur de profil client")
    st.caption("Modifiez les attributs commerciaux pour estimer le risque d'un profil hypothétique.")
    c1, c2, c3 = st.columns(3)
    tenure_sim = c1.slider("Ancienneté (mois)", 0, 72, 12)
    monthly_sim = c2.slider("Facture mensuelle ($)", 18.0, 120.0, 70.0, 1.0)
    contract_sim = c3.selectbox("Contrat", sorted(df["Contract"].unique()))
    c4, c5, c6 = st.columns(3)
    internet_sim = c4.selectbox("Service internet", sorted(df["InternetService"].unique()))
    payment_sim = c5.selectbox("Mode de paiement", sorted(df["PaymentMethod"].unique()))
    support_sim = c6.selectbox("Support technique", sorted(df["TechSupport"].unique()))

    simulated = df.iloc[[0]].copy()
    simulated["tenure"] = tenure_sim
    simulated["MonthlyCharges"] = monthly_sim
    simulated["TotalCharges"] = monthly_sim * tenure_sim
    simulated["Contract"] = contract_sim
    simulated["InternetService"] = internet_sim
    simulated["PaymentMethod"] = payment_sim
    simulated["TechSupport"] = support_sim
    # Rejoue les mêmes règles de feature engineering que l'ETL.
    simulated["tenure_years"] = simulated["tenure"] / 12
    simulated["avg_monthly_spend"] = simulated["TotalCharges"] / simulated["tenure"].replace(0, 1)
    simulated["charge_per_tenure_month"] = simulated["TotalCharges"] / simulated["tenure"].clip(lower=1)
    simulated["is_new_customer"] = (simulated["tenure"] <= 6).astype("int8")
    simulated["is_long_tenure"] = (simulated["tenure"] >= 48).astype("int8")
    simulated["is_month_to_month"] = (simulated["Contract"] == "Month-to-month").astype("int8")
    simulated["is_electronic_check"] = (simulated["PaymentMethod"] == "Electronic check").astype("int8")
    simulated["auto_payment"] = simulated["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype("int8")
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    simulated["num_services"] = simulated[service_cols].apply(
        lambda row: sum(value == "Yes" for value in row), axis=1
    )
    simulated["has_premium_support"] = simulated[["OnlineSecurity", "TechSupport"]].eq("Yes").any(axis=1).astype("int8")
    simulated["has_streaming"] = simulated[["StreamingTV", "StreamingMovies"]].eq("Yes").any(axis=1).astype("int8")
    sim_x = prepare_features(simulated, bundle["feature_cols"])
    sim_proba = float(model.predict_proba(sim_x)[0, 1])
    st.metric("Probabilité de churn", f"{sim_proba:.1%}")
    st.progress(sim_proba)