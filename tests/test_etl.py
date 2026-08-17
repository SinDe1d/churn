import pandas as pd

from src.etl import extract, transform
from src.model import prepare_features, top_risk_reasons


def test_real_telco_etl_has_no_missing_values():
    transformed = transform(extract())
    assert len(transformed) == 7043
    assert int(transformed.isna().sum().sum()) == 0
    assert set(transformed["Churn"].unique()) == {0, 1}


def test_telco_service_columns_are_kept_and_engineered():
    transformed = transform(extract())
    for column in ["InternetService", "OnlineSecurity", "TechSupport", "StreamingTV", "StreamingMovies"]:
        assert column in transformed.columns
    assert transformed["num_services"].between(0, 8).all()


def test_feature_matrix_and_local_reasons_are_aligned():
    transformed = transform(extract()).head(12)
    encoded = prepare_features(transformed)
    fake_shap = pd.DataFrame(0.0, index=encoded.index, columns=encoded.columns).to_numpy().copy()
    fake_shap[:, 0] = 1.0
    reasons = top_risk_reasons(encoded, fake_shap)
    assert encoded.shape[0] == reasons.shape[0] == 12
    assert reasons["top_risk_reason"].notna().all()