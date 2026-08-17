"""Lance tout le pipeline reproductible dans le bon ordre."""

from src.etl import run as run_etl
from src.model import train_and_save
from src.memo_pdf import build_memo


if __name__ == "__main__":
    print("[1/3] ETL")
    run_etl()
    print("[2/3] Modèle, tuning et SHAP")
    train_and_save()
    print("[3/3] Mémo exécutif PDF")
    print(f"Mémo généré : {build_memo()}")