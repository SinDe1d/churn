# Telco Churn Intelligence

Pipeline complet d'analyse et de prédiction du churn sur le vrai dataset
**Telco Customer Churn**. Le projet couvre l'ETL, l'exploration, la comparaison
de modèles, le tuning d'hyperparamètres, l'explicabilité SHAP, un dashboard
Streamlit et un mémo exécutif PDF.

## Résultats livrés

- Le dataset réel est déjà placé dans `data/raw/telco_churn.csv` (7 043 lignes).
- Les colonnes de services ont été intégrées: internet, sécurité en ligne,
  sauvegarde, protection appareil, support technique, streaming TV et films.
- LightGBM et XGBoost sont comparés à une régression logistique de référence.
- `GridSearchCV` optimise les modèles arbres selon la PR-AUC.
- Les contributions SHAP globales et locales sont générées dans `reports/`.
- Le tableau de clients à risque expose `top_risk_reason` et les trois raisons
  principales par client.
- Le mémo `reports/executive_memo.pdf` est généré automatiquement.
- Trois tests automatisés couvrent l'ETL, les variables de service et
  l'alignement des contributions locales.

## Installation

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Exécution complète

Depuis la racine du dépôt:

```bash
python run_pipeline.py
```

Ou étape par étape:

```bash
python src/etl.py
python src/eda.py
python src/model.py
python src/memo_pdf.py
streamlit run src/dashboard.py
```

Le dashboard est disponible à l'URL affichée par Streamlit.

## Sorties principales

| Chemin | Contenu |
| --- | --- |
| `data/raw/telco_churn.csv` | Source réelle versionnée |
| `data/features/churn_features.parquet` | Feature store après ETL |
| `src/model_final.joblib` | Bundle du modèle retenu + colonnes |
| `reports/model_comparison.csv` | PR-AUC et ROC-AUC par modèle |
| `reports/metrics.json` | Métriques, split et hyperparamètres |
| `reports/shap_global_importance.csv` | Drivers SHAP globaux |
| `reports/client_risk_scores.csv` | Probabilité et raisons locales par client |
| `reports/executive_memo.pdf` | Mémo exécutif prêt à présenter |

## Split temporel

Le fichier Telco public ne contient pas de date d'inscription. Le pipeline
utilise donc un split stratifié aléatoire reproductible et le documente dans
`reports/metrics.json`. Si une colonne `signup_date`, `start_date`,
`registration_date` ou `tenure_start_date` est ajoutée au feature store, le
code la détecte et utilise automatiquement les observations les plus anciennes
pour l'entraînement et les plus récentes pour le test.

## Interprétation métier

Les raisons SHAP ne constituent pas une causalité. Elles expliquent la
contribution des variables à la prédiction du modèle et servent à prioriser
les actions commerciales. Avant production, il faut calibrer le seuil avec le
coût des campagnes, tester le lift par expérimentation et surveiller les biais
entre segments.

## Tests

```bash
pytest -q
```

## Mise sur GitHub

Le dossier contient un projet autonome: créez un dépôt GitHub, copiez-y le
contenu de cette archive, puis exécutez les commandes ci-dessous.

```bash
git init
git add .
git commit -m "Build Telco churn analytics pipeline"
git branch -M main
git remote add origin https://github.com/<votre-compte>/<votre-depot>.git
git push -u origin main
```