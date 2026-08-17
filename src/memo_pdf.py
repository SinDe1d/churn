"""Génère le mémo exécutif PDF à partir des rapports du pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.etl import PROJECT_ROOT


REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "executive_memo.pdf"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def build_memo() -> Path:
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    clients = pd.read_csv(REPORTS_DIR / "client_risk_scores.csv")
    drivers = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv").head(8)
    comparison = pd.read_csv(REPORTS_DIR / "model_comparison.csv")

    high_risk = clients["churn_proba"] >= 0.5
    churn_rate = metrics["churn_rate"]
    risk_mrr = clients.loc[high_risk, "MonthlyCharges"].sum()

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#153B50"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#153B50"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Mémo exécutif — Telco Churn",
        author="Telco Churn Analytics",
    )
    story = [
        Paragraph("Mémo exécutif — Telco Customer Churn", styles["TitleCentered"]),
        Paragraph(
            "Objectif: identifier les clients susceptibles de résilier leur abonnement et prioriser "
            "les actions de rétention sur le portefeuille existant.",
            styles["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
    ]

    kpi_data = [
        ["Clients analysés", f"{metrics['dataset_rows']:,}", "Taux de churn historique", f"{churn_rate:.1%}"],
        ["Clients à risque élevé", f"{high_risk.sum():,}", "MRR à risque", _money(risk_mrr)],
        ["Modèle retenu", metrics["selected_model"], "Split", metrics["split_strategy"]],
    ]
    kpi_table = Table(kpi_data, colWidths=[4 * cm, 4 * cm, 4.2 * cm, 5 * cm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF4F4")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8D8D8")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story += [kpi_table, Spacer(1, 0.5 * cm)]

    story.append(Paragraph("Lecture business", styles["Section"]))
    story.append(
        Paragraph(
            f"Le portefeuille présente un churn historique de <b>{churn_rate:.1%}</b>. "
            f"Le modèle classe <b>{high_risk.sum():,} clients</b> au-dessus du seuil opérationnel de 50 %, "
            f"représentant environ <b>{_money(risk_mrr)} de revenus mensuels</b> à protéger. "
            "Cette estimation est un outil de priorisation et non une décision automatique.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Modèles et qualité prédictive", styles["Section"]))
    model_rows = [["Modèle", "PR-AUC test", "ROC-AUC test"]]
    for _, row in comparison.iterrows():
        model_rows.append([row["model"], f"{row['pr_auc']:.3f}", f"{row['roc_auc']:.3f}"])
    model_table = Table(model_rows, colWidths=[8 * cm, 4 * cm, 4 * cm])
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153B50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [model_table, Spacer(1, 0.25 * cm)]
    story.append(
        Paragraph(
            "Le tuning compare une régression logistique de référence à XGBoost et LightGBM via "
            "GridSearchCV optimisé sur la PR-AUC. La PR-AUC est privilégiée car la classe churn est minoritaire.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Drivers de risque et actions recommandées", styles["Section"]))
    driver_rows = [["Driver SHAP", "Impact absolu moyen", "Lecture"]]
    for _, row in drivers.iterrows():
        direction = "augmente plutôt le risque" if row["mean_shap"] > 0 else "réduit plutôt le risque"
        driver_rows.append([row["label"], f"{row['mean_abs_shap']:.4f}", direction])
    driver_table = Table(driver_rows, colWidths=[8 * cm, 4 * cm, 5 * cm])
    driver_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#153B50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(driver_table)
    story += [
        Spacer(1, 0.35 * cm),
        Paragraph(
            "<b>Plan d'action suggéré:</b> prioriser les clients mensuels à forte facture, proposer "
            "des offres de migration vers un contrat annuel, traiter les irritants liés au support et "
            "à la sécurité en ligne, puis mesurer le taux de conversion par segment et la marge retenue.",
            styles["BodyText"],
        ),
        Paragraph(
            "<b>Interprétabilité:</b> chaque ligne du fichier client_risk_scores.csv contient la top raison "
            "et les trois principales raisons du risque. Elles proviennent de contributions SHAP locales, "
            "et sont destinées à guider l'action commerciale.",
            styles["BodyText"],
        ),
        Paragraph("Limites et prochaines étapes", styles["Section"]),
        Paragraph(
            "Le dataset Telco public ne fournit pas de date d'inscription ni d'historique longitudinal. "
            "Le split utilisé ici est donc stratifié et aléatoire. Dès qu'une date d'acquisition ou de "
            "début de contrat est disponible, le pipeline basculera automatiquement vers un split temporel. "
            "Avant mise en production, il faudra calibrer les probabilités, définir le coût d'une campagne "
            "de rétention et suivre le lift réel par expérimentation contrôlée.",
            styles["BodyText"],
        ),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "Sources artefactées: data/raw/telco_churn.csv, reports/metrics.json, "
            "reports/shap_global_importance.csv et reports/client_risk_scores.csv.",
            styles["Small"],
        ),
    ]

    doc.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(f"Mémo généré : {build_memo()}")