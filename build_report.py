from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from PIL import ImageChops
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "Titanic-Dataset.csv"
LOGO_PATH = PROJECT_ROOT / "logo.jpg"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_ASSETS_DIR = PROJECT_ROOT / "report_assets"
REPORT_FIGURES_DIR = REPORT_ASSETS_DIR / "figures"
REPORT_DATA_DIR = REPORT_ASSETS_DIR / "data"
REPORT_TABLES_DIR = REPORT_ASSETS_DIR / "tables"
REPORT_PATH = PROJECT_ROOT / "report.pdf"

# All figures extracted from the latest notebook execution
FIG_DIR = OUTPUTS_DIR / "figures"
SOURCE_FIGURES = {
    "target_distribution":  FIG_DIR / "01_target_distribution_and_missing_values.png",
    "numeric_boxplots":     FIG_DIR / "02_numeric_boxplots_outliers.png",
    "feature_importances":  FIG_DIR / "07_feature_importances_decision_tree.png",
    "decision_tree":        FIG_DIR / "08_decision_tree_visualization_top3_levels.png",
    "knn_performance":      FIG_DIR / "10_knn_f1_by_k_and_metric.png",
    "final_comparison":     FIG_DIR / "15_final_model_comparison_f1_rocauc.png",
}

# ── Hardcoded model results (from latest notebook execution) ──────────────────

TREE_TUNING = {
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "cv_f1_score": 0.7502,
}

TOP_FEATURE_IMPORTANCE = [
    ("Sex_female",  0.3150),
    ("Age",         0.2698),
    ("Fare",        0.2349),
    ("Pclass",      0.1097),
    ("Embarked_S",  0.0227),
    ("SibSp",       0.0195),
    ("Parch",       0.0121),
]

READABLE_RULES = [
    {
        "rule": "Male, age > 3.5, Pclass > 1.5, fare ≤ 51.70",
        "prediction": "Did Not Survive",
        "samples": 339,
        "purity": 0.900,
    },
    {
        "rule": "Female, Pclass ≤ 2.5, fare > 28.86, Parch ≤ 1.5",
        "prediction": "Survived",
        "samples": 71,
        "purity": 1.000,
    },
    {
        "rule": "Female, Pclass ≤ 2.5, fare ≤ 28.86, age ≤ 37",
        "prediction": "Survived",
        "samples": 36,
        "purity": 0.944,
    },
    {
        "rule": "Female, Pclass > 2.5, Embarked=S, fare > 17.25",
        "prediction": "Did Not Survive",
        "samples": 27,
        "purity": 0.852,
    },
    {
        "rule": "Female, Pclass > 2.5, Embarked≠S, fare ≤ 8.08",
        "prediction": "Survived",
        "samples": 23,
        "purity": 0.870,
    },
]

BEST_KNN = {"metric": "Manhattan", "k": 9, "cv_f1_score": 0.7574,
            "test_accuracy": 0.7989, "test_f1_score": 0.7188, "test_roc_auc": 0.8432}

BASELINE_RESULTS = [
    ("Logistic Regression",  0.8045, 0.7244, 0.8437),
    ("Gaussian Naive Bayes", 0.7877, 0.7164, 0.8191),
]


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_directories() -> None:
    for d in (REPORT_ASSETS_DIR, REPORT_FIGURES_DIR, REPORT_DATA_DIR, REPORT_TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_dataset_summary() -> dict:
    df = pd.read_csv(DATASET_PATH)
    tc = df["Survived"].value_counts().sort_index()
    mv = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": "Survived",
        "class_balance": {"Did Not Survive (0)": int(tc.get(0, 0)),
                          "Survived (1)": int(tc.get(1, 0))},
        "missing_values": {"Age": int(mv.get("Age", 0)),
                           "Cabin": int(mv.get("Cabin", 0)),
                           "Embarked": int(mv.get("Embarked", 0))},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def trim_white(img: PILImage.Image, pad: int = 14) -> PILImage.Image:
    bg = PILImage.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img
    return img.crop((max(bbox[0]-pad, 0), max(bbox[1]-pad, 0),
                     min(bbox[2]+pad, img.width), min(bbox[3]+pad, img.height)))


def curate_figures() -> dict[str, str]:
    out: dict[str, str] = {}

    # 1. Target distribution — trim and save
    src = PILImage.open(SOURCE_FIGURES["target_distribution"]).convert("RGB")
    p = REPORT_FIGURES_DIR / "target_distribution.png"
    trim_white(src, pad=12).save(p, quality=95)
    out["target_distribution"] = str(p)

    # 2. Numeric boxplots — crop top row (Age + Fare) only
    src = PILImage.open(SOURCE_FIGURES["numeric_boxplots"]).convert("RGB")
    top = src.crop((0, 0, src.width, int(src.height * 0.50)))
    p = REPORT_FIGURES_DIR / "preprocessing_boxplots.png"
    trim_white(top, pad=12).save(p, quality=95)
    out["preprocessing_boxplots"] = str(p)

    # 3. Feature importances bar chart
    src = PILImage.open(SOURCE_FIGURES["feature_importances"]).convert("RGB")
    p = REPORT_FIGURES_DIR / "feature_importances.png"
    trim_white(src, pad=12).save(p, quality=95)
    out["feature_importances"] = str(p)

    # 4. Decision tree visualization
    src = PILImage.open(SOURCE_FIGURES["decision_tree"]).convert("RGB")
    p = REPORT_FIGURES_DIR / "decision_tree.png"
    trim_white(src, pad=16).save(p, quality=95)
    out["decision_tree"] = str(p)

    # 5. kNN performance chart
    src = PILImage.open(SOURCE_FIGURES["knn_performance"]).convert("RGB")
    p = REPORT_FIGURES_DIR / "knn_performance.png"
    trim_white(src, pad=14).save(p, quality=95)
    out["knn_performance"] = str(p)

    # 6. Final model comparison chart
    src = PILImage.open(SOURCE_FIGURES["final_comparison"]).convert("RGB")
    p = REPORT_FIGURES_DIR / "final_comparison.png"
    trim_white(src, pad=12).save(p, quality=95)
    out["final_comparison"] = str(p)

    return out


def curate_tables(figures: dict[str, str]) -> pd.DataFrame:
    comparison_df = pd.read_csv(OUTPUTS_DIR / "tables" / "final_model_comparison.csv")
    report_df = comparison_df[["model", "accuracy", "precision", "recall",
                                "f1_score", "roc_auc"]].copy()
    report_df.to_csv(REPORT_TABLES_DIR / "final_model_comparison_report.csv", index=False)

    pd.DataFrame(TOP_FEATURE_IMPORTANCE, columns=["feature", "importance"]
                 ).to_csv(REPORT_TABLES_DIR / "top_feature_importance.csv", index=False)
    pd.DataFrame(READABLE_RULES
                 ).to_csv(REPORT_TABLES_DIR / "readable_rules.csv", index=False)

    facts = {
        "title": "Machine Learning Assignment Report: Titanic Survival Prediction",
        "institution": "BITS Pilani Digital",
        "student_name": "Anik Das",
        "roll_number": "2025em1100026",
        "subject": "Machine Learning",
        "term": "Trimester 2",
        "dataset_summary": load_dataset_summary(),
        "tree_tuning": TREE_TUNING,
        "best_knn": BEST_KNN,
        "curated_figures": figures,
    }
    (REPORT_DATA_DIR / "report_facts.json").write_text(
        json.dumps(facts, indent=2), encoding="utf-8")

    return report_df


# ── ReportLab helpers ─────────────────────────────────────────────────────────

# Cell paragraph helper — use inside table data so text wraps properly
def _cp(text: str, font: str = "Helvetica", fs: float = 8.0,
        bold: bool = False) -> "Paragraph":
    """Return a Paragraph suitable for use as a table cell value."""
    from reportlab.lib.styles import ParagraphStyle as _PS
    style = _PS(
        name=f"_cell_{id(text)}",
        fontName="Helvetica-Bold" if bold else font,
        fontSize=fs,
        leading=fs + 1.8,
        spaceAfter=0,
        spaceBefore=0,
    )
    return Paragraph(text, style)


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    defs = [
        ("ReportTitle", "Title", "Helvetica-Bold", 18, 22, TA_CENTER,
         "#0F2742", 8, 0),
        ("SectionHeading", "Heading1", "Helvetica-Bold", 13, 15, None,
         "#0F2742", 6, 0),
        ("SubHeading", "Heading2", "Helvetica-Bold", 10, 12, None,
         "#17375E", 3, 3),
        ("BodySmall", "BodyText", "Helvetica", 8.7, 10.8, None,
         None, 3, 0),
        ("PanelText", "BodyText", "Helvetica", 8.0, 9.8, None,
         None, 2, 0),
        ("Caption", "BodyText", "Helvetica-Oblique", 7.7, 9.3, TA_CENTER,
         "#555555", 4, 2),
        ("CoverMeta", "BodyText", "Helvetica", 10.5, 13, TA_CENTER,
         "#22313F", 4, 0),
        ("CompactList", "BodyText", "Helvetica", 8.5, 10.2, None,
         None, 2, 0),
    ]
    for name, parent, font, fs, lead, align, color, after, before in defs:
        kwargs = dict(parent=styles[parent], fontName=font, fontSize=fs,
                      leading=lead, spaceAfter=after, spaceBefore=before)
        if align is not None:
            kwargs["alignment"] = align
        if color is not None:
            kwargs["textColor"] = colors.HexColor(color)
        if name == "CompactList":
            kwargs.update(leftIndent=10, bulletIndent=0)
        styles.add(ParagraphStyle(name=name, **kwargs))
    return styles


def make_table(data, col_widths, highlight_row=None, fs=8.0) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    cmds = [
        ("BACKGROUND",    (0, 0),  (-1, 0),  colors.HexColor("#D9E8F5")),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  colors.HexColor("#0F2742")),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), fs),
        ("LEADING",       (0, 0),  (-1, -1), fs + 2),
        ("GRID",          (0, 0),  (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("VALIGN",        (0, 0),  (-1, -1), "TOP"),        # TOP so wrapped cells align
        ("LEFTPADDING",   (0, 0),  (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0),  (-1, -1), 4),
        ("TOPPADDING",    (0, 0),  (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [colors.white, colors.HexColor("#F8FBFE")]),
    ]
    if highlight_row is not None:
        cmds += [
            ("BACKGROUND", (0, highlight_row), (-1, highlight_row),
             colors.HexColor("#FFF2CC")),
            ("FONTNAME",   (0, highlight_row), (-1, highlight_row),
             "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(cmds))
    return t


def make_panel(data, col_widths, fs=8.0) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F8FBFE")),
        ("BOX",          (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C6D3")),
        ("INNERGRID",    (0, 0), (-1, -1), 0.35, colors.HexColor("#D3DDE7")),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE",     (0, 0), (-1, -1), fs),
        ("LEADING",      (0, 0), (-1, -1), fs + 1.7),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    return t


def pp(title: str, body: str, styles) -> Paragraph:
    return Paragraph(f"<b>{title}</b><br/>{body}", styles["PanelText"])


def bullets(items: list[str]) -> str:
    return "<br/>".join(f"&#8226; {it}" for it in items)


def fit_image(path: str | Path, max_w: float, max_h: float) -> Image:
    img = Image(str(path))
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


def add_page_number(canvas, doc) -> None:
    if canvas.getPageNumber() == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.35 * inch,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


# ── Page builders ─────────────────────────────────────────────────────────────

def page_cover(styles, facts) -> list:
    logo = fit_image(LOGO_PATH, 4.2 * inch, 2.35 * inch)
    panel = make_panel([[
        pp("Assignment objective",
           "Build and compare a complete supervised learning pipeline for Titanic survival "
           "prediction, covering preprocessing, baseline models, decision trees, rule extraction, "
           "kNN, and ensemble methods.",
           styles),
        pp("Report basis",
           "Dataset: Titanic (891 passengers, 12 columns).<br/>"
           "Results validated from executed notebook (43 cells, 0 errors, 15 figures).",
           styles),
    ]], [3.12 * inch, 3.18 * inch], fs=7.8)

    return [
        Spacer(1, 0.5 * inch),
        logo,
        Spacer(1, 0.35 * inch),
        Paragraph(facts["institution"], styles["ReportTitle"]),
        Paragraph(facts["title"],       styles["ReportTitle"]),
        Spacer(1, 0.25 * inch),
        Paragraph("<b>Subject:</b> Machine Learning",            styles["CoverMeta"]),
        Paragraph("<b>Trimester:</b> Trimester 2",               styles["CoverMeta"]),
        Paragraph("<b>Student:</b> Anik Das",                    styles["CoverMeta"]),
        Paragraph("<b>Roll Number:</b> 2025em1100026",           styles["CoverMeta"]),
        Spacer(1, 0.35 * inch),
        panel,
        PageBreak(),
    ]


def page_dataset_preprocessing(styles, facts) -> list:
    ds = facts["dataset_summary"]
    mv = ds["missing_values"]
    cb = ds["class_balance"]

    _s = lambda t, bold=False: _cp(t, fs=8.0, bold=bold)
    summary_tbl = make_table([
        [_s("Property", bold=True), _s("Value", bold=True)],
        [_s("Rows"),          _s(str(ds["rows"]))],
        [_s("Columns"),       _s(str(ds["columns"]))],
        [_s("Target column"), _s(ds["target_column"])],
        [_s("Class balance"),
         _s(f"Not Survived: {cb['Did Not Survive (0)']}   |   Survived: {cb['Survived (1)']}")],
        [_s("Missing values"),
         _s(f"Age: {mv['Age']} (19.9%)     Cabin: {mv['Cabin']} (77.1%)     Embarked: {mv['Embarked']}")],
        [_s("Duplicate rows"),     _s(str(ds["duplicate_rows"]))],
        [_s("Train / Test split"), _s("712 / 179  (80% / 20%, stratified by target)")],
    ], [1.6 * inch, 4.7 * inch], fs=8.0)

    _a = lambda t, bold=False: _cp(t, fs=7.4, bold=bold)
    attr_tbl = make_table([
        [_a("Attribute type", bold=True), _a("Columns", bold=True), _a("Handling", bold=True)],
        [_a("Ordinal"),       _a("Pclass"),
         _a("Used as-is (numeric: 1=First class, 3=Third class)")],
        [_a("Nominal"),       _a("Sex, Embarked"),
         _a("One-hot encoded: Sex_female/male, Embarked_C/Q/S")],
        [_a("Numeric cont."), _a("Age, Fare"),
         _a("Median imputation; StandardScaler applied for LR / NB / kNN")],
        [_a("Numeric disc."), _a("SibSp, Parch"),
         _a("Used as-is; IQR outlier analysis performed")],
        [_a("Excluded"),      _a("PassengerId, Name,\nTicket, Cabin"),
         _a("Identifier-like, high-cardinality, or >77% missing — dropped")],
    ], [0.95 * inch, 1.50 * inch, 3.85 * inch], fs=7.4)

    target_img = fit_image(
        facts["curated_figures"]["target_distribution"],
        max_w=6.2 * inch, max_h=1.8 * inch,
    )
    boxplot_img = fit_image(
        facts["curated_figures"]["preprocessing_boxplots"],
        max_w=6.2 * inch, max_h=1.85 * inch,
    )
    rationale = make_panel([[pp(
        "Preprocessing rationale",
        "PassengerId, Name, Ticket and Cabin were dropped — they are identifiers or mostly missing. "
        "Age was imputed with the median (robust to its right skew). Embarked used mode imputation "
        "(only 2 missing). Fare outliers were retained because very high fares correspond to genuine "
        "first-class passengers, not data errors. Two preprocessor pipelines were built: one with "
        "StandardScaler (for LR, NB, kNN) and one without scaling (for all tree-based models).",
        styles,
    )]], [6.3 * inch], fs=7.8)

    return [
        Paragraph("1. Dataset and Preprocessing", styles["SectionHeading"]),
        Paragraph(
            "The Titanic dataset (Kaggle) contains information on 891 passengers with 12 features. "
            "The binary target Survived (0/1) is moderately imbalanced — 549 non-survivors vs 342 survivors — "
            "so evaluation emphasized F1-score alongside accuracy.",
            styles["BodySmall"],
        ),
        Spacer(1, 3),
        summary_tbl,
        Spacer(1, 4),
        target_img,
        Paragraph(
            "Figure 1. Target class distribution (left) and missing-value counts by column (right). "
            "Cabin is too sparse to impute reliably.",
            styles["Caption"],
        ),
        Spacer(1, 3),
        attr_tbl,
        Spacer(1, 4),
        rationale,
        Spacer(1, 4),
        boxplot_img,
        Paragraph(
            "Figure 2. Age and Fare boxplots confirm right-skewed distributions and high-value outliers. "
            "Outliers were kept — they represent real fare differences between passenger classes.",
            styles["Caption"],
        ),
        PageBreak(),
    ]


def page_baseline_and_tree(styles, facts) -> list:
    tuning = facts["tree_tuning"]

    _b = lambda t, bold=False: _cp(t, fs=7.8, bold=bold)
    baseline_rows = [[_b("Model", bold=True), _b("Accuracy", bold=True),
                      _b("F1", bold=True), _b("ROC-AUC", bold=True)]] + [
        [_b(m), _b(f"{a:.4f}"), _b(f"{f:.4f}"), _b(f"{r:.4f}")]
        for m, a, f, r in BASELINE_RESULTS
    ]
    baseline_tbl = make_table(
        baseline_rows,
        [2.85 * inch, 1.05 * inch, 0.82 * inch, 1.05 * inch], fs=7.8)

    _f = lambda t, bold=False: _cp(t, fs=7.6, bold=bold)
    fi_rows = [[_f("Feature", bold=True), _f("Gini Importance", bold=True)]] + [
        [_f(feat), _f(f"{imp:.4f}")] for feat, imp in TOP_FEATURE_IMPORTANCE
    ]
    fi_tbl = make_table(fi_rows, [2.10 * inch, 1.25 * inch], fs=7.6)

    fi_img   = fit_image(facts["curated_figures"]["feature_importances"],
                         max_w=2.60 * inch, max_h=2.40 * inch)
    tree_img = fit_image(facts["curated_figures"]["decision_tree"],
                         max_w=6.2 * inch, max_h=3.4 * inch)

    # Feature importance: table left, chart right — side by side
    fi_side_tbl = Table(
        [[fi_tbl, fi_img]],
        colWidths=[3.55 * inch, 2.75 * inch],
    )
    fi_side_tbl.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))

    return [
        Paragraph("2. Baseline Models and Decision Tree", styles["SectionHeading"]),

        Paragraph("Baseline model results", styles["SubHeading"]),
        Paragraph(
            "Logistic Regression and Gaussian Naive Bayes were trained as baselines on the "
            "scaled feature set (StandardScaler + one-hot encoding, 80/20 stratified split).",
            styles["BodySmall"],
        ),
        baseline_tbl,
        Spacer(1, 3),
        Paragraph(
            "Logistic Regression (accuracy 0.8045, F1 0.7244, ROC-AUC 0.8437) outperformed Naive Bayes "
            "across all metrics. Both establish a reference floor — any more complex model that cannot "
            "beat these numbers is not adding useful predictive value.",
            styles["BodySmall"],
        ),
        Spacer(1, 4),

        Paragraph("Decision tree training and tuning", styles["SubHeading"]),
        Paragraph(
            f"An untuned Decision Tree was trained first (accuracy 0.8268, F1 0.7634). "
            f"5-fold GridSearchCV then searched max_depth ∈ {{3,4,5,6,None}}, "
            f"min_samples_split ∈ {{2,5,10,20}}, min_samples_leaf ∈ {{1,2,4,6}}. "
            f"Best CV params: max_depth=None, min_samples_split=10, min_samples_leaf=1 "
            f"(CV F1={tuning['cv_f1_score']:.4f}). The tuned tree scored 0.7111 F1 on the test set — "
            f"worse than the untuned tree. CV ranking and held-out ranking can diverge, especially with "
            f"small datasets.",
            styles["BodySmall"],
        ),
        Spacer(1, 4),

        Paragraph("Feature importance and decision paths", styles["SubHeading"]),
        Paragraph(
            "Sex_female is the dominant predictor, followed by Age, Fare, and Pclass. "
            "Embarkation port contributes very little. "
            "Two representative paths from the untuned tree:",
            styles["BodySmall"],
        ),
        Paragraph(
            "&#8226; <b>Path 1 (Survived):</b> Female, Pclass ≤ 2.5, Fare > 28.86, Parch ≤ 1.5 "
            "— 71 training passengers, purity 1.00.<br/>"
            "&#8226; <b>Path 2 (Did Not Survive):</b> Male, Age > 3.5, Pclass > 1.5, Fare ≤ 7.13 "
            "— large group, purity 0.90+.",
            styles["CompactList"],
        ),
        Spacer(1, 4),
        fi_side_tbl,
        Paragraph(
            "Figure 3 (left). Feature importance scores. Figure 3 (right). Horizontal bar confirms "
            "Sex_female, Age, and Fare dominate — Embarked_S and SibSp/Parch contribute very little.",
            styles["Caption"],
        ),
        Spacer(1, 4),
        tree_img,
        Paragraph(
            "Figure 4. Decision tree top 3 levels. First split on Sex_female; subsequent splits "
            "on Fare, Age, and Pclass reflect known survival patterns from the historical record.",
            styles["Caption"],
        ),
        PageBreak(),
    ]


def page_rules_and_knn(styles, facts) -> list:
    knn = facts["best_knn"]
    knn_img = fit_image(facts["curated_figures"]["knn_performance"],
                        max_w=6.0 * inch, max_h=3.0 * inch)

    rule_rows = [
        [
            Paragraph("<b>Rule</b>", styles["BodySmall"]),
            Paragraph("<b>Prediction</b>", styles["BodySmall"]),
            Paragraph("<b>n</b>", styles["BodySmall"]),
            Paragraph("<b>Purity</b>", styles["BodySmall"]),
        ]
    ]
    for idx, r in enumerate(READABLE_RULES, 1):
        rule_rows.append([
            Paragraph(f"{idx}. IF {r['rule']}", styles["BodySmall"]),
            Paragraph(r["prediction"], styles["BodySmall"]),
            Paragraph(str(r["samples"]), styles["BodySmall"]),
            Paragraph(f"{r['purity']:.3f}", styles["BodySmall"]),
        ])
    rule_tbl = make_table(
        rule_rows,
        [3.55 * inch, 1.08 * inch, 0.48 * inch, 0.59 * inch],
        fs=7.2,
    )

    return [
        Paragraph("3. Rule-Based Classification and kNN", styles["SectionHeading"]),

        Paragraph("Rule-based classification", styles["SubHeading"]),
        Paragraph(
            "A constrained decision tree (max_depth=4, min_samples_split=20, min_samples_leaf=10) "
            "was trained to generate short, human-readable rules. It achieved accuracy 0.7765 and "
            "F1 0.6552 — lower than the full tree, but every decision is expressible as a plain "
            "if-then statement with measurable support and purity.",
            styles["BodySmall"],
        ),
        Spacer(1, 3),
        rule_tbl,
        Paragraph(
            "Table 3. Five rules extracted from the constrained tree with support (n = training "
            "passengers) and purity. Rule 2 (purity 1.00, 71 passengers) is perfectly reliable.",
            styles["Caption"],
        ),
        Spacer(1, 4),
        Paragraph(
            "Interpretability trade-off: the rule-based model is easy to audit and communicate — "
            "a stakeholder can verify each rule without ML knowledge. The cost is a drop in recall "
            "(0.55 vs 0.72 for the full tree), meaning the model misses more survivors.",
            styles["BodySmall"],
        ),
        Spacer(1, 6),

        Paragraph("kNN (Lazy Learning)", styles["SubHeading"]),
        Paragraph(
            "kNN was evaluated with k ∈ {1, 3, 5, 7, 9} and two distance metrics (Euclidean, Manhattan). "
            "All experiments used 5-fold cross-validation on the training set with preprocessing inside "
            "each fold to avoid data leakage. Best result: Manhattan distance, k=9 "
            f"(CV F1={knn['cv_f1_score']:.4f}, test accuracy={knn['test_accuracy']:.4f}, "
            f"test F1={knn['test_f1_score']:.4f}, ROC-AUC={knn['test_roc_auc']:.4f}).",
            styles["BodySmall"],
        ),
        Paragraph(
            "Why scaling is essential for kNN: Fare ranges up to £512 while SibSp ranges 0–8. "
            "Without StandardScaler, distance is dominated entirely by Fare, making the other features "
            "irrelevant. After scaling, all features contribute proportionally.",
            styles["BodySmall"],
        ),
        Paragraph(
            "Bias-variance trade-off: k=1 memorises training data (low bias, high variance — overfits). "
            "As k increases, the boundary smooths and variance falls but bias rises. "
            "Performance peaked at k=7–9, indicating the data contains enough noise that very local "
            "neighbourhoods hurt generalisation.",
            styles["BodySmall"],
        ),
        Spacer(1, 4),
        knn_img,
        Paragraph(
            "Figure 5. Cross-validated F1 score vs k for Euclidean and Manhattan distance. "
            "Manhattan consistently outperforms Euclidean; both peak near k=7–9.",
            styles["Caption"],
        ),
        PageBreak(),
    ]


def page_ensemble_comparison(styles, facts, comparison_df: pd.DataFrame) -> list:
    # Build full comparison table with precision and recall too
    _c = lambda t, bold=False: _cp(t, fs=7.5, bold=bold)
    rows = [[_c("Model", bold=True), _c("Acc", bold=True), _c("Prec", bold=True),
             _c("Rec", bold=True), _c("F1", bold=True), _c("AUC", bold=True)]]
    highlight = None
    for i, row in comparison_df.iterrows():
        rows.append([
            _c(row["model"]),
            _c(f"{row['accuracy']:.3f}"),
            _c(f"{row['precision']:.3f}"),
            _c(f"{row['recall']:.3f}"),
            _c(f"{row['f1_score']:.3f}"),
            _c(f"{row['roc_auc']:.3f}"),
        ])
        if row["model"] == "Decision Tree (Baseline)":
            highlight = i + 1

    comp_tbl = make_table(
        rows,
        [2.35 * inch, 0.68 * inch, 0.68 * inch, 0.60 * inch, 0.60 * inch, 0.68 * inch],
        highlight_row=highlight, fs=7.5,
    )

    comp_img = fit_image(facts["curated_figures"]["final_comparison"],
                         max_w=6.2 * inch, max_h=2.45 * inch)

    return [
        Paragraph("4. Ensemble Learning and Final Comparison", styles["SectionHeading"]),

        Paragraph("Ensemble results", styles["SubHeading"]),
        Paragraph(
            "Three ensemble methods were trained: Random Forest (300 trees, bagging), "
            "Gradient Boosting (sequential boosting), and AdaBoost (adaptive boosting). "
            "All used the unscaled tree preprocessor (imputation + one-hot encoding only).",
            styles["BodySmall"],
        ),
        Paragraph(
            "Surprisingly, none of the ensemble models beat the plain untuned Decision Tree. "
            "Random Forest matched Logistic Regression on accuracy (0.8045) but trailed on F1 "
            "(0.7154 vs 0.7634). Gradient Boosting and AdaBoost were weaker still. "
            "This is expected for small, low-dimensional datasets where the dominant features "
            "(sex, age, fare) are already captured by a single well-structured tree.",
            styles["BodySmall"],
        ),
        Spacer(1, 4),

        Paragraph("Final model comparison — all 9 models", styles["SubHeading"]),
        comp_tbl,
        Paragraph(
            "Table 4. All models ranked by F1 score (descending). Highlighted row = best overall. "
            "Acc = Accuracy, Prec = Precision, Rec = Recall, AUC = ROC-AUC.",
            styles["Caption"],
        ),
        Spacer(1, 4),
        comp_img,
        Paragraph(
            "Figure 6. F1 score (left) and ROC-AUC (right) for all 9 models. "
            "Decision Tree (Baseline) leads on F1; Logistic Regression leads on ROC-AUC.",
            styles["Caption"],
        ),
        Spacer(1, 6),

        Paragraph("Key insights", styles["SubHeading"]),
        Paragraph(
            bullets([
                "Decision Tree (Baseline): best F1 (0.763) and accuracy (0.827) — top overall model.",
                "Logistic Regression: best ROC-AUC (0.844) — most reliable probability ranking.",
                "kNN: competitive only after scaling; confirms preprocessing importance.",
                "Rule-Based Tree: lowest F1 but most interpretable — best for explanation-first use.",
                "Ensembles: did not gain over simpler models here — complexity ≠ guaranteed improvement.",
            ]),
            styles["CompactList"],
        ),
        PageBreak(),
    ]


def page_conclusions(styles) -> list:
    return [
        Paragraph("5. Conclusions, Risks, and Recommendations", styles["SectionHeading"]),

        Paragraph("Best-performing model", styles["SubHeading"]),
        Paragraph(
            "The untuned Decision Tree achieved the highest F1 (0.7634) and accuracy (0.8268) "
            "on the held-out test set. It captures non-linear interactions between sex, class, "
            "age, and fare that a linear model cannot represent, without requiring the additional "
            "complexity of ensemble aggregation. Interestingly, the GridSearchCV-tuned variant "
            "scored worse on the test set (F1 0.7111), showing that CV performance does not "
            "always translate to held-out performance on small datasets.",
            styles["BodySmall"],
        ),
        Spacer(1, 5),

        Paragraph("Most interpretable model", styles["SubHeading"]),
        Paragraph(
            "The Rule-Based Tree is the most interpretable. Its five extracted rules can be "
            "written as plain if-then statements that a non-technical audience can read and "
            "verify without understanding machine learning. Each rule includes the number of "
            "training passengers it covers and its purity, giving statistical backing to the "
            "plain-language explanation. The cost is a drop in recall from 0.72 to 0.55.",
            styles["BodySmall"],
        ),
        Spacer(1, 5),

        Paragraph("Deployment risks", styles["SubHeading"]),
        Paragraph(
            bullets([
                "Historical bias: the model is trained on 1912 data — survival patterns reflect "
                "social norms of that era and cannot generalize to modern contexts.",
                "Fairness: sex and passenger class are the strongest predictors, so the model makes "
                "systematically different predictions for men vs women and rich vs poor passengers.",
                "Threshold brittleness: tree splits create hard boundaries (e.g. Fare ≤ 7.13 → "
                "non-survival). A passenger just above or below that threshold gets opposite predictions.",
                "Missing data sensitivity: Age is missing for ~20% of passengers; the imputation "
                "strategy directly affects predictions for those passengers in deployment.",
                "Small dataset: 891 rows is insufficient for stable deployment or strong confidence "
                "intervals on out-of-distribution inputs.",
            ]),
            styles["CompactList"],
        ),
        Spacer(1, 5),

        Paragraph("Limitations of this analysis", styles["SubHeading"]),
        Paragraph(
            bullets([
                "Single train-test split (random_state=42) — results may vary with different seeds.",
                "No repeated cross-validation or bootstrap resampling for confidence intervals.",
                "Feature engineering (e.g. family size, title from Name) was intentionally excluded "
                "to keep the pipeline transparent and aligned with the assignment scope.",
                "XGBoost was not tested (not available in the environment); Gradient Boosting from "
                "sklearn was used as the boosting representative.",
            ]),
            styles["CompactList"],
        ),
        Spacer(1, 5),

        Paragraph("Recommendations", styles["SubHeading"]),
        Paragraph(
            "For maximum predictive performance on this dataset: use the baseline Decision Tree. "
            "For stakeholder communication and explainability: use the Rule-Based Tree. "
            "For a compact statistical baseline: Logistic Regression offers stable ROC-AUC (0.844) "
            "with a simple, well-understood model form. Future work could improve robustness through "
            "repeated cross-validation, threshold tuning, and title-based feature engineering.",
            styles["BodySmall"],
        ),
        Spacer(1, 5),

        Paragraph("Conclusion", styles["SubHeading"]),
        Paragraph(
            "This assignment built a complete supervised learning pipeline from raw Titanic data "
            "to a final model comparison across nine classifiers. Careful preprocessing, stratified "
            "evaluation, and explicit interpretability analysis showed that a simple Decision Tree "
            "can outperform ensemble methods when features are low-dimensional and highly predictive. "
            "The most important result is not which model won numerically, but that the analysis "
            "exposes a concrete trade-off: the Rule-Based Tree loses 10 F1 points relative to the "
            "best tree, but gains full human-readable explainability — a trade-off that matters "
            "enormously in any real deployment context.",
            styles["BodySmall"],
        ),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def build_pdf(facts: dict, comparison_df: pd.DataFrame) -> None:
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        pageCompression=0,
    )
    story: list = []
    story += page_cover(styles, facts)
    story += page_dataset_preprocessing(styles, facts)
    story += page_baseline_and_tree(styles, facts)
    story += page_rules_and_knn(styles, facts)
    story += page_ensemble_comparison(styles, facts, comparison_df)
    story += page_conclusions(styles)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def count_pages(pdf_path: Path) -> int:
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def main() -> None:
    ensure_directories()
    dataset_summary = load_dataset_summary()
    curated = curate_figures()
    comparison_df = curate_tables(curated)

    facts = json.loads((REPORT_DATA_DIR / "report_facts.json").read_text(encoding="utf-8"))
    build_pdf(facts, comparison_df)

    pages = count_pages(REPORT_PATH)
    print(f"Report written to : {REPORT_PATH}")
    print(f"Page count        : {pages}")
    if pages > 6:
        print(f"WARNING: report is {pages} pages — exceeds the 6-page limit.")
    else:
        print("Page count is within the 6-page limit.")


if __name__ == "__main__":
    main()
