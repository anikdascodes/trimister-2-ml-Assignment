from __future__ import annotations

import json
import re
import shutil
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
NOTEBOOK_PATH = PROJECT_ROOT / "assignment_notebook.ipynb"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_ASSETS_DIR = PROJECT_ROOT / "report_assets"
REPORT_FIGURES_DIR = REPORT_ASSETS_DIR / "figures"
REPORT_DATA_DIR = REPORT_ASSETS_DIR / "data"
REPORT_TABLES_DIR = REPORT_ASSETS_DIR / "tables"
REPORT_PATH = PROJECT_ROOT / "report.pdf"

SOURCE_FIGURES = {
    "numeric_boxplots.png": OUTPUTS_DIR / "figures" / "numeric_boxplots.png",
    "decision_tree_top_levels.png": OUTPUTS_DIR / "figures" / "decision_tree_top_levels.png",
    "knn_f1_by_k.png": OUTPUTS_DIR / "figures" / "knn_f1_by_k.png",
}

CURATED_FIGURES = {
    "preprocessing_outliers.png": "numeric_boxplots.png",
    "decision_tree_structure.png": "decision_tree_top_levels.png",
    "knn_performance.png": "knn_f1_by_k.png",
}

TREE_TUNING = {
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "cv_f1_score": 0.7502,
}

TOP_FEATURE_IMPORTANCE = [
    ("Sex_female", 0.3150),
    ("Age", 0.2698),
    ("Fare", 0.2349),
    ("Pclass", 0.1097),
    ("Embarked_S", 0.0227),
]

READABLE_RULES = [
    {
        "rule": "Male passenger, age > 3.5, class > 1.5, fare <= 51.698",
        "prediction": "Did Not Survive",
        "samples": 339,
        "purity": 0.900,
    },
    {
        "rule": "Female passenger, class <= 2.5, fare > 28.856, parch <= 1.5",
        "prediction": "Survived",
        "samples": 71,
        "purity": 1.000,
    },
    {
        "rule": "Female passenger, class <= 2.5, fare <= 28.856, age <= 37",
        "prediction": "Survived",
        "samples": 36,
        "purity": 0.944,
    },
    {
        "rule": "Female passenger, class > 2.5, Embarked = S, fare > 17.250",
        "prediction": "Did Not Survive",
        "samples": 27,
        "purity": 0.852,
    },
    {
        "rule": "Female passenger, class > 2.5, Embarked != S, fare <= 8.083",
        "prediction": "Survived",
        "samples": 23,
        "purity": 0.870,
    },
]

BEST_KNN = {
    "metric": "Manhattan",
    "k": 9,
    "cv_accuracy": 0.8273,
    "cv_f1_score": 0.7574,
    "cv_roc_auc": 0.8690,
    "test_accuracy": 0.7989,
    "test_f1_score": 0.7188,
}

BASELINE_RESULTS = [
    ("Logistic Regression", 0.8045, 0.7244, 0.8437),
    ("Gaussian Naive Bayes", 0.7877, 0.7164, 0.8191),
]


def ensure_directories() -> None:
    for directory in (REPORT_ASSETS_DIR, REPORT_FIGURES_DIR, REPORT_DATA_DIR, REPORT_TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset_summary() -> dict[str, object]:
    df = pd.read_csv(DATASET_PATH)
    target_counts = df["Survived"].value_counts().sort_index()
    missing = df.isna().sum()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": "Survived",
        "class_balance": {
            "Did Not Survive (0)": int(target_counts.get(0, 0)),
            "Survived (1)": int(target_counts.get(1, 0)),
        },
        "missing_values": {
            "Age": int(missing.get("Age", 0)),
            "Cabin": int(missing.get("Cabin", 0)),
            "Embarked": int(missing.get("Embarked", 0)),
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }


def trim_white_border(image: PILImage.Image, padding: int = 16) -> PILImage.Image:
    background = PILImage.new(image.mode, image.size, "white")
    difference = ImageChops.difference(image, background)
    bbox = difference.getbbox()
    if bbox is None:
        return image

    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image.width)
    lower = min(bbox[3] + padding, image.height)
    return image.crop((left, upper, right, lower))


def curate_figures() -> dict[str, str]:
    curated_paths: dict[str, str] = {}

    preprocessing_source = PILImage.open(SOURCE_FIGURES["numeric_boxplots.png"]).convert("RGB")
    top_row = preprocessing_source.crop((0, 0, preprocessing_source.width, int(preprocessing_source.height * 0.495)))
    preprocessing_report_image = trim_white_border(top_row, padding=14)
    preprocessing_target = REPORT_FIGURES_DIR / "preprocessing_outliers.png"
    preprocessing_report_image.save(preprocessing_target, quality=95)
    curated_paths["preprocessing_outliers.png"] = str(preprocessing_target)

    for target_name in ("decision_tree_structure.png", "knn_performance.png"):
        source_name = CURATED_FIGURES[target_name]
        source_path = SOURCE_FIGURES[source_name]
        target_path = REPORT_FIGURES_DIR / target_name
        trimmed_image = trim_white_border(PILImage.open(source_path).convert("RGB"), padding=18)
        trimmed_image.save(target_path, quality=95)
        curated_paths[target_name] = str(target_path)

    return curated_paths


def curate_tables() -> pd.DataFrame:
    comparison_path = OUTPUTS_DIR / "tables" / "final_model_comparison.csv"
    comparison_df = pd.read_csv(comparison_path)
    report_table_df = comparison_df[["model", "accuracy", "f1_score", "roc_auc"]].copy()
    report_table_df.to_csv(REPORT_TABLES_DIR / "final_model_comparison_report.csv", index=False)

    feature_df = pd.DataFrame(TOP_FEATURE_IMPORTANCE, columns=["feature", "importance"])
    feature_df.to_csv(REPORT_TABLES_DIR / "top_feature_importance.csv", index=False)

    rules_df = pd.DataFrame(READABLE_RULES)
    rules_df.to_csv(REPORT_TABLES_DIR / "readable_rules.csv", index=False)

    return report_table_df


def write_report_facts(dataset_summary: dict[str, object], curated_paths: dict[str, str]) -> None:
    facts = {
        "title": "Machine Learning Assignment Report: Titanic Survival Prediction",
        "institution": "BITS Pilani Digital",
        "student_name": "Anik Das",
        "roll_number": "2025em1100026",
        "subject": "Machine Learning",
        "term": "Trimester 2",
        "dataset_summary": dataset_summary,
        "tree_tuning": TREE_TUNING,
        "top_feature_importance": TOP_FEATURE_IMPORTANCE,
        "best_knn": BEST_KNN,
        "curated_figures": curated_paths,
    }
    (REPORT_DATA_DIR / "report_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F2742"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=15,
            textColor=colors.HexColor("#0F2742"),
            spaceAfter=6,
            spaceBefore=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#17375E"),
            spaceAfter=3,
            spaceBefore=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=10.8,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PanelText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=9.8,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.7,
            leading=9.3,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=4,
            spaceBefore=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#22313F"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CompactList",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.2,
            leftIndent=10,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    return styles


def build_table(data, col_widths, highlight_row: int | None = None, font_size: float = 8.0) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E8F5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F2742")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFE")]),
    ]
    if highlight_row is not None:
        style_commands.extend(
            [
                ("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#FFF2CC")),
                ("FONTNAME", (0, highlight_row), (-1, highlight_row), "Helvetica-Bold"),
            ]
        )
    table.setStyle(TableStyle(style_commands))
    return table


def build_panel_table(data, col_widths, font_size: float = 8.0) -> Table:
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FBFE")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C6D3")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D3DDE7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 1.7),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def panel_paragraph(title: str, body: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"<b>{title}</b><br/>{body}", styles["PanelText"])


def bullet_html(items: list[str]) -> str:
    return "<br/>".join(f"&#8226; {item}" for item in items)


def report_image(path: Path, max_width: float, max_height: float | None = None) -> Image:
    image = Image(str(path))
    width_scale = max_width / image.imageWidth
    height_scale = (max_height / image.imageHeight) if max_height is not None else width_scale
    scale = min(width_scale, height_scale)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def add_page_number(canvas, doc) -> None:
    if canvas.getPageNumber() == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.35 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def compact_rule_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[
        Paragraph("<b>Rule</b>", styles["BodySmall"]),
        Paragraph("<b>Prediction</b>", styles["BodySmall"]),
        Paragraph("<b>Support</b>", styles["BodySmall"]),
        Paragraph("<b>Purity</b>", styles["BodySmall"]),
    ]]
    for index, rule in enumerate(READABLE_RULES, start=1):
        rows.append([
            Paragraph(f"{index}. {rule['rule']}", styles["BodySmall"]),
            Paragraph(rule["prediction"], styles["BodySmall"]),
            Paragraph(str(rule["samples"]), styles["BodySmall"]),
            Paragraph(f"{rule['purity']:.3f}", styles["BodySmall"]),
        ])
    return build_table(rows, [3.72 * inch, 0.98 * inch, 0.62 * inch, 0.58 * inch], font_size=7.1)


def compact_rule_paragraphs(styles: dict[str, ParagraphStyle]) -> list:
    story = [Paragraph("Readable rule set extracted from the shallower tree:", styles["BodySmall"])]
    for index, rule in enumerate(READABLE_RULES, start=1):
        story.append(
            Paragraph(
                f"<b>{index}.</b> {rule['rule']} <b>Prediction:</b> {rule['prediction']}. "
                f"<b>Support:</b> {rule['samples']} passengers. <b>Purity:</b> {rule['purity']:.3f}.",
                styles["BodySmall"],
            )
        )
    return story


def page_two_story(styles: dict[str, ParagraphStyle], facts: dict[str, object]) -> list:
    dataset = facts["dataset_summary"]
    missing = dataset["missing_values"]
    class_balance = dataset["class_balance"]
    attribute_table = build_table(
        [
            ["Attribute type", "Columns"],
            ["Ordinal", "Pclass"],
            ["Nominal", "Sex, Embarked"],
            ["Numeric", "Age, Fare, SibSp, Parch"],
        ],
        [1.35 * inch, 1.75 * inch],
        font_size=7.6,
    )
    summary_table = build_table(
        [
            ["Property", "Value"],
            ["Rows", str(dataset["rows"])],
            ["Columns", str(dataset["columns"])],
            ["Target", str(dataset["target_column"])],
            ["Class balance", f"0: {class_balance['Did Not Survive (0)']}, 1: {class_balance['Survived (1)']}"],
            ["Missing values", f"Age {missing['Age']}, Cabin {missing['Cabin']}, Embarked {missing['Embarked']}"],
            ["Duplicate rows", str(dataset["duplicate_rows"])],
        ],
        [1.55 * inch, 4.7 * inch],
        font_size=8.0,
    )
    preprocessing_image = report_image(
        Path(facts["curated_figures"]["preprocessing_outliers.png"]),
        max_width=6.2 * inch,
        max_height=2.9 * inch,
    )
    summary_row = build_panel_table([[summary_table, attribute_table]], [3.75 * inch, 2.55 * inch], font_size=7.6)
    rationale_box = build_panel_table(
        [[
            panel_paragraph(
                "Preprocessing rationale",
                "Identifier-like or sparse fields (`PassengerId`, `Name`, `Ticket`, `Cabin`) were excluded because they were non-predictive, high-cardinality, or mostly missing. "
                "`Age` used median imputation and `Embarked` used mode imputation so rows could be retained. Outliers were kept because they represented plausible passengers, especially expensive fares, rather than obvious data-entry errors.",
                styles,
            )
        ]],
        [6.3 * inch],
        font_size=7.8,
    )

    return [
        Paragraph("1. Introduction, Dataset, and Preprocessing", styles["SectionHeading"]),
        Paragraph(
            "This report summarizes a complete supervised learning workflow for Titanic survival prediction. "
            "The objective was to clean the data, compare multiple classifiers, and balance predictive performance with interpretability.",
            styles["BodySmall"],
        ),
        Paragraph(
            "The class distribution was moderately imbalanced at 549 non-survivors versus 342 survivors, so the analysis emphasized stratified splitting and F1-score rather than relying on accuracy alone.",
            styles["BodySmall"],
        ),
        summary_row,
        Spacer(1, 3),
        Paragraph(
            "The dataset mixes ordinal, nominal, discrete, and continuous attributes. `Pclass` was treated as ordinal, "
            "`Sex` and `Embarked` as nominal, and `Age`, `Fare`, `SibSp`, and `Parch` as numeric features.",
            styles["BodySmall"],
        ),
        rationale_box,
        Spacer(1, 4),
        preprocessing_image,
        Paragraph(
            "Figure 1. Age and fare boxplots show skew and outliers, especially for `Fare`, which motivated discussion but not deletion.",
            styles["Caption"],
        ),
        Paragraph(
            "After preprocessing, the modeling feature set contained `Pclass`, one-hot encoded `Sex` and `Embarked`, and the numeric fields `Age`, `Fare`, `SibSp`, and `Parch`. "
            "This retained the strongest interpretable variables while avoiding sparse or identifier-like columns that would add noise without supporting explanation.",
            styles["BodySmall"],
        ),
        PageBreak(),
    ]


def page_three_story(styles: dict[str, ParagraphStyle], facts: dict[str, object]) -> list:
    tuning = facts["tree_tuning"]
    feature_rows = [["Feature", "Importance"]] + [[feature, f"{score:.4f}"] for feature, score in TOP_FEATURE_IMPORTANCE]
    feature_table = build_table(feature_rows, [2.45 * inch, 1.0 * inch], font_size=7.7)
    baseline_rows = [["Model", "Accuracy", "F1", "ROC-AUC"]]
    for model, accuracy, f1_score, roc_auc in BASELINE_RESULTS:
        baseline_rows.append([model, f"{accuracy:.4f}", f"{f1_score:.4f}", f"{roc_auc:.4f}"])
    baseline_table = build_table(baseline_rows, [2.52 * inch, 0.88 * inch, 0.68 * inch, 0.88 * inch], font_size=7.5)
    tree_image = report_image(
        Path(facts["curated_figures"]["decision_tree_structure.png"]),
        max_width=6.15 * inch,
        max_height=3.8 * inch,
    )

    return [
        Paragraph("2. Modeling Setup and Decision Tree Analysis", styles["SectionHeading"]),
        Paragraph(
            "Evaluation setup: the data was split using an 80/20 stratified train-test split with `random_state = 42`. "
            "Performance was compared using accuracy, precision, recall, F1-score, and ROC-AUC where available.",
            styles["BodySmall"],
        ),
        baseline_table,
        Spacer(1, 2),
        Paragraph(
            f"Decision tree tuning used 5-fold cross-validation on the training data only. The best search result was "
            f"`max_depth = {tuning['max_depth']}`, `min_samples_split = {tuning['min_samples_split']}`, "
            f"`min_samples_leaf = {tuning['min_samples_leaf']}` with cross-validated F1 = {tuning['cv_f1_score']:.4f}.",
            styles["BodySmall"],
        ),
        Paragraph(
            "The baseline tree was retained for interpretation because it achieved the stronger held-out result while also keeping the decision paths easy to explain in a classroom setting. "
            "That made it a better fit for the final report than the tuned alternative, even though the tuned model won the training-only search.",
            styles["BodySmall"],
        ),
        Paragraph(
            "Top decision-tree feature importance scores are shown below. The values reinforce that survival in this dataset is driven mostly by sex, age, fare, and passenger class, with embarkation contributing much less.",
            styles["BodySmall"],
        ),
        feature_table,
        Spacer(1, 8),
        tree_image,
        Paragraph(
            "Figure 2. The top levels of the decision tree show that sex, age, fare, and passenger class dominate the early splits.",
            styles["Caption"],
        ),
        Paragraph(
            "Decision tree interpretation: the first major splits are driven by sex, then by age, fare, and passenger class. "
            "This indicates that a small number of strong predictors capture most of the useful survival structure in the selected feature set.",
            styles["BodySmall"],
        ),
        Paragraph(
            "The structure also highlights useful interaction effects. For example, the meaning of fare changes depending on class and sex, which helps explain why a tree-based model outperformed the purely linear baseline on this dataset.",
            styles["BodySmall"],
        ),
        PageBreak(),
    ]


def page_four_story(styles: dict[str, ParagraphStyle], facts: dict[str, object]) -> list:
    knn = facts["best_knn"]
    knn_image = report_image(
        Path(facts["curated_figures"]["knn_performance.png"]),
        max_width=6.0 * inch,
        max_height=3.3 * inch,
    )

    story = [
        Paragraph("3. Rule-Based Classification and kNN", styles["SectionHeading"]),
        Paragraph(
            "A shallower tree was converted into a compact rule-based classifier so that the logic could be presented directly as human-readable conditions rather than only as a plotted tree.",
            styles["BodySmall"],
        ),
    ]
    story.extend(compact_rule_paragraphs(styles))
    story.extend(
        [
            Spacer(1, 4),
            Paragraph(
                "These extracted rules capture the most common survival patterns in a directly explainable form. They are especially useful for showing how the model links gender, age, fare, embarkation, and passenger class to the final prediction.",
                styles["BodySmall"],
            ),
            Paragraph(
                "Interpretability tradeoff: the rule-based tree is easier to explain because its logic can be written as a short set of explicit if-then rules. "
                "Its held-out performance was weaker than the strongest decision tree, so the gain in interpretability came with a measurable predictive tradeoff.",
                styles["BodySmall"],
            ),
            Paragraph(
                f"kNN interpretation: scaling mattered because kNN depends directly on feature distance. Manhattan distance with k = {knn['k']} performed best, "
                "which suggests that a smoother neighborhood definition was more stable for this dataset than more local, noise-sensitive settings.",
                styles["BodySmall"],
            ),
            Paragraph(
                "Larger k values reduce variance by averaging over more neighbors, while smaller k values react more sharply to local variation and therefore behave with lower bias but higher variance.",
                styles["BodySmall"],
            ),
            Spacer(1, 6),
            knn_image,
            Paragraph(
                "Figure 3. Cross-validated kNN F1-scores improved after scaling, with Manhattan distance at k = 9 giving the best overall result.",
                styles["Caption"],
            ),
            PageBreak(),
        ]
    )
    return story


def page_five_story(styles: dict[str, ParagraphStyle], comparison_df: pd.DataFrame) -> list:
    table_df = comparison_df.copy()
    rows = [["Model", "Accuracy", "F1", "ROC-AUC"]]
    highlight_row = None
    for index, row in table_df.iterrows():
        rows.append([
            row["model"],
            f"{row['accuracy']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['roc_auc']:.4f}",
        ])
        if row["model"] == "Decision Tree (Baseline)":
            highlight_row = index + 1

    return [
        Paragraph("4. Ensemble Learning and Final Comparison", styles["SectionHeading"]),
        Paragraph(
            "Three ensemble methods were evaluated: Random Forest, Gradient Boosting, and AdaBoost. "
            "In this split they remained competitive but did not clearly outperform the best simpler models. "
            "That finding is important because it shows that higher model complexity did not automatically lead to the strongest result.",
            styles["BodySmall"],
        ),
        Paragraph(
            "The final comparison table below summarizes the main held-out metrics used in the report.",
            styles["BodySmall"],
        ),
        build_table(rows, [2.75 * inch, 1.05 * inch, 0.82 * inch, 1.05 * inch], highlight_row=highlight_row, font_size=8.0),
        Spacer(1, 12),
        Paragraph("Ensemble findings", styles["SubHeading"]),
        Paragraph(
            "Random Forest, Gradient Boosting, and AdaBoost all produced reasonable results, but none clearly surpassed the strongest simpler models. "
            "Random Forest matched Logistic Regression on accuracy but not on F1, while Gradient Boosting and AdaBoost were slightly weaker overall. "
            "The main lesson is that additional ensemble complexity did not create enough class-sensitive improvement to justify a more complex final choice on this dataset.",
            styles["BodySmall"],
        ),
        Spacer(1, 10),
        Paragraph("Why simpler models worked well here", styles["SubHeading"]),
        Paragraph(
            "The dataset is modest in size, the selected features are low-dimensional, and survival is strongly influenced by a few high-signal variables such as sex, age, fare, and class. "
            "Under those conditions, a well-shaped tree and a strong linear baseline can already capture much of the useful structure without very deep or highly aggregated models.",
            styles["BodySmall"],
        ),
        Paragraph(
            "This is an important assignment result because it shows that choosing an interpretable model family did not require sacrificing strong predictive quality. In other words, the simpler models were not just easier to explain; they were also highly competitive on the actual task.",
            styles["BodySmall"],
        ),
        Spacer(1, 10),
        Paragraph("Key comparison insights", styles["SubHeading"]),
        Paragraph(
            bullet_html(
                [
                    "Decision Tree (Baseline) achieved the top F1-score and accuracy, showing that non-linear splits were valuable.",
                    "Logistic Regression remained the strongest non-tree baseline and offered stable overall behavior.",
                    "kNN became competitive only after scaling and cross-validated tuning, confirming the importance of preprocessing.",
                    "Rule-Based Tree gave the clearest explanations, but that interpretability came with lower predictive strength.",
                ]
            ),
            styles["CompactList"],
        ),
        Spacer(1, 10),
        Paragraph("Top practical choices", styles["SubHeading"]),
        Paragraph(
            bullet_html(
                [
                    "Decision Tree: best overall predictive choice for this assignment dataset.",
                    "Logistic Regression: strongest compact baseline with stable behavior and solid ROC-AUC.",
                    "Rule-Based Tree: best option when interpretability and stakeholder communication are the main priorities.",
                ]
            ),
            styles["CompactList"],
        ),
        Paragraph(
            "Taken together, the final table supports a clear ranking: the baseline Decision Tree is the strongest overall submission model, Logistic Regression is the best compact statistical baseline, and the Rule-Based Tree is the most suitable option for explanation-focused use cases.",
            styles["BodySmall"],
        ),
        PageBreak(),
    ]


def page_six_story(styles: dict[str, ParagraphStyle]) -> list:
    return [
        Paragraph("5. Final Answers, Risks, and Conclusion", styles["SectionHeading"]),
        Paragraph("Best-performing model", styles["SubHeading"]),
        Paragraph(
            "The baseline Decision Tree was the strongest overall model in this analysis. It achieved accuracy = 0.8268 and F1-score = 0.7634, "
            "giving the best balance between overall correctness and class-sensitive performance on the held-out test set. "
            "Its strong result is consistent with the way the dataset responds to non-linear interactions among sex, age, fare, and passenger class.",
            styles["BodySmall"],
        ),
        Spacer(1, 4),
        Paragraph("Most interpretable model", styles["SubHeading"]),
        Paragraph(
            "The Rule-Based Tree was the most interpretable approach because its logic can be communicated as a short set of explicit if-then rules. "
            "Its lower performance is acceptable when transparency and explanation matter more than maximum predictive strength, making it a useful choice for low-risk educational or illustrative settings.",
            styles["BodySmall"],
        ),
        Spacer(1, 6),
        Paragraph("Deployment risks", styles["SubHeading"]),
        Paragraph(
            bullet_html(
                [
                    "The Titanic dataset is small and historical, so generalization is limited.",
                    "Sex and passenger class may encode social or historical bias in the predictions.",
                    "Tree thresholds can create brittle near-boundary decisions for similar passengers.",
                    "Missing or low-quality operational data would reduce reliability and confidence.",
                ]
            ),
            styles["CompactList"],
        ),
        Spacer(1, 6),
        Paragraph("Ethical considerations", styles["SubHeading"]),
        Paragraph(
            "Even in an educational setting, survival prediction can reflect historical social structure rather than purely individual characteristics. "
            "That means apparently strong performance does not remove the need to question fairness, representativeness, and the meaning of the target itself.",
            styles["BodySmall"],
        ),
        Spacer(1, 6),
        Paragraph("Limitations", styles["SubHeading"]),
        Paragraph(
            bullet_html(
                [
                    "Results come from a single train-test split rather than repeated resampling.",
                    "The dataset is too small for strong real-world claims or stable deployment promises.",
                    "Historical survival patterns may not transfer to modern or unrelated settings.",
                    "The chosen feature set deliberately favored clarity over aggressive feature engineering.",
                ]
            ),
            styles["CompactList"],
        ),
        Spacer(1, 6),
        Paragraph("Recommendation", styles["SubHeading"]),
        Paragraph(
            "If predictive performance is the priority, the baseline Decision Tree is the best choice from this assignment. "
            "If explainability is more important than marginal performance gains, the Rule-Based Tree is the better deployment candidate. "
            "Logistic Regression remains a useful fallback when a simpler statistical baseline is preferred for comparison or classroom discussion.",
            styles["BodySmall"],
        ),
        Spacer(1, 6),
        Paragraph("Future improvement opportunities", styles["SubHeading"]),
        Paragraph(
            "The analysis could be strengthened through repeated cross-validation, probability calibration, more careful threshold selection, and modest feature engineering such as family-size or title-based features. "
            "Those extensions might improve robustness, but they were intentionally left out here to keep the workflow transparent and aligned with the assignment scope.",
            styles["BodySmall"],
        ),
        Spacer(1, 6),
        Paragraph("Conclusion", styles["SubHeading"]),
        Paragraph(
            "This assignment demonstrated that careful preprocessing, disciplined model comparison, and explicit interpretation can produce a strong supervised learning report. "
            "For the Titanic dataset, a relatively simple Decision Tree delivered the best predictive result, while the Rule-Based Tree offered the clearest explanation. "
            "The overall comparison therefore highlights a practical tradeoff between accuracy and transparency rather than a single universally best model. "
            "That balance is one of the most important lessons from the full workflow.",
            styles["BodySmall"],
        ),
    ]


def build_pdf(facts: dict[str, object], comparison_df: pd.DataFrame) -> None:
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

    story = []

    logo = report_image(LOGO_PATH, max_width=4.2 * inch, max_height=2.35 * inch)
    cover_bottom = build_panel_table(
        [[
            panel_paragraph(
                "Assignment objective",
                "Build and compare a complete supervised learning pipeline for Titanic survival prediction, covering preprocessing, baseline models, decision trees, rule extraction, kNN, and ensembles.",
                styles,
            ),
            panel_paragraph(
                "Report basis",
                "Dataset used: Titanic Dataset.<br/>This PDF summarizes validated results from the executed notebook and curated report visuals.",
                styles,
            ),
        ]],
        [3.12 * inch, 3.18 * inch],
        font_size=7.8,
    )
    story.extend(
        [
            Spacer(1, 0.5 * inch),
            logo,
            Spacer(1, 0.35 * inch),
            Paragraph(facts["institution"], styles["ReportTitle"]),
            Paragraph(facts["title"], styles["ReportTitle"]),
            Spacer(1, 0.25 * inch),
            Paragraph("<b>Subject:</b> Machine Learning", styles["CoverMeta"]),
            Paragraph("<b>Trimester:</b> Trimester 2", styles["CoverMeta"]),
            Paragraph("<b>Student:</b> Anik Das", styles["CoverMeta"]),
            Paragraph("<b>Roll Number:</b> 2025em1100026", styles["CoverMeta"]),
            Spacer(1, 0.35 * inch),
            cover_bottom,
            PageBreak(),
        ]
    )

    story.extend(page_two_story(styles, facts))
    story.extend(page_three_story(styles, facts))
    story.extend(page_four_story(styles, facts))
    story.extend(page_five_story(styles, comparison_df))
    story.extend(page_six_story(styles))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def count_pdf_pages(pdf_path: Path) -> int:
    pdf_bytes = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", pdf_bytes))


def main() -> None:
    ensure_directories()
    dataset_summary = load_dataset_summary()
    curated_paths = curate_figures()
    comparison_df = curate_tables()
    write_report_facts(dataset_summary, curated_paths)

    facts = json.loads((REPORT_DATA_DIR / "report_facts.json").read_text(encoding="utf-8"))
    build_pdf(facts, comparison_df)

    page_count = count_pdf_pages(REPORT_PATH)
    if page_count != 6:
        raise RuntimeError(f"Expected a 6-page report, but generated {page_count} pages.")

    print(f"Curated assets written to: {REPORT_ASSETS_DIR}")
    print(f"Report created at: {REPORT_PATH}")
    print(f"Verified page count: {page_count}")


if __name__ == "__main__":
    main()
