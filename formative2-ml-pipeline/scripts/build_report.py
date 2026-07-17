"""Create the final PDF report and supporting figures."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.signal import spectrogram
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_pipeline import load_audio
from src.tabular_pipeline import (
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET,
    make_product_pipeline,
)

FIGURES = ROOT / "reports" / "figures"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT = OUTPUT_DIR / "formative2_multimodal_preprocessing_report.pdf"
FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F95")
TEAL = colors.HexColor("#4A9D9C")
LIGHT = colors.HexColor("#EAF1F5")
PALE = colors.HexColor("#F6F8FA")
DARK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#52606D")

sns.set_theme(style="whitegrid")


def save_figures() -> dict[str, Path]:
    transactions = pd.read_csv("data/raw/customer_transactions.csv")
    merged = pd.read_csv("data/processed/merged_dataset.csv")

    eda_path = FIGURES / "tabular_eda.png"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    order = merged[TARGET].value_counts().index
    sns.countplot(data=merged, x=TARGET, order=order, color="#2E6F95", ax=axes[0])
    axes[0].set_title("Merged Product Categories")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Transactions")
    axes[0].tick_params(axis="x", rotation=28)
    sns.histplot(merged["purchase_amount"], bins=12, kde=True, color="#E07A5F", ax=axes[1])
    axes[1].set_title("Purchase Amount Distribution")
    axes[1].set_xlabel("Purchase amount")
    axes[1].set_ylabel("Rows")
    sns.boxplot(
        data=merged,
        x=TARGET,
        y="purchase_amount",
        order=order,
        color="#81B29A",
        ax=axes[2],
    )
    axes[2].set_title("Outliers by Product Category")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Purchase amount")
    axes[2].tick_params(axis="x", rotation=28)
    fig.tight_layout()
    fig.savefig(eda_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    corr_path = FIGURES / "correlation_heatmap.png"
    social = pd.read_csv("data/raw/customer_social_profiles.csv")
    numeric = pd.concat(
        [
            social[["engagement_score", "purchase_interest_score"]].reset_index(drop=True),
            transactions[["purchase_amount", "customer_rating"]].reset_index(drop=True),
        ],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title("Numeric Feature Correlations")
    fig.tight_layout()
    fig.savefig(corr_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    image_path = FIGURES / "christian_image_samples.png"
    photos = [
        ("Neutral", Path("data/images/christian_tonny_gentil_iradukunda/neutral.jpeg")),
        ("Smiling", Path("data/images/christian_tonny_gentil_iradukunda/smiling.jpeg")),
        ("Surprised", Path("data/images/christian_tonny_gentil_iradukunda/surprised.jpeg")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5))
    for ax, (label, path) in zip(axes, photos):
        with PILImage.open(path) as photo:
            ax.imshow(ImageOps.exif_transpose(photo))
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Christian: Required Facial Expressions", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(image_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    audio_path = FIGURES / "audio_waveforms_spectrograms.png"
    audio_samples = [
        (
            "Christian",
            Path("data/audio/christian_tonny_gentil_iradukunda/yes_approve.m4a"),
        ),
        ("Mahlet", Path("data/audio/mahlet_tilahun/voice_010858.m4a")),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for row, (member, path) in enumerate(audio_samples):
        audio, sample_rate = load_audio(path)
        times = np.arange(len(audio)) / sample_rate
        frequencies, spec_times, power = spectrogram(audio, fs=sample_rate)
        axes[row, 0].plot(times, audio, color="#2E6F95", linewidth=0.6)
        axes[row, 0].set_title(f"{member} Waveform")
        axes[row, 0].set_xlabel("Time (seconds)")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 1].pcolormesh(
            spec_times,
            frequencies,
            10 * np.log10(power + 1e-10),
            shading="auto",
            cmap="magma",
        )
        axes[row, 1].set_ylim(0, 8000)
        axes[row, 1].set_title(f"{member} Spectrogram")
        axes[row, 1].set_xlabel("Time (seconds)")
        axes[row, 1].set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(audio_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    confusion_path = FIGURES / "product_confusion_matrix.png"
    x = merged[MODEL_FEATURES]
    y = merged[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    model = make_product_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    fig, ax = plt.subplots(figsize=(6.7, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
        xticks_rotation=25,
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Product Recommendation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "eda": eda_path,
        "correlation": corr_path,
        "images": image_path,
        "audio": audio_path,
        "confusion": confusion_path,
    }


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def build_pdf(figures: dict[str, Path]) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            alignment=TA_LEFT,
            textColor=NAVY,
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=MUTED,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=BLUE,
            spaceBefore=5,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body2",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.6,
            leading=14,
            textColor=DARK,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallWhite",
            parent=styles["Small"],
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=6,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.6 * cm,
        title="Formative 2: Multimodal Data Preprocessing",
        author="Christian Tonny Gentil Iradukunda and Group 19",
    )

    def page_frame(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 0.55 * cm, width, 0.55 * cm, stroke=0, fill=1)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.7 * cm, 0.75 * cm, "Group 19 - Multimodal Data Preprocessing")
        canvas.drawRightString(
            width - 1.7 * cm, 0.75 * cm, f"Page {document.page}"
        )
        canvas.restoreState()

    body = styles["Body2"]
    section = styles["Section"]
    subsection = styles["Subsection"]
    story = []

    story.extend(
        [
            Spacer(1, 2.0 * cm),
            Paragraph("FORMATIVE 2", styles["CoverSub"]),
            Paragraph(
                "Multimodal Data<br/>Preprocessing",
                styles["CoverTitle"],
            ),
            Paragraph(
                "User Identity and Product Recommendation System",
                styles["CoverSub"],
            ),
            Spacer(1, 0.45 * cm),
            Table(
                [
                    ["Group", "19"],
                    ["Prepared by", "Christian Tonny Gentil Iradukunda and Group 19"],
                    ["Models", "Face recognition, voice verification, product recommendation"],
                    ["Implementation", "Python, Jupyter Notebook, command-line application"],
                ],
                colWidths=[3.2 * cm, 12.4 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.white),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 0.7 * cm),
            Table(
                [
                    [
                        Paragraph(
                            "<b>System result:</b> The full approved path works. "
                            "Christian's face and voice were accepted and the model "
                            "predicted Clothing. Both face rejection and voice mismatch "
                            "denial paths were also tested.",
                            styles["Callout"],
                        )
                    ]
                ],
                colWidths=[15.6 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                        ("BOX", (0, 0), (-1, -1), 1, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("1. Approach and System Logic", section),
            Paragraph(
                "The system uses sequential security checks. A face image is processed "
                "first. If the prediction confidence is below 85%, access is denied. "
                "When the face passes, the product model runs but its answer is held. "
                "The voice sample must then match the face identity before the product "
                "is displayed.",
                body,
            ),
            Paragraph("Processing sequence", subsection),
            Table(
                [
                    [
                        Paragraph("<b>1</b><br/>Face input", styles["Small"]),
                        Paragraph("<b>2</b><br/>Face check", styles["Small"]),
                        Paragraph("<b>3</b><br/>Product model", styles["Small"]),
                        Paragraph("<b>4</b><br/>Voice check", styles["Small"]),
                        Paragraph("<b>5</b><br/>Display / deny", styles["Small"]),
                    ]
                ],
                colWidths=[3.1 * cm] * 5,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
                        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.white),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                    ]
                ),
            ),
            Spacer(1, 0.3 * cm),
            Paragraph("Main design choices", subsection),
            Paragraph(
                "<b>Merge:</b> social observations were aggregated before a validated "
                "many-to-one inner join. This avoided transaction duplication.",
                body,
            ),
            Paragraph(
                "<b>Leakage control:</b> customer rating was excluded from product "
                "features because it is collected after purchase. Augmented copies from "
                "one media source stayed in the same model split.",
                body,
            ),
            Paragraph(
                "<b>Models:</b> Random Forest was used because it supports nonlinear "
                "patterns, mixed engineered features, multiclass prediction, and "
                "probability-based confidence checks.",
                body,
            ),
            Paragraph("2. Data Cleaning and Merge Validation", section),
            Table(
                [
                    ["Check", "Observed result"],
                    ["Social rows", "155 original; 5 exact duplicates removed"],
                    ["Transaction rows", "150 original; 10 ratings imputed with median 3.0"],
                    ["Social customers", "84 unique customer IDs"],
                    ["Validated merge", "117 rows; many-to-one; 0 duplicate transaction IDs"],
                    ["Unmatched transactions", "33 rows without a social profile"],
                ],
                colWidths=[5.1 * cm, 10.5 * cm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("3. Exploratory Data Analysis", section),
            Paragraph(
                "The merged target was fairly balanced: Sports 28, Electronics 27, "
                "Clothing 22, Groceries 20, and Books 20. Purchase amount ranged from "
                "62 to 495, with a mean of 287.91. Categories overlap strongly in the "
                "boxplot and numeric correlations are weak.",
                body,
            ),
            scaled_image(figures["eda"], 16.0 * cm, 7.0 * cm),
            Spacer(1, 0.25 * cm),
            Table(
                [
                    [
                        scaled_image(figures["correlation"], 7.3 * cm, 6.1 * cm),
                        Paragraph(
                            "<b>Interpretation</b><br/><br/>No numeric feature has a "
                            "strong relationship with another feature. The product "
                            "classes also share similar purchase amount ranges. This "
                            "supports the later finding that the product model has weak "
                            "predictive performance.<br/><br/>Date features were still "
                            "added because shopping behavior may vary by month, weekday, "
                            "and quarter.",
                            body,
                        ),
                    ]
                ],
                colWidths=[7.8 * cm, 7.8 * cm],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("4. Image Collection and Processing", section),
            Paragraph(
                "Each original image produced four rows: original, 15-degree rotation, "
                "horizontal flip, and grayscale. Features include RGB and grayscale "
                "histograms, brightness statistics, and a 16 by 16 pixel embedding. "
                "The final CSV contains 16 rows, 322 numeric features, and four metadata "
                "columns.",
                body,
            ),
            scaled_image(figures["images"], 15.8 * cm, 8.7 * cm),
            Spacer(1, 0.25 * cm),
            Table(
                [
                    ["Member", "Original images", "Feature rows"],
                    ["Christian Tonny Gentil Iradukunda", "3", "12"],
                    ["Yvette Uwimpaye", "1", "4"],
                ],
                colWidths=[9.6 * cm, 3.0 * cm, 3.0 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            Spacer(1, 0.25 * cm),
            Paragraph(
                "<b>Coverage note:</b> the image model learned two identities, but only "
                "Christian had enough original files for a held-out identity evaluation. "
                "The image score is therefore incomplete even though the pipeline works.",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("5. Audio Collection and Processing", section),
            Paragraph(
                "Audio was decoded to mono at 16 kHz. Every original produced pitch "
                "shift, time stretch, and background noise variants. The feature file "
                "contains 13 MFCC means, 13 MFCC standard deviations, spectral roll-off, "
                "RMS energy, zero-crossing rate, and duration.",
                body,
            ),
            scaled_image(figures["audio"], 15.8 * cm, 9.3 * cm),
            Spacer(1, 0.25 * cm),
            Table(
                [
                    ["Member", "Original audios", "Feature rows"],
                    ["Christian Tonny Gentil Iradukunda", "2", "8"],
                    ["Mahlet Tilahun", "2", "8"],
                ],
                colWidths=[9.6 * cm, 3.0 * cm, 3.0 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("6. Model Evaluation", section),
            Table(
                [
                    ["Model", "Accuracy", "Macro F1", "Log loss"],
                    ["Product recommendation", "0.167", "0.177", "1.746"],
                    ["Facial recognition", "1.000", "1.000", "0.088"],
                    ["Voiceprint verification", "1.000", "1.000", "0.096"],
                ],
                colWidths=[7.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 0.3 * cm),
            Table(
                [
                    [
                        scaled_image(figures["confusion"], 7.7 * cm, 6.8 * cm),
                        Paragraph(
                            "<b>Product model</b><br/><br/>The model was tested on 30 "
                            "held-out transactions. Its 0.167 accuracy is weak and the "
                            "confusion matrix shows errors across all five categories. "
                            "This matches the EDA, where features had weak relationships "
                            "and category distributions overlapped.<br/><br/>"
                            "<b>Authentication models</b><br/><br/>The high scores are "
                            "based on very small datasets. They show that the code and "
                            "decision flow work, but they do not prove general performance.",
                            body,
                        ),
                    ]
                ],
                colWidths=[8.0 * cm, 7.6 * cm],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            ),
            Paragraph("7. System Demonstration", section),
            Table(
                [
                    ["Scenario", "Face result", "Voice result", "Final decision"],
                    ["Approved", "Christian 89.7%", "Christian 99.3%", "Clothing displayed"],
                    ["Unauthorized image", "Below 85% threshold", "Not checked", "Access Denied"],
                    ["Identity mismatch", "Yvette 98.7%", "Mahlet 97.3%", "Access Denied"],
                ],
                colWidths=[4.1 * cm, 3.7 * cm, 3.7 * cm, 4.1 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            ),
            PageBreak(),
        ]
    )

    contribution_rows = [
        [
            Paragraph("Member", styles["SmallWhite"]),
            Paragraph("Contribution", styles["SmallWhite"]),
        ],
        [
            Paragraph("Christian Tonny Gentil Iradukunda", styles["Small"]),
            Paragraph(
                "Three facial expressions, two voice phrases, data cleaning, merge "
                "logic, feature engineering, model integration, testing, CLI, notebook, "
                "and report.",
                styles["Small"],
            ),
        ],
        [
            Paragraph("Hassan", styles["Small"]),
            Paragraph(
                "Created and shared the group data folders and coordinated member uploads.",
                styles["Small"],
            ),
        ],
        [
            Paragraph("Mahlet Tilahun", styles["Small"]),
            Paragraph(
                "Submitted two voice recordings used in audio preprocessing and voice "
                "model training.",
                styles["Small"],
            ),
        ],
        [
            Paragraph("Yvette Uwimpaye", styles["Small"]),
            Paragraph(
                "Submitted one image used in image preprocessing and face model training.",
                styles["Small"],
            ),
        ],
    ]
    story.extend(
        [
            Paragraph("8. Contributions", section),
            Table(
                contribution_rows,
                colWidths=[5.0 * cm, 10.6 * cm],
                repeatRows=1,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("9. Limitations and Conclusion", section),
            Paragraph(
                "The system demonstrates the complete multimodal sequence, produces "
                "all three required feature datasets, evaluates three models, and "
                "handles both approval and denial paths. The merge is validated and "
                "the notebook is reproducible from top to bottom.",
                body,
            ),
            Paragraph(
                "The main limitation is media quantity. At processing time, Hassan had "
                "not uploaded media, Mahlet had no images, and Yvette had one image and "
                "no audio. More original samples per member are needed for stronger and "
                "fully representative authentication metrics. The product dataset also "
                "has weak predictive signals, which explains its low test score.",
                body,
            ),
            Paragraph("10. Submission Links", section),
            Table(
                [
                    ["GitHub repository", "https://github.com/irachrist1/alu-machine_learning"],
                    ["Demonstration video", "Add the uploaded video URL before submission."],
                ],
                colWidths=[4.3 * cm, 11.3 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 0.35 * cm),
            Paragraph("Data Sources", section),
            Paragraph(
                'Customer social profiles: <link href="https://docs.google.com/spreadsheets/d/10up-WdC0a6egYaXLKiMQUotpOvaKZRZvYuWFQx4-RPQ/">Google Sheet</link>',
                body,
            ),
            Paragraph(
                'Customer transactions: <link href="https://docs.google.com/spreadsheets/d/1s4WOVm49lmLQ8d9QbbbdgAcRTNh3m0KCH_5ciiaRZw0/">Google Sheet</link>',
                body,
            ),
        ]
    )

    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    print(OUTPUT)


if __name__ == "__main__":
    figures = save_figures()
    build_pdf(figures)
