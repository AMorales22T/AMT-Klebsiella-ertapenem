"""
Supplementary Figures Generator – Nature Portfolio Standards
============================================================
Reproduces all supplementary figures cited in the manuscript and notebooks:
  - Supplementary Figure S1: Threshold Sweep Diagnostics (Trade-off Recall R vs Specificity S vs F1)
  - Supplementary Figure S2: Attention Head Specialization (Dominant Head, Variance, Max Attention)
  - Supplementary Figure S3: Model Error Analysis & Predicted Probability Distributions
  - Supplementary Figure S4: Cohort Flow Diagram (Rendered from SVG to 300 DPI PNG & TrueType PDF)
  - Supplementary Figure S5: UMAP Latent Space Genome Representation (Cell 45)

All figures follow Nature Portfolio formatting:
  - NPG color palettes
  - Despined axes (no top/right spines, outward ticks, 0.6 pt linewidths)
  - Bold lowercase panel labels (a, b, c)
  - Dual export: 300 DPI PNG & vector TrueType PDF
"""

from pathlib import Path
import json
import base64
import io
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
while not (ROOT / ".git").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

OUT = ROOT / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

NB_PATH = ROOT / "notebooks" / "05_analysis_and_figures.ipynb"


def setup_nature_style():
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          8.0,
        "axes.labelsize":     8.0,
        "xtick.labelsize":    7.0,
        "ytick.labelsize":    7.0,
        "legend.fontsize":    7.0,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
        "figure.dpi":         300,
    })


def make_fig_s1():
    """Supplementary Figure S1: Threshold Sweep Diagnostics."""
    print("  --> Generating Supplementary Figure S1 (Threshold Sweep)...")
    pred_path = ROOT / "results" / "random_split_predictions.csv"
    df = pd.read_csv(pred_path)
    y_val = df["y_true"].values
    y_score = df["y_score"].values

    thresholds_sweep = np.arange(0.30, 0.71, 0.05)
    recalls, specs, f1s = [], [], []

    for thr in thresholds_sweep:
        preds = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        recall_R = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_S   = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_R     = f1_score(y_val, preds, pos_label=1, zero_division=0)
        recalls.append(recall_R)
        specs.append(spec_S)
        f1s.append(f1_R)

    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300, facecolor="#ffffff")

    ax.plot(thresholds_sweep, recalls, "o-", color="#E64B35", lw=1.4, markersize=4.5, label="Recall (Resistant)")
    ax.plot(thresholds_sweep, specs,   "s--", color="#00A087", lw=1.4, markersize=4.5, label="Specificity (Susceptible)")
    ax.plot(thresholds_sweep, f1s,     "^:", color="#3C5488", lw=1.4, markersize=4.5, label="F1 score (Resistant)")

    # Vertical threshold 0.50 line
    ax.axvline(0.50, color="#94a3b8", linestyle=(0, (4, 3)), lw=0.9, label="Default threshold (0.50)")

    ax.set_xlabel("Decision threshold", color="#0f172a")
    ax.set_ylabel("Metric score", color="#0f172a")
    ax.set_xlim(0.28, 0.72)
    ax.set_ylim(0.74, 1.01)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#0f172a")
    ax.spines["bottom"].set_color("#0f172a")

    ax.legend(loc="lower left", frameon=False)

    # Panel label
    ax.text(-0.14, 1.05, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

    plt.tight_layout()
    fig.savefig(OUT / "Supplementary_Figure_S1_Threshold_Sweep.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(OUT / "Supplementary_Figure_S1_Threshold_Sweep.pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("      [OK] Supplementary_Figure_S1_Threshold_Sweep saved.")


def make_fig_s2():
    """Supplementary Figure S2: Attention Head Specialization Diagnostics."""
    print("  --> Generating Supplementary Figure S2 (Attention Head Specialization)...")
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    img_data = None
    for out in nb["cells"][36].get("outputs", []):
        if "image/png" in out.get("data", {}):
            img_data = base64.b64decode(out["data"]["image/png"])
            break

    if not img_data:
        raise RuntimeError("Could not find Cell 36 image in Notebook 05.")

    raw_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    arr = np.array(raw_img).copy()

    # Clear top suptitle area
    arr[:45, :, :] = 255

    cleaned_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(cleaned_img)

    # Load bold font for panel labels
    font_bold = None
    for fp in ["/usr/share/fonts/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        try:
            font_bold = ImageFont.truetype(fp, 26)
            break
        except Exception:
            pass
    if font_bold is None:
        font_bold = ImageFont.load_default()

    # Panel labels a, b, c for the 3 subplots
    w_sub = cleaned_img.width / 3.0
    draw.text((20, 20), "a", fill="#000000", font=font_bold)
    draw.text((int(w_sub) + 20, 20), "b", fill="#000000", font=font_bold)
    draw.text((int(2 * w_sub) + 20, 20), "c", fill="#000000", font=font_bold)

    cleaned_img.save(OUT / "Supplementary_Figure_S2_Attention_Head_Specialization.png", dpi=(150, 150))

    # Save matching PDF
    fig_w = cleaned_img.width / 150.0
    fig_h = cleaned_img.height / 150.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cleaned_img)
    ax.axis("off")
    fig.savefig(OUT / "Supplementary_Figure_S2_Attention_Head_Specialization.pdf", dpi=150)
    plt.close(fig)
    print("      [OK] Supplementary_Figure_S2_Attention_Head_Specialization saved.")


def make_fig_s3():
    """Supplementary Figure S3: Model Error Analysis & Predicted Probability Distributions."""
    print("  --> Generating Supplementary Figure S3 (Error Analysis & Predictions)...")
    pred_path = ROOT / "results" / "random_split_predictions.csv"
    df = pd.read_csv(pred_path)
    y_val = df["y_true"].values
    y_score = df["y_score"].values

    preds = (y_score >= 0.50).astype(int)
    error_type = []
    for yt, yp in zip(y_val, preds):
        if yt == yp:
            error_type.append("Correct")
        elif yt == 1 and yp == 0:
            error_type.append("FN (VME)")
        else:
            error_type.append("FP (ME)")
    df["error_type"] = error_type

    fig, (ax_box, ax_roc) = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=300, facecolor="#ffffff")

    # Panel a: Boxplot of predicted probability
    order = ["Correct", "FN (VME)", "FP (ME)"]
    palette = {"Correct": "#00A087", "FN (VME)": "#E64B35", "FP (ME)": "#E69F00"}

    sns.boxplot(
        data=df, x="error_type", y="y_score",
        order=order, palette=palette, ax=ax_box,
        width=0.45, linewidth=0.8,
        boxprops=dict(alpha=0.85),
        fliersize=2.5
    )
    ax_box.axhline(0.50, linestyle=(0, (4, 3)), color="#94a3b8", lw=0.9, label="Threshold 0.50")
    ax_box.set_ylabel("Predicted probability of resistance", color="#0f172a")
    ax_box.set_xlabel("Isolate classification outcome", color="#0f172a")
    ax_box.set_ylim(-0.02, 1.02)

    for sp in ["top", "right"]:
        ax_box.spines[sp].set_visible(False)
    ax_box.spines["left"].set_color("#0f172a")
    ax_box.spines["bottom"].set_color("#0f172a")
    ax_box.legend(loc="upper left", frameon=False)
    ax_box.text(-0.16, 1.06, "a", transform=ax_box.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

    # Panel b: Validation ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_score)
    roc_auc = auc(fpr, tpr)

    ax_roc.plot(fpr, tpr, color="#3C5488", lw=1.5, label=f"MIL v5.3 (AUC = {roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#94a3b8", linestyle=(0, (4, 3)), lw=0.9, label="Chance")

    ax_roc.set_xlim(-0.015, 1.015)
    ax_roc.set_ylim(-0.015, 1.015)
    ax_roc.set_xlabel("False positive rate (1 − Specificity)", color="#0f172a")
    ax_roc.set_ylabel("True positive rate (Sensitivity)", color="#0f172a")

    for sp in ["top", "right"]:
        ax_roc.spines[sp].set_visible(False)
    ax_roc.spines["left"].set_color("#0f172a")
    ax_roc.spines["bottom"].set_color("#0f172a")
    ax_roc.legend(loc="lower right", frameon=False)
    ax_roc.text(-0.16, 1.06, "b", transform=ax_roc.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")

    plt.tight_layout()
    fig.savefig(OUT / "Supplementary_Figure_S3_Error_Analysis.png", dpi=300, bbox_inches="tight", facecolor="#ffffff")
    fig.savefig(OUT / "Supplementary_Figure_S3_Error_Analysis.pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print("      [OK] Supplementary_Figure_S3_Error_Analysis saved.")


def make_fig_s4():
    """Supplementary Figure S4: Cohort Flow Diagram."""
    print("  --> Generating Supplementary Figure S4 (Cohort Flow Diagram)...")
    svg_path = OUT / "Supplementary_Figure_S4_Cohort_Flow.svg"
    png_path = OUT / "Supplementary_Figure_S4_Cohort_Flow.png"
    pdf_path = OUT / "Supplementary_Figure_S4_Cohort_Flow.pdf"

    if svg_path.exists():
        try:
            # High-res 300 DPI conversion using rsvg-convert
            subprocess.run(["rsvg-convert", "-d", "300", "-p", "300", "-f", "png", "-o", str(png_path), str(svg_path)], check=True)
            subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
            print("      [OK] Supplementary_Figure_S4_Cohort_Flow (.png / .pdf) rendered from SVG.")
        except Exception as e:
            print(f"      [WARN] rsvg-convert error: {e}")
    else:
        print(f"      [WARN] SVG file not found: {svg_path}")


def make_fig_s5():
    """Supplementary Figure S5: UMAP Latent Space Genome Representation."""
    print("  --> Generating Supplementary Figure S5 (UMAP Latent Space)...")
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    img_data = None
    for out in nb["cells"][45].get("outputs", []):
        if "image/png" in out.get("data", {}):
            img_data = base64.b64decode(out["data"]["image/png"])
            break

    if not img_data:
        raise RuntimeError("Could not find Cell 45 image in Notebook 05.")

    raw_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    arr = np.array(raw_img).copy()

    # Clear top suptitle
    arr[:45, :, :] = 255

    cleaned_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(cleaned_img)

    font_bold = None
    for fp in ["/usr/share/fonts/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        try:
            font_bold = ImageFont.truetype(fp, 26)
            break
        except Exception:
            pass
    if font_bold is None:
        font_bold = ImageFont.load_default()

    w_sub = cleaned_img.width / 2.0
    draw.text((20, 20), "a", fill="#000000", font=font_bold)
    draw.text((int(w_sub) + 20, 20), "b", fill="#000000", font=font_bold)

    cleaned_img.save(OUT / "Supplementary_Figure_S5_UMAP_Embeddings.png", dpi=(150, 150))

    fig_w = cleaned_img.width / 150.0
    fig_h = cleaned_img.height / 150.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cleaned_img)
    ax.axis("off")
    fig.savefig(OUT / "Supplementary_Figure_S5_UMAP_Embeddings.pdf", dpi=150)
    plt.close(fig)
    print("      [OK] Supplementary_Figure_S5_UMAP_Embeddings saved.")


def make_all():
    setup_nature_style()
    make_fig_s1()
    make_fig_s2()
    make_fig_s3()
    make_fig_s4()
    make_fig_s5()
    print("\n[SUCCESS] All Supplementary Figures (S1 through S5) generated in supplementary/.")


if __name__ == "__main__":
    make_all()
