#!/usr/bin/env python3
"""
draw_architecture_diagrams.py
=============================
Generates three publication-quality architecture diagrams for the
Two-Stage Dynamic Hyperthyroid Surveillance Framework paper.

Standalone script -- only uses matplotlib/numpy, no project imports.

Outputs (300 DPI):
  Figure_00_Overall_Framework.png
  Figure_00_Stage1_Design.png
  Figure_00_Stage2_Design.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── colour palette ────────────────────────────────────────────────
C_BLUE   = "#457b9d"   # Stage 1
C_BLUE_L = "#a8d5e2"   # Stage 1 light
C_GREEN  = "#2a9d8f"   # Stage 2
C_GREEN_L= "#b5e2db"   # Stage 2 light
C_ORANGE = "#e76f51"   # Outputs / endpoints
C_ORANGE_L="#f4c4b3"   # Outputs light
C_GRAY   = "#6c757d"   # Annotations
C_GRAY_L = "#e9ecef"   # Backgrounds / dividers
C_WHITE  = "#ffffff"
C_BLACK  = "#264653"   # Dark text
C_YELLOW = "#e9c46a"   # Accent

_default_out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "results", "stage2_multihorizon_relapse_v4", "figures"
)
# Allow --out override for integration with main pipeline
import sys as _sys
OUT_DIR = _default_out
for _i, _a in enumerate(_sys.argv[:-1]):
    if _a == "--out":
        OUT_DIR = _sys.argv[_i + 1]
        break
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 300
FONT_TITLE = 14
FONT_SECTION = 12
FONT_BODY = 10
FONT_SMALL = 8.5
FONT_TINY = 7.5

# ── helpers ───────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, fc, ec=None, fs=FONT_BODY,
         text_color=C_BLACK, bold=False, alpha=1.0, lw=1.2,
         ha="center", va="center", zorder=3, radius=0.02,
         multi_line=False):
    """Draw a rounded rectangle with centred text."""
    if ec is None:
        ec = fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
        zorder=zorder, transform=ax.transData
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if multi_line:
        ax.text(x + w/2, y + h/2, text, fontsize=fs, color=text_color,
                ha=ha, va=va, fontweight=weight, zorder=zorder+1,
                linespacing=1.4, transform=ax.transData)
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=fs, color=text_color,
                ha=ha, va=va, fontweight=weight, zorder=zorder+1,
                transform=ax.transData)
    return box


def _arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.5, style="-|>",
           connectionstyle="arc3,rad=0", zorder=2, shrinkA=0, shrinkB=0):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=zorder,
        shrinkA=shrinkA, shrinkB=shrinkB,
        mutation_scale=14
    )
    ax.add_patch(arrow)
    return arrow


def _section_bg(ax, x, y, w, h, color, alpha=0.10, zorder=0):
    """Draw a subtle section background."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    return box


def _section_label(ax, x, y, text, color, fs=FONT_SECTION):
    ax.text(x, y, text, fontsize=fs, color=color, fontweight="bold",
            ha="left", va="top", zorder=5)


def _setup_ax(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def _note_text(ax, x, y, text, fs=FONT_TINY, color=C_GRAY):
    ax.text(x, y, text, fontsize=fs, color=color, ha="left", va="top",
            style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 1  --  Overall Framework
# ══════════════════════════════════════════════════════════════════

def draw_overall_framework():
    fig = plt.figure(figsize=(14, 8))
    ax = _setup_ax(fig)

    # Title
    ax.text(0.50, 0.96, "Two-Stage Dynamic Hyperthyroid Surveillance Framework",
            fontsize=FONT_TITLE, fontweight="bold", color=C_BLACK,
            ha="center", va="top", zorder=5)
    ax.plot([0.08, 0.92], [0.935, 0.935], color=C_GRAY_L, lw=1.5, zorder=1)

    # ── Section backgrounds ──
    _section_bg(ax, 0.02, 0.38, 0.22, 0.54, C_GRAY)       # Cohort
    _section_bg(ax, 0.26, 0.38, 0.22, 0.54, C_BLUE)        # Stage 1
    _section_bg(ax, 0.50, 0.38, 0.24, 0.54, C_GREEN)       # Stage 2
    _section_bg(ax, 0.76, 0.38, 0.22, 0.54, C_ORANGE)      # Patient-level

    # ── Section labels ──
    _section_label(ax, 0.04, 0.91, "Cohort", C_GRAY)
    _section_label(ax, 0.28, 0.91, "Stage 1", C_BLUE)
    _section_label(ax, 0.52, 0.91, "Stage 2", C_GREEN)
    _section_label(ax, 0.78, 0.91, "Patient-Level Triage", C_ORANGE)

    # ── COHORT PANEL ──
    _box(ax, 0.04, 0.80, 0.18, 0.07,
         "RAI-Treated Cohort\n1003 treatment episodes", C_GRAY_L, C_GRAY,
         fs=FONT_BODY, bold=True, multi_line=True)

    _box(ax, 0.04, 0.70, 0.18, 0.07,
         "Visits: 0M, 1M, 3M,\n6M, 12M, 18M, 24M", C_WHITE, C_GRAY,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.04, 0.58, 0.18, 0.08,
         "Chronological Split\nDevelopment: < 2020\nTemporalTest: >=2020",
         C_WHITE, C_GRAY, fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.04, 0.45, 0.18, 0.08,
         "5-Fold GroupKFold\n(by patient_id)\nOOF for selection",
         C_WHITE, C_GRAY, fs=FONT_SMALL, multi_line=True)

    _arrow(ax, 0.13, 0.80, 0.13, 0.775, C_GRAY)
    _arrow(ax, 0.13, 0.70, 0.13, 0.665, C_GRAY)
    _arrow(ax, 0.13, 0.58, 0.13, 0.535, C_GRAY)

    # ── STAGE 1 PANEL ──
    _box(ax, 0.28, 0.80, 0.18, 0.07,
         "Fixed-Landmark\nPrediction", C_BLUE_L, C_BLUE,
         fs=FONT_BODY, bold=True, multi_line=True)

    _box(ax, 0.28, 0.70, 0.18, 0.06,
         "Baseline + 1M Labs", C_WHITE, C_BLUE,
         fs=FONT_SMALL)

    _box(ax, 0.28, 0.61, 0.18, 0.06,
         "L2 Logistic Regression", C_BLUE_L, C_BLUE,
         fs=FONT_SMALL, bold=True)

    _box(ax, 0.28, 0.50, 0.18, 0.07,
         "3M Hyper/Non-Hyper\n6M Hyper/Non-Hyper", C_WHITE, C_BLUE,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.285, 0.42, 0.17, 0.05,
         "Stage1_3M_Risk\nStage1_6M_Risk", C_YELLOW, C_BLUE,
         fs=FONT_SMALL, bold=True, multi_line=True, text_color=C_BLACK)

    _arrow(ax, 0.37, 0.80, 0.37, 0.77, C_BLUE)
    _arrow(ax, 0.37, 0.70, 0.37, 0.67, C_BLUE)
    _arrow(ax, 0.37, 0.61, 0.37, 0.575, C_BLUE)
    _arrow(ax, 0.37, 0.50, 0.37, 0.475, C_BLUE)

    # Arrow from Stage1 risk to Stage2
    _arrow(ax, 0.455, 0.445, 0.52, 0.445, C_YELLOW, lw=2.5, style="-|>")

    # ── STAGE 2 PANEL ──
    _box(ax, 0.52, 0.80, 0.20, 0.07,
         "Rolling-Landmark\nDynamic Prediction", C_GREEN_L, C_GREEN,
         fs=FONT_BODY, bold=True, multi_line=True)

    _box(ax, 0.52, 0.70, 0.20, 0.06,
         "Static + Labs + Trajectory\n+ Stage1 Risk", C_WHITE, C_GREEN,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.52, 0.61, 0.20, 0.06,
         "L2 Logistic Regression\n+ StandardScaler", C_GREEN_L, C_GREEN,
         fs=FONT_SMALL, bold=True, multi_line=True)

    _box(ax, 0.52, 0.50, 0.20, 0.07,
         "P(H1) per landmark\n(3M, 6M, 12M, 18M)", C_WHITE, C_GREEN,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.52, 0.42, 0.20, 0.05,
         "Interval-Level Risk", C_GREEN_L, C_GREEN,
         fs=FONT_SMALL, bold=True)

    _arrow(ax, 0.62, 0.80, 0.62, 0.77, C_GREEN)
    _arrow(ax, 0.62, 0.70, 0.62, 0.67, C_GREEN)
    _arrow(ax, 0.62, 0.61, 0.62, 0.575, C_GREEN)
    _arrow(ax, 0.62, 0.50, 0.62, 0.475, C_GREEN)

    # Arrow from Stage2 to Patient-level
    _arrow(ax, 0.72, 0.445, 0.78, 0.445, C_GREEN, lw=2.0)

    # ── PATIENT-LEVEL PANEL ──
    _box(ax, 0.78, 0.80, 0.19, 0.07,
         "Patient-Level\nAggregation", C_ORANGE_L, C_ORANGE,
         fs=FONT_BODY, bold=True, multi_line=True)

    _box(ax, 0.78, 0.70, 0.19, 0.06,
         "Mean_H1_Risk\nacross landmarks", C_WHITE, C_ORANGE,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.78, 0.61, 0.19, 0.06,
         "Quartile Stratification\nQ1 < Q2 < Q3 < Q4", C_WHITE, C_ORANGE,
         fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.78, 0.50, 0.19, 0.07,
         "Clinical Triage\nLow / Moderate / High\nRisk", C_ORANGE_L, C_ORANGE,
         fs=FONT_SMALL, bold=True, multi_line=True)

    _arrow(ax, 0.875, 0.80, 0.875, 0.765, C_ORANGE)
    _arrow(ax, 0.875, 0.70, 0.875, 0.675, C_ORANGE)
    _arrow(ax, 0.875, 0.61, 0.875, 0.575, C_ORANGE)

    # ── Horizontal flow arrows between panels ──
    _arrow(ax, 0.22, 0.835, 0.28, 0.835, C_GRAY, lw=2.0, style="-|>")
    _arrow(ax, 0.46, 0.835, 0.52, 0.835, C_GRAY, lw=2.0, style="-|>")
    _arrow(ax, 0.72, 0.835, 0.78, 0.835, C_GRAY, lw=2.0, style="-|>")

    # ── BOTTOM PRINCIPLES BAR ──
    _section_bg(ax, 0.02, 0.04, 0.96, 0.28, C_GRAY, alpha=0.06)
    ax.text(0.50, 0.30, "Key Design Principles",
            fontsize=FONT_SECTION, fontweight="bold", color=C_BLACK,
            ha="center", va="top", zorder=5)
    ax.plot([0.15, 0.85], [0.285, 0.285], color=C_GRAY_L, lw=1, zorder=1)

    principles = [
        ("Time-Safe", "No future information\nleakage; features frozen\nat each landmark",
         C_BLUE, 0.14),
        ("OOF-Only Selection", "Model & threshold chosen\non out-of-fold predictions;\ntest never seen",
         C_GREEN, 0.37),
        ("Cross-Stage\nInheritance", "Stage1 OOF risk reused\nin Stage2 as domain-\naware scalar features",
         C_YELLOW, 0.60),
        ("Clinical Utility", "Patient-level quartile\ntriage; actionable risk\nstratification",
         C_ORANGE, 0.83),
    ]
    for title, desc, color, cx in principles:
        _box(ax, cx - 0.09, 0.16, 0.18, 0.10, title, color,
             fs=FONT_BODY, bold=True, text_color=C_WHITE, alpha=0.85)
        ax.text(cx, 0.14, desc, fontsize=FONT_TINY, color=C_GRAY,
                ha="center", va="top", zorder=5, linespacing=1.3)

    # Save
    path = os.path.join(OUT_DIR, "Figure_00_Overall_Framework.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C_WHITE,
                edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 2  --  Stage 1 Design
# ══════════════════════════════════════════════════════════════════

def draw_stage1_design():
    fig = plt.figure(figsize=(14, 8))
    ax = _setup_ax(fig)

    # Title
    ax.text(0.50, 0.96, "Stage 1: Fixed-Landmark Early Response Prediction",
            fontsize=FONT_TITLE, fontweight="bold", color=C_BLUE,
            ha="center", va="top", zorder=5)
    ax.plot([0.08, 0.92], [0.935, 0.935], color=C_BLUE_L, lw=1.5, zorder=1)

    # ── LEFT: Feature groups ──
    _section_bg(ax, 0.02, 0.35, 0.28, 0.57, C_BLUE, alpha=0.06)
    _section_label(ax, 0.04, 0.91, "Input Features", C_BLUE)
    ax.text(0.04, 0.885, "(available at baseline / 1M)",
            fontsize=FONT_TINY, color=C_GRAY, ha="left", va="top",
            style="italic", zorder=5)

    feature_groups = [
        ("Demographics", "Age, Sex", 0.80),
        ("Thyroid Parameters", "Weight, Dose, Dose/g", 0.72),
        ("Baseline Labs", "FT3, FT4, TSH,\nTRAb, TPOAb, TGAb", 0.62),
        ("1M Follow-up Labs", "FT3, FT4, TSH,\nTRAb at 1 month", 0.52),
        ("Pre-RAI History", "ATD duration, GD\nduration, relapse hx", 0.42),
    ]
    for title, desc, cy in feature_groups:
        _box(ax, 0.04, cy, 0.24, 0.07, "", C_WHITE, C_BLUE, lw=1.0)
        ax.text(0.06, cy + 0.055, title, fontsize=FONT_BODY,
                fontweight="bold", color=C_BLUE, ha="left", va="center", zorder=5)
        ax.text(0.06, cy + 0.025, desc, fontsize=FONT_SMALL,
                color=C_GRAY, ha="left", va="center", zorder=5,
                linespacing=1.2)

    # ── CENTER: Model ──
    _section_bg(ax, 0.33, 0.35, 0.34, 0.57, C_BLUE, alpha=0.04)
    _section_label(ax, 0.38, 0.91, "Model & Validation", C_BLUE)

    # Arrows from features to model
    for cy in [0.835, 0.755, 0.655, 0.555, 0.455]:
        _arrow(ax, 0.28, cy, 0.38, 0.68, C_BLUE_L, lw=1.0)

    # Model box
    _box(ax, 0.38, 0.63, 0.24, 0.10,
         "L2 Logistic Regression\n(StandardScaler + LR)", C_BLUE_L, C_BLUE,
         fs=FONT_SECTION, bold=True, multi_line=True, text_color=C_BLACK)

    # CV details
    _box(ax, 0.38, 0.53, 0.24, 0.08,
         "5-Fold GroupKFold CV\n(grouped by patient_id)\nOOF predictions collected",
         C_WHITE, C_BLUE, fs=FONT_SMALL, multi_line=True)

    # Temporal safety note
    _box(ax, 0.38, 0.42, 0.24, 0.07,
         "Temporal Safety\nFeatures frozen at 1M\nNo data after landmark",
         C_YELLOW, C_BLUE, fs=FONT_SMALL, multi_line=True,
         alpha=0.35, text_color=C_BLACK)

    _arrow(ax, 0.50, 0.63, 0.50, 0.615, C_BLUE)
    _arrow(ax, 0.50, 0.53, 0.50, 0.495, C_BLUE)

    # ── RIGHT: Outputs ──
    _section_bg(ax, 0.70, 0.35, 0.28, 0.57, C_BLUE, alpha=0.06)
    _section_label(ax, 0.72, 0.91, "Outputs", C_BLUE)

    # 3M prediction
    _box(ax, 0.72, 0.78, 0.24, 0.07,
         "3M Prediction\nHyper vs Non-Hyper", C_BLUE_L, C_BLUE,
         fs=FONT_BODY, bold=True, multi_line=True)

    # 6M prediction
    _box(ax, 0.72, 0.68, 0.24, 0.07,
         "6M Prediction\nHyper vs Non-Hyper", C_BLUE_L, C_BLUE,
         fs=FONT_BODY, bold=True, multi_line=True)

    # Arrows from model to outputs
    _arrow(ax, 0.62, 0.71, 0.72, 0.815, C_BLUE, lw=1.5)
    _arrow(ax, 0.62, 0.68, 0.72, 0.715, C_BLUE, lw=1.5)

    # Risk scores
    _box(ax, 0.72, 0.57, 0.24, 0.06,
         "OOF Probability Outputs", C_WHITE, C_BLUE, fs=FONT_BODY)

    _box(ax, 0.74, 0.48, 0.20, 0.06,
         "Stage1_3M_Risk", C_YELLOW, C_BLUE,
         fs=FONT_SECTION, bold=True, text_color=C_BLACK)

    _box(ax, 0.74, 0.40, 0.20, 0.06,
         "Stage1_6M_Risk", C_YELLOW, C_BLUE,
         fs=FONT_SECTION, bold=True, text_color=C_BLACK)

    _arrow(ax, 0.84, 0.68, 0.84, 0.635, C_BLUE)
    _arrow(ax, 0.84, 0.57, 0.84, 0.545, C_BLUE)
    _arrow(ax, 0.84, 0.48, 0.84, 0.465, C_BLUE)

    # Downstream arrow
    _arrow(ax, 0.84, 0.40, 0.84, 0.35, C_YELLOW, lw=2.5, style="-|>")
    ax.text(0.84, 0.33, "Passed to Stage 2\nas scalar features",
            fontsize=FONT_SMALL, color=C_ORANGE, ha="center", va="top",
            fontweight="bold", zorder=5)

    # ── BOTTOM: Data split detail ──
    _section_bg(ax, 0.02, 0.04, 0.96, 0.24, C_GRAY, alpha=0.05)
    ax.text(0.50, 0.27, "Data Handling for Cross-Stage Inheritance",
            fontsize=FONT_SECTION, fontweight="bold", color=C_BLACK,
            ha="center", va="top", zorder=5)
    ax.plot([0.15, 0.85], [0.25, 0.25], color=C_GRAY_L, lw=1, zorder=1)

    # Development column
    _box(ax, 0.08, 0.12, 0.35, 0.11,
         "Development Set (< 2020)\nOOF risk used as Stage2 features\n(no data leakage: each fold\npredicts on its held-out patients)",
         C_BLUE_L, C_BLUE, fs=FONT_SMALL, multi_line=True, alpha=0.6)

    # Test column
    _box(ax, 0.55, 0.12, 0.35, 0.11,
         "Temporal Test Set (>= 2020)\nModel-predicted risk used\n(full Development model retrained,\nthen applied to unseen test patients)",
         C_ORANGE_L, C_ORANGE, fs=FONT_SMALL, multi_line=True, alpha=0.6)

    _arrow(ax, 0.43, 0.175, 0.55, 0.175, C_GRAY, lw=1.5)
    ax.text(0.49, 0.19, "independent", fontsize=FONT_TINY, color=C_GRAY,
            ha="center", va="bottom", style="italic", zorder=5)

    # Save
    path = os.path.join(OUT_DIR, "Figure_00_Stage1_Design.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C_WHITE,
                edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════
# DIAGRAM 3  --  Stage 2 Design
# ══════════════════════════════════════════════════════════════════

def draw_stage2_design():
    fig = plt.figure(figsize=(14, 8))
    ax = _setup_ax(fig)

    # Title
    ax.text(0.50, 0.96,
            "Stage 2: Rolling Landmark Dynamic Hyperthyroid Event Prediction",
            fontsize=FONT_TITLE, fontweight="bold", color=C_GREEN,
            ha="center", va="top", zorder=5)
    ax.plot([0.08, 0.92], [0.935, 0.935], color=C_GREEN_L, lw=1.5, zorder=1)

    # ── TOP: Rolling landmark concept ──
    _section_bg(ax, 0.02, 0.74, 0.96, 0.18, C_GREEN, alpha=0.05)
    _section_label(ax, 0.04, 0.91, "Rolling Landmark Concept", C_GREEN)

    # Timeline
    timeline_y = 0.82
    ax.plot([0.08, 0.92], [timeline_y, timeline_y],
            color=C_GRAY, lw=2, zorder=2)
    landmarks = [
        ("0M\nRAI", 0.10), ("1M", 0.20), ("3M", 0.35),
        ("6M", 0.50), ("12M", 0.65), ("18M", 0.78), ("24M", 0.90)
    ]
    for label, lx in landmarks:
        ax.plot(lx, timeline_y, "o", color=C_GREEN, markersize=8, zorder=4)
        ax.text(lx, timeline_y - 0.025, label, fontsize=FONT_SMALL,
                color=C_BLACK, ha="center", va="top", zorder=5,
                fontweight="bold")

    # Prediction windows (arrows below timeline)
    windows = [
        (0.35, 0.50, "3M: predict\nnext event"),
        (0.50, 0.65, "6M: predict\nnext event"),
        (0.65, 0.78, "12M: predict\nnext event"),
        (0.78, 0.90, "18M: predict\nnext event"),
    ]
    for x1, x2, label in windows:
        mid = (x1 + x2) / 2
        _arrow(ax, x1, timeline_y + 0.015, x2, timeline_y + 0.015,
               C_GREEN, lw=1.5, style="<|-|>")
        ax.text(mid, timeline_y + 0.06, label, fontsize=FONT_TINY,
                color=C_GREEN, ha="center", va="bottom", zorder=5)

    # ── LEFT: Features ──
    feat_top = 0.72
    _section_bg(ax, 0.02, 0.24, 0.30, 0.49, C_GREEN, alpha=0.05)
    _section_label(ax, 0.04, feat_top, "Feature Input (per landmark)", C_GREEN)

    feat_groups = [
        ("Static Baseline", "Demographics, thyroid\nweight/dose, pre-RAI hx", 0.60),
        ("Current Labs", "FT3, FT4, TSH, TRAb\nat current landmark", 0.51),
        ("Trajectory History", "Lab deltas, trends\nfrom prior visits", 0.42),
        ("Stage1 Scalar Risk", "Stage1_3M_Risk\nStage1_6M_Risk", 0.33),
    ]
    for title, desc, cy in feat_groups:
        col = C_YELLOW if "Stage1" in title else C_WHITE
        ecol = C_BLUE if "Stage1" in title else C_GREEN
        _box(ax, 0.04, cy, 0.26, 0.07, "", col, ecol, lw=1.2,
             alpha=0.7 if "Stage1" in title else 1.0)
        ax.text(0.06, cy + 0.055, title, fontsize=FONT_BODY,
                fontweight="bold", color=ecol, ha="left", va="center",
                zorder=5)
        ax.text(0.06, cy + 0.025, desc, fontsize=FONT_SMALL,
                color=C_GRAY, ha="left", va="center", zorder=5,
                linespacing=1.2)

    # Inherited arrow label
    ax.text(0.17, 0.305, "inherited from Stage 1",
            fontsize=FONT_TINY, color=C_BLUE, ha="center", va="top",
            style="italic", fontweight="bold", zorder=5)

    # ── CENTER: Model ──
    _section_bg(ax, 0.35, 0.24, 0.30, 0.49, C_GREEN, alpha=0.04)
    _section_label(ax, 0.40, feat_top, "Model", C_GREEN)

    # Arrows from features to model
    for cy in [0.635, 0.545, 0.455, 0.365]:
        _arrow(ax, 0.30, cy, 0.40, 0.555, C_GREEN_L, lw=1.0)

    _box(ax, 0.38, 0.51, 0.26, 0.09,
         "StandardScaler\n+\nL2 Logistic Regression", C_GREEN_L, C_GREEN,
         fs=FONT_SECTION, bold=True, multi_line=True, text_color=C_BLACK)

    _box(ax, 0.38, 0.40, 0.26, 0.08,
         "5-Fold GroupKFold CV\nOOF predictions for\nselection & thresholding",
         C_WHITE, C_GREEN, fs=FONT_SMALL, multi_line=True)

    _arrow(ax, 0.51, 0.51, 0.51, 0.485, C_GREEN)

    _box(ax, 0.38, 0.28, 0.26, 0.08,
         "SelectionScore\n= 0.6 * H1_PR_AUC\n+ 0.4 * H12_PR_AUC",
         C_WHITE, C_GREEN, fs=FONT_SMALL, multi_line=True)

    _arrow(ax, 0.51, 0.40, 0.51, 0.365, C_GREEN)

    # ── RIGHT: Outputs & Endpoints ──
    _section_bg(ax, 0.68, 0.24, 0.30, 0.49, C_ORANGE, alpha=0.05)
    _section_label(ax, 0.70, feat_top, "Outputs & Endpoints", C_ORANGE)

    # Interval-level output
    _arrow(ax, 0.64, 0.555, 0.72, 0.635, C_GREEN, lw=1.5)

    _box(ax, 0.70, 0.60, 0.26, 0.07,
         "P(H1) at each landmark\n3M, 6M, 12M, 18M", C_GREEN_L, C_GREEN,
         fs=FONT_BODY, bold=True, multi_line=True)

    # Endpoint definitions
    _box(ax, 0.70, 0.50, 0.26, 0.07,
         "H1 (Primary)\nNext-interval hyper event\nwindow = 1 interval ahead",
         C_ORANGE_L, C_ORANGE, fs=FONT_SMALL, multi_line=True)

    _box(ax, 0.70, 0.41, 0.26, 0.06,
         "H6 / H12 (Sensitivity)\n6- and 12-month hyper occurrence",
         C_ORANGE_L, C_ORANGE, fs=FONT_SMALL, multi_line=True, alpha=0.7)

    _arrow(ax, 0.83, 0.60, 0.83, 0.575, C_ORANGE)
    _arrow(ax, 0.83, 0.50, 0.83, 0.475, C_ORANGE)

    # Patient-level aggregation
    _box(ax, 0.70, 0.31, 0.26, 0.06,
         "Patient-Level: Mean_H1_Risk\nacross all landmarks", C_WHITE, C_ORANGE,
         fs=FONT_SMALL, bold=True, multi_line=True)

    _arrow(ax, 0.83, 0.41, 0.83, 0.375, C_ORANGE)

    # Evaluation note
    _box(ax, 0.70, 0.245, 0.26, 0.05,
         "Quartile triage\nQ1 | Q2 | Q3 | Q4", C_ORANGE_L, C_ORANGE,
         fs=FONT_SMALL, bold=True, multi_line=True)
    _arrow(ax, 0.83, 0.31, 0.83, 0.30, C_ORANGE)

    # ── BOTTOM: Evaluation protocol ──
    _section_bg(ax, 0.02, 0.02, 0.96, 0.18, C_GRAY, alpha=0.05)
    ax.text(0.50, 0.19, "Evaluation Protocol",
            fontsize=FONT_SECTION, fontweight="bold", color=C_BLACK,
            ha="center", va="top", zorder=5)
    ax.plot([0.15, 0.85], [0.175, 0.175], color=C_GRAY_L, lw=1, zorder=1)

    _box(ax, 0.06, 0.06, 0.25, 0.09,
         "OOF (Development)\nModel selection\nThreshold calibration\nNO test peeking",
         C_GREEN_L, C_GREEN, fs=FONT_SMALL, multi_line=True, alpha=0.5)

    _box(ax, 0.37, 0.06, 0.25, 0.09,
         "Temporal Test\nReporting only\nPre-locked thresholds\nIndependent validation",
         C_ORANGE_L, C_ORANGE, fs=FONT_SMALL, multi_line=True, alpha=0.5)

    _box(ax, 0.68, 0.06, 0.25, 0.09,
         "Metrics\nROC-AUC, PR-AUC\nCalibration (ECE)\nDCA (Net Benefit)",
         C_GRAY_L, C_GRAY, fs=FONT_SMALL, multi_line=True, alpha=0.5)

    _arrow(ax, 0.31, 0.105, 0.37, 0.105, C_GRAY, lw=1.5)
    _arrow(ax, 0.62, 0.105, 0.68, 0.105, C_GRAY, lw=1.5)

    # Save
    path = os.path.join(OUT_DIR, "Figure_00_Stage2_Design.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C_WHITE,
                edgecolor="none", pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating architecture diagrams ...")
    draw_overall_framework()
    draw_stage1_design()
    draw_stage2_design()
    print("Done.")
