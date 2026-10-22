"""
Figure 1 – Nature Portfolio Publication Architecture Diagram
============================================================
Refactored with clean, balanced spacing across all panels:
  - Panel a: Title "INPUT PREPROCESSING" fits perfectly within the badge.
  - Generous, uniform gaps (0.035-0.045) between all blocks, matching Panels b and c.
  - Biological bacterium illustration with capsule, chromosome, and blaKPC-2 plasmid.
  - Clean orthogonal routing for the purple dashed interpretability connector:
    routes completely clear of Concat contexts and the Stage 4 title, entering
    SHAP attribution cleanly from the left channel.
"""

from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge, Polygon,
)
import numpy as np

ROOT = Path(__file__).resolve().parent
while not (ROOT / ".git").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
OUT = ROOT / "results" / "new"
OUT.mkdir(parents=True, exist_ok=True)

# ── Typography & Export Setup ─────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ── Color Palette ────────────────────────────────────────────────
BG       = "#ffffff"
TEXT     = "#0f172a"
MUTED    = "#334155"  # High-contrast secondary text
LIGHT    = "#475569"  # High-contrast annotation text

# Stage accent colours
S1       = "#1e40af"   # preprocessing  (deep blue)
S2       = "#b91c1c"   # embedding      (deep red)
S3       = "#0e7490"   # classification (dark cyan)
S4       = "#7e22ce"   # interpretability (purple)

S1_BG    = "#eff6ff"
S2_BG    = "#fef2f2"
S3_BG    = "#ecfeff"
S4_BG    = "#faf5ff"

GREEN    = "#15803d"
ARROW_C  = "#334155"
CORAL    = "#e11d48"   # Coral red for AMR plasmid / resistance


def make_figure():
    YMAX = 0.985
    fig, ax = plt.subplots(figsize=(16.0, 8.4 * YMAX / 0.925), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, YMAX)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Helpers ────────────────────────────────────────────────────

    def stage_bg(x, y, w, h, letter, title, color, fill):
        """Stage background with Nature panel badge (a, b, c, d) + title."""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            linewidth=1.2, edgecolor=color, facecolor=fill,
            linestyle="solid", alpha=0.60, zorder=0,
        ))
        badge_x = x + 0.012
        badge_y = y + h - 0.038
        ax.add_patch(FancyBboxPatch(
            (badge_x, badge_y - 0.012), 0.022, 0.024,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            linewidth=0, facecolor=color, alpha=0.95, zorder=3,
        ))
        ax.text(badge_x + 0.011, badge_y, str(letter),
                fontsize=11, fontweight="bold", color="#ffffff",
                ha="center", va="center", zorder=4,
                fontfamily="sans-serif")
        ax.text(badge_x + 0.027, badge_y - 0.002, title, color=color,
                fontsize=9.2, fontweight="bold", ha="left", va="center",
                zorder=3, fontfamily="sans-serif",
                fontstyle="normal")

    def box(x, y, w, h, title, edge=ARROW_C, fill="#ffffff",
            lw=1.1, fs=9.0, title_weight="semibold",
            subtitle=None, sub_fs=7.2, sub_color=None,
            linestyle="solid"):
        """Process block – clean, no shadow. Optional subtitle inside."""
        if sub_color is None:
            sub_color = MUTED
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.007,rounding_size=0.009",
            linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
            linestyle=linestyle,
        ))
        if subtitle:
            if "\n" in title:
                title_y = y + h * 0.65
                sub_y = y + h * 0.22
            else:
                title_y = y + h * 0.60
                sub_y = y + h * 0.24
            ax.text(x + w / 2, title_y, title,
                    ha="center", va="center", fontsize=fs, color=TEXT,
                    zorder=3, linespacing=1.1, fontfamily="sans-serif",
                    fontweight=title_weight)
            ax.text(x + w / 2, sub_y, subtitle,
                    ha="center", va="center", fontsize=sub_fs,
                    color=sub_color, zorder=3, fontfamily="sans-serif",
                    fontstyle="italic", linespacing=1.1)
        else:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center", fontsize=fs, color=TEXT,
                    zorder=3, linespacing=1.2, fontfamily="sans-serif",
                    fontweight=title_weight)

    def annotation(x, y, txt, fs=7.8, color=LIGHT, ha="center"):
        """Subordinate annotation text."""
        ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=color,
                zorder=4, fontfamily="sans-serif")

    def arrow(a, b, dashed=False, rad=0.0, color=ARROW_C, lw=1.0, ms=10):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
            color=color, linestyle=(0, (4, 3)) if dashed else "solid",
            connectionstyle=f"arc3,rad={rad}", zorder=2.5,
        ))

    def heavy_arrow(a, b, color=ARROW_C):
        """Thicker inter-stage arrow."""
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=13, linewidth=1.4,
            color=color, zorder=3,
        ))

    # ═══════════════════════════════════════════════════════════════
    # STAGE BACKGROUNDS
    # ═══════════════════════════════════════════════════════════════
    # Shift panel a by dy = -0.185 so that top aligns with b and c (0.775),
    # badge a aligns with badge b and c (0.737),
    # and bottom aligns with panel d (0.020).
    dy = -0.185

    stage_bg(0.020, 0.205 + dy, 0.200, 0.755,
             "a", "INPUT PREPROCESSING", S1, S1_BG)

    stage_bg(0.250, 0.280, 0.295, 0.495,
             "b", "EMBEDDING & POOLING", S2, S2_BG)

    stage_bg(0.575, 0.280, 0.405, 0.495,
             "c", "RESISTANCE CLASSIFICATION", S3, S3_BG)

    stage_bg(0.250, 0.020, 0.555, 0.185,
             "d", "INTERPRETABILITY & VALIDATION", S4, S4_BG)

    # ═══════════════════════════════════════════════════════════════
    # 1. INPUT PREPROCESSING (Panel a: Clean Bacterium + Genome Assemblies)
    # ═══════════════════════════════════════════════════════════════
    cx = 0.120
    bw_card = 0.160

    # 1.1 Klebsiella pneumoniae + zoom on the blaKPC-2 plasmid
    #     Drawn in an inset with true 1:1 aspect (inches), so circles are
    #     circles (the main axes is anisotropic: x and y units differ).
    ax_w_in = fig.get_figwidth() * ax.get_position().width
    ax_h_in = fig.get_figheight() * ax.get_position().height
    ins_w_in, ins_h_in = 2.30, 0.80
    ins_w = ins_w_in / ax_w_in
    ins_h = ins_h_in * YMAX / ax_h_in
    ia = ax.inset_axes([cx - ins_w / 2, 0.790 + dy, ins_w, ins_h], transform=ax.transData)
    ia.set_xlim(0, ins_w_in)
    ia.set_ylim(0, ins_h_in)
    ia.axis("off")
    ia.patch.set_alpha(0)

    # --- cell: capsule halo + rod-shaped body -------------------------
    ccx, ccy = 0.66, 0.40
    ia.add_patch(FancyBboxPatch(
        (ccx - 0.62, ccy - 0.27), 1.24, 0.54,
        boxstyle="round,pad=0,rounding_size=0.27",
        fc="#e0f2fe", ec="#38bdf8", lw=0.8, ls="--", alpha=0.75, zorder=1))
    ia.add_patch(FancyBboxPatch(
        (ccx - 0.55, ccy - 0.20), 1.10, 0.40,
        boxstyle="round,pad=0,rounding_size=0.20",
        fc="#bae6fd", ec="#0284c7", lw=1.1, zorder=2))

    # --- chromosome (supercoiled loop) ---------------------------------
    tt = np.linspace(0, 2 * np.pi, 600)
    for k, ph, al in [(5, 0.0, 0.95), (7, 1.1, 0.55)]:
        rr = 1 + 0.30 * np.cos(k * tt + ph)
        ia.plot(0.46 + 0.26 * rr * np.cos(tt), ccy + 0.092 * rr * np.sin(tt),
                color="#0369a1", lw=0.9, alpha=al, zorder=3)

    # --- small plasmid inside the cell ---------------------------------
    pcx, pcy, pr = 1.00, ccy, 0.075
    ia.add_patch(Circle((pcx, pcy), pr, fc="#ffe4e6", ec=CORAL, lw=1.4, zorder=4))

    # --- zoom bubble ---------------------------------------------------
    bcx, bcy, bR = 1.90, ccy, 0.38
    a1, a2 = np.deg2rad(135), np.deg2rad(225)
    top_small = (pcx + pr * np.cos(np.deg2rad(50)), pcy + pr * np.sin(np.deg2rad(50)))
    bot_small = (pcx + pr * np.cos(np.deg2rad(-50)), pcy + pr * np.sin(np.deg2rad(-50)))
    top_big = (bcx + bR * np.cos(a1), bcy + bR * np.sin(a1))
    bot_big = (bcx + bR * np.cos(a2), bcy + bR * np.sin(a2))
    ia.add_patch(Polygon([top_small, top_big, bot_big, bot_small], closed=True,
                         fc=CORAL, ec="none", alpha=0.07, zorder=1.5))
    ia.plot([top_small[0], top_big[0]], [top_small[1], top_big[1]],
            color=CORAL, lw=0.7, ls=(0, (3, 2)), alpha=0.8, zorder=3)
    ia.plot([bot_small[0], bot_big[0]], [bot_small[1], bot_big[1]],
            color=CORAL, lw=0.7, ls=(0, (3, 2)), alpha=0.8, zorder=3)
    ia.add_patch(Circle((bcx, bcy), bR, fc="#ffffff", ec=CORAL, lw=1.1, zorder=4))

    # plasmid map: grey backbone + blaKPC-2 gene as coloured arrow-arc
    ring_r, ring_w = 0.28, 0.055
    ia.add_patch(Wedge((bcx, bcy), ring_r + ring_w / 2, 0, 360, width=ring_w,
                       fc="#cbd5e1", ec="none", zorder=5))
    ia.add_patch(Wedge((bcx, bcy), ring_r + ring_w / 2, 52, 140, width=ring_w,
                       fc=CORAL, ec="none", zorder=6))
    def _pt(r, deg):
        return (bcx + r * np.cos(np.deg2rad(deg)), bcy + r * np.sin(np.deg2rad(deg)))
    ia.add_patch(Polygon([_pt(ring_r + ring_w / 2 + 0.028, 52),
                          _pt(ring_r - ring_w / 2 - 0.028, 52),
                          _pt(ring_r, 28)], closed=True, fc=CORAL, ec="none", zorder=6))
    ia.text(bcx, bcy + 0.035, "blaKPC-2", fontsize=6.4, fontweight="bold",
            fontstyle="italic", color=CORAL, ha="center", va="center", zorder=7)
    ia.text(bcx, bcy - 0.075, "plasmid", fontsize=5.8, color=MUTED,
            ha="center", va="center", zorder=7)

    # Caption under the illustration
    ax.text(cx, 0.780 + dy, "Klebsiella pneumoniae", fontsize=8.6, fontweight="bold",
            fontstyle="italic", ha="center", va="center", color=TEXT, zorder=4)
    ax.text(cx, 0.768 + dy, "Clinical AMR isolates (BV-BRC / EuSCAPE)",
            fontsize=6.4, color=MUTED, ha="center", va="top", zorder=4)

    # Arrow: Bacterium -> Genome Assemblies
    arrow((cx, 0.756 + dy), (cx, 0.739 + dy))

    # 1.2 Visual Genome Assemblies Block (WGS Contigs & Scaffolds)
    asmb_y = 0.655 + dy
    asmb_h = 0.082
    box(cx - bw_card/2, asmb_y, bw_card, asmb_h, "", edge="#0284c7", fill="#f0f9ff", lw=1.0)

    # Header inside assembly box
    ax.text(cx, asmb_y + asmb_h - 0.010, "Genome assemblies (WGS)",
            fontsize=8.0, fontweight="bold", color="#0369a1", ha="center", va="center", zorder=4)

    # Visual Contigs / Assembly Tracks
    contig_x0 = cx - bw_card/2 + 0.014
    contig_w = bw_card - 0.028

    # Track 1: Long contig with highlighted blaKPC-2 AMR locus
    t1_y = asmb_y + 0.040
    ax.add_patch(Rectangle((contig_x0, t1_y), contig_w, 0.007, facecolor="#cbd5e1", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((contig_x0 + 0.010, t1_y), 0.016, 0.007, facecolor="#94a3b8", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((contig_x0 + 0.045, t1_y), 0.025, 0.007, facecolor=CORAL, edgecolor="none", zorder=5))
    ax.text(contig_x0 + 0.045 + 0.0125, t1_y + 0.0085, "blaKPC-2",
            fontsize=5.8, fontstyle="italic", fontweight="bold",
            color=CORAL, ha="center", va="bottom", zorder=6)
    ax.add_patch(Rectangle((contig_x0 + 0.085, t1_y), 0.020, 0.007, facecolor="#94a3b8", edgecolor="none", zorder=4))

    # Track 2: Medium contig
    t2_y = asmb_y + 0.028
    ax.add_patch(Rectangle((contig_x0 + 0.005, t2_y), contig_w * 0.75, 0.006, facecolor="#cbd5e1", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((contig_x0 + 0.025, t2_y), 0.018, 0.006, facecolor="#64748b", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((contig_x0 + 0.055, t2_y), 0.014, 0.006, facecolor="#94a3b8", edgecolor="none", zorder=4))

    # Track 3: Short contig
    t3_y = asmb_y + 0.0175
    ax.add_patch(Rectangle((contig_x0 + 0.015, t3_y), contig_w * 0.60, 0.005, facecolor="#cbd5e1", edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((contig_x0 + 0.030, t3_y), 0.022, 0.005, facecolor="#64748b", edgecolor="none", zorder=4))

    # Assembly metrics
    ax.text(cx, asmb_y + 0.0065, "3,702 genomes | Ertapenem: 1,448 R / 2,254 S",
            fontsize=6.2, color=MUTED, ha="center", va="center", zorder=4)

    # Arrow: Genome Assemblies -> Lineage Partitioning
    arrow((cx, asmb_y), (cx, asmb_y - 0.024))

    # 1.3 Lineage Partitioning Block (y: 0.558 to 0.631, h = 0.073)
    lp_y = 0.558 + dy
    lp_h = 0.073
    box(cx - bw_card/2, lp_y, bw_card, lp_h,
        "Lineage partitioning\nby Sequence Type", edge=S1, fs=8.2,
        subtitle="ST-blocked CV (primary)\nvs Historical comparator",
        sub_fs=6.4, sub_color=MUTED)

    # Arrow: Lineage Partitioning -> CARD-RGI
    arrow((cx, lp_y), (cx, lp_y - 0.024))

    # 1.4 CARD-RGI Annotation & Extraction (y: 0.461 to 0.534, h = 0.073)
    rgi_y = 0.461 + dy
    rgi_h = 0.073
    box(cx - bw_card/2, rgi_y, bw_card, rgi_h,
        "CARD-RGI resistome\nannotation", edge=S1, fs=8.2,
        subtitle="v6.0.3 / CARD 4.0.1 ontology\nStrict & Perfect criteria",
        sub_fs=6.4, sub_color=MUTED)

    # Arrow: CARD-RGI -> Gene bag (TOUCHES TOP CARD EXACTLY!)
    arrow((cx, rgi_y), (cx, rgi_y - 0.024))

    # 1.5 Variable Resistome Gene Bag
    top_card_top = rgi_y - 0.024
    card_w = 0.145
    card_h = 0.046
    top_base_y = top_card_top - card_h

    n_cards = 4
    for i in range(n_cards):
        off_x = (i - 3) * 0.004
        off_y = (i - 3) * 0.005
        is_top = (i == n_cards - 1)
        ax.add_patch(FancyBboxPatch(
            (cx - card_w/2 + off_x, top_base_y + off_y), card_w, card_h,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor="#ffffff" if is_top else "#f8fafc",
            edgecolor=CORAL if is_top else "#cbd5e1",
            linewidth=1.1 if is_top else 0.6, zorder=2 + i
        ))

    # Top card text (Gene 1: blaKPC-2)
    ax.text(cx, top_base_y + card_h * 0.68, r"Gene 1: $\bf{bla}_{KPC-2}$",
            fontsize=8.0, ha="center", va="center", color=CORAL, zorder=8)
    ax.text(cx, top_base_y + card_h * 0.26, "Carbapenemase (882 bp | CARD:3000596)",
            fontsize=6.2, ha="center", va="center", color=MUTED, fontstyle="italic", zorder=8)

    # Labels below gene bag cards
    ax.text(cx, 0.355 + dy, r"$\bf{Variable\ resistome\ bag\ } B_j = \{g_1, \dots, g_{M_j}\}$",
            fontsize=7.4, ha="center", va="center", color=TEXT, zorder=4)
    ax.text(cx, 0.337 + dy, r"Resistome profile ($M_j \in [1, 50]$ genes per isolate)",
            fontsize=6.6, ha="center", va="center", color=MUTED, zorder=4)
    ax.text(cx, 0.320 + dy, "Extracted CDS nucleotide sequences",
            fontsize=6.3, fontstyle="italic", ha="center", va="center", color=LIGHT, zorder=4)

    # Container badge for Multi-Instance Learning representation
    ax.add_patch(FancyBboxPatch(
        (cx - bw_card/2 + 0.010, 0.230 + dy), bw_card - 0.020, 0.065,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        facecolor="#eff6ff", edgecolor="#93c5fd", linewidth=0.8, linestyle=":", zorder=2
    ))
    ax.text(cx, 0.276 + dy, "Multi-Instance Learning Input",
            fontsize=7.2, fontweight="bold", color=S1, ha="center", va="center", zorder=4)
    ax.text(cx, 0.258 + dy, "Unordered set of variable length",
            fontsize=6.4, color=MUTED, ha="center", va="center", zorder=4)
    ax.text(cx, 0.244 + dy, "Instances: individual resistance genes",
            fontsize=6.2, fontstyle="italic", color=LIGHT, ha="center", va="center", zorder=4)

    # Heavy arrow from Gene Bag in Stage 1 into NTv3 in Stage 2
    gene_bag_exit_y = top_base_y + card_h / 2
    heavy_arrow((0.220, gene_bag_exit_y), (0.260, 0.575))

    # ═══════════════════════════════════════════════════════════════
    # 2. EMBEDDING & POOLING (Original Layout)
    # ═══════════════════════════════════════════════════════════════

    # NTv3 block (subtitle inside)
    box(0.260, 0.550, 0.130, 0.090,
        "NTv3 foundation\nmodel", edge=S2, fs=9.2,
        subtitle="end-to-end 100M encoder")

    # MIL encoder block (subtitle inside)
    box(0.415, 0.550, 0.130, 0.090,
        "Multi-head\nattention MIL", edge=S2, fs=9.2,
        subtitle="attention pooling")

    # ── Gene embeddings (matrix representation) ────────────────
    emb_x0 = 0.270
    emb_y0 = 0.365
    emb_rows = 6
    emb_cols = 10
    cell_w = 0.011
    cell_h = 0.018
    np.random.seed(42)
    blues = plt.cm.Blues(np.linspace(0.15, 0.65, emb_rows * emb_cols))
    np.random.shuffle(blues)
    idx = 0
    for r in range(emb_rows):
        for c in range(emb_cols):
            ax.add_patch(Rectangle(
                (emb_x0 + c * cell_w, emb_y0 + r * cell_h),
                cell_w * 0.92, cell_h * 0.85,
                facecolor=blues[idx], edgecolor="none",
                linewidth=0, zorder=2, alpha=0.80,
            ))
            idx += 1
    ax.add_patch(Rectangle(
        (emb_x0 - 0.002, emb_y0 - 0.002),
        emb_cols * cell_w + 0.004, emb_rows * cell_h + 0.004,
        facecolor="none", edgecolor="#93c5fd",
        linewidth=0.8, zorder=3,
    ))
    annotation(0.325, emb_y0 - 0.020,
              r"$e_i \in \mathbb{R}^{768}$", fs=8.5, color=MUTED)
    annotation(0.325, emb_y0 - 0.042,
              r"$M \times 768$ embeddings", fs=7.8, color=LIGHT)

    # ── Attention heads (heatmap bars) ─────────────────────────
    head_x0 = 0.430
    head_y0 = 0.375
    head_w = 0.100
    head_h = 0.020
    head_gap = 0.005
    head_labels = ["Head 1", "Head 2", "Head 3", "Head 4"]
    head_cmaps = [plt.cm.Reds, plt.cm.Oranges, plt.cm.YlOrRd, plt.cm.PuRd]
    np.random.seed(7)
    for i in range(4):
        yy = head_y0 + i * (head_h + head_gap)
        n_seg = 16
        vals = np.random.dirichlet(np.ones(n_seg) * 2)
        vals = vals / vals.max()
        cmap = head_cmaps[i]
        for s in range(n_seg):
            seg_w = head_w / n_seg
            ax.add_patch(Rectangle(
                (head_x0 + s * seg_w, yy),
                seg_w, head_h,
                facecolor=cmap(0.2 + 0.5 * vals[s]),
                edgecolor="none", linewidth=0, zorder=2,
            ))
        ax.add_patch(Rectangle(
            (head_x0, yy), head_w, head_h,
            facecolor="none", edgecolor="#fca5a5",
            linewidth=0.5, zorder=3,
        ))
    annotation(0.480, head_y0 - 0.018,
              r"$H = 4$ attention heads", fs=8.0, color=MUTED)
    annotation(0.480, head_y0 - 0.040,
              "cosine penalty", fs=7.5, color=LIGHT)

    # Arrows within embedding stage
    arrow((0.325, 0.550), (0.325, 0.480))            # NTv3 -> emb matrix
    arrow((0.390, 0.595), (0.415, 0.595))            # NTv3 -> MIL
    arrow((0.480, 0.550), (0.480, 0.480))            # MIL -> heads

    # ═══════════════════════════════════════════════════════════════
    # 3. RESISTANCE CLASSIFICATION (Original Layout)
    # ═══════════════════════════════════════════════════════════════

    col1_x = 0.595
    col1_w = 0.175
    col2_x = 0.805
    col2_w = 0.160
    row1_y = 0.415
    row1_h = 0.075
    row2_y = 0.550
    row2_h = 0.085

    # 1. Multi-head context – visual bar showing concatenation (3072-D)
    ge_x = col1_x
    ge_y = 0.320
    ge_w = col1_w
    ge_h = 0.036
    seg_w = ge_w / 4
    seg_colors = ["#bfdbfe", "#c7d2fe", "#ddd6fe", "#e9d5ff"]
    seg_borders = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7"]
    for i in range(4):
        ax.add_patch(Rectangle(
            (ge_x + i * seg_w, ge_y), seg_w, ge_h,
            facecolor=seg_colors[i], edgecolor=seg_borders[i],
            linewidth=0.7, zorder=2,
        ))
        ax.text(ge_x + (i + 0.5) * seg_w, ge_y + ge_h / 2,
                f"$c^{{({i+1})}}$", fontsize=7.2,
                ha="center", va="center", color=TEXT, zorder=3,
                fontfamily="sans-serif")
    ax.add_patch(Rectangle(
        (ge_x, ge_y), ge_w, ge_h,
        facecolor="none", edgecolor="#6366f1",
        linewidth=1.0, zorder=3,
    ))
    ax.text(ge_x + ge_w / 2, 0.300,
            r"Concat contexts (3072-D): [$c^{(1)}; c^{(2)}; c^{(3)}; c^{(4)}$]",
            fontsize=6.8, ha="center", va="center", color=MUTED,
            zorder=4, fontfamily="sans-serif", fontstyle="italic")

    # 2. Linear projection & LayerNorm layer (3072 -> 768-D)
    box(col1_x, row1_y, col1_w, row1_h,
        "Linear projection\n& LayerNorm", edge=S3, fill="#ffffff",
        fs=8.8, title_weight="semibold",
        subtitle="Genome embedding (768-D)", sub_fs=7.5, sub_color=S3)

    # 3. Neural classification head (MIL-only) – Primary model
    box(col1_x, row2_y, col1_w, row2_h,
        "Neural head\n(MIL-only)", edge=S3, fill="#f0fdfa",
        lw=1.5, fs=9.2, title_weight="bold",
        subtitle="primary model", sub_fs=7.8, sub_color=S3)

    # 4. XGBoost classifier – Hybrid ablation branch
    box(col2_x, row1_y, col2_w, row1_h,
        "XGBoost\nclassifier", edge="#64748b", fill="#f8fafc",
        lw=1.2, linestyle="dashed", fs=8.8, title_weight="semibold",
        subtitle="hybrid ablation", sub_fs=7.5, sub_color=MUTED)

    # 5. Output Prediction block
    box(col2_x, row2_y, col2_w, row2_h,
        "Prediction\n(S / R)", edge=GREEN, fill="#f0fdf4",
        lw=1.4, fs=9.5, title_weight="bold",
        subtitle="susceptible / resistant", sub_fs=7.2, sub_color="#15803d")

    # Arrows within classification
    arrow((0.545, 0.380), (ge_x, ge_y + ge_h / 2))                        # heads -> concat context
    arrow((col1_x + col1_w / 2, ge_y + ge_h), (col1_x + col1_w / 2, row1_y)) # concat -> projection (3072->768)
    arrow((col1_x + col1_w / 2, row1_y + row1_h), (col1_x + col1_w / 2, row2_y),
          color=S3, lw=1.5)                                                # projection -> neural head (primary)
    arrow((col1_x + col1_w, row2_y + row2_h / 2), (col2_x, row2_y + row2_h / 2),
          color=S3, lw=1.5)                                                # neural head -> prediction (primary)
    arrow((col1_x + col1_w, row1_y + row1_h / 2), (col2_x, row1_y + row1_h / 2),
          dashed=True, color="#64748b", lw=1.1)                             # projection -> xgboost (ablation)
    arrow((col2_x + col2_w / 2, row1_y + row1_h), (col2_x + col2_w / 2, row2_y),
          dashed=True, color="#64748b", lw=1.1)                             # xgboost -> prediction (ablation)

    # ═══════════════════════════════════════════════════════════════
    # 4. INTERPRETABILITY & VALIDATION (Clean Stepped Routing)
    # ═══════════════════════════════════════════════════════════════

    # Three blocks across the bottom row
    ib_y = 0.050
    ib_h = 0.080

    box(0.270, ib_y, 0.155, ib_h,
        "SHAP attribution", edge=S4, fs=9.2,
        subtitle="768 latent dimensions ranked")

    box(0.455, ib_y, 0.165, ib_h,
        "Gene backtracking", edge=S4, fs=9.2,
        subtitle=r"attention $\alpha_i$ + SHAP correlation")

    box(0.650, ib_y, 0.155, ib_h,
        "BLAST validation", edge=S4, fs=9.2,
        subtitle="NCBI / CARD RefSeq alignment")

    # Flow arrows between interpretability blocks
    heavy_arrow((0.425, ib_y + ib_h/2), (0.455, ib_y + ib_h/2))
    heavy_arrow((0.620, ib_y + ib_h/2), (0.650, ib_y + ib_h/2))

    # Clean Stepped Connector from XGBoost to SHAP:
    # 1. Drops from bottom of XGBoost into the clear corridor (y = 0.240)
    #    COMPLETELY BELOW Concat contexts (y >= 0.300)
    # 2. Runs left along y = 0.240 above Stage 4 (y <= 0.205)
    # 3. Drops down in the left vertical channel (x = 0.238)
    #    COMPLETELY CLEAR of the Stage 4 title (x >= 0.289)
    # 4. Enters the left side of the SHAP box at y = 0.090
    corridor_y = 0.240
    left_ch = 0.238
    shap_entry_y = ib_y + ib_h / 2  # 0.090

    pts = [
        (col2_x + 0.025, row1_y),         # Exit XGBoost bottom
        (col2_x + 0.025, corridor_y),    # Down to corridor
        (left_ch, corridor_y),           # Left across corridor
        (left_ch, shap_entry_y),         # Down left channel
    ]
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
            linestyle=(0, (4, 3)), color=S4, lw=1.1, zorder=3)
    ax.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]],
            linestyle=(0, (4, 3)), color=S4, lw=1.1, zorder=3)
    ax.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]],
            linestyle=(0, (4, 3)), color=S4, lw=1.1, zorder=3)

    # Arrow entering left face of SHAP box
    arrow((left_ch, shap_entry_y), (0.270, shap_entry_y),
          dashed=True, color=S4, lw=1.1, ms=9)

    # Label on horizontal corridor
    ax.text(0.550, corridor_y + 0.007,
            "768-D representation to SHAP",
            fontsize=6.8, fontstyle="italic", color=S4, ha="center", va="bottom", zorder=4)

    # ═══════════════════════════════════════════════════════════════
    # SAVE (High-res 300 DPI & TrueType Vector PDF)
    # ═══════════════════════════════════════════════════════════════
    fig.savefig(OUT / "Figure1_updated.png",
                bbox_inches="tight", pad_inches=0.08, dpi=300,
                facecolor=BG)
    fig.savefig(OUT / "Figure1_updated.pdf",
                bbox_inches="tight", pad_inches=0.08,
                facecolor=BG)
    plt.close(fig)

    # Standardize exact canvas dimensions to match legacy (3792 × 2151 px @ 300 DPI)
    try:
        from PIL import Image
        png_path = OUT / "Figure1_updated.png"
        im = Image.open(png_path)
        target_size = (3792, 2151)
        if im.size != target_size:
            canvas = Image.new("RGB", target_size, BG)
            offset_x = (target_size[0] - im.size[0]) // 2
            offset_y = (target_size[1] - im.size[1]) // 2
            canvas.paste(im, (offset_x, offset_y))
            canvas.save(png_path, dpi=(299.9994, 299.9994))
    except Exception as e:
        print(f"  [WARN] Canvas normalization warning: {e}")

    print("  [OK] Figure1_updated (.png / .pdf) saved to results/new/ (3792 × 2151 px, 300 DPI)")


def update_docx():
    """Re-inject the updated Figure 1 into the DOCX copy if present."""
    extracted = ROOT / "docx_extracted"
    dst_docx  = ROOT / "v6.2_dlf_figures_only_updated.docx"
    temp      = ROOT / "_docx_figures_updated"
    if not extracted.exists():
        return
    if temp.exists():
        shutil.rmtree(temp)
    shutil.copytree(extracted, temp)
    shutil.copy2(OUT / "Figure1_updated.png",
                 temp / "word" / "media" / "image1.png")
    for fig_name, img_name in [("Figure2_updated.png", "image2.png"),
                                ("Figure5_updated.png", "image5.png")]:
        p = OUT / fig_name
        if p.exists():
            shutil.copy2(p, temp / "word" / "media" / img_name)
    if dst_docx.exists():
        dst_docx.unlink()
    with ZipFile(dst_docx, "w", ZIP_DEFLATED) as zf:
        for path in temp.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(temp).as_posix())
    shutil.rmtree(temp)
    print("  [OK] v6.2_dlf_figures_only_updated.docx")


if __name__ == "__main__":
    make_figure()
    update_docx()
