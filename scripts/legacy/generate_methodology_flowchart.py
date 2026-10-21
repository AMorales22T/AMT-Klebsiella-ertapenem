#!/usr/bin/env python3
"""
generate_methodology_flowchart.py
=================================
Generates a publication-grade Nature-style methodology flowchart using pure Graphviz (DOT)
and renders it to both vector PDF and 600 DPI PNG without using Matplotlib.

Rules Implemented:
  1. Rounded rectangles with 1.0 pt borders and zero drop shadows.
  2. Specific color fills:
     - "Metastable Structure Search": Red (#E64B35) with white text.
     - "Equilibrium phase diagram": Light Blue (#4DBBD5).
     - "Phase Boundaries using SVM": Orange (#F39B7F / #EEA236).
     - "Metastable phase diagram": Purple (#8491B4 / #9370DB).
     - Internal calculation blocks: White background (#FFFFFF), grey borders (#777777).
     - Mathematical equations: Green box (#E8F5E9 fill, #2E7D32 border) with LaTeX math.
  3. Dashed feedback loops at the bottom:
     - "Perform DFT if energetically favorable"
     - "Update phase diagram when new phases are discovered"
  4. Strict 3-column alignment (Left, Center, Right) using Graphviz `rank=same`.
  5. High-resolution export via rsvg-convert: 600 DPI PNG + vector PDF.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DOT_SOURCE = """digraph MethodologyFlowchart {
    graph [
        rankdir=TB,
        splines=ortho,
        nodesep=1.0,
        ranksep=1.1,
        bgcolor="#FFFFFF",
        fontname="Arial",
        compound=true,
        pad="0.3,0.3"
    ];

    node [
        fontname="Arial",
        fontsize=9.0,
        penwidth=1.0,
        style="rounded,filled",
        shape=box,
        color="#777777",
        fillcolor="#FFFFFF",
        fontcolor="#111111",
        margin="0.22,0.16"
    ];

    edge [
        fontname="Arial",
        fontsize=7.8,
        color="#333333",
        fontcolor="#333333",
        penwidth=1.0,
        arrowsize=0.75
    ];

    // =========================================================================
    // COLUMN 1: LEFT (Structure Sampling & Exploration)
    // =========================================================================
    L1 [
        label=< <B>Candidate Structure</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Prototype sampling &amp; ionic substitution</FONT> >
    ];

    L2 [
        label=< <B>Metastable Structure Search</B><BR/><FONT POINT-SIZE="7.5" COLOR="#FFFFFF">Active learning &amp; Bayesian exploration</FONT> >,
        fillcolor="#E64B35",
        fontcolor="#FFFFFF",
        color="#B91C1C",
        penwidth=1.2
    ];

    L3 [
        label=< <B>Density Functional Theory</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Electronic structure &amp; force minimization</FONT> >
    ];

    L4 [
        label=< <B>Local Structure Relaxation</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Stress tensor &amp; atomic coordinate audit</FONT> >
    ];

    // =========================================================================
    // COLUMN 2: CENTER (Thermodynamics, Equations & Convex Hull)
    // =========================================================================
    C1 [
        label=< <B>Total Energy Evaluation</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Reference chemical potential integration</FONT> >
    ];

    C2 [
        label=<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="2">
                <TR><TD><B>Formation Energy &amp; Stability Equations</B></TD></TR>
                <TR><TD><FONT POINT-SIZE="8.5" FACE="Courier,monospace">&Delta;<I>H</I><SUB>f</SUB> = <I>E</I><SUB>tot</SUB> &minus; &sum; <I>x</I><SUB><I>i</I></SUB> &mu;<SUB><I>i</I></SUB></FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8.5" FACE="Courier,monospace"><I>E</I><SUB>hull</SUB> = <I>E</I>(<I>x</I>) &minus; min<SUB>hull</SUB> <I>E</I>(<I>x</I>) &le; &delta;<I>E</I><SUB>tol</SUB></FONT></TD></TR>
                <TR><TD><FONT POINT-SIZE="8.5" FACE="Courier,monospace">&Delta;<I>G</I>(<I>T</I>, <I>P</I>) = &Delta;<I>H</I><SUB>f</SUB> &minus; <I>T</I>&Delta;<I>S</I><SUB>vib</SUB></FONT></TD></TR>
            </TABLE>
        >,
        fillcolor="#E8F5E9",
        color="#2E7D32",
        fontcolor="#1B5E20",
        penwidth=1.2
    ];

    C3 [
        label=< <B>Thermodynamic Convex Hull</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Enthalpy lower-bound construction</FONT> >
    ];

    C4 [
        label=< <B>Chemical Potential Space</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Grand potential &Delta;&Phi; mapping</FONT> >
    ];

    // =========================================================================
    // COLUMN 3: RIGHT (Phase Diagrams & Machine Learning Phase Boundaries)
    // =========================================================================
    R1 [
        label=< <B>Equilibrium phase diagram</B><BR/><FONT POINT-SIZE="7.5" COLOR="#111111">Global ground-state stability regions</FONT> >,
        fillcolor="#4DBBD5",
        fontcolor="#111111",
        color="#1E88E5",
        penwidth=1.2
    ];

    R2 [
        label=< <B>Phase Boundaries using SVM</B><BR/><FONT POINT-SIZE="7.5" COLOR="#111111">Support Vector Machine kernel separation</FONT> >,
        fillcolor="#F39B7F",
        fontcolor="#111111",
        color="#E65100",
        penwidth=1.2
    ];

    R3 [
        label=< <B>Decision Margin Optimization</B><BR/><FONT POINT-SIZE="7.5" COLOR="#666666">Confidence intervals &amp; boundary curvature</FONT> >
    ];

    R4 [
        label=< <B>Metastable phase diagram</B><BR/><FONT POINT-SIZE="7.5" COLOR="#111111">Free-energy kinetic &amp; thermal persistence</FONT> >,
        fillcolor="#8491B4",
        fontcolor="#111111",
        color="#475569",
        penwidth=1.2
    ];

    // =========================================================================
    // STRICT 3-COLUMN ALIGNMENT VIA rank=same
    // =========================================================================
    { rank=same; L1; C1; R1; }
    { rank=same; L2; C2; R2; }
    { rank=same; L3; C3; R3; }
    { rank=same; L4; C4; R4; }

    // =========================================================================
    // VERTICAL FLOW WITHIN COLUMNS
    // =========================================================================
    L1 -> L2;
    L2 -> L3;
    L3 -> L4;

    C1 -> C2;
    C2 -> C3;
    C3 -> C4;

    R1 -> R2;
    R2 -> R3;
    R3 -> R4;

    // =========================================================================
    // HORIZONTAL METHODOLOGY FLOW
    // =========================================================================
    L1 -> C1;
    C1 -> R1;

    L2 -> C2;
    C2 -> R2;

    L3 -> C3;
    C3 -> R3;

    L4 -> C4;
    C4 -> R4;

    // =========================================================================
    // FEEDBACK LOOPS AT THE BOTTOM
    // =========================================================================
    // Loop 1: Perform DFT if energetically favorable
    R4 -> L2 [
        style="dashed",
        color="#666666",
        penwidth=1.0,
        constraint=false,
        label=" Perform DFT if energetically favorable "
    ];

    // Loop 2: Update phase diagram when new phases are discovered
    L4 -> R1 [
        style="dashed",
        color="#666666",
        penwidth=1.0,
        constraint=false,
        label=" Update phase diagram when new phases are discovered "
    ];
}
"""

def generate_flowchart():
    dot_file = OUT / "Methodology_Flowchart.dot"
    svg_file = OUT / "Methodology_Flowchart.svg"
    png_file = OUT / "Methodology_Flowchart.png"
    pdf_file = OUT / "Methodology_Flowchart.pdf"

    # 1. Write the clean DOT file
    with open(dot_file, "w", encoding="utf-8") as f:
        f.write(DOT_SOURCE)
    print(f"  [OK] Saved DOT source: {dot_file}")

    # 2. Render DOT to SVG via local dot CLI (WASM-based engine)
    dot_cmd = ["/home/dolfius/.local/bin/dot", "-Tsvg", "-o", str(svg_file), str(dot_file)]
    res = subprocess.run(dot_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing dot: {res.stderr}")
        sys.exit(1)
    print(f"  [OK] Rendered SVG: {svg_file}")

    # 3. Render SVG to 600 DPI PNG via rsvg-convert
    png_cmd = ["rsvg-convert", "-d", "600", "-p", "600", "-f", "png", "-o", str(png_file), str(svg_file)]
    res_png = subprocess.run(png_cmd, capture_output=True, text=True)
    if res_png.returncode != 0:
        print(f"Error rendering PNG with rsvg-convert: {res_png.stderr}")
        sys.exit(1)
    print(f"  [OK] Rendered 600 DPI PNG: {png_file}")

    # 4. Render SVG to Vector PDF via rsvg-convert
    pdf_cmd = ["rsvg-convert", "-f", "pdf", "-o", str(pdf_file), str(svg_file)]
    res_pdf = subprocess.run(pdf_cmd, capture_output=True, text=True)
    if res_pdf.returncode != 0:
        print(f"Error rendering PDF with rsvg-convert: {res_pdf.stderr}")
        sys.exit(1)
    print(f"  [OK] Rendered Vector PDF: {pdf_file}")

    print("\nFlowchart generation successful!")

if __name__ == "__main__":
    generate_flowchart()
