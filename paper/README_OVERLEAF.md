# Overleaf LaTeX Deployment Guide for DART-X Research Paper

This folder contains a complete, camera-ready IEEE-format research paper ready for online deployment on Overleaf.

## Files Included:
- **`main.tex`**: The complete academic paper manuscript (formatted with IEEEtran two-column styling, equations, tables, pseudocode algorithm, and figure references).
- **`references.bib`**: Complete BibTeX bibliographic citations (Almeida et al. ESWA 2018, PKLot 2015, Gama 2014, Bifet 2010, Domingos 2000, etc.).
- **`figures/`**: All 5 high-resolution publication figures:
  1. `accuracy_over_time.png` (82-day chronological timeline)
  2. `per_location_accuracy.png` (UFPR04, UFPR05, PUCPR grouped bar chart)
  3. `confusion_matrix.png` (Normalized confusion matrices across 619,466 instances)
  4. `ablation_study.png` (Component ablation error-bar chart)
  5. `synthetic_drifts.png` (4-panel synthetic drift stress tests)

---

## How to Deploy to Overleaf in 3 Simple Steps:

### Step 1: Locate the Zip File
A pre-packaged archive is ready at:
`dartx_paper_overleaf.zip` (in the project root directory or `~/Documents/DART-X/`).

### Step 2: Upload to Overleaf
1. Open [https://www.overleaf.com](https://www.overleaf.com) and log in.
2. Click the green **"New Project"** button in the top left.
3. Click **"Upload Project"**.
4. Drag and drop `dartx_paper_overleaf.zip` into the dialog.

### Step 3: Compile
Overleaf will automatically unpack all files and compile the document. Click **"Recompile"** if not triggered automatically. The PDF will render with all tables, equations, citations, and figures in IEEE format.

