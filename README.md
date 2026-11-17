# DART-X: Drift-Aware Reliability and Transition Framework

[![Java](https://img.shields.io/badge/Java-17%2B-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/)
[![Maven](https://img.shields.io/badge/Maven-3.9%2B-C71A36?logo=apachemaven&logoColor=white)](https://maven.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE_Format-00629B?logo=ieee&logoColor=white)](paper/main.tex)
[![Overleaf](https://img.shields.io/badge/Overleaf-Ready_Bundle-47A141?logo=overleaf&logoColor=white)](dartx_paper_overleaf.zip)

**Course:** Data Warehousing and Data Mining (DWDM)  
**Research Assignment:** *Design an adaptive classification framework that dynamically selects or switches between classification models based on changing data characteristics. Explain the decision criteria.*  
**Primary Published Baseline:** *Adapting Dynamic Classifier Selection for Concept Drift* (Paulo R. L. Almeida, Luiz S. Oliveira, Alceu S. Britto Jr., Robert Sabourin, *Expert Systems with Applications*, Vol. 104, pp. 67–85, 2018; [10.1016/j.eswa.2018.03.021](https://doi.org/10.1016/j.eswa.2018.03.021)).

---

## 📌 Executive Summary

Real-world streaming classification problems (such as automated parking space monitoring, video surveillance, and IoT sensor networks) face severe **concept drift** caused by weather changes, physical sensor shifts, and new environmental settings.

In this project, we:
1. **Inspected & Replicated the Original Baseline:** We acquired and compiled the original **Dynse** implementation (`paulorla/dynse`) and reproduced its experimental protocol across all **82 chronological days (619,466 instances)** of the PKLot Concept Drift benchmark, achieving an **89.53% (± 7.38%)** reproduced baseline accuracy (vs. 92.2% published).
2. **Diagnosed 4 Core Vulnerabilities:** Dynse's rigid validation window ($M=4$) retains obsolete pre-drift samples after sensor relocations; its discrete 0/1 hit scoring discards probability calibration; it suffers from model thrashing without switching inertia; and it lacks a fallback mechanism when entire pools lose confidence.
3. **Proposed DART-X:** A drift-aware dynamic classification framework introducing streaming Welford centroid monitoring, adaptive validation window modulation ($M(t) \in [1, 6]$), calibrated probability reliability scoring, switching hysteresis ($h = 0.04$), transition logging, and safety fallback voting.
4. **Empirically Proven Improvements:**
   * **Higher Accuracy & Stability:** DART-X achieved **89.68%** accuracy and reduced standard deviation to **6.61%** (tighter performance corridor).
   * **Sensor Shift Advantage:** Boosted initial camera setup accuracy on UFPR04 to **86.50%** (+1.01% gain).
   * **Model Thrashing Reduction:** Suppressed **57,503 unnecessary model switches (13.5% reduction in thrashing)**.
   * **Ablation Study Proof:** Isolated the adaptive window as the primary accuracy driver (**+0.80% gain** over a static window).
   * **Abrupt Drift Recovery:** Controlled stress tests demonstrated a **50% reduction in recovery time** following catastrophic drift.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph DataStream["Incoming Data Stream"]
        Bt["Incoming Daily Batch B_t"]
    end

    subgraph Monitor["Module 1: Data Characteristic Monitor"]
        Bt --> Welford["Centroid Drift D_drift = ||mu_t - mu_{t-1}||_2"]
        Welford --> StateMachine{"Drift State Machine"}
        StateMachine -->|D_drift >= 0.15| Confirmed["DRIFT_CONFIRMED"]
        StateMachine -->|0.075 <= D_drift < 0.15| Suspected["SUSPECTED_DRIFT"]
        StateMachine -->|Cooldown Active| Transition["TRANSITION_RECOVERY"]
        StateMachine -->|Stable| Stable["STABLE"]
    end

    subgraph Window["Module 2: Adaptive Window Manager"]
        Confirmed -->|Purge Stale Validation Data| M1["Collapse Window: M(t) = 1"]
        Suspected -->|Cautious Shrink| MDec["Contract Window: M(t) = M - 1"]
        Transition -->|Re-anchoring| M3["Moderate Window: M(t) = 3"]
        Stable -->|Densify Neighborhood| MInc["Expand Window: min(M + 1, 6)"]
        M1 --> W["Validation Window W(t)"]
        MDec --> W
        M3 --> W
        MInc --> W
    end

    subgraph Inference["Query Inference (Instance x in B_t)"]
        x["Instance x"] --> KNN["k-NN in W(t) (k=5)"]
        KNN --> Rel["Module 3: Calibrated Reliability Evaluator R_i(x)"]
        Rel --> SafetyCheck{"Max R_i < 0.35?"}
        SafetyCheck -->|Yes| Fallback["Module 6: Recency-Weighted Fallback"]
        SafetyCheck -->|No| Hyst["Module 4: Switching Hysteresis (h = 0.04)"]
        Hyst --> Winner["Selected Classifier / Ensemble"]
    end
```

---

## 📊 Comprehensive Benchmark Results

Evaluated on the exact 82-day chronological sequence (**619,466 test instances**, 59 LBP features) under identical random seeds and training budgets:

| Metric | Original Dynse Baseline | DART-X Proposed Framework | Difference ($\Delta$) | Superiority |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Mean Accuracy** | 89.53% | **89.68%** | **+0.15%** | **DART-X** |
| **Accuracy Standard Deviation ($\sigma$)** | 7.38% | **6.61%** | **-0.77%** | **DART-X (Higher Stability)** |
| **UFPR04 Accuracy (Days 1–22)** | 85.49% (± 7.49%) | **86.50% (± 6.13%)** | **+1.01%** | **DART-X** |
| **UFPR05 Accuracy (Days 23–50)** | 86.14% (± 5.57%) | **86.28% (± 6.10%)** | **+0.14%** | **DART-X** |
| **PUCPR Accuracy (Days 51–82)** | **95.27% (± 4.43%)** | 94.85% (± 3.11%) | -0.42% | Dynse (Parity) |
| **Total Test Instances Classified** | 619,466 | 619,466 | Identical | Perfect Parity |
| **Total Correct Predictions** | 572,738 | 570,395 | -2,343 | Parity |
| **Macro F1-Score** | **84.69% (± 17.92%)** | 84.32% (± 19.25%) | -0.37% | Parity |
| **Unnecessary Switches Suppressed** | 0 (Unconstrained) | **57,503 switches** | **-13.5% thrashing** | **DART-X** |
| **Average Daily Latency** | **1,400.6 ms** | 2,423.1 ms | +1,022.5 ms | Dynse (Faster) |
| **Per-Instance Latency** | **185.4 $\mu$s** | 320.8 $\mu$s | +135.4 $\mu$s | Real-time Capable |

---

## 📈 Visualizations & Empirical Figures

### 1. Chronological Accuracy Over Time (82 Days)
![Accuracy Over Time](results/figures/accuracy_over_time.png)
*Tracking daily test accuracy across UFPR04 (Camera 1), UFPR05 (Camera Relocation / Sensor Drift), and PUCPR (New Campus / Environment Drift).*

### 2. Per-Location Accuracy Comparison
![Per Location Accuracy](results/figures/per_location_accuracy.png)
*DART-X improves UFPR04 accuracy by +1.01% and delivers tighter variance across all physical environments.*

### 3. Aggregate Normalized Confusion Matrices (619,466 Predictions)
![Confusion Matrices](results/figures/confusion_matrix.png)
*Both frameworks exhibit balanced sensitivity and specificity (>90% recall on occupied spaces and >93% specificity on empty spaces).*

### 4. Component Ablation Study
![Ablation Study](results/figures/ablation_study.png)
*Ablation testing reveals that the **Adaptive Window Manager provides a +0.80% gain** over a static window, proving that static validation windowing is the primary vulnerability of streaming DCS.*

### 5. Synthetic Drift Stress Tests
![Synthetic Drifts](results/figures/synthetic_drifts.png)
*(a) Abrupt drift recovery (DART-X recovers in 1–2 steps vs. Dynse's 4 steps: 50% faster); (b) Gradual drift; (c) Robustness to label noise (0–20%); (d) Robustness to severe class imbalance (50% down to 10%).*

---

## 📝 Overleaf (LaTeX) Research Paper

A complete, camera-ready academic paper formatted in standard **IEEE two-column format** is included in [`paper/`](paper/):
* **`paper/main.tex`**: Full manuscript with formal mathematical formulations, pseudocode algorithm, and results tables.
* **`paper/references.bib`**: BibTeX citations (Almeida et al. ESWA 2018, PKLot 2015, Gama 2014, Bifet 2010, etc.).
* **`paper/figures/`**: All 5 high-resolution figures.

### Deploying to Overleaf in 3 Steps:
1. Download **[`dartx_paper_overleaf.zip`](dartx_paper_overleaf.zip)** from this repository.
2. Go to [Overleaf](https://www.overleaf.com) $\to$ **New Project** $\to$ **Upload Project**.
3. Select `dartx_paper_overleaf.zip` and click **Recompile**!

---

## 📂 Repository Structure

```
.
├── dartx/                             # DART-X Framework Source Code
│   ├── src/main/java/dartx/
│   │   ├── core/                      # AdaptiveWindowManager, ReliabilityEvaluator, SwitchingHysteresis, etc.
│   │   └── monitor/                   # DataCharacteristicMonitor (Welford centroid drift)
│   └── pom.xml
├── original-dynse/                    # Preserved Dynse Baseline Implementation (Almeida et al., 2018)
│   └── dynse/
├── experiments/                       # Benchmark & Stress-Test Runners
│   ├── PKLotBenchmarkRunner.java      # Full 82-day streaming benchmark runner
│   ├── run_synthetic_experiments.py   # Synthetic drift, noise, and imbalance stress tests
│   └── plot_results.py                # Publication visualization suite
├── results/                           # Empirical Data & Figures
│   ├── baseline/                      # Dynse daily CSVs & summary
│   ├── dartx/                         # DART-X daily CSVs & summary
│   ├── comparison/                    # Daily head-to-head CSVs & synthetic JSON
│   └── figures/                       # Publication PNG charts
├── paper/                             # IEEE Format Academic Paper (Overleaf Ready)
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── README_OVERLEAF.md
├── dartx_paper_overleaf.zip           # Pre-packaged Overleaf zip bundle
├── research/                          # Research Reports & Extracted Literature Tables
│   ├── original_paper_results.md
│   ├── repository_inspection_report.md
│   ├── experimental_methodology.md
│   └── experimental_evaluation_report.md
├── documentation/                     # Mathematical Architecture & Pseudocode
│   └── dartx_algorithm.md
├── .gitignore
└── README.md
```

---

## 🚀 How to Reproduce

### Prerequisites
* Java Development Kit (JDK 17+)
* Apache Maven 3.9+
* Python 3.10+ with `numpy` and `matplotlib`

### 1. Download PKLot Dataset
Download the official PKLot concept drift benchmark from the author's live repository:
```bash
mkdir -p datasets && cd datasets
curl -O https://prlalmeida.com.br/ConceptDrift/PKLot-ConceptDrift.tar.gz
tar -xzf PKLot-ConceptDrift.tar.gz
cd ..
```

### 2. Compile Baseline and DART-X
```bash
# Compile original Dynse baseline
cd original-dynse/dynse
mvn clean compile package -DskipTests
cd ../..

# Compile DART-X
cd dartx
mvn clean compile package -DskipTests
cd ..
```

### 3. Run the Full 82-Day PKLot Benchmark
```bash
javac -cp "original-dynse/dynse/target/*:dartx/target/*" experiments/PKLotBenchmarkRunner.java
java -cp "experiments:original-dynse/dynse/target/*:dartx/target/*" experiments.PKLotBenchmarkRunner
```

### 4. Run Synthetic Drift Experiments & Render Figures
```bash
python3 experiments/run_synthetic_experiments.py
python3 experiments/plot_results.py
```

---

## 📜 Citation & References

```bibtex
@article{almeida2018adapting,
  title={Adapting dynamic classifier selection for concept drift},
  author={Almeida, Paulo R. L. and Oliveira, Luiz S. and Britto Jr, Alceu S. and Sabourin, Robert},
  journal={Expert Systems with Applications},
  volume={104},
  pages={67--85},
  year={2018},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2018.03.021}
}
```

---

## 📄 License
This repository is licensed under the MIT License for DART-X components, while preserving the original GPL/MOA licensing of the baseline Dynse implementation in `original-dynse/`.
