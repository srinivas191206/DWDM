# DART-X Experimental Methodology & Protocol Specification

**Project:** DART-X (Drift-Aware Reliability and Transition Framework)  
**Primary Benchmark:** PKLot Parking Space Classification under Concept Drift  
**Baseline Model:** Dynse (Almeida et al., *Expert Systems with Applications*, 2018)  
**Date:** 2026-11-18  

---

## 1. Overview and Scientific Motivation

The university research assignment requires:
> *"Design an adaptive classification framework that dynamically selects or switches between classification models based on changing data characteristics. Explain the decision criteria."*

This research project evaluates Dynamic Classifier Selection (DCS) for non-stationary streaming environments and proposes **DART-X (Drift-Aware Reliability and Transition Framework)**. 

To ensure strict scientific reproducibility and experimental integrity, this methodology establishes:
1. An exact replication of the published experimental protocol from Almeida et al. (2018).
2. A controlled comparative framework where all models (Dynse baseline and DART-X) process the exact same sequence of data batches, feature vectors, random seeds, and training subsets.
3. Complete ablation testing to isolate the individual contributions of DART-X components.
4. Additional controlled synthetic experiments exploring specific drift dynamics (abrupt, gradual, noise, and class imbalance).

---

## 2. PKLot Streaming Benchmark Protocol

### 2.1 Dataset Characteristics & Temporal Ordering
- **Total Parking Spaces:** 619,466 individual segmented parking space image features across 82 days.
- **Feature Extraction:** Uniform Local Binary Patterns ($LBP^{u2}_{8,1}$) yielding 59 numeric histogram attributes normalized to $[0, 1]$.
- **Target Classes:** Binary occupancy status:
  - `-1`: Empty / Vacant space
  - `+1`: Occupied space
- **Chronological Sequence:**
  Data is presented in strict chronological order across three physical camera setups:
  1. **Phase 1: UFPR04 (Days 1 to 22)** — Static camera angle 1 on parking lot 1 (Curitiba, Brazil).
  2. **Phase 2: UFPR05 (Days 23 to 50)** — Sudden camera angle relocation on parking lot 1 (*Sensor Drift*).
  3. **Phase 3: PUCPR (Days 51 to 82)** — Relocation to a completely different university parking lot with distinct pavement texture, tree foliage, and lighting conditions (*Environment / Background Drift*).

### 2.2 Streaming Train-Test Simulation Protocol
To simulate an active operational system with a human-in-the-loop supervisor:
1. **Test Phase (Day $t$):** When Day $t$ arrives, **all** instances of Day $t$ are treated as unlabelled test instances and classified by the active model/ensemble. Daily performance metrics are computed over the complete unlabelled batch.
2. **Supervised Feedback & Update Phase (Day $t \to t+1$):** At the end of Day $t$, a human supervisor annotates a small budget of 50 positive instances and 50 negative instances ($N = 100$ total labeled instances) randomly sampled from Day $t$.
3. **Model Training:** These 100 labeled instances are fed to the framework:
   - A new base classifier (Hoeffding Tree) is trained on this batch.
   - The labeled batch enters the validation window $W$.
   - The pool pruning engine maintains pool size $|P| \le D$ ($D=25$).
4. **Day 1 Initialization:** At Day 1 ($t=1$), 50 instances per class from Day 1 are sampled to train the initial classifier, after which Day 1 is evaluated.

---

## 3. Evaluated Model Architectures

### 3.1 Original Dynse Baseline Configuration
Following Table 11 of Almeida et al. (2018):
- **Base Classifier ($BC$):** Hoeffding Tree (`moa.classifiers.trees.HoeffdingTree`).
- **Validation Window Size ($M$):** Fixed $M = 4$ batches ($400$ instances).
- **Classification Engine ($CE$):** KNORA-Eliminate with $k = 5$ nearest neighbors and slack variable $l = 0$.
- **Pool Capacity ($D$):** Maximum $D = 25$ classifiers.
- **Pruning Engine ($PE$):** Age-based pruning (FIFO eviction of oldest model).

### 3.2 DART-X Proposed Framework Configuration
DART-X introduces six modular improvements while preserving the same base learner (`HoeffdingTree`) and pool capacity ($D=25$):
1. **Data Characteristic Monitor:** Computes streaming centroid divergence $D_{\text{drift}} = \|\mu_t - \mu_{t-1}\|_2$ and error jump rate to classify drift state into `STABLE`, `SUSPECTED_DRIFT`, `DRIFT_CONFIRMED`, and `TRANSITION_RECOVERY`.
2. **Adaptive Window Manager:** Dynamically modulates window size $M(t) \in [1, 6]$ batches. Upon confirmed drift, $M$ collapses to $1$ to instantly purge misleading historical neighbors; during stable periods, $M$ expands up to $6$ to increase local density.
3. **Reliability-Based Classifier Evaluation:** Computes distance-weighted local calibration score $R_i(x) = \sum_{j=1}^k w_j \cdot P(y_j \mid x_j, C_i)$ replacing pure binary correctness hits.
4. **Switching Hysteresis:** Requires candidates to beat the incumbent model by a margin $h = 0.04$ ($4\%$) to suppress erratic model thrashing on ambiguous boundary samples.
5. **Model Transition Logger:** Records all switches, suppressed transitions, and stability metrics.
6. **Reliability Fallback:** Engages recency-weighted pool voting when maximum candidate reliability drops below safety threshold $\theta_{\text{safety}} = 0.35$.

---

## 4. Evaluation Metrics & Statistical Verification

For each day $t \in [1, 82]$:
- **Accuracy (%):** $\frac{TP + TN}{TP + TN + FP + FN} \times 100$
- **Precision (%):** $\frac{TP}{TP + FP} \times 100$
- **Recall (%):** $\frac{TP}{TP + FN} \times 100$
- **Macro F1-Score (%):** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Confusion Matrix:** Aggregated $TP, TN, FP, FN$.
- **Per-Location Accuracy:** Sub-averages for `UFPR04`, `UFPR05`, and `PUCPR`.
- **Recovery Time:** Number of days required to regain $\ge 90\%$ accuracy after a camera or lot transition.
- **Model Switching Frequency:** Total model switches vs. switches suppressed by hysteresis.
- **Computational Latency:** Inference time per day (ms) and per instance ($\mu$s).

---

## 5. Ablation Study Plan

To verify which components of DART-X are responsible for observed gains, four configurations are evaluated on identical splits:
1. **Full DART-X:** All modules active.
2. **DART-X (No Adaptive Window):** Fixed $M=4$ (monitoring, reliability, hysteresis, and fallback active).
3. **DART-X (No Hysteresis):** Hysteresis threshold $h=0$ (unconstrained switching).
4. **DART-X (No Reliability Scoring):** Pure binary hit count (KNORA-E selection without probability weighting).

---

## 6. Synthetic Controlled Experiments Plan

To test generalized robustness beyond PKLot, additional controlled experiments are run using:
1. **Abrupt Concept Drift:** STAGGER (abrupt rule inversion every 10 steps) and SEA Concepts (abrupt boundary shifts).
2. **Gradual Concept Drift:** Rotating Checkerboard (continuous rotation across 400 steps).
3. **Class Imbalance Stress Test:** Skewing class priors from 50:50 down to 10:90.
4. **Label Noise Stress Test:** Injecting 0%, 5%, 10%, and 20% random label noise.

