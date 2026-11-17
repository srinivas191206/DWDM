# Original Paper Results & Benchmark Reference

**Paper:** Adapting Dynamic Classifier Selection for Concept Drift  
**Authors:** Paulo R. L. Almeida, Luiz S. Oliveira, Alceu S. Britto Jr., Robert Sabourin  
**Journal:** *Expert Systems with Applications*, Vol. 104, pp. 67–85, 2018  
**DOI:** [10.1016/j.eswa.2018.03.021](https://doi.org/10.1016/j.eswa.2018.03.021)  
**Official Repository:** `https://github.com/paulorla/dynse`  
**Official Project Page:** `https://web.inf.ufpr.br/vri/software/dynse` and `https://prlalmeida.com.br/pklotConceptDrift.html`  

---

## 1. Executive Summary & Core Contributions

The paper investigates extending **Dynamic Classifier Selection (DCS)** and **Dynamic Ensemble Selection (DES)** techniques from stationary environments to streaming environments undergoing concept drift. In stationary environments, DCS selects a classifier or ensemble based on local competence in the neighborhood of a query instance ($N_x$) within a static validation dataset ($Q$). 

In streaming data with concept drift, the authors propose **Dynse (Dynamic Selection Based Drift Handler)**:
1. **Time-Dependent Validation Window ($W$):** Modifies DCS by replacing static $Q$ with a sliding validation window $W$ containing the $M$ latest labeled batches ($|W| = M$).
2. **Spatiotemporal Diversity ("Concept Diversity"):** Maintains a pool of classifiers $P$ trained across different batches (and hence under different concepts). Since drifts may be localized, older classifiers trained under past concepts can remain valid in unshifted subregions of the feature space.
3. **Pruning Mechanism ($PE$):** Restricts unbounded pool growth to a maximum capacity $D$ (default $D = 25$) using strategies like Age-based pruning (evicting the oldest model) or Accuracy-based pruning (evaluating models on current window $W$).
4. **Drift-Adaptive Window Sizing:**
   - **Real Concept Drift ($P(y|x)$ changes):** Requires small $M$ (e.g., $M = 4$ batches) for fast eviction of obsolete validation labels.
   - **Virtual Concept Drift ($P(x)$ changes, $P(y|x)$ constant):** Requires large $M$ (e.g., $M = 32$ or $\infty$) to maximize feature-space coverage.
5. **PKLot Concept Drift Benchmark:** Establishes a standardized real-world concept drift benchmark using 695,899 parking spot images across three parking configurations (`UFPR04` $\to$ `UFPR05` $\to$ `PUCPR`) with uniform Local Binary Pattern (LBP) features.

---

## 2. Tested Algorithms & Taxonomy

### 2.1 Dynamic Classifier Selection Engines (Table 1 of Paper)

| Acronym | Description | Selection Mechanism | Reference |
| :--- | :--- | :--- | :--- |
| **OLA** | DCS-LA Overall Local Accuracy | Selects the single classifier with highest overall accuracy on $k$ nearest neighbors in $W$. | Woods et al. (1996) |
| **LCA** | DCS-LA Local Class Accuracy | Selects the classifier with highest accuracy on $k$ neighbors belonging to the class predicted by the classifier. | Woods et al. (1996) |
| **Priori** | A Priori DCS | Computes probability of correct classification via distance-weighted posterior probabilities of neighbors. | Giacinto & Roli (1999) |
| **Posteriori** | A Posteriori DCS | Uses Bayes' rule combining class priors and neighbor confidence distributions. | Giacinto & Roli (1999) |
| **K-U** | KNORA-Union | Selects all classifiers that correctly classify at least one neighbor in $N_x$; combines via majority vote. | Ko et al. (2008) |
| **K-UW** | KNORA-Union Weighted | Selects all classifiers that correctly classify at least one neighbor; weights votes inversely by distance. | Ko et al. (2008) |
| **K-E** | KNORA-Eliminate (Modified) | Selects classifiers correctly classifying at least $k - l$ neighbors ($l$ is slack variable). Increments $l$ if no model qualifies. | Almeida et al. (2016) |
| **Oracle** | Theoretical Upper Bound | Selects the correct classifier if at least one classifier in pool $P$ predicts the true class for test instance $x$. | Theoretical limit |

### 2.2 State-of-the-Art Baseline Comparison Methods (Table 2 of Paper)

| Acronym | Method | Category | Reference |
| :--- | :--- | :--- | :--- |
| **AUE** | Accuracy Updated Ensemble | Ensemble | Brzeziński & Stefanowski (2011) |
| **AWE** | Accuracy-Weighted Ensemble | Ensemble | Wang et al. (2003) |
| **ADACC** | Anticipative Dynamic Adaptation to Concept Change | Ensemble | Jaber et al. (2013) |
| **DDM** | Drift Detection Method (with pool of Hoeffding Trees) | Trigger | Gama et al. (2004) |
| **EDDM** | Early Drift Detection Method (with pool of Hoeffding Trees) | Trigger | Baena-García et al. (2006) |
| **HAT** | Hoeffding Adaptive Tree | Trigger | Bifet & Gavaldà (2007) |
| **LevBag** | Leveraging Bagging | Trigger/Ensemble | Bifet et al. (2010) |
| **OzaAD** | Oza Bagging with ADWIN Trigger | Trigger/Ensemble | Bifet et al. (2009) |
| **OzaAS** | Oza Bagging with Adaptive-Size Hoeffding Trees | Trigger/Ensemble | Bifet et al. (2009) |
| **ARF** | Adaptive Random Forests | Trigger/Ensemble | Gomes et al. (2017) |
| **NaiveComb** | Naive Combination (Majority vote of all accumulated models) | Baseline (Lower Bound) | Unadapted ensemble |

---

## 3. Benchmark Datasets & Experimental Protocols (Table 3 of Paper)

| Benchmark | Drift Type | Test Protocol | Features | Classes | Total Steps | Train Batch Size | Test Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STAGGER** | Abrupt real | Holdout | 3 | 2 | 40 | 20 | 200 |
| **SEA** | Abrupt real | Holdout | 3 | 2 | 200 | 250 | 250 |
| **SEARec** | Abrupt real (recurrent) | Holdout | 3 | 2 | 200 | 250 | 250 |
| **CkrC** | Gradual real (constant) | Holdout | 2 | 2 | 400 | 25 | 1024 |
| **CkrP** | Gradual real (pulse) | Holdout | 2 | 2 | 400 | 25 | 1024 |
| **CkrS** | Gradual real (sinusoidal) | Holdout | 2 | 2 | 400 | 25 | 1024 |
| **CkrE** | Gradual real (exponential) | Holdout | 2 | 2 | 400 | 25 | 1024 |
| **Gauss** | Gradual real (+/- class) | Holdout | 4 | 2 | 300 | 20 | 1024 |
| **Nebraska** | Real-world climate | Test-then-train | 8 | 2 | 604 | 30 | 30 |
| **Forest** | Real-world forestry | Holdout | 54 | 7 | 264 | 200 | 2000 |
| **Digit** | Virtual drift | Holdout | 64 | 10 | 56 | 50 | 50 |
| **Letters** | Virtual drift | Holdout | 16 | 26 | 200 | 50 | 50 |
| **PKLot** | Real-world camera/lot drift | Batch Holdout | 59 (LBP) | 2 | Daily steps | 100 (50/class) | Variable (all day) |

---

## 4. Published Numerical Results

### 4.1 Concept Reuse & Memory Validation on SEA Concepts (Table 4)

Proportion of classifiers trained under each concept selected to classify test instances under current concept (values in parentheses represent theoretical optimal proportion):

| Test Concept | Concept 1 | Concept 2 | Concept 3 | Concept 4 | Concept 1 (Recurrent) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train C1** | 100% (100%) | 46.5% (47.8%) | 34.5% (33.5%) | 23.3% (24.0%) | 26.3% (27.0%) |
| **Train C2** | 0.0% (0.0%) | 53.5% (52.2%) | 30.8% (30.4%) | 26.7% (26.4%) | 25.9% (24.7%) |
| **Train C3** | 0.0% (0.0%) | 0.0% (0.0%) | 34.7% (36.2%) | 21.3% (21.9%) | 23.8% (24.9%) |
| **Train C4** | 0.0% (0.0%) | 0.0% (0.0%) | 0.0% (0.0%) | 28.7% (27.7%) | 23.9% (23.4%) |
| **Mean Accuracy** | **97.2%** | **97.6%** | **97.7%** | **97.4%** | **97.9%** |

*Key finding:* Dynse accurately reuses models from previous concepts in subregions unaffected by drift, and performance on the recurrent concept improves over its initial presentation (97.9% vs. 97.2%).

---

### 4.2 Impact of Validation Window Size $M$ (Tables 5 & 6)

#### Real Concept Drift: SEA Concepts with Noise (Table 5)

| Classification Engine | $M = 1$ | $M = 4$ | $M = 8$ | $M = 16$ | $M = 32$ | $M = \infty$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K-E ($k=5, l=0$)** | 87.07% | **87.48%** | 87.38% | 86.98% | 85.95% | 84.22% |
| **K-U ($k=5$)** | 85.48% | 86.09% | **86.46%** | 86.33% | 85.54% | 84.09% |
| **NaiveComb** | 83.71% | 83.71% | 83.71% | 83.71% | 83.71% | 83.71% |
| **Oracle** | 93.90% | 93.90% | 93.90% | 93.90% | 93.90% | 93.90% |

#### Virtual Concept Drift: Digit Benchmark (Table 6)

| Classification Engine | $M = 1$ | $M = 4$ | $M = 8$ | $M = 16$ | $M = 32$ | $M = \infty$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K-E ($k=5, l=0$)** | 13.78% | 36.02% | 53.46% | 68.56% | 76.79% | **77.17%** |
| **K-U ($k=5$)** | 15.02% | 38.90% | 54.00% | 68.70% | 74.90% | **75.80%** |
| **NaiveComb** | 12.89% | 12.89% | 12.89% | 12.89% | 12.89% | 12.89% |
| **Oracle** | 86.15% | 86.15% | 86.15% | 86.15% | 86.15% | 86.15% |

---

### 4.3 Classification Engine Comparison Across 12 Benchmarks (Table 7)

Mean test batch accuracy (%) with standard deviation in parentheses (best non-Oracle results in bold):

| Benchmark | OLA | LCA | Priori | Posteriori | K-U | K-UW | K-E | Oracle | NaiveComb |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stagger** | 91.9 (17.7) | 75.2 (13.9) | 93.6 (12.0) | **93.9 (12.3)** | 86.1 (17.9) | 91.8 (14.3) | 92.0 (16.6) | 99.7 (1.2) | 66.3 (17.6) |
| **SEA** | 87.4 (1.3) | 83.5 (3.1) | 86.5 (1.4) | 86.9 (1.5) | 86.1 (1.9) | 86.3 (1.9) | **87.5 (1.4)** | 93.9 (1.4) | 83.7 (3.0) |
| **SEARec** | **87.1 (1.8)** | 83.5 (3.0) | 86.3 (1.6) | 86.6 (1.7) | 86.3 (2.0) | 86.3 (1.9) | **87.1 (1.8)** | 94.1 (1.4) | 83.8 (3.2) |
| **CkrC** | 85.6 (3.1) | 68.7 (7.4) | **86.4 (2.7)** | 83.9 (2.9) | 83.7 (3.5) | 86.1 (3.0) | 85.6 (3.1) | 99.9 (2.5) | 51.1 (8.6) |
| **CkrP** | 86.5 (4.9) | 68.4 (6.9) | **86.8 (4.8)** | 85.2 (5.0) | 85.1 (5.1) | 86.7 (4.7) | 86.5 (4.9) | 99.8 (2.8) | 49.2 (8.8) |
| **CkrS** | 86.2 (5.1) | 70.4 (9.8) | **86.7 (4.6)** | 84.2 (4.7) | 84.0 (5.3) | 86.3 (5.0) | 86.2 (5.1) | 99.6 (3.6) | 63.7 (18.9) |
| **CkrE** | 85.3 (3.3) | 64.0 (6.1) | **86.1 (3.5)** | 83.9 (3.4) | 83.2 (3.5) | 85.8 (3.5) | 85.3 (3.3) | 99.6 (3.8) | 51.7 (10.2) |
| **Gauss** | 89.5 (5.7) | 85.9 (8.1) | **90.1 (5.4)** | 89.9 (6.6) | 89.4 (7.1) | 89.5 (7.0) | 89.5 (5.7) | — | 78.4 (13.3) |
| **Nebr** | 74.5 (1.0) | 70.0 (1.1) | 73.7 (1.0) | 71.9 (1.1) | 74.4 (1.1) | **74.7 (1.0)** | 74.5 (1.0) | 97.2 (4.3) | 70.4 (1.1) |
| **For** | **78.3 (10.6)**| 50.2 (19.8)| 77.5 (10.5) | 76.3 (11.0) | 77.5 (11.1) | 77.8 (11.0) | **78.3 (10.6)**| 98.6 (7.4) | 68.8 (11.9) |
| **Dig** | 74.9 (20.6) | 11.2 (2.4) | 75.4 (20.1) | 72.6 (19.8) | 74.9 (22.1) | 74.5 (21.3) | **76.8 (20.5)**| 86.2 (24.4)| 12.9 (3.7) |
| **Let** | 66.2 (14.9) | 23.1 (8.4) | 64.6 (14.3) | 58.5 (11.6) | 67.1 (15.7) | **70.0 (16.1)**| 66.5 (14.0) | 93.2 (18.7)| 41.9 (16.0) |
| **Avg Rank**| 3.1 | 7.5 | 2.6 | 4.4 | 5.2 | 3.1 | **2.5** | — | 7.5 |

*Statistical Significance (Table 8):* Bonferroni-Dunn test ($\alpha = 0.05$, Critical Distance $CD = 2.69$) showed K-E, Priori, OLA, K-UW, and Posteriori form an equivalent clique. Pairwise tests with Bergmann-Hommel and Wilcoxon signed-ranks showed K-E significantly outperformed Posteriori ($p = 0.0117$) and LCA ($p < 0.01$).

---

### 4.4 Impact of Pruning Strategies: Infinite vs. Age vs. Accuracy (Table 9)

Maximum pool size $D = 25$ for pruned configurations:

| Benchmark | Infinite Accuracy | Infinite Memory | Age Prune Accuracy | Age Prune Memory | Accuracy Prune Accuracy | Accuracy Prune Memory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **STAGGER** | **92.0% (16.6)** | 0.24 MB | 91.4% (17.4) | 0.18 MB | 91.7% (17.2) | 0.18 MB |
| **SEA** | 87.5% (1.4) | 16.23 MB | 87.6% (1.5) | 2.77 MB | **87.8% (1.6)** | 2.80 MB |
| **SEARec** | 87.1% (1.8) | 16.17 MB | **87.2% (2.2)** | 2.77 MB | 87.1% (2.2) | 2.77 MB |
| **CkrC** | **85.6% (3.1)** | 2.22 MB | 83.9% (3.0) | 0.29 MB | 83.8% (3.1) | 0.31 MB |
| **CkrP** | **86.5% (4.9)** | 3.23 MB | 84.6% (4.8) | 0.29 MB | 84.9% (5.0) | 0.31 MB |
| **CkrS** | **86.2% (5.1)** | 3.20 MB | 84.3% (4.7) | 0.29 MB | 84.3% (4.8) | 0.30 MB |
| **CkrE** | **85.3% (3.3)** | 3.24 MB | 83.4% (3.5) | 0.30 MB | 83.5% (3.6) | 0.31 MB |
| **Gauss** | 89.5% (5.7) | 1.77 MB | 89.6% (5.8) | 0.34 MB | **89.7% (5.7)** | 0.35 MB |
| **Nebr** | 74.5% (1.0) | 8.19 MB | 74.1% (1.1) | 0.53 MB | **74.7% (0.9)** | 0.55 MB |
| **For** | **78.3% (10.6)** | 20.67 MB | 78.2% (10.8) | 4.89 MB | 78.2% (10.8) | 4.96 MB |
| **Dig** | **76.8% (20.5)** | 10.18 MB | 74.3% (20.7) | 8.21 MB | 68.0% (17.3) | 8.23 MB |
| **Let** | **66.5% (14.0)** | 15.85 MB | 60.3% (13.5) | 4.62 MB | 62.0% (12.7) | 5.24 MB |

*Statistical Significance (Table 10):* Infinite pool was significantly better than Age pruning ($p_{Berg} = 0.0963, p_{Wilcoxon} = 0.0107$). Between Age and Accuracy pruning, no statistically significant difference was detected ($p_{Wilcoxon} = 0.2168$). Age pruning ($D=25$) was selected as default due to its linear memory ceiling and simplicity.

---

### 4.5 State-of-the-Art Benchmark Comparison (Table 12)

Default Dynse Configuration: $M = 4$ (real drift) / $M = 32$ (virtual drift), $CE = \text{K-E}$ ($k=5, l=0$), $D = 25$, $PE = \text{Age}$.

| Benchmark | Dynse | AUE | AWE | ADACC | DDM | EDDM | HAT | LevBag | OzaAD | OzaAS | ARF | NaiveComb |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stagger** | 91.4 (17.4) | 93.7 (10.3) | 97.0 (3.0) | **97.7 (5.2)** | 93.4 (7.1) | 88.7 (16.3)| 66.3 (20.1)| 85.4 (20.3)| 79.8 (20.8)| 67.2 (19.1)| 91.0 (16.5)| 66.3 (17.6) |
| **SEARec** | **87.2 (2.2)** | 86.3 (1.5) | 86.7 (1.8) | 85.0 (1.7) | 84.4 (3.6) | 82.5 (4.5) | 84.9 (2.6) | 87.1 (2.4) | 85.7 (2.2) | 85.1 (2.2) | 86.2 (2.8) | 83.8 (3.2) |
| **CkrC** | **83.9 (3.0)** | 58.8 (7.5) | 57.3 (7.2) | 60.1 (7.9) | 51.0 (9.4) | 57.3 (8.6) | 55.4 (9.4) | 75.2 (6.4) | 57.3 (7.6) | 57.0 (13.1)| 69.2 (6.2) | 51.1 (8.6) |
| **CkrP** | **84.6 (4.8)** | 54.3 (7.2) | 54.2 (7.9) | 56.8 (7.7) | 51.4 (8.8) | 55.0 (6.8) | 56.4 (9.4) | 82.4 (11.8)| 59.5 (12.4)| 63.7 (13.6)| 79.2 (13.8)| 49.2 (8.8) |
| **CkrS** | **84.3 (4.7)** | 59.2 (9.4) | 56.2 (7.6) | 55.3 (6.7) | 46.2 (9.5) | 55.7 (7.1) | 58.4 (11.6)| 79.4 (10.6)| 50.7 (8.6) | 57.1 (10.9)| 75.7 (9.9) | 63.7 (18.9) |
| **CkrE** | **83.4 (3.5)** | 58.6 (7.8) | 57.8 (7.6) | 56.4 (6.9) | 57.9 (8.9) | 58.5 (7.8) | 59.0 (8.7) | 77.0 (7.9) | 59.0 (7.8) | 58.4 (10.4)| 75.3 (7.6) | 51.7 (10.2) |
| **Gauss** | 89.61 (5.8)| 90.5 (5.2) | 90.4 (5.5) | 89.2 (6.3) | **91.2 (7.3)**| 91.0 (7.0) | 61.3 (23.6)| 90.8 (5.5) | 90.2 (5.3) | 86.8 (9.0) | 88.9 (7.3) | 78.4 (13.3) |
| **Nebr** | 74.1 (1.1) | 73.7 (0.6) | 73.8 (0.8) | 71.7 (1.1) | 71.1 (1.0) | 67.7 (2.1) | 72.3 (2.2) | 77.0 (1.3) | 73.4 (1.5) | 73.5 (1.6) | **77.6 (2.2)** | 70.4 (1.1) |
| **For** | **78.2 (10.8)**| 73.7 (11.9)| 66.4 (15.8)| 64.1 (19.3)| 51.9 (24.9)| 46.2 (21.9)| 67.2 (12.5)| 75.1 (13.6)| 49.7 (19.8)| 74.2 (11.5)| 72.1 (13.3)| 68.8 (11.9) |
| **Let** | **60.3 (13.5)**| 58.4 (12.1)| 23.2 (10.1)| 31.4 (12.6)| 42.9 (17.4)| 42.3 (15.2)| 56.7 (11.3)| 56.8 (12.2)| 58.3 (11.5)| 57.2 (11.1)| 42.5 (9.8) | 41.9 (16.0) |
| **Avg Rank**| **2.2** | 4.7 | 7.0 | 7.7 | 8.6 | 8.4 | 8.1 | 3.0 | 7.0 | 6.8 | 4.6 | 10.0 |

*Statistical Significance (Fig. 11 & Table 13):* Bonferroni-Dunn test at $\alpha = 0.05$ ($CD = 4.58$) showed Dynse significantly outperformed 7 of 11 state-of-the-art baselines (DDM, EDDM, HAT, OzaAD, OzaAS, AWE, ADACC, NaiveComb). Pairwise Bergmann-Hommel testing confirmed Dynse was significantly superior to AUE ($p = 0.0194$) and ARF ($p = 0.0194$) at $\alpha = 0.05$.

---

### 4.6 PKLot Real-World Parking Lot Benchmark (Table 15)

Evaluated across UFPR04 $\to$ UFPR05 $\to$ PUCPR:

| Rank | Method | Category | Average Accuracy (%) | Std Dev (%) |
| :---: | :--- | :--- | :---: | :---: |
| **1** | **Dynse** | Dynamic Classifier Selection | **92.2%** | **8.7%** |
| 2 | Leveraging Bagging (LevBag) | Trigger/Ensemble | 89.7% | 13.1% |
| 3 | OzaAD | Trigger/Ensemble | 88.6% | 11.1% |
| 4 | OzaAS | Trigger/Ensemble | 88.3% | 10.9% |
| 5 | AUE | Ensemble | 87.1% | 14.9% |
| 6 | ARF | Trigger/Ensemble | 86.7% | 16.2% |
| 7 | DDM | Trigger | 85.5% | 10.2% |
| 8 | EDDM | Trigger | 85.4% | 10.3% |
| 9 | HAT | Trigger | 84.5% | 10.4% |
| 10 | AWE | Ensemble | 84.0% | 18.2% |
| 11 | NaiveComb | Baseline Ensemble | 81.4% | 11.7% |
| 12 | ADACC | Ensemble | 73.9% | 27.2% |

---

## 5. Distinction Between Published Results and Reproduced/Future Experiments

> [!IMPORTANT]
> **Strict Research Integrity Note:**
> All tables and numbers recorded above represent the peer-reviewed results published in Almeida et al. (*Expert Systems with Applications*, 2018).
>
> In our local test run of the compiled `original-dynse` repository using `StaggerTestbed` with 10 random seeds on OpenJDK 17:
> - Measured Mean Accuracy: **90.59%**
> - Minimum Batch Accuracy: **24.20%**
> - Published Paper Accuracy: **92.0% (16.6)**
>
> The local run confirms consistent behavior with the published dynamics: sharp drops during concept switches (at batches 10, 20, 30) followed by immediate recovery within 2–3 batches. Any future experiments run for DART-X will be clearly isolated and documented separately.

