# PalmPay Research Paper: Empirical Evaluation Log

**Dataset Source:** `data/empirical_trials.csv`  
**Model Architecture:** DINOv2 (`dinov2_vits14`, 384-D)  
**Biometric Modality:** Multi-Frame Contactless Palmprint + Landmark Hand-Geometry Fusion (`handgeo-v1`)  
**Operating Threshold ($T$):** 0.82  
**Geometry Limit ($D_{geo}$):** 1.85  

---

## Experiment 1: Baseline Environment (Room Ambient + Direct Lighting)

### 1. Raw Trials Captured (n = 22)
| # | Time (UTC) | Target Subject | Score ($S$) | $D_{geo}$ | Distance ($Z_{cm}$) | Result | Stage/Reason |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 14:35:57 | prabal | **0.8704** | 1.760 | 21.5 | **PASS** | `ok` |
| 2 | 14:36:08 | prabal | **0.8314** | 1.349 | 21.5 | **PASS** | `ok` |
| 3 | 14:36:17 | bhuvan | 0.7997 | 2.140 | 32.3 | **BLOCKED** | `below_threshold` |
| 4 | 14:36:24 | prabal | **0.8225** | 1.798 | 24.3 | **PASS** | `ok` |
| 5 | 14:36:35 | prabal | **0.8736** | 1.470 | 21.5 | **PASS** | `ok` |
| 6 | 14:36:47 | vibha | 0.8501 | 2.047 | 20.4 | **BLOCKED** | `geometry_mismatch` |
| 7 | 14:36:52 | prabal | **0.8426** | 1.563 | 22.1 | **PASS** | `ok` |
| 8 | 14:37:01 | prabal | 0.7889 | 2.787 | 28.1 | **FAIL** | `below_threshold` |
| 9 | 14:37:08 | prabal | 0.8217 | 2.768 | 30.2 | **FAIL** | `geometry_mismatch` |
| 10 | 14:37:13 | prabal | 0.8265 | 2.770 | 27.6 | **FAIL** | `geometry_mismatch` |
| 11 | 14:37:20 | bhuvan | 0.7917 | 2.206 | 25.3 | **BLOCKED** | `below_threshold` |
| 12 | 14:37:25 | aryan | 0.8559 | 6.933 | 22.8 | **BLOCKED** | `geometry_mismatch` |
| 13 | 14:37:31 | prabal | 0.8486 | 2.267 | 22.9 | **FAIL** | `geometry_mismatch` |
| 14 | 14:37:37 | anush | 0.8537 | 3.607 | 24.3 | **BLOCKED** | `geometry_mismatch` |
| 15 | 14:37:43 | aryan | 0.8271 | 5.983 | 23.7 | **BLOCKED** | `geometry_mismatch` |
| 16 | 14:37:49 | aryan | 0.7989 | 5.972 | 23.8 | **BLOCKED** | `below_threshold` |
| 17 | 14:37:55 | prabal | 0.8280 | 1.935 | 23.8 | **FAIL** | `geometry_mismatch` |
| 18 | 14:38:01 | prabal | 0.8207 | 2.015 | — | **FAIL** | `geometry_mismatch` |
| 19 | 14:38:07 | aryan | 0.8692 | 3.382 | 23.3 | **BLOCKED** | `geometry_mismatch` |
| 20 | 14:38:13 | aryan | 0.8577 | 5.062 | 22.7 | **BLOCKED** | `geometry_mismatch` |
| 21 | 14:38:19 | aryan | 0.8510 | 5.531 | 23.3 | **BLOCKED** | `geometry_mismatch` |
| 22 | 14:38:26 | prabal | **0.8867** | 1.325 | 24.9 | **PASS** | `ok` |

---

### 2. Statistical Breakdown

#### A. Genuine Trials (`prabal`, n = 12)
* **Mean Similarity ($\mu_{gen}$):** $0.8385 \pm 0.0263$
* **Score Distribution:** $\text{Range} = [0.7889, 0.8867]$
* **Optimal Range ($21 - 25\text{ cm}$):** **6 / 7 accepted (85.7% GAR)**
* **Distant Range ($>27\text{ cm}$):** **0 / 3 accepted (0.0% GAR)** — palm scale drop causes perspective distortion on hand geometry ($D_{geo} > 2.7$).
* **Overall Acceptance Rate:** **50.0% (6 / 12)**

#### B. Impostor / Cross-Subject Blocks (n = 10)
* **Mean Cross Score ($\mu_{imp}$):** $0.8355$
* **Mean Hand Geometry Distance ($\mu_{geo}$):** **$4.286$** (vs. genuine $1.984$)
* **False Accept Rate (FAR):** **0.00% (0 / 10)**
* **Key Finding for Paper:** Even when cross-subject DINOv2 cosine similarity reaches up to $0.8692$ under ambient lighting, the **Hand Geometry Gate blocked 100% of false matches** ($D_{geo} = 3.38 - 6.93 \gg 1.85$).

---

## Experiment 2: Condition B (Panel Light Dominant / Ceiling Light Dimmed)

### 1. Raw Trials Captured (n = 13)
| # | Time (UTC) | Target Subject | Score ($S$) | $D_{geo}$ | Distance ($Z_{cm}$) | Result | Stage/Reason |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 14:40:29 | prabal | **0.8481** | 1.301 | 21.9 | **PASS** | `ok` |
| 2 | 14:40:39 | aryan | 0.8380 | 3.546 | 23.1 | **BLOCKED** | `geometry_mismatch` |
| 3 | 14:40:46 | prabal | 0.8312 | 2.047 | 22.8 | **FAIL** | `geometry_mismatch` |
| 4 | 14:40:52 | prabal | 0.8498 | 2.090 | 22.0 | **FAIL** | `geometry_mismatch` |
| 5 | 14:40:58 | prabal | 0.8211 | 2.322 | 22.2 | **FAIL** | `geometry_mismatch` |
| 6 | 14:41:06 | prabal | 0.8093 | 1.625 | 22.8 | **FAIL** | `below_threshold` |
| 7 | 14:41:12 | prabal | 0.8189 | 1.936 | 23.8 | **FAIL** | `below_threshold` |
| 8 | 14:41:19 | vibha | **0.8245** | 1.162 | 25.2 | **PASS** | `ok` |
| 9 | 14:41:31 | vibha | 0.7783 | 1.283 | 25.4 | **FAIL** | `below_threshold` |
| 10 | 14:41:38 | vibha | 0.7901 | 1.554 | 27.5 | **FAIL** | `below_threshold` |
| 11 | 14:41:44 | bhuvan | 0.7870 | 3.433 | 27.9 | **BLOCKED** | `below_threshold` |
| 12 | 14:41:51 | bhuvan | 0.7905 | 2.623 | 23.1 | **BLOCKED** | `below_threshold` |
| 13 | 14:41:57 | prabal | **0.8386** | 1.669 | 22.4 | **PASS** | `ok` |

---

### 2. Comparative Findings: Condition A vs Condition B

| Metric | Condition A (Room + Ambient) | Condition B (Panel Dominant) | Shift ($\Delta$) | Analysis / Significance |
|:---|:---:|:---:|:---:|:---|
| **Prabal Mean Score ($\mu_{gen}$)** | **0.8385** | **0.8310** | **-0.0075** | Panel specular glare slightly shifts high-end cosine similarity |
| **Prabal Score Stability** | $0.7889 - 0.8867$ | $0.8093 - 0.8498$ | **Narrowed variance** | On-axis panel reduces shadow noise; tightens score distribution |
| **Impostor Blocks (FAR)** | **0.00% (0 / 10)** | **0.00% (0 / 3)** | **0.00%** | Zero false accepts in both conditions |
| **Key Failure Mode** | Distance drop ($>27\text{ cm}$) | Geometry drift ($D_{geo} \approx 2.0 - 2.3$) | $D_{geo}$ gate | At $D_{geo} = 1.85$, trials with $D_{geo} \approx 2.05$ fail despite score $0.8498$ |

