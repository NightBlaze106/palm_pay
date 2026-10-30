# PalmPay Research Paper: Empirical Evaluation Log & Metric Formulations

**Dataset Sources:** `data/empirical_trials.csv` (1,276 live physical presentations) & `database/palmpay.db` (26 enrolled subjects, 144 CCNet templates)  
**Primary Architecture (Production):** CCNet ONNX (`ccnet-onnx-v1`, 2048-D)  
**Secondary Architectures:** DINOv2 (`dinov2_vits14`, 384-D), Classical Fusion (`HogLbpFusion-v1-fm`, 3652-D)  
**Biometric Modality:** Multi-Frame Contactless Palmprint (Burst of 4) + Option B Score-Weighted Adaptive Hand Geometry (`handgeo-v1`)  
**Operating Threshold ($T$):** 0.73 (Multi-signal production) / 0.78 (Zero-FAR pure texture boundary for 26-subject cohort)  
**Adaptive Geometry Gates:** $D_{geo} \le 3.20$ (if $S \ge 0.76$), $D_{geo} \le 2.40$ (if $0.73 \le S < 0.76$)

---

## Mathematical Formulations & Biometric Metric Definitions

The quantitative metrics reported across all experiments adhere to ISO/IEC 2382-37 and ISO/IEC 19795 biometric evaluation standards.

### 1. Vector Normalization & Cosine Similarity
For any raw convolutional or transformer feature vector $\mathbf{x} \in \mathbb{R}^D$ (where $D=2048$ for CCNet, $D=384$ for DINOv2, $D=3652$ for Classical), the template is projected onto the unit hypersphere:
$$\mathbf{v} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^D x_i^2}}$$

The primary similarity score $S(\mathbf{q}, \mathbf{t})$ between query vector $\mathbf{q}$ and enrolled template $\mathbf{t}$ is the dot product in unit space:
$$S(\mathbf{q}, \mathbf{t}) = \frac{\mathbf{q} \cdot \mathbf{t}}{\|\mathbf{q}\|_2 \|\mathbf{t}\|_2} = \sum_{i=1}^D q_i t_i \quad \in [-1.0, 1.0]$$

### 2. Multi-Frame Temporal Fusion Embedding
Given a temporal burst of $K = 4$ video frames captured across a $\approx 220\text{ ms}$ interval, individual feature vectors $\{\mathbf{q}_1, \dots, \mathbf{q}_K\}$ are fused via $L_2$-normalized vector summation:
$$\bar{\mathbf{q}} = \frac{\sum_{k=1}^K \mathbf{q}_k}{\left\|\sum_{k=1}^K \mathbf{q}_k\right\|_2}$$

Matching requires **frame agreement voting**:
$$\text{Pass}_{\text{temporal}} \iff \left( S(\bar{\mathbf{q}}, \mathbf{t}) \ge T \right) \;\land\; \left( \sum_{k=1}^K \mathbb{I}[S(\mathbf{q}_k, \mathbf{t}) \ge T] \ge 2 \right)$$

### 3. Normalized Mahalanobis Hand Geometry Distance ($D_{geo}$)
Let $\mathbf{g} \in \mathbb{R}^{20}$ denote the query hand geometry vector (containing finger length-to-palm ratios, knuckle width proportions, and signed inter-finger splay angles). Given enrolled user geometry mean $\boldsymbol{\mu} \in \mathbb{R}^{20}$ and standard deviation $\boldsymbol{\sigma} \in \mathbb{R}^{20}$:
$$D_{geo}(\mathbf{g}, \boldsymbol{\mu}, \boldsymbol{\sigma}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} \left( \frac{g_i - \mu_i}{\max(\sigma_i, \epsilon)} \right)^2}$$
where $d = 20$ dimensions and $\epsilon = 10^{-6}$ provides numerical stabilization against zero-variance enrollment dimensions.

### 4. Option B Score-Weighted Adaptive Geometry Gate
Rather than a rigid single threshold, the decision boundary adapts based on texture confidence:
$$\text{Decision}(S, D_{geo}) = \begin{cases}
\text{ACCEPT} & \text{if } S \ge 0.76 \;\land\; D_{geo} \le 3.20 \quad \text{(High-confidence texture)} \\
\text{ACCEPT} & \text{if } 0.73 \le S < 0.76 \;\land\; D_{geo} \le 2.40 \quad \text{(Borderline texture)} \\
\text{REJECT} & \text{if } S < 0.73 \;\lor\; D_{geo} > D_{\text{adaptive}}
\end{cases}$$

### 5. Pinhole Projective Distance Estimation
From camera focal length $f_{px} = 554.0\text{ px}$ and canonical human palm breadth $W_{mm} = 78.0\text{ mm}$, physical distance $Z_{cm}$ is derived from the Euclidean landmark span between index MCP ($P_5$) and pinky MCP ($P_{17}$):
$$W_{px} = \|P_5 - P_{17}\|_2 = \sqrt{(x_5 - x_{17})^2 + (y_5 - y_{17})^2}$$
$$Z_{cm} = \frac{W_{mm} \times f_{px}}{W_{px} \times 10} = \frac{78.0 \times 554.0}{10 \cdot W_{px}} = \frac{4321.2}{W_{px}}$$
Valid capture requires $20.0\text{ cm} \le Z_{cm} \le 33.0\text{ cm}$.

### 6. Standard Biometric Error Metrics

Let $\mathcal{S}_{gen}$ be the set of genuine comparison scores and $\mathcal{S}_{imp}$ be the set of impostor comparison scores:

- **Genuine Accept Rate (GAR):**
  $$\text{GAR}(T) = \frac{|\{s \in \mathcal{S}_{gen} : s \ge T\}|}{|\mathcal{S}_{gen}|} \times 100\%$$

- **False Accept Rate (FAR):**
  $$\text{FAR}(T) = \frac{|\{s \in \mathcal{S}_{imp} : s \ge T\}|}{|\mathcal{S}_{imp}|} \times 100\%$$

- **False Reject Rate (FRR):**
  $$\text{FRR}(T) = \frac{|\{s \in \mathcal{S}_{gen} : s < T\}|}{|\mathcal{S}_{gen}|} \times 100\% = 100\% - \text{GAR}(T)$$

- **Separation Gap ($\Delta\mu$):**
  $$\Delta\mu = \mu_{gen} - \mu_{imp} = \frac{1}{|\mathcal{S}_{gen}|}\sum_{s \in \mathcal{S}_{gen}} s - \frac{1}{|\mathcal{S}_{imp}|}\sum_{s \in \mathcal{S}_{imp}} s$$

- **Equal Error Rate (EER):**
  $$\text{EER} = \text{FAR}(T^*) = \text{FRR}(T^*) \quad \text{where} \quad T^* = \arg\min_T |\text{FAR}(T) - \text{FRR}(T)|$$

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
* **Key Finding:** Even when cross-subject DINOv2 cosine similarity reaches up to $0.8692$ under ambient lighting, the **Hand Geometry Gate blocked 100% of false matches** ($D_{geo} = 3.38 - 6.93 \gg 1.85$).

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

### 2. Comparative Findings: Condition A vs Condition B
| Metric | Condition A (Room + Ambient) | Condition B (Panel Dominant) | Shift ($\Delta$) | Analysis / Significance |
|:---|:---:|:---:|:---:|:---|
| **Prabal Mean Score ($\mu_{gen}$)** | **0.8385** | **0.8310** | **-0.0075** | Panel specular glare slightly shifts high-end cosine similarity |
| **Prabal Score Stability** | $0.7889 - 0.8867$ | $0.8093 - 0.8498$ | **Narrowed variance** | On-axis panel reduces shadow noise; tightens score distribution |
| **Impostor Blocks (FAR)** | **0.00% (0 / 10)** | **0.00% (0 / 3)** | **0.00%** | Zero false accepts in both conditions |
| **Key Failure Mode** | Distance drop ($>27\text{ cm}$) | Geometry drift ($D_{geo} \approx 2.0 - 2.3$) | $D_{geo}$ gate | At $D_{geo} = 1.85$, trials with $D_{geo} \approx 2.05$ fail despite score $0.8498$ |

---

## Experiment 3: Model Architecture Benchmark — DINOv2 vs CCNet ONNX

**Evaluation Date:** October 13, 2026 (15:05 UTC)  
**Hardware / Pipeline:** Contactless RGB Webcam, Pinhole Distance Gated ($24.8 - 25.7\text{ cm}$), CLAHE $128\times128$ ROI  
**Subject Tested:** `prabal` (6 enrollment templates, 10 continuous probe trials)

### 1. Comparative Performance Matrix
| Metric | **DINOv2 (ViT-S/14)** [Port 5001] | **CCNet ONNX (TIFS 2021)** [Port 5002] | Shift ($\Delta$) / Significance |
|:---|:---:|:---:|:---|
| **Embedding Dimension** | 384-D | **2048-D** | $5.3\times$ higher representation capacity |
| **Input Shape** | $224\times224\times3$ RGB | **$128\times128\times1$ Grayscale** | Direct CLAHE ROI without RGB upscaling |
| **Inference CPU Latency** | $\approx 28-35\text{ ms}$ | **$\approx 24-31\text{ ms}$** | ONNX Runtime CPU execution |
| **Genuine Mean Score ($\mu_{gen}$)** | $0.8385 \pm 0.026$ | **$0.9101 \pm 0.029$** | **$+0.0716$ boost in genuine similarity** |
| **Genuine Score Range** | $[0.7889, 0.8867]$ | **$[0.8351, 0.9380]$** | High floor: 9 of 10 trials scored $>0.91$ |
| **First-Pass Accept Rate** | $50.0\% - 85.7\%$ | **$100.0\%$ (10 / 10)** | **Zero false rejects at $T = 0.82$** |
| **Intra-Template Consistency** | $0.8437$ | **$0.8939$** | Templates adhere closer in feature space |
| **Mean Hand Geometry Distance** | $1.984$ | **$0.654$** | Extreme pose stability ($0.399 - 1.473$) |

---

## Experiment 4: 7-Subject Multi-User Classroom Cohort Evaluation (CCNet ONNX)

**Evaluation Date:** October 14, 2026  
**Model Architecture:** CCNet ONNX (`ccnet-onnx-v1`, 2048-D)  
**Enrolled Subjects (n = 7, total 37 templates):** `ccbhuvan` (6), `ccjee` (6), `cckiki` (6), `ccpunii` (6), `ccviii` (5), `ccyagna` (4), `pppcc` (4)

| Metric | Pairwise Formulation | Leave-One-Out vs Mean Template (1:N Matching) |
|:---|:---:|:---:|
| **Genuine Sample Count ($N_{gen}$)** | 82 pairs | 37 probes |
| **Genuine Mean Score ($\mu_{gen}$)** | **$0.8518 \pm 0.1102$** | **$0.9076 \pm 0.0671$** |
| **Genuine Score Range** | $[0.5428, 0.9745]$ | $[0.6733, 0.9737]$ |
| **Impostor Sample Count ($N_{imp}$)** | 584 cross-pairs | 222 cross-comparisons |
| **Impostor Mean Score ($\mu_{imp}$)** | **$0.4497 \pm 0.0858$** | **$0.4775 \pm 0.0889$** |
| **Impostor Score Range** | $[0.2080, 0.6746]$ | $[0.2485, 0.7155]$ |
| **Absolute Impostor Ceiling** | **$0.6746$** | **$0.7155$** |
| **Separation Gap ($\Delta \mu$)** | **$+0.4021$** | **$+0.4301$** |

---

## Experiment 5: 26-Subject Expanded Multi-User Cohort Benchmark & Adaptive Hardening (CCNet ONNX)

**Evaluation Date:** October 30, 2026  
**Model Architecture:** CCNet ONNX (`ccnet-onnx-v1`, 2048-D fp32, ONNX Runtime CPU)  
**Hardware Platform:** Standard Consumer Laptop Webcam (720p RGB, Intel i3 CPU, no NIR/depth sensor)  
**Enrolled Cohort:** **26 unique subjects**, **144 biometric templates** enrolled in SQLite (`database/palmpay.db`) across 3 progressive enrollment sessions:
1. *Wave 1 (Oct 14):* Initial cohort (`pppcc`, `ccyagna`, `ccbhuvan`, `ccpunii`, `ccjee`) — 28 templates.
2. *Wave 2 (Oct 16):* Classroom batch (`ccviii`, `cckii`, `ccsaaaa`, `ccaryan`, `user1`, `xxvishak`, `ccbindushree`, `ccrohith`, `ccanu`, `ccmaurya`, `cctanush`, `cckeerthana`, `ccchinmayi`, `cckrupa`, `ccchaitra`, `ccsarvesh`, `ccdevansh`, `ccyogesh`) — 95 templates.
3. *Wave 3 (Oct 18):* Extended diversity batch (`ccdarshil`, `viii-mother`, `viii-grandmother`) — 18 templates.

> **Biometric Challenge Note:** Wave 3 deliberately enrolled biological relatives (`viii-mother`, `viii-grandmother`) of student `ccviii` to evaluate discrimination among genetically correlated palm morphologies and older populations with fainter epidermal creases.

### 1. Exhaustive Statistical Distributions (26 Subjects, 144 Templates)

| Evaluation Metric | Pairwise Cross-Template Analysis | Leave-One-Out (LOO) 1:N Simulation |
|:---|:---:|:---:|
| **Genuine Comparisons ($N_{gen}$)** | **333 intra-subject pairs** | 144 probe trials |
| **Genuine Mean Score ($\mu_{gen}$)** | **$0.8918 \pm 0.0789$** | — |
| **Genuine Score Distribution** | $\text{Min} = 0.5428, \;\text{Max} = 0.9870$ | — |
| **Impostor Comparisons ($N_{imp}$)** | **9,963 inter-subject pairs** | 144 LOO cross-evaluations |
| **Impostor Mean Score ($\mu_{imp}$)** | **$0.5158 \pm 0.0933$** | — |
| **Impostor Score Distribution** | $\text{Min} = 0.1805, \;\text{Max} = 0.7611$ | $\text{Max} = 0.7611$ |
| **Absolute Impostor Ceiling** | **$0.7611$** | **$0.7611$** |
| **Separation Gap ($\Delta \mu = \mu_{gen} - \mu_{imp}$)** | **$+0.3760$** (Vast discriminative margin) | — |
| **Equal Error Rate (EER)** | **$2.466\%$** at threshold $T^* = 0.6693$ | — |

### 2. Per-Subject Genuine Template Consistency Analysis

| Subject | Templates | Pairs | Mean $\mu_{gen}$ | Min Score | Standard Deviation ($\sigma$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `cckrupa` | 5 | 10 | **0.9599** | 0.9417 | $\pm 0.006$ |
| `ccsarvesh` | 6 | 15 | **0.9587** | 0.9004 | $\pm 0.024$ |
| `cctanush` | 5 | 10 | **0.9560** | 0.9439 | $\pm 0.007$ |
| `ccdevansh` | 6 | 15 | **0.9557** | 0.9204 | $\pm 0.016$ |
| `cckii` | 6 | 15 | **0.9454** | 0.8992 | $\pm 0.022$ |
| `ccyagna` | 4 | 6 | **0.9370** | 0.9174 | $\pm 0.012$ |
| `ccbindushree` | 6 | 15 | **0.9302** | 0.8776 | $\pm 0.028$ |
| `cckeerthana` | 5 | 10 | **0.9269** | 0.8594 | $\pm 0.038$ |
| `pppcc` | 4 | 6 | **0.9264** | 0.8985 | $\pm 0.023$ |
| `ccjee` | 6 | 15 | **0.9263** | 0.8603 | $\pm 0.033$ |
| `ccanu` | 5 | 10 | **0.9227** | 0.8782 | $\pm 0.030$ |
| `ccsaaaa` | 6 | 15 | **0.9178** | 0.8688 | $\pm 0.027$ |
| `viii-mother` | 6 | 15 | **0.9137** | 0.8475 | $\pm 0.035$ |
| `ccrohith` | 5 | 10 | **0.9078** | 0.8080 | $\pm 0.046$ |
| `ccdarshil` | 6 | 15 | **0.8989** | 0.8449 | $\pm 0.034$ |
| `user1` | 6 | 15 | **0.8929** | 0.7786 | $\pm 0.063$ |
| `ccaryan` | 4 | 6 | **0.8866** | 0.8315 | $\pm 0.032$ |
| `ccmaurya` | 6 | 15 | **0.8815** | 0.8256 | $\pm 0.043$ |
| `ccpunii` | 6 | 15 | **0.8760** | 0.7712 | $\pm 0.052$ |
| `ccyogesh` | 6 | 15 | **0.8738** | 0.6815 | $\pm 0.076$ |
| `ccchinmayi` | 5 | 10 | **0.8668** | 0.8132 | $\pm 0.043$ |
| `xxvishak` | 6 | 15 | **0.8623** | 0.8204 | $\pm 0.035$ |
| `ccviii` | 6 | 15 | **0.8447** | 0.7053 | $\pm 0.073$ |
| `ccchaitra` | 6 | 15 | **0.8414** | 0.6624 | $\pm 0.087$ |
| `viii-grandmother` | 6 | 15 | **0.7706** | 0.6109 | $\pm 0.079$ |
| `ccbhuvan` | 6 | 15 | **0.7146** | 0.5428 | $\pm 0.125$ |

### 3. Exhaustive Threshold Sweep & Operating Points (9,963 Impostor Pairs, 333 Genuine Pairs)

| Threshold ($T$) | GAR (%) | FAR (%) | FRR (%) | Margin vs Ceiling (0.7611) | Security Profile & Operating Context |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$0.65$** | 98.50% | 5.490% | 1.50% | $-0.1111$ | Permissive research mode |
| **$0.67$ ($T^*_{\text{EER}}$)** | 97.60% | 2.529% | 2.40% | $-0.0911$ | Theoretical Equal Error Rate Point |
| **$0.70$** | 96.10% | 0.642% | 3.90% | $-0.0611$ | Relaxed consumer retail mode |
| **$0.73$ (Active Terminal)** | **93.99%** | **0.281%** | **6.01%** | **$-0.0311$** | **Production threshold with Option B geometry gate** |
| **$0.75$** | 91.89% | 0.080% | 8.11% | $-0.0111$ | Conservative threshold |
| **$0.76$** | 91.29% | 0.010% | 8.71% | $-0.0011$ | Near-zero boundary |
| **$0.78$ (Zero-FAR Pure Texture)** | **89.79%** | **0.000%** | **10.21%** | **$+0.0189$** | **Absolute zero FAR in standalone texture space** |
| **$0.80$** | 89.79% | 0.000% | 10.21% | $+0.0389$ | Strict high-value payment |
| **$0.82$ (Prior Default)** | 86.79% | 0.000% | 13.21% | $+0.0589$ | Overly strict; induces unnecessary false rejects |

### 4. Live Physical Transactions Log Analysis (`data/empirical_trials.csv`)

Across development, calibration, and live multi-user trials, **1,276 live physical presentations** were executed at the terminal camera:
- **Recognized & Authorized (`ok`):** **`388` trials (30.4%)**, mean similarity score $= \mathbf{0.8512 \pm 0.0669}$.
- **Rejected & Blocked:** **`888` trials (69.6%)**, mean similarity score $= \mathbf{0.6400 \pm 0.1374}$.
- **Breakdown of Rejections:**
  - `below_threshold` ($S < T$): 798 presentations (62.5%) — primarily non-enrolled subjects, wrong palms, or deliberate impostor attack attempts.
  - `geometry_mismatch` ($D_{geo} > D_{\text{threshold}}$): 80 presentations (6.3%) — occurred predominantly under strict Option A before Option B adaptive gating was enabled.
  - `unstable_capture` (burst agreement $< 2$ frames): 10 presentations (0.8%) — rapid hand movement or camera auto-exposure flutter.
- **Option B Acceptance Impact:** Following the rollout of the Option B adaptive geometry gate, first-pass genuine transaction acceptance rose from **`28.6%` to `80.0%`**, completely resolving false geometry rejections while maintaining an empirical **`0.00%` False Accept Rate (FAR)** across all live physical interactions.
