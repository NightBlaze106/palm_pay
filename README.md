# PalmPay — Contactless Palm Biometric Payment System

**Project:** PalmPay — Pay Without Your Phone (Contactless Palm Biometrics on RGB Camera)  
**Course Context:** 6th-Semester CSE Major Project · **Status:** Multi-Model Architecture (Classical Vision + Foundation Transformer + Domain Pretrained + Deep Metric Embeddings)

> **What it does:** a merchant enters an amount at a POS terminal, the customer shows their
> open palm to an ordinary webcam, and the system recognizes them and debits their wallet —
> automatically, in a few seconds, with no phone, card, or touch.

Companion docs:

| Doc | Contents |
|---|---|
| [`accuracy_report.md`](file:///home/nightblaze/Projects/palm_pay_4/accuracy_report.md) | Full architectural benchmark, comparative brand survey, 26-subject cohort analysis & metric formulas |
| [`RESEARCH_EXPERIMENTS.md`](file:///home/nightblaze/Projects/palm_pay_4/RESEARCH_EXPERIMENTS.md) | Exhaustive empirical evaluation log (Experiments 1–5), mathematical formulations & threshold sweeps |
| [`PROJECT_CONTEXT.md`](file:///home/nightblaze/Projects/palm_pay_4/PROJECT_CONTEXT.md) | Session-continuation context: architecture decisions, decision log, current state |
| [`ROADMAP.md`](file:///home/nightblaze/Projects/palm_pay_4/ROADMAP.md) | Product phases (Demo → Pilot → Robustness → Productization) and gates |
| [`AGENTS.md`](file:///home/nightblaze/Projects/palm_pay_4/AGENTS.md) | Dev-environment rules (venv discipline, node version manager, test policy) |

---

## 1. System Overview

```
                 PALMPAY END-TO-END ARCHITECTURE (payment era)

   MERCHANT POS (browser)                     CONSUMER (palm only)
   ┌────────────────────────┐                 ┌──────────────────┐
   │ amount + store select  │   640x480 RGB   │ presents open    │
   │ "Start Payment"        │◄────frames──────┤ palm to webcam   │
   └───────────┬────────────┘   base64 JPEG   └──────────────────┘
               │ POST /api/recognize {images:[burst of 4]}
               ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                FLASK BACKEND — BIOMETRIC PIPELINE                │
   │                                                                  │
   │  1. Hand Landmark Detection      (core/hand_detector.py)         │
   │     MediaPipe Tasks HandLandmarker → 21 landmarks                │
   │  1.5 Hand Geometry Features      (core/geometry.py)              │
   │     20-dim shape descriptor: ratios + signed splay angles        │
   │  2. Palm ROI + Affine Normalization (core/roi.py)                │
   │     wrist/MCP anchors → rotation+scale-normalized 128×128 crop   │
   │  3. Preprocessing                (core/preprocessing.py)         │
   │     grayscale → CLAHE(2.0,(8,8)) — Gaussian blur DISABLED        │
   │  4. Capture Quality Gate         (core/pipeline.py)              │
   │     rejects blur/glare/dark/too-far BEFORE scoring               │
   │  5. Finger Masking               (core/pipeline.py)              │
   │     rows above MCP knuckle line → feathered neutral fill         │
   │                                                                  │
   │  6. MULTI-MODEL FEATURE EXTRACTION (core/palm_model.py)          │
   │     ├─► [CCNET] (PRODUCTION) Palmprint ONNX (2048-D):            │
   │     │   CCNet (Yang et al., TIFS 2023) ONNX Runtime CPU (~26ms)  │
   │     ├─► [DINOv2] Foundation Vision Transformer (384-D):          │
   │     │   ViT-S/14 self-supervised representation via PyTorch      │
   │     ├─► [CLASSICAL] Hand-Crafted Fusion (3652-D):                │
   │     │   HOG (1764-D) + MB-uLBP (1888-D) [CompCode 1536-D opt]   │
   │     └─► [DEEPPALM] Custom Inverted-Residual Embedder (512-D):    │
   │         MobileNetV3-Small + ArcFace via ONNX Runtime CPU (<15ms) │
   │                                                                  │
   │  7. Hardened Matching Engine     (core/matcher.py)               │
   │     • Option B Adaptive Geometry Gate (handgeo-v1):              │
   │       D_geo ≤ 3.20 (if S ≥ 0.76) or D_geo ≤ 2.40 (0.73 ≤ S < 0.76)│
   │     • Multi-candidate ambiguity guard (margin 0.02)              │
   │  8. Multi-Frame Fusion           (app.py /api/recognize)         │
   │     Burst of 4 frames + L2-normalized mean query vector          │
   │     ≥2 of 4 frames must individually agree above threshold       │
   └───────────┬─────────────────────────────────────────────────────┘
               │ success ⇒ server-side biometric session (opaque token,
               │          30 s TTL, single-use) — never a user_id from JS
               ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │            PAYMENT LEDGER (SQLite, database/db.py)               │
   │  /api/payment → atomic debit/credit + transaction row + receipt │
   │  wallets · top-up requests · merchant cash-outs · audit rows     │
   │  Raw palm images are NEVER stored — only float32 embeddings.     │
   └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | **Python 3.11** (managed via `mise`), Flask 3.x | CPU-only inference, fast prototyping |
| Hand tracking | **MediaPipe Tasks `HandLandmarker`** (`mediapipe>=0.10,<1.0`) | Real-time 21-landmark detection on CPU |
| Classical Vision | **OpenCV 4.x, NumPy** | Affine warps, CLAHE, HOG line gradients, MB-uLBP |
| Foundation Vision Model | **PyTorch / Torchvision (`torch.hub`)** | DINOv2 ViT-S/14 self-supervised representation (384-D) |
| Pretrained Domain Model | **ONNX Runtime (`onnxruntime>=1.17.0`)** | CCNet (Yang et al., 2023) 2048-D palmprint embeddings |
| Model Distribution | **Hugging Face Hub (`huggingface_hub`)** | Automated download of CCNet ONNX weights (~257 MB) |
| Deep Learning (Serving) | **ONNX Runtime (CPU)**, `DeepPalmModel` | Lightweight 512-D neural embeddings in <15 ms |
| Deep Learning (Training) | **PyTorch (CPU-only)**, `training/` | MobileNetV3-Small + ArcFace loss, disjoint validation |
| Liveness Detection | **MediaPipe 3D Landmark Sequences** | Dynamic challenge-response anti-spoofing (`OPEN_PALM`, `FIST`, `SPREAD`) |
| C++ Acceleration | **C++17 Native Core** (CMake, optional) | SIMD-friendly matching, feature extraction fallback |
| Storage | SQLite (single file, WAL-free demo mode) | Model-isolated template tables, zero-config kiosk |
| Frontend | Vanilla HTML/CSS/JS, WebRTC `getUserMedia`, Fetch API | No build step, no CDN dependency, runs offline |
| Hardware target | Intel i3-class laptop, 8 GB RAM, any USB RGB webcam | Deliberately low-cost, no GPU / NIR required |

> ⚠️ **mediapipe is pinned `<1.0`:** the 1.x series' bundled TF-Lite runtime is killed by the
> kernel during XNNPACK init on current Arch kernels (Tiger Lake). 0.10.35 is verified good.
> See `AGENTS.md` for environment rules.

---

## 3. The Recognition Pipeline (algorithm details)

### Stage 1 — Hand landmark detection
MediaPipe Tasks `HandLandmarker` (IMAGE mode) returns 21 3D landmarks per frame.
Anchors used downstream: **wrist (0)**, **index MCP (5)**, **middle MCP (9)**,
**ring MCP (13)**, **pinky MCP (17)**. Handedness is ignored (matching must be
hand-agnostic).

### Stage 2 — Palm ROI extraction & affine normalization (`core/roi.py`)
All geometry is derived from landmarks, so no pegs/guides hardware is needed:

$$C_{palm} = 0.45 \cdot P_0 + 0.55 \cdot \tfrac{P_5 + P_{17}}{2} \qquad \vec{v} = P_9 - P_0 \qquad \theta = \operatorname{atan2}(dy, dx) - 90^\circ$$

$$S = \max\big(\lVert P_5 - P_{17}\rVert,\; 0.75\lVert\vec{v}\rVert\big) \cdot 1.05$$

A single affine transform (rotation θ, scale $\text{roi\_width}/2S$) warps the frame onto a
canonical **128×128** canvas (`BORDER_REPLICATE`). The same matrix is reused to project the
four MCP joints into ROI coordinates, yielding the **knuckle cut-line** `knuckle_y_roi`
consumed by Stage 5.

### Stage 3 — Preprocessing (`core/preprocessing.py`)
Grayscale → **CLAHE** (clipLimit 2.0, tile 8×8) to amplify principal flexion creases under
variable illumination. **Gaussian smoothing is disabled** (`GAUSSIAN_BLUR_KERNEL=(0,0)`):
measured impostor scores reached ~0.80 with blur enabled because it erased the exact
micro-texture that separates palms. Noise is instead controlled by Stage 4 + Stage 8.

### Stage 4 — Capture quality gate (`FrameQualityGate`, `core/pipeline.py`)
Frames are rejected *before feature extraction*, with human-readable coaching messages
shown live in the UI:

| Check | Threshold / Criteria | Mathematical Metric | Rejection message |
|---|---|---|---|
| Blur (Dual Tenengrad) | Tenengrad Sobel $\ge 60.0$ | $\frac{1}{N}\sum (\nabla_x^2 + \nabla_y^2)$ | "Image too blurry — hold palm steady" |
| Blur (Global Laplacian) | Variance $\ge 18.0$ | $\operatorname{Var}(\nabla^2 I)$ | "Image too blurry — hold palm steady" |
| Blur (Patch Laplacian) | Min 3×3 patch var $\ge 16.0$ | $\min_{p} \operatorname{Var}(\nabla^2 I_p)$ | "Image too blurry — hold palm steady" |
| Glare (Over-exposure) | Blown-out fraction $\le 8\%$ | $\frac{|\{p : I(p) > 245\}|}{N} \le 0.08$ | "Glare detected — tilt palm away" |
| Mean brightness | Grayscale mean $\in [40, 232]$ | $\mu_I \in [40, 232]$ | "Too dark" / "Over-exposed" |
| Pinhole hand distance | Strictly within $[20.0, 33.0]\text{ cm}$ | $Z_{cm} = \frac{78.0 \times 554.0}{10 \cdot W_{px}}$ | "Move closer ($<33\text{cm}$)" / "Move back ($>20\text{cm}$)" |

### Stage 5 — Finger masking (`apply_finger_mask`)
Rows **above** the projected knuckle line are replaced with a neutral fill equal to the mean
brightness of the kept palm area, using a Gaussian-feathered mask edge (σ=3) and byte-exact
rounding below the cut. Purpose: finger spread / pitch stops leaking into block histograms —
a measured cause of room→hall recognition drift. Applied identically at enrollment and
recognition (single code path ⇒ consistent by construction). Config: `FINGER_MASK_ENABLED`,
`FINGER_MASK_OFFSET_PX`.

### Stage 6 — Feature extraction: HOG + Multi-Block Uniform LBP Fusion (`core/features.py`)
The classical pipeline fuses macroscopic crease edge gradients with microscopic uniform texture patterns into a joint **3652-D** feature vector:

1. **Multi-Scale Uniform LBP (1888-D):**
   For each radius $R \in \{1, 2\}$ the enhanced ROI is coded with the operator:
   $$LBP_{8,R}^{u2}(c) = \sum_{p=0}^{7} s(g_p - g_c)\,2^{\,p}, \qquad s(x)=\begin{cases}1 & x \ge 0\\ 0 & \text{else}\end{cases}$$
   Mapped through a 256→59 uniform lookup table (58 uniform patterns + 1 non-uniform bin) across a **4×4 spatial grid** ($4 \times 4 \times 59 \times 2 = 1888\text{-D}$).

2. **Histogram of Oriented Gradients / HOG (1764-D):**
   Extracts directional line gradients over $8\times8$ pixel cells with $2\times2$ cell block normalization and 9 orientation bins ($7 \times 7 \times 4 \times 9 = 1764\text{-D}$), capturing strong principal lines and crease topology.

3. **Concatenation & Normalization:**
   The block-normalized sub-vectors are concatenated and globally $L_2$ normalized:
   $$\mathbf{v} = [\mathbf{v}_{\text{HOG}} \,\|\, \mathbf{v}_{\text{LBP}}] \in \mathbb{R}^{3652}, \qquad \lVert\mathbf{v}\rVert_2 = 1$$

*Alternate baselines:* CompCode (1536-D Gabor orientation coding) and pure MB-uLBP (1888-D) remain fully supported in `config.py`.

### Stage 6b — Deep Learning Palm Embedder (`DeepPalmModel`, `core/palm_model.py`)
When `DEEP_ENABLED=True` in `config.py`, the pipeline switches from hand-crafted texture filtering to neural representation learning:

1. **Convolutional Backbone (`training/model.py`):**
   Uses **MobileNetV3-Small** with inverted residual blocks and Squeeze-and-Excitation (SE) channel attention for high-speed feature extraction on standard CPUs.
2. **Channel Expansion & Normalization:**
   Replicates the 128×128 enhanced grayscale crop to 3 channels and applies symmetric $[-1.0, 1.0]$ normalization.
3. **512-D L2-Normalized Embedding:**
   The neural network maps the palm image directly onto a 512-dimensional hypersphere:
   $$\mathbf{z} = \frac{f_{\theta}(\mathbf{x})}{\lVert f_{\theta}(\mathbf{x})\rVert_2} \in \mathbb{R}^{512}, \qquad \lVert\mathbf{z}\rVert_2 = 1$$
4. **ONNX Runtime Serving:**
   Inference executes on CPU via ONNX Runtime (`models/palm_embedder.onnx`) in **<15 ms**, requiring no discrete GPU or external cloud API.

### Stage 6c — Foundation Vision Transformer: DINOv2 ViT-S/14 (`DinoV2Model`, `core/palm_model.py`)
When `ACTIVE_MODEL="dinov2"` in `config.py`, the pipeline leverages Meta's self-supervised DINOv2 foundation model:

1. **Vision Transformer Architecture:**
   Uses `dinov2_vits14` (Vision Transformer Small with $14\times14$ patch size, 21M parameters) loaded seamlessly via `torch.hub`.
2. **Resolution & Color Adaptation:**
   The $128\times128$ CLAHE grayscale crop is upsampled to $224\times224$ (`cv2.INTER_LINEAR`), expanded to 3 RGB channels, scaled to $[0.0, 1.0]$, and standardized with standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
3. **384-D [CLS] Hypersphere Projection:**
   The network extracts the 384-dimensional `[CLS]` token embedding, followed by explicit $L_2$ unit normalization:
   $$\mathbf{z} = \frac{f_{\text{DINO}}(\mathbf{x})}{\lVert f_{\text{DINO}}(\mathbf{x})\rVert_2} \in \mathbb{R}^{384}, \qquad \lVert\mathbf{z}\rVert_2 = 1$$
4. **Foundation Generalization:**
   Trained on 142M curated natural images (LVD-142M), DINOv2 yields robust semantic features that generalize across sensor noise and subtle palm rotation without requiring specialized palmprint supervision.

### Stage 6d — Domain-Specific Pretrained Palmprint CNN: CCNet ONNX (`CcnetOnnxModel`, `core/palm_model.py`) — PRODUCTION
When `ACTIVE_MODEL="ccnet"` in `config.py` (**production default**), the pipeline routes inference to a specialized palmprint neural network:

1. **Network Architecture & Paper:**
   Implements **CCNet** (Yang et al., *"CCNet: A Cross-layer Contrastive Network for Contactless Palmprint Recognition"*, IEEE TIFS 2023), pre-trained on the benchmark Tongji contactless RGB palmprint dataset.
2. **Tensor Formatting:**
   Ingests the $128\times128$ CLAHE grayscale crop directly as a single-channel float32 tensor of shape $[1, 1, 128, 128]$ with pixel values scaled to $[0.0, 1.0]$ (no duplicate CLAHE filtering applied).
3. **2048-D Normalized Embedding:**
   Produces a 2048-dimensional dense feature representation with $L_2$ unit normalization:
   $$\mathbf{z} = \frac{f_{\text{CCNet}}(\mathbf{x})}{\lVert f_{\text{CCNet}}(\mathbf{x})\rVert_2} \in \mathbb{R}^{2048}, \qquad \lVert\mathbf{z}\rVert_2 = 1$$
4. **ONNX CPU Serving:**
   Runs via ONNX Runtime CPU (`models/palm_ccnet.onnx`, ~257 MB fp32 weights downloaded via `scripts/download_ccnet.py`). Latency is ~24–28 ms on commodity laptop CPUs.

### Stage 7 — Hardened 1:N Matching & Option B Adaptive Geometry Gate (`core/matcher.py`)

Cosine similarity between $L_2$-normalized vectors is computed as the dot product:
$$S(\mathbf{q}, \mathbf{t}) = \sum_{i=1}^D q_i t_i \quad \in [-1.0, 1.0]$$

In addition to texture correlation, the system extracts a **20-dimensional hand-geometry vector** $\mathbf{g}$ (finger length-to-palm ratios, knuckle width proportions, and signed splay angles) and evaluates the normalized Mahalanobis distance:
$$D_{geo}(\mathbf{g}, \boldsymbol{\mu}, \boldsymbol{\sigma}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}\left(\frac{g_i - \mu_i}{\max(\sigma_i, \epsilon)}\right)^2}$$

#### Option B Adaptive Gate Decision Rule:
Rather than a fixed rigid cutoff that rejects natural genuine hand relaxation, PalmPay deploys the **Option B Score-Weighted Adaptive Gate**:
$$\text{Decision} = \begin{cases}
\text{ACCEPT} & \text{if } S \ge 0.76 \;\land\; D_{geo} \le 3.20 \quad \text{(High-Confidence Texture)} \\
\text{ACCEPT} & \text{if } 0.73 \le S < 0.76 \;\land\; D_{geo} \le 2.40 \quad \text{(Borderline Texture)} \\
\text{REJECT} & \text{if } S < 0.73 \;\lor\; D_{geo} > D_{\text{threshold}}
\end{cases}$$

Additionally, if top-1 and top-2 candidates both qualify within `MATCHER_AMBIGUITY_MARGIN` (0.02), the match is **rejected as ambiguous** rather than guessed.

### Stage 8 — Multi-frame Temporal Burst Fusion (`/api/recognize`)
POS sends a burst of $K = 4$ frames (~220 ms temporal window). Recognition evaluates:

1. **Temporal Vector Average:** Computes $L_2$-normalized mean query embedding:
   $$\bar{\mathbf{q}} = \frac{\sum_{k=1}^K \mathbf{q}_k}{\left\|\sum_{k=1}^K \mathbf{q}_k\right\|_2}$$
2. **Frame Agreement Voting:** Requires at least $2$ of the $4$ individual frames to independently clear threshold $T$:
   $$\sum_{k=1}^4 \mathbb{I}[S(\mathbf{q}_k, \mathbf{t}) \ge T] \ge 2$$
Otherwise the response returns `match_reason: unstable_capture` and the POS retries automatically.

### Threshold Calibration Policy & Mathematical Metric Formulas

All biometric evaluation formulas adhere strictly to international standards:
- **Cosine Similarity:** $S(\mathbf{q}, \mathbf{t}) = \frac{\mathbf{q} \cdot \mathbf{t}}{\|\mathbf{q}\|_2 \|\mathbf{t}\|_2}$
- **Genuine Accept Rate:** $\text{GAR} = \frac{\text{Genuine Accepts}}{\text{Total Genuine Probes}} \times 100\%$
- **False Accept Rate:** $\text{FAR} = \frac{\text{False Accepts}}{\text{Total Impostor Probes}} \times 100\%$
- **False Reject Rate:** $\text{FRR} = 100\% - \text{GAR}$
- **Separation Gap:** $\Delta\mu = \mu_{gen} - \mu_{imp}$
- **Equal Error Rate (EER):** Point where $\text{FAR}(T) = \text{FRR}(T)$

#### Production Calibrated Findings (26 Enrolled Subjects, 144 CCNet Templates, 1,276 Physical Trials)
- **Genuine Pairwise Matches ($N=333$):** $\mu_{gen} = \mathbf{0.8918 \pm 0.0789}$, Range: $[0.5428, 0.9870]$
- **Impostor Pairwise Matches ($N=9,963$):** $\mu_{imp} = \mathbf{0.5158 \pm 0.0933}$, Range: $[0.1805, 0.7611]$
- **Separation Gap ($\Delta\mu$):** **`+0.3760`** (Clean separation margin; $>9\times$ wider than classical LBP)
- **Absolute Impostor Ceiling:** **`0.7611`**
- **Operating Decision Point:**
  - **$T = 0.73$ (Active Terminal):** $\text{GAR} = 93.99\%$, combined with Option B adaptive geometry gate to maintain **$0.00\%$ live FAR**.
  - **$T = 0.78$ (Zero-FAR Pure Texture):** $\text{GAR} = 89.79\%$, $\text{FAR} = 0.000\%$, operating with $+0.0189$ safety buffer above the highest impostor.
  - **Equal Error Rate (EER):** $\mathbf{\approx 2.47\%}$ at threshold $T \approx 0.669$.
- **Live Physical Presentations:** **1,276 logged trials** in `data/empirical_trials.csv` (388 authorized, 888 rejected; 0.00% live false accepts).

---

## 4. Automated Enrollment (`/enroll`)

Fingerprint-setup style — the user never clicks per-sample:

1. Enter name + USN → **🚀 Start Enrollment Scan** (camera opens only for the scan).
2. A 400 ms probe loop sends low-quality preview frames to `/api/process_frame`; the quality
   gate's messages coach positioning live ("move closer", "too blurry").
3. When aligned + quality-passed, a high-quality (JPEG q=0.92) frame is auto-captured;
   **1.3 s cooldown**, then the next prompt appears.
4. **Condition script** (6 samples): neutral baseline → tilt LEFT → tilt RIGHT → move closer →
   same-spread reminder → flat-palm reminder. **Pitch/foreshortening prompts were removed in
   Sprint v4**: out-of-plane rotation changes projected hand geometry and was measured to
   destroy geometry-template consistency (session drift was the dominant error source).
5. **Near-duplicate guard:** each candidate is diffed against the previous sample on a 32×32
   grayscale snapshot (< 1.5% mean-abs difference ⇒ skipped with "vary a bit more").
6. **Pose-consistency gate (server-side):** samples whose geometry deviates more than
   `ENROLL_GEO_CONSISTENCY_MAX_DIST` (absolute physical units, per-dim median) from the
   enrollment's median are auto-dropped; the enrollment is rejected with coaching when fewer
   than `MIN_ENROLLMENT_SAMPLES` survive. Texture+geometry are always dropped as a pair.
7. Auto-submit to `/api/enroll` → wallet auto-created → done.

Config: `DEFAULT_ENROLLMENT_SAMPLES=6` (min 3 / max 8), `ENROLL_CAPTURE_COOLDOWN_MS`,
`ENROLL_POSE_PROMPTS`, `ENROLL_DUPLICATE_DIFF_PCT`, `ENROLL_GEO_CONSISTENCY_MAX_DIST`.

---

## 5. Payment System (Demo / Prototype)

### Security model
- **Merchant-side POS only.** Customers never operate a device to pay.
- After a successful match, the backend stores the identity in a **server-side session store**
  (`bio_session.py`) keyed by an opaque token held in the browser's Flask session cookie.
  **30 s TTL · single-use** (burned on successful payment) · invalidated by any failed scan.
- `/api/payment` resolves the payer **only** from that session — a client-sent `user_id` is
  ignored (regression-tested).
- Amount validated server-side: numeric, > 0, ≤ ₹5000. Failure paths are audit-logged
  (`FAILED_AMOUNT`, `FAILED_BALANCE`); unknown identities log nothing.

### Flows
| Actor | Flow |
|---|---|
| Merchant | `/merchant?merchant_id=m1&view=pos` or standalone `/pay` → pick store, enter amount, Start Payment → customer palm → auto-debit → success screen + receipt link. Camera lives only while the sale is active. |
| Consumer | `/consumer?user_id=N` — read-only wallet: balance, history, **top-up request** (admin-approved). No pay button anywhere. |
| Merchant cash-out | Console tab → request → logged `PENDING` until admin approves (simulated settlement). |
| Admin | `/admin/login` (`admin` / `palmpay2025` demo cred): direct top-ups, approve consumer top-up requests, approve cash-outs, system stats. |

Ledger: every transaction row records type (PAYMENT/TOPUP/CASHOUT), status, amount, parties,
and the **biometric score** — this log is the pilot-phase evaluation dataset (data flywheel).

---

## 6. Research Basis & References

1. **Zhang, D., Kong, W.-K., You, J., & Wong, M. (2003).** *Online palmprint identification.*
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(9), 1041–1050.
   → Establishes palmprint (principal lines, wrinkles, ridges) as a distinct biometric modality.
2. **Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002).** *Multiresolution gray-scale and rotation
   invariant texture classification with local binary patterns.* IEEE TPAMI, 24(7), 971–987.
   → The $LBP_{P,R}^{u2}$ operator and its multiresolution extension — basis for our R=(1,2)
   concatenation and the 59-bin uniform mapping (>90% of texture energy lands in uniform patterns).
3. **Zhang, L., Chu, R., Xiang, S., Liao, S., & Li, S. Z. (2007).** *Face Detection Based on
   Multi-Block LBP Representation.* International Conference on Biometrics (ICB).
   → Block-wise LBP histograms (our 4×4 spatial grid) capture coarse structure while staying
   robust to micro-pose noise.
4. **Amrouni, M., et al. (2022).** *Contactless palmprint preprocessing and recognition.*
   Journal of Biometrics.
   → CLAHE on contactless ROIs enhances principal lines under variable ambient illumination.
5. **Zhang, F., Bazarevsky, V., Vakunov, A., et al. (2020).** *MediaPipe Hands: On-device
   real-time hand tracking.* arXiv:2006.10214.
   → Real-time 21-landmark estimation on CPU; provides our geometric anchor points.
6. **Sánchez-Brizuela, G., et al. (2023).** *Landmark-based geometric normalization for
   contactless hand/palm biometric systems.* Sensors.
   → Wrist→middle-MCP axis formulation for rotation/scale-invariant ROI extraction.
7. **Zhang, D., Guo, Z., Lu, G., Zhang, L., & Wang, Y. (2010).** *Online finger-knuckle-print
   verification for personal authentication.* Pattern Recognition, 43(4).
   → Scientific grounding for the Phase-3 multi-modal roadmap item (palmprint +
   knuckle-print fusion); knuckle texture is RGB-camera-friendly unlike dorsal veins (NIR).
8. **Ross, A. & Jain, A. K. (2003).** *Information fusion in biometrics.* Pattern Recognition
   Letters, 24(13), 2115–2125.
   → AND/score-level fusion framework behind our texture⊕geometry decision rule.
9. **Sanchez-Reillo, R., Sanchez-Avila, C., & Gonzalez-Macian, A. (2000).** *Biometric
   identification through hand geometry measurements.* IEEE TPAMI, 22(10), 1168–1171.
   → Establishes hand-geometry features (finger length/width ratios) as a deployable
   biometric trait — the basis of our geometry channel.
10. **Oquab, M., Darcet, T., Moutakanni, T., et al. (2023).** *DINOv2: Learning Robust Visual Features without Supervision.* arXiv:2304.07193.
    → Self-supervised Vision Transformer foundation model; provides generalizable 384-D [CLS] embeddings without palmprint supervision.
11. **Yang, W., et al. (2023).** *CCNet: A Cross-layer Contrastive Network for Contactless Palmprint Recognition.* IEEE Transactions on Information Forensics and Security (TIFS), 18, 5913–5925.
    → Domain-specific contactless palmprint contrastive network; basis for the 2048-D CCNet ONNX model trained on the Tongji dataset.

## 7. Multi-Model Biometric Architecture: Classical, Foundation & Deep Models

PalmPay features a pluggable, multi-model biometric architecture governed by the [`BasePalmModel`](file:///home/nightblaze/Projects/palm_pay_4/core/palm_model.py) abstraction. The system can run on hand-crafted classical computer vision, self-supervised foundation vision transformers, domain-specific pretrained palmprint networks, or custom ArcFace neural embeddings:

| Model Option | Identifier | Dimension | Backbone / Source | Runtime Engine | Typical Latency | Training Requirement |
|---|---|---|---|---|---|---|
| **CCNet Pretrained** (Production) | `ccnet-onnx-v1` | **2048-D** | CCNet (Yang et al., TIFS 2023) | ONNX Runtime CPU | ~24–28 ms | Zero (Pretrained on Tongji) |
| **DINOv2 Foundation** | `dinov2-vits14-v1` | **384-D** | Meta ViT-S/14 (`torch.hub`) | PyTorch CPU | ~32–35 ms | Zero (Self-supervised foundation) |
| **Classical Fusion** | `HogLbpFusion-v1-fm` | **3652-D** | HOG (1764-D) + MB-uLBP (1888-D) | NumPy / OpenCV CPU | ~18 ms | Zero (Analytical filters) |
| **DeepPalm Embedder** | `deep-palm-embedder-v1` | **512-D** | MobileNetV3-Small + ArcFace | ONNX Runtime CPU | ~12 ms | Supervised ArcFace training |

---

### Three-Stage Empirical Architecture Scorecard

```
┌────────────────────────┬──────────────────┬─────────────────┬──────────────────────┐
│ Metric / Trait         │ Phase 1: HOG+LBP │ Phase 2: DINOv2 │ Phase 3: CCNet (Prod)│
├────────────────────────┼──────────────────┼─────────────────┼──────────────────────┤
│ Model Backbone         │ Handcrafted LBP  │ ViT-S/14 (Meta) │ CCNet ONNX (TIFS)    │
│ Vector Dimension       │ 3652-D           │ 384-D           │ 2048-D               │
│ Input Size             │ 128x128 Grayscale│ 224x224 RGB     │ 128x128 Grayscale    │
│ CPU Latency            │ ~18 ms           │ ~32 ms          │ ~26 ms               │
│ Enrolled Subjects      │ 2                │ 2               │ 26 subjects          │
│ Live Trials Logged     │ 35               │ 35              │ 1,276 presentations  │
│ Genuine Pairs          │ —                │ —               │ 333 pairs            │
│ Impostor Pairs         │ —                │ —               │ 9,963 pairs          │
│ Genuine Mean (μ_gen)   │ 0.9214           │ 0.8385          │ 0.8918 ± 0.0789      │
│ Impostor Mean (μ_imp)  │ 0.8812           │ 0.6264          │ 0.5158 ± 0.0933      │
│ Separation Gap (Δμ)    │ +0.0402          │ +0.2121         │ +0.3760 (9.4x gap!)  │
│ Max Impostor Ceiling   │ 0.9841           │ 0.8692          │ 0.7611               │
│ Zero-FAR Threshold     │ N/A              │ N/A             │ T = 0.78             │
│ Production Threshold   │ T = 0.95 (geom)  │ T = 0.82 (geom) │ T = 0.73 (Option B)  │
│ Equal Error Rate (EER) │ 8.5%             │ 4.53%           │ ≈ 2.47%              │
└────────────────────────┴──────────────────┴─────────────────┴──────────────────────┘
```

---

### Step-by-Step Guide: How to Switch Biometric Models

To change the recognition model used by the system, open **[`config.py`](file:///home/nightblaze/Projects/palm_pay_4/config.py)** and modify the `ACTIVE_MODEL` setting:

#### 1. File to Edit
* Open **`config.py`** (line 230).

#### 2. Settings You Can Change

```python
# ── Model selection ──────────────────────────────────────────────────
# Options: "classical" | "ccnet" | "dinov2"
ACTIVE_MODEL = "ccnet"                  # Production default

# CCNet ONNX (fp32) path
CCNET_MODEL_PATH = "models/palm_ccnet.onnx"

# DINOv2 — model variant: "dinov2_vits14" (384-D) or "dinov2_vitb14" (768-D)
DINOV2_VARIANT = "dinov2_vits14"
```

#### 3. What Each Option Does

| Value of `ACTIVE_MODEL` | Model Used | Dimension | What It Does & When to Use It |
|---|---|---|---|
| `"ccnet"` *(Production)* | **CCNet ONNX (fp32)** | **2048-D** | Specialized contactless palmprint CNN (Yang et al., 2023) pretrained on the Tongji dataset. Delivers wide $+0.3760$ separation gap. Real-time CPU inference (~26 ms). |
| `"dinov2"` | **Meta DINOv2 ViT-S/14** | **384-D** | Self-supervised Vision Transformer foundation model. Robust against lighting shifts and palm tilt. Weights (~330 MB) download automatically on first run via `torch.hub`. |
| `"classical"` | **HOG + MB-uLBP Fusion** | **3652-D** | Handcrafted computer vision (crease line gradients + multi-block texture). 100% offline, zero neural weights needed, baseline benchmark. |

#### 4. Action Checklist When Changing Models

1. **If selecting `"ccnet"`:** Download the ONNX weights before starting the server:
   ```bash
   .venv/bin/python scripts/download_ccnet.py
   ```
   *(If the file is missing, the pipeline will display a warning and fall back to `"classical"` automatically.)*

2. **Restart the server** so the new model is loaded into memory:
   ```bash
   .venv/bin/python app.py
   ```
   *(Look for `[PalmPay] Active model: ...` in the startup console logs to confirm the switch).*

3. **Re-enroll your palm on [`/enroll`](http://127.0.0.1:5001/enroll):**
   * **Why?** The database strictly isolates templates by `model_name` and `feature_dim`. A 3652-D classical template cannot be compared to a 384-D DINOv2 vector.
   * Templates from the previous model remain safely stored in SQLite, but the matcher will only look for templates created by the currently active model.
   * If you try to recognize a user who was enrolled under a *different* model, the system will report 0% match rather than crash.

4. **Verify / Calibrate Threshold on [`/test`](http://127.0.0.1:5001/test):**
   * Go to the `/test` page and click **Run Evaluation**.
   * Classical models typically operate around `T = 0.95`.
   * Pretrained neural models (DINOv2 / CCNet) typically produce genuine cosine scores around `0.80 – 0.90`.
   * Click **Apply Suggested Threshold** to set the optimal decision boundary for the selected model.

---

### Engine 1 — Classical Vision (HOG + MB-uLBP + Hand Geometry)
The classical engine requires no pre-existing dataset of human palms, making it 100% operational out of the box:
1. **HOG Crease Filter (1764-D):** Extracts directional gradient histograms across $8\times8$ cells to map major flexion creases (heart, head, life line).
2. **MB-uLBP Microtexture (1888-D):** Extracts multi-scale uniform local binary patterns across radii (1,2) over a $4\times4$ grid to map fine skin ridges.
3. **Hand Geometry Channel (20-D):** Independent physical landmark descriptor (finger lengths, knuckle breadths, and signed splay angles) that stops impostors even when palm texture scores correlate.
4. *Alternative Baselines:* **CompCode** (1536-D Gabor filter bank) and pure **MB-uLBP** (1888-D) remain configurable in `config.py`.

---

### Engine 2 — Foundation Vision Transformer (Meta DINOv2 ViT-S/14)
The foundation model engine maps palm crops into high-dimensional semantic space via self-supervised vision transformer features:
1. **Architecture:** Vision Transformer Small (`dinov2_vits14`, 21M parameters) with $14\times14$ patch tokens.
2. **Input Processing:** $128\times128$ grayscale crop resized to $224\times224$, converted to 3 RGB channels, and standardized with ImageNet mean/variance.
3. **Embedding:** 384-D $L_2$-normalized `[CLS]` token vector.
4. **Acquisition:** PyTorch automatically downloads and caches model weights (~330 MB) on first launch via `torch.hub`.

---

### Engine 3 — Pretrained Contactless Palmprint Network (CCNet ONNX)
A state-of-the-art specialized contactless palmprint model trained on the Tongji dataset:
1. **Architecture:** Cross-layer Contrastive Network (Yang et al., IEEE TIFS 2023).
2. **Input Processing:** Direct single-channel float32 tensor $[1, 1, 128, 128]$ scaled to $[0.0, 1.0]$.
3. **Embedding:** 2048-D $L_2$-normalized feature vector.
4. **Acquisition:** Download the ONNX weights (~257 MB) via `python scripts/download_ccnet.py`.

---

### Engine 4 — Deep Learning Embedder (MobileNetV3-Small + ArcFace 512-D)
The custom metric learning engine trained on local or pilot transaction data:
1. **Backbone Architecture (`training/model.py`):**
   MobileNetV3-Small pre-trained on ImageNet, adapted for $128\times128$ palm crops using lightweight inverted residual blocks and Squeeze-and-Excitation (SE) channel attention.
2. **ArcFace Metric Learning (`training/train.py`):**
   Trained using Additive Angular Margin Loss ($\text{margin}=0.5, \text{scale}=64.0$):
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$
3. **Disjoint Open-Set Verification:**
   Evaluated strictly on held-out subjects unseen during training via pairwise genuine/impostor cosine distributions, Equal Error Rate (EER), and Zero-FAR GAR.
4. **ONNX Export & CPU Serving (`core/palm_model.py`):**
   Exported to `models/palm_embedder.onnx`. Inference runs in `<15 ms` on standard CPU hardware via ONNX Runtime without external cloud or GPU requirements.

---

### The Privacy-Preserving Data Flywheel
How do we train the Deep Model without violating our standing rule: *"Never store raw palm images"*?
1. **Ephemeral Consent Capture:**
   In `config.py`, enable `DATASET_CAPTURE_ENABLED = True`. During enrollment on `/enroll`, a volunteer ticks the consent checkbox.
2. **Isolated Preprocessed Crops:**
   The backend saves only preprocessed $128\times128$ CLAHE crops with knuckle-line finger masking into `data/pilot_rois/` (gitignored). Raw camera frames are never saved.
3. **Training & Export:**
   Run `.venv/bin/python -m training.train --epochs 25` to train the model, then `.venv/bin/python scripts/export_deep_model.py` to compile the ONNX artifact.
4. **Privacy Purge:**
   Execute `.venv/bin/python scripts/purge_dataset.py --dataset data/pilot_rois` to completely erase the training crops from disk.
5. **Production Deployment:**
   Set `DEEP_ENABLED = True` in `config.py`. The system now serves neural embeddings exclusively.

---

## 8. Installation & Running

### Linux (dev machine — managed via mise)
```bash
mise exec python@3.11 -- python -m venv .venv      # create once (gitignored)
.venv/bin/pip install -r requirements.txt          # deps ONLY inside .venv
.venv/bin/python scripts/download_ccnet.py         # download CCNet ONNX weights (~257 MB)
.venv/bin/python app.py                            # http://127.0.0.1:5001
```

### Windows (target demo laptop)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_ccnet.py
python app.py                                      # http://127.0.0.1:5001
```

Server flags live in `config.py` (`SERVER_HOST`, `SERVER_PORT`, debug ON, reloader OFF —
restart manually after edits).

### Utilities
```bash
ACTIVE_MODEL="classical" .venv/bin/python -m unittest discover tests # 113 tests — must stay green after ANY change
ctest --test-dir cpp/build --output-on-failure                       # 4 C++ unit tests
python scripts/download_ccnet.py                                     # Download CCNet fp32 ONNX weights (~257 MB)
python scripts/verify_demo_script.py                                 # 13-step end-to-end demo rehearsal (pipeline mocked)
python scripts/smoke_test_deep.py                                    # 8-step controlled DeepPalm end-to-end smoke test
python scripts/eval_deep_model.py                                    # Head-to-head Classical vs Deep evaluation benchmark
python scripts/purge_dataset.py --dataset data/pilot_rois           # Privacy purge for ephemeral training crops
python scripts/reset_biometrics.py --yes                             # wipe biometric users/templates, PRESERVES merchants
python scripts/seed_demo_dataset.py                                  # synthetic templates for offline evaluation testing
python scripts/demo_pipeline_test.py                                 # visual pipeline stage dump (needs webcam)
node --check static/js/enroll.js                                     # JS syntax gate after editing any controller
```

---

## 9. How-To: Every User Journey

### First-time setup checklist (demo venue)
1. Mount laptop + small fixed LED light at the counter (consistency beats any camera upgrade).
2. Start server; open `/enroll`; enroll operators first, then volunteers — follow the prompts.
3. **Enroll your other hand as a decoy user** (it becomes the impostor test subject).
4. `/test` → Run Evaluation → read genuine_min vs impostor_max → **Use Suggested → Apply
   Threshold**.
5. Acceptance: decoy rejected 10/10 at `/pay`; every registered palm pays; wrong palms bounce.

### Daily demo flow
Top-up consumers (admin) → merchant opens POS tab → sale → palm → green screen → receipt →
show dashboards → deliberately fail a scan (wrong palm) → low-balance orange screen →
merchant cash-out request → admin approval.

### Recalibrating later
Whenever several new users enroll (rule of thumb: ~every 10× cohort growth), re-run `/test`
and apply the fresh suggested threshold. Never hand-tune T against intuition.

---

## 10. Page & API Map

**Pages:** `/` overview · `/enroll` auto-scan · `/recognize` legacy live HUD · `/users`
management · `/test` evaluation/calibration · `/consumer?user_id=` mobile wallet · `/pay`
standalone POS · `/merchant?merchant_id=&view=dashboard|pos|transactions|cashout` console ·
`/transactions?role=&user_id=` shared history · `/receipt/<uuid>` · `/admin(/login|dashboard|
topup|cashouts)`.

**APIs:** biometric — `POST /api/process_frame`, `POST /api/enroll`, `POST /api/recognize`
(single image or burst), `GET/POST /api/config` (threshold), `POST /api/evaluate`,
`GET /api/users…`. Payment — `POST /api/payment`, `POST /api/topup`,
`POST /api/topup/request` (+ `/approve`), `POST /api/cashout` (+ `/approve`),
`GET /api/wallet/<id|m1..m4>`, `GET /api/transactions[?]`, `GET /api/transactions/export`,
`GET /api/merchant/stats/<mid>`.

**DB schema highlights:** `users(usn UNIQUE)` · `templates(feature_blob, feature_dim,
model_name)` — loader serves **only** rows matching the active model, so model upgrades
invalidate old templates safely · `wallets(role CHECK, balance)` · `transactions(UUID PK,
status CHECK, biometric_score, note-as-type)` · `cashout_requests` / `topup_requests(PENDING→APPROVED)`.

---

## 11. Verified vs Pending

**Verified (code-level):**
- [x] End-to-end pipeline on CPU: frame → landmarks → ROI → CLAHE → mask → 3652-D HOG+LBP → match
- [x] Quality-gate rejection paths with live coaching messages
- [x] Finger-mask geometry (pure-function unit tests incl. byte-exactness below the cut)
- [x] **Hand-geometry channel** (20-dim invariant descriptor; rotation/scale/translation
      invariance + mirror-sensitivity unit-tested) with AND-fusion matching gate
- [x] **Enrollment pose-consistency gate** (absolute-unit outlier rejection, fail-fast coaching)
- [x] Hardened matcher semantics (dual-pass, ambiguity guard, dimension safety, dynamic active threshold routing)
- [x] 4-frame fusion agreement counting + `unstable_capture` path
- [x] Payment ledger atomicity, session gating, IDOR-resistance (payer from server session)
- [x] **113/113 automated Python unit tests** · **4/4 C++ tests** · 13-step demo script green
- [x] **Selectable Multi-Model Engine Architecture:**
      - Domain Pretrained Palmprint CNN (`ccnet-onnx-v1`, 2048-D ONNX via CPU) — **Active Production**
      - Foundation Vision Transformer (`dinov2-vits14-v1`, 384-D via `torch.hub`)
      - Classical HOG+LBP Fusion (`HogLbpFusion-v1-fm`, 3652-D)
      - DeepPalm Inverted Residual Embedder (`deep-palm-embedder-v1`, 512-D ONNX / PyTorch)
- [x] **Model Isolation & Database Integrity:** Database template loader safely isolates rows by active `model_name` and `feature_dim`, preventing cross-model matching collisions
- [x] **Phase-2 Spike (flag-gated):** Deep palm embedding interface (`DeepPalmModel`), challenge-response landmark liveness (`OPEN_PALM`, `FIST`, `SPREAD`), native C++ accelerated core, and data flywheel training pipeline
- [x] **Disjoint Validation:** `training/train.py` evaluates open-set biometric verification (pairwise genuine/impostor distributions, EER, Zero-FAR GAR) rather than cross-entropy on unseen validation subjects
- [x] **Integration Smoke Test:** End-to-end controlled smoke test script passes with full enrollment, recognition, rejection (`deep_below_threshold`), and cleanup

**Empirically Verified (On Real Human Palms):**
- [x] **Production Multi-User Cohort Benchmark:** 26 subjects, 144 CCNet templates enrolled in `database/palmpay.db`.
- [x] **Exhaustive Biometric Cross-Evaluation:** 333 genuine pairs ($\mu = 0.8918 \pm 0.0789$), 9,963 impostor pairs ($\mu = 0.5158 \pm 0.0933$), separation margin $\Delta\mu = +0.3760$, impostor ceiling $0.7611$.
- [x] **Live POS Presentation Benchmark:** 1,276 physical presentations logged in `data/empirical_trials.csv` with 0.00% empirical False Accept Rate (FAR) under Option B adaptive geometry fusion.
- [x] **Cross-Condition Illumination Resilience:** Verified under ambient fluorescent and directional specular panel lighting conditions.
- [x] **Decoy / Left-Hand Rejection:** Hand-geometry mirror-sensitivity blocks opposite-hand and stranger impersonation.

---

## 12. Known Limitations & Threat Model (state honestly — also viva material)

1. **Illumination sensitivity mitigated, not solved.** RGB + classical features remain
   environment-sensitive. Mitigations shipped: quality gate, diversity-scripted enrollment,
   finger masking. Definitive fixes (NIR sensor / learned invariant embeddings) are ROADMAP
   Phases 2–3. Interim policy: **enroll where you pay.**
2. **Liveness & Anti-Spoofing posture (Phase 2):** Server-side challenge-response liveness (`OPEN_PALM`,
   `FIST`, `SPREAD`) rejects static printouts, photo replays, and rigid translations without finger
   articulation. However, liveness is an anti-spoofing defense layer rather than cryptographic proof;
   high-effort physical video masks or synthetic replay attacks remain a threat model consideration
   until NIR multispectral sensors are deployed.
3. **Deep Model Status:** The ONNX embedder interface (`DeepPalmModel`) is ready for inference. A
   scaffold script (`scripts/export_deep_model.py`) generates a deterministic local model without network
   dependencies; real neural weights must be trained/fine-tuned on pilot transaction data before
   calibrating `DEEP_MATCH_THRESHOLD`.
4. **Small-cohort statistics.** Thresholds calibrated on few subjects generalize weakly;
   recalibrate as the cohort grows.
5. **Security floor deferred by agreement** (documented in ROADMAP Phase 1): merchant terminal
   auth, consumer PIN binding, rate limiting, HTTPS/secrets management. Acceptable only
   because the system holds no real money.
6. **Privacy posture:** raw biometric images and raw liveness frames are **never stored**; only
   derived unit-norm float32 embeddings, ephemeral pilot ROIs (only when consent checked), and single-use session tokens are maintained.

---

## 13. Multi-Model Configuration, C++ Core & Tests

### Feature Flags in `config.py`
```python
# ── Active Model Selection ───────────────────────────────────────────
ACTIVE_MODEL = "ccnet"                  # Options: "classical" | "ccnet" | "dinov2"
CCNET_MODEL_PATH = "models/palm_ccnet.onnx"
DINOV2_VARIANT = "dinov2_vits14"        # "dinov2_vits14" (384-D) or "dinov2_vitb14" (768-D)

# ── Phase-2 Deep Model & Liveness ────────────────────────────────────
DEEP_ENABLED = False                    # Set True to switch to DeepPalmModel (512-D ONNX / CPU)
DEEP_MATCH_THRESHOLD = 0.85             # Calibrated matching threshold for DeepPalm
LIVENESS_ENABLED = False                # Enables challenge-response liveness API
LIVENESS_MIN_SCORE = 0.50               # Minimum composite score for liveness pass
PAYMENT_REQUIRES_LIVENESS = False       # Enforces verified liveness token before debiting
NATIVE_CPP_ENABLED = False              # Bridges performance-critical ops to C++ binary
DATASET_CAPTURE_ENABLED = False         # Enables consent-gated ephemeral pilot ROI capture
```

### Building the C++ Native Core
```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure
```

### Running Test Suites & Model Tools
```bash
# Full Python test suite (113 tests across all models and payment subsystems)
ACTIVE_MODEL="classical" .venv/bin/python -m unittest discover tests

# C++ Native Core tests (4 tests)
ctest --test-dir cpp/build --output-on-failure

# Download CCNet ONNX checkpoint (~257 MB)
python scripts/download_ccnet.py

# End-to-end demo verification (13 scripted steps)
.venv/bin/python scripts/verify_demo_script.py

# Controlled deep model smoke test (enrollment, db schema, matching, rejection)
.venv/bin/python scripts/smoke_test_deep.py

# Benchmark classical vs deep model (synthetic / pilot)
.venv/bin/python scripts/eval_deep_model.py

# Purge ephemeral pilot ROIs (privacy compliance)
.venv/bin/python scripts/purge_dataset.py --dataset data/pilot_rois
```


