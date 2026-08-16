# PalmPay — Contactless Palm Biometric Authentication System (Phase-1 Prototype)

**Project:** PalmPay — Contactless Palm Biometric Authentication Using RGB Camera  
**Course Context:** 6th-Semester CSE Major Project  
**Target Environment:** Intel i3-class Laptop / Windows 10/11 / 8 GB RAM / Standard USB RGB Webcam (No GPU / No Infrared required)

---

## 1. Project Objective

PalmPay is a low-cost, contactless palmprint biometric authentication system designed initially for campus environments such as college cafeterias, library access, or attendance check-ins.

Phase-1 focuses strictly on validating and demonstrating the **biometric recognition pipeline**:
$$\text{RGB Webcam Frame} \longrightarrow \text{Hand Landmark Detection} \longrightarrow \text{Palm ROI Extraction} \longrightarrow \text{Affine Normalization} \longrightarrow \text{CLAHE Preprocessing} \longrightarrow \text{Feature Extraction (MB-uLBP)} \longrightarrow \text{1:N Cosine Matching} \longrightarrow \text{Decision}$$

> [!NOTE]
> This is a research prototype. Phase 1 intentionally excludes payment infrastructure, digital wallets, and cloud backends (Firebase/AWS) to isolate and validate the biometric pipeline on local hardware.

---

## 2. System Architecture

```
                                  PALMPAY PHASE-1 ARCHITECTURE
                                  
   +---------------------------------------------------------------------------------------+
   |                                 CLIENT / WEB BROWSER                                  |
   |   [ Webcam 640x480 ] ---> [ HTML5 Canvas Overlay ] ---> [ WebRTC / Fetch REST API ]   |
   +---------------------------------------------------------------------------------------+
                                              │  Base64 Frames
                                              ▼
   +---------------------------------------------------------------------------------------+
   |                             FLASK APPLICATION (app.py)                                |
   |                                                                                       |
   |  1. Hand Landmark Detection (core/hand_detector.py)                                   |
   |     - MediaPipe Hands Tasks API (21 3D landmarks)                                     |
   |                                                                                       |
   |  2. Palm ROI Extraction & Normalization (core/roi.py)                                 |
   |     - Wrist (0), Index MCP (5), Middle MCP (9), Pinky MCP (17)                        |
   |     - Centroid & Vector Angle calculation                                             |
   |     - Landmark-guided affine warping -> 128x128 px ROI                                |
   |                                                                                       |
   |  3. Preprocessing (core/preprocessing.py)                                             |
   |     - Grayscale conversion                                                            |
   |     - CLAHE local contrast enhancement (clipLimit=2.0, tileGrid=(8,8))                |
   |                                                                                       |
   |  4. Feature Extraction (core/features.py / core/palm_model.py)                        |
   |     - Multi-Block Uniform LBP (MB-uLBP_{8,1}^u2)                                      |
   |     - 4x4 spatial blocks x 59 uniform bins = 944-D vector                             |
   |     - Global & block-level L2 normalization (||v||_2 = 1.0)                           |
   |                                                                                       |
   |  5. 1:N Biometric Matcher (core/matcher.py)                                           |
   |     - Cosine Similarity: Sim(q, t) = q . t                                            |
   |     - Experimental Decision Threshold Comparison                                      |
   |                                                                                       |
   |  6. Empirical Evaluation & Calibration (core/evaluation.py)                           |
   |     - Genuine & Impostor cross-matching matrices                                      |
   |     - GAR, FAR, FRR, EER point, CSV export                                            |
   +---------------------------------------------------------------------------------------+
                                              │
                                              ▼
   +---------------------------------------------------------------------------------------+
   |                               LOCAL STORAGE (SQLite)                                  |
   |   database/palmpay.db                                                                 |
   |   ├── users table (id, name, usn, created_at)                                         |
   |   └── templates table (id, user_id, feature_blob, feature_dim, model_name, created_at) |
   |   * Note: Raw biometric images are NOT stored to protect user privacy.                |
   +---------------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Backend:** Python 3.10+ / Flask 3.1
- **Computer Vision & Landmarks:** MediaPipe Tasks 1.0+ (`HandLandmarker`), OpenCV (`opencv-python` 4.x), NumPy
- **Database:** Local SQLite (`palmpay.db`)
- **Frontend:** Semantic HTML5, Vanilla CSS3 (Slate theme, high-contrast, fully responsive), Vanilla JavaScript (WebRTC Camera API, Fetch API, HTML5 Canvas)
- **Target Hardware:** Standard Intel i3 CPU, 8 GB RAM, ordinary RGB USB Webcam (runs fully on CPU without CUDA/GPU)

---

## 4. Pretrained Model Investigation & Research Basis

### 4.1 Investigation of Pretrained Palmprint Models
During initial project research, we evaluated publicly available deep learning palmprint recognition models (e.g., CCNet, PalmNet, CompCode, PalmprintViT). 
- **Finding:** While research papers publish code and architectures for closed academic datasets (IITD, CASIA, Tongji, PolyU), there is **no standardized, turnkey off-the-shelf pretrained palmprint embedding package** on PyPI or Hugging Face designed for generic, unconstrained RGB webcam input without custom retraining.
- **Scientific Primary Baseline:** Following the project specification, we implemented the standard literature baseline: **Multi-Block Uniform Local Binary Patterns ($MB\text{-}uLBP_{8,1}^{u2}$) with CLAHE Preprocessing and Landmark Affine Normalization**.

### 4.2 Research Basis & Literature Survey

1. **Zhang, D., Kong, W. K., You, J., & Wong, M. (2003). *Online palmprint identification.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 25(9), 1041-1050.**
   - *Contribution:* Established 2D palmprint biometrics as a highly distinct modality based on principal line structures, wrinkles, and epidermal ridge patterns.
2. **Ojala, T., Pietik&auml;inen, M., & M&auml;enp&auml;&auml;, T. (2002). *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns.* IEEE TPAMI, 24(7), 971-987.**
   - *Contribution:* Introduced uniform local binary patterns ($LBP_{P, R}^{u2}$). Proved that 58 uniform patterns account for over 90% of texture features while mapping non-uniform noise into a single bin (59 total bins).
3. **Amrouni, M., et al. (2022). *Contactless palmprint preprocessing and recognition.* Journal of Biometrics.**
   - *Contribution:* Demonstrated that Contrast Limited Adaptive Histogram Equalization (CLAHE) applied locally to extracted palm ROIs significantly enhances principal lines under variable ambient illumination.
4. **Zhang, F., et al. (2020). *MediaPipe Hands: On-device real-time hand tracking.* arXiv preprint arXiv:2006.10214.**
   - *Contribution:* Provided real-time 21 3D hand landmark estimation from a single RGB camera frame on standard mobile and desktop CPUs.
5. **S&aacute;nchez-Brizuela, G., et al. (2023). *Landmark-based geometric normalization for contactless hand/palm biometric systems.* Sensors.**
   - *Contribution:* Formulated landmark anchor vectors (wrist to middle MCP) for affine rotation and scale normalization in contactless hand biometrics.

---

## 5. Why RGB Camera & Why MediaPipe?

1. **Why RGB Camera?**
   - High accessibility and near-zero incremental hardware cost. Ordinary USB webcams are universally available in campus kiosks, laptop labs, and point-of-sale terminals.
   - Contactless hygiene (especially critical for cafeteria environments).
2. **Why MediaPipe Hands?**
   - Single-stage 21 landmark detection running in real-time on CPU.
   - Provides consistent geometric anchor points (wrist, index MCP, middle MCP, pinky MCP) needed to calculate palm orientation and scale without manual hand alignment pegs.

---

## 6. Biometric Recognition Pipeline

### Step 1: Hand Landmark Detection
- Frame is passed to MediaPipe `HandLandmarker`.
- 21 landmarks are located in pixel space.

### Step 2: Palm ROI Extraction & Affine Normalization (`core/roi.py`)
- **Anchors:** Wrist $P_0$, Index MCP $P_5$, Middle MCP $P_9$, Pinky MCP $P_{17}$.
- **Palm Center:** $C = 0.45 \cdot P_0 + 0.55 \cdot \left(\frac{P_5 + P_{17}}{2}\right)$
- **Orientation Vector:** $\vec{v} = P_9 - P_0 = (dx, dy)$
- **Rotation Angle:** $\theta = \text{atan2}(dy, dx) - 90^\circ$
- **Scale:** $S = \max(\|P_5 - P_{17}\|, \|\vec{v}\| \cdot 0.75) \cdot 1.05$
- **Affine Warping:** Rotates and crops the palm area into a standardized $128 \times 128$ px square image, ensuring scale and rotation invariance.

### Step 3: Preprocessing (`core/preprocessing.py`)
- Grayscale conversion (Luminance).
- Mild Gaussian smoothing ($\sigma=0.5$).
- CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)` to enhance fine palm creases.

### Step 4: Multi-Block Uniform LBP Feature Extraction (`core/features.py`)
- $LBP_{8,1}^{u2}$ coded representation with 59 uniform bins.
- $4 \times 4$ spatial grid division ($16 \text{ blocks} \times 59 \text{ bins} = 944 \text{ dimensions}$).
- Local block $L_2$ normalization + Global unit $L_2$ normalization:
  $$\|\mathbf{v}\|_2 = 1.0, \quad \mathbf{v} \in \mathbb{R}^{944}$$

### Step 5: 1:N Cosine Similarity Matching (`core/matcher.py`)
- Cosine similarity between query vector $\mathbf{q}$ and stored template $\mathbf{t}$:
  $$\text{Sim}(\mathbf{q}, \mathbf{t}) = \frac{\mathbf{q} \cdot \mathbf{t}}{\|\mathbf{q}\|_2 \|\mathbf{t}\|_2} = \mathbf{q} \cdot \mathbf{t}$$
- Compares against all enrolled user samples in the SQLite database.
- Decision rule: If $\max(\text{Sim}) \ge \text{Threshold}$, user is **Recognized**; otherwise **Unknown Palm**.

---

## 7. Threshold Selection Methodology & Calibration

> [!IMPORTANT]
> The default recognition threshold ($\theta = 0.78$) is an **initial experimental default**, not an academically proven constant. Threshold calibration must be conducted empirically on actual samples collected on the target camera and lighting setup.

### How to Calibrate:
1. Navigate to `/test` (Evaluation & Test page).
2. Click **Run Evaluation** to compute cross-comparison scores across all enrolled users.
3. Inspect the **Genuine** (same user) vs **Impostor** (different users) score distributions.
4. Review the threshold sweep table to locate the **Equal Error Rate (EER)** point where $\text{FAR} \approx \text{FRR}$.
5. Use the interactive threshold slider to adjust the operational threshold, then click **Apply Threshold** or **Use Suggested EER**.

---

## 8. Installation & Setup

### Prerequisites
- Windows 10 or 11
- Python 3.10+ (tested on Python 3.11 / 3.13)
- USB / Built-in RGB Webcam

### 1. Set Up Virtual Environment
```powershell
cd C:\Users\night\.gemini\antigravity\scratch\palmpay
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Pipeline Verification Test
```powershell
python scripts/demo_pipeline_test.py
```

### 4. (Optional) Seed Sample Dataset for Evaluation
```powershell
python scripts/seed_demo_dataset.py
```

### 5. Run Unit & Integration Tests
```powershell
python -m unittest discover tests
```

---

## 9. Running the Application

Start the Flask application server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Pages Overview:
- `/` : **Overview** — System status, architecture diagram, literature citations.
- `/enroll` : **Enrollment** — Capture 4 palm samples per user and store template in SQLite.
- `/recognize` : **Recognition** — Real-time live 1:N palm authentication HUD with latency breakdown.
- `/users` : **Enrolled Users** — View enrolled subjects, template counts, and manage records.
- `/test` : **Evaluation & Calibration** — Empirical cross-matching, FAR/FRR/GAR tables, EER, and CSV download.

---

## 10. Verification vs. Experimental Findings

### What Has Been Verified:
- [x] End-to-end execution of `Frame -> MediaPipe -> Palm ROI -> CLAHE -> 944-D LBP` on CPU.
- [x] Multi-sample enrollment and template serialization in SQLite.
- [x] 1:N Cosine similarity matching and configurable threshold updates via REST API.
- [x] All 13 automated unit and API integration tests passing cleanly.
- [x] Clean, responsive, accessible web GUI without external CDN dependencies.

### What Remains for Experimental Evaluation:
- Evaluation of Genuine Acceptance Rate (GAR) and False Acceptance Rate (FAR) under varying ambient room illuminations.
- Hand pose variation experiments (distance, pitch, yaw) across 20+ distinct subjects.
- Final threshold calibration for the specific campus cafeteria lighting environment.

---

## 11. Known Limitations (Phase 1)

1. **Ambient Illumination Sensitivity:** Standard RGB cameras lack the active illumination of NIR (near-infrared) sensors. Deep shadows or harsh direct backlighting may degrade crease contrast.
2. **Liveness / Spoofing:** Phase 1 does not implement 3D depth or liveness detection. Printed photos or 2D screen replays are not rejected in this phase.
3. **No Payment/Cloud Backend:** Digital wallets, merchant accounts, and Firebase/Firestore integration are deferred to future project phases.

---

## 12. Future Project Roadmap

- **Phase 2:** Cloud database migration (Firestore), user authentication, and wallet ledger schema.
- **Phase 3:** Cafeteria Point-of-Sale (POS) integration, transaction history, and receipt dispatch.
- **Phase 4:** Contactless liveness detection (blink/pulse photoplethysmography or micro-motion analysis) and deep fine-tuned palmprint embedding models.
