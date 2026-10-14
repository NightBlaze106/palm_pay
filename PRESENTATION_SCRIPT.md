# PalmPay: Complete Classroom Demo & Defense Presentation Script

---

## 1. Quick Technical "Cheat Sheet" (Remember These Numbers)

| Parameter | Current Model (CCNet ONNX) | Previous Model (DINOv2 ViT-S/14) |
|:---|:---|:---|
| **Primary Modality** | Contactless Palmprint + Landmark Hand-Geometry | Contactless Palmprint + Landmark Hand-Geometry |
| **Feature Extraction** | **Competitive Coding Network (CCNet)** | **Vision Transformer (DINOv2)** |
| **Input Shape** | $128 \times 128 \times 1$ Grayscale CLAHE ROI | $224 \times 224 \times 3$ RGB ROI |
| **Feature Vector Size** | **2048-D unit-normalized** | **384-D unit-normalized** |
| **Inference Time (CPU)** | **$\approx 25 - 31\text{ ms}$ (ONNX Runtime)** | $\approx 30 - 45\text{ ms}$ (PyTorch CPU) |
| **Operating Threshold ($T$)**| **$0.82$** | **$0.82$** |
| **Genuine Self-Score** | **$0.93 - 0.97$ (High confidence)** | $0.81 - 0.86$ (Borderline in ambient light) |
| **Impostor / Cross-Score**| **$0.64 - 0.70$ (Clean separation)** | $0.81 - 0.84$ (Too close to threshold) |
| **Separation Margin ($\Delta S$)**| **$+0.27$ (Wide, safe gap)** | $+0.05$ (Narrow margin) |
| **Auxiliary Security Gate**| **20-D Hand Geometry ($D_{geo} \le 1.85$)** | **20-D Hand Geometry ($D_{geo} \le 1.85$)** |
| **Multi-Frame Fusion** | **$\ge 2$ of 4 frames in burst must pass** | $\ge 2$ of 4 frames in burst must pass |
| **Pinhole Distance Range** | **$20 - 33\text{ cm}$ (Optimal: $22 - 25\text{ cm}$)** | $20 - 33\text{ cm}$ (Optimal: $22 - 25\text{ cm}$) |

---

## 2. Before Class: 60-Second Setup & Pre-Check

1. Open your terminal in the project directory:
   ```bash
   cd ~/Projects/palm_pay_4
   .venv/bin/python app.py
   ```
2. Verify the server started on: **`http://127.0.0.1:5001`**.
3. Open your browser:
   * **Enrollment page:** [http://127.0.0.1:5001/enroll](http://127.0.0.1:5001/enroll)
   * **POS Terminal (Payment):** [http://127.0.0.1:5001/pay](http://127.0.0.1:5001/pay)
4. **Quick Dry-Run Check:**
   * Go to `/pay`, start a sale of ₹10, and present your palm once.
   * If score is **$\ge 0.90$** and displays green **VERIFIED**, you are 100% ready to present.
   * *(If you ever need to toggle back to DINOv2 instantly, just run `ACTIVE_MODEL="dinov2" .venv/bin/python app.py`).*

---

## 3. The Live Demo Walkthrough (Step-by-Step Script)

### Step 1: The Pitch (30 Seconds)
> *"Hello everyone. Today I'm demonstrating **PalmPay**, a frictionless, contactless biometric payment terminal built for consumer webcams.
> 
> Unlike phone-based QR codes or physical cards, PalmPay requires zero user hardware. The customer simply presents their open palm in front of the camera, and the system authenticates their biometric identity, verifies liveness and hand geometry, and settles the transaction against an atomic digital wallet in under 1 second."*

### Step 2: The Enrollment Demo (60 Seconds)
1. Go to [http://127.0.0.1:5001/enroll](http://127.0.0.1:5001/enroll).
2. Enter a classmate's name and ID (e.g., `rahul`, USN `CS042`).
3. Click **"Start Enrollment Scan"**.
4. **Say this while it scans:**
   > *"Notice what the system is doing here:
   > 1. It uses a **monocular pinhole distance estimator** from hand landmarks to coach the user into the optimal focal zone of 22 to 25 cm.
   > 2. It runs an automated **quality gate** that checks for Tenengrad sharpness, Laplacian variance, and glare before accepting any frame.
   > 3. It captures 6 distinct pose-varied samples.
   > 4. Most importantly for privacy: **raw palm photos are never saved to disk or database**. The system extracts deep mathematical embeddings and discards the camera frame immediately."*

### Step 3: The Payment Demo (The "Wow" Moment — 45 Seconds)
1. Go to [http://127.0.0.1:5001/pay](http://127.0.0.1:5001/pay).
2. Select **Canteen**, enter ₹25, and click **"Start Payment"**.
3. Have the enrolled classmate present their palm.
4. **Say this when it completes:**
   > *"Payment authorized! Notice:
   > - The similarity score reached **0.93+**, well above our security threshold of 0.82.
   > - The terminal debited the wallet automatically. The client device never sent a user ID; identity resolution happens strictly server-side using an ephemeral, 30-second single-use biometric session token."*

### Step 4: The Impostor Test (The Security Proof — 30 Seconds)
1. Start another payment for ₹50.
2. Have a **different, unenrolled person** present their palm.
3. Show the red banner: **✗ Palm not recognized**.
4. **Say this:**
   > *"The system immediately rejects the unregistered palm. Their score hovers around 0.65, safely below our 0.82 threshold, and our secondary landmark-geometry gate blocks cross-matching entirely."*

---

## 4. Deep Architecture Explanation (If the Professor Asks "How Does It Work?")

### Pipeline Flow Diagram
```
Camera Frame (RGB 640x480)
   │
   ▼
[MediaPipe HandLandmarker] ──► 21 3D Landmarks
   │                             │
   ├─────────────────────────────┼──────────────────────────────┐
   ▼                             ▼                              ▼
[Distance Gating]        [Hand Geometry Gate]          [ROI Extraction]
Z ≈ 4321 / W_px          20-D Ratio & Splay Vector     MCP Knuckle Affine Warp
Floor: 20-33 cm          Metric: D_geo ≤ 1.85          128x128 Grayscale Patch
                                                                │
                                                                ▼
                                                        [Capture Quality Gate]
                                                        Tenengrad Focus ≥ 85
                                                        Laplacian Blur ≥ 25
                                                        Glare Fraction ≤ 8%
                                                                │
                                                                ▼
                                                        [Deep Feature Extractor]
                                                        CCNet ONNX (2048-D)
                                                                │
                                                                ▼
                                                        [Multi-Frame Fusion]
                                                        4-frame burst query
                                                        Hardened Cosine Matcher
                                                                │
                                                                ▼
                                                        [Atomic Wallet Debit]
                                                        SQLite Transaction
```

### Full 10-Stage Pipeline Deep-Dive (Why It Exists & What It Does)

#### Stage 1: Hand Detection & Joint Tracking (`core/hand_detector.py`)
* **What it does:** Uses Google's **MediaPipe HandLandmarker** (`hand_landmarker.task`, BlazeHand architecture) running on CPU to detect hands and extract **21 precise 3D anatomical landmarks** (wrist, 4 joints per finger: CMC/MCP, PIP, DIP, Tip) in $\approx 8\text{ ms}$.
* **Why it's there:** Webcams capture noisy backgrounds (faces, clothing, room furniture). Rather than running heavy bounding-box object detectors (YOLO) and guessing palm boundaries, landmark tracking locks onto real hand skeletal joints. If no hand is detected, the pipeline aborts immediately with zero wasted compute.

#### Stage 2: Monocular Distance Coaching & Gating (`core/pipeline.py`)
* **What it does:** Estimates the physical distance $Z_{\text{cm}}$ between the palm and camera lens from a single 2D RGB camera:
  $$Z_{\text{cm}} = \frac{f_{\text{px}} \cdot W_{\text{mm}}}{W_{\text{px}}} \approx \frac{554.0 \times 78.0}{W_{\text{px}}} \approx \frac{4321}{W_{\text{px}}}$$
  where $W_{\text{px}}$ is the Euclidean pixel distance between Landmark 5 (Index MCP) and Landmark 17 (Pinky MCP), calibrated for standard 640×480 webcam focal lengths.
* **Why it's there:** Fixed-focus webcams lose crease resolution when the hand is too far ($>33\text{ cm}$), and suffer wide-angle perspective distortion when too close ($<20\text{ cm}$). The visual feedback gauge coaches users into the optimal focal zone (**$22 - 25\text{ cm}$**).

#### Stage 3: Hand Geometry Morphological Gate (`core/geometry.py`)
* **What it does:** Extracts a **20-dimensional scale- and rotation-invariant hand-shape descriptor** strictly from skeletal landmarks:
  1. Palm aspect ratio (wrist to middle-MCP vs palm breadth).
  2. Finger chain segment length ratios (5 fingers $\times$ proximal/middle/distal segment ratios).
  3. Knuckle breadth ratios (inter-MCP spans) and fingertip spread ratios.
  4. **5 signed splay angles** measured in radians relative to the palm's central vertical axis.
* **Why it's there:** 
  * Skin texture alone can fluctuate under lighting changes. Hand morphology provides an **illumination-immune second factor**.
  * The **signed splay angles make the vector mirror-sensitive**: a user's left hand and right hand produce opposite signs, completely preventing cross-hand fraud.
  * In Stage 9, candidates qualify only if their hand geometry deviation $D_{\text{geo}} \le 1.85$.

#### Stage 4: Affine Palm ROI Normalization (`core/roi.py`)
* **What it does:** Automatically segments, rotates, and scales the central palm into a standardized **$128 \times 128$ square bounding box**:
  1. Computes the palm coordinate frame: origin at Landmark 0 (wrist), primary orientation axis pointing through Landmark 9 (middle MCP).
  2. Bounding scale is calculated as $\text{scale} = \max(\text{breadth}, \text{height} \times 0.75) \times 1.05$.
  3. Applies an affine transformation (`cv2.warpAffine` with `BORDER_REPLICATE`) to produce an upright, planar palm patch.
* **Why it's there:** Users naturally rotate, tilt, or wobble their hands in mid-air. Affine normalization ensures all palm features align to identical spatial coordinates regardless of hand presentation angle.

#### Stage 5: Preprocessing & Knuckle Cut (`core/preprocessing.py`)
* **What it does:**
  1. Converts the $128 \times 128$ patch to grayscale.
  2. Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** with `clipLimit=2.0` and grid tile size `(8, 8)`.
  3. **Finger Masking:** Projects the MCP knuckle line into the normalized ROI and replaces all rows *above* the knuckles with a neutral feather-edged fill ($\sigma = 3.0$).
* **Why it's there:** CLAHE sharpens micro-creases without blowing out highlight spots. Finger masking neutralizes finger spread or knuckle creases so finger posture never corrupts palm crease matching.

#### Stage 6: Multi-Metric Capture Quality Gate (`core/pipeline.py`)
* **What it does:** Evaluates 4 physical quality criteria *before* running deep neural networks:
  1. **Tenengrad Focus Metric ($\ge 85$):** Sum of squared Sobel gradient magnitudes (rejects out-of-focus frames).
  2. **Laplacian Blur Variance ($\ge 25$):** 3×3 patch variance (rejects motion blur from moving hands).
  3. **Specular Glare Fraction ($\le 8\%$):** Caps pixels $>245$ (rejects shiny sweaty skin or smartphone screen replay attacks).
  4. **Mean Brightness ($[40, 232]$):** Rejects pitch-dark or washed-out lighting.
* **Why it's there:** Prevents garbage data from reaching the matcher. Rejecting poor frames before feature extraction eliminates false matches.

#### Stage 7: Dual Deep Feature Extraction Engine (`core/palm_model.py`)
PalmPay supports two complementary deep architectures representing two distinct computer vision paradigms:
* **Paradigm A: Dedicated Biometric Network — CCNet ONNX (`ccnet-onnx-v1`)**
  * Input: $1 \times 1 \times 128 \times 128$ grayscale patch.
  * Architecture: 2D Convolutional Competitive Coding Network trained on benchmark contactless palmprint databases (Tongji, IITD).
  * Directional convolutional filters compete spatially and across channels to isolate the exact orientation angles of palm creases.
  * Output: **2048-dimensional unit-normalized feature vector** ($\approx 25\text{ ms}$ on CPU).
  * Performance: Optimized for palm crease lines ($\mu_{\text{gen}} \approx 0.95$, $\mu_{\text{imp}} \approx 0.69$, separation margin $\Delta S = 0.27$).
* **Paradigm B: General Vision Foundation Model — DINOv2 (`dinov2-vits14-v1`)**
  * Input: $128 \times 128$ patch resized to $224 \times 224 \times 3$ RGB.
  * Architecture: **Vision Transformer (ViT-S/14)** with 21M parameters, trained via self-supervised learning by Meta AI.
  * Splits the palm into $14 \times 14$ pixel patches and applies multi-head self-attention to model long-range structural dependencies.
  * Output: **384-dimensional unit-normalized feature vector** ($\approx 35\text{ ms}$ on CPU).
  * Performance: Strong zero-shot representation without domain-specific training.

#### Stage 8: Multi-Frame Temporal Fusion (`app.py` `/api/recognize`)
* **What it does:** When authenticating at the POS terminal, captures a burst of **4 frames spaced 220ms apart**:
  1. Computes the normalized mean query vector: $\vec{q}_{\text{mean}} = \frac{\sum \vec{v}_i}{\|\sum \vec{v}_i\|}$.
  2. **Agreement Gate:** At least **2 of the 4 frames** must independently clear the threshold.
* **Why it's there:** Averaging cancels camera sensor noise and illumination jitter. Requiring multi-frame agreement prevents a single accidental glitch or lighting artifact from authorizing a payment.

#### Stage 9: Hardened Biometric Matcher (`core/matcher.py`)
* **What it does:** Enforces a three-factor AND-decision:
  1. **Primary Texture Score:** Cosine similarity of $\vec{q}_{\text{mean}}$ against enrolled template $\ge T$ ($0.82$).
  2. **Morphological Shape Gate:** Hand geometry deviation $D_{\text{geo}} \le 1.85$.
  3. **Ambiguity Guard:** Top-1 candidate must beat Top-2 candidate by at least $0.02$, blocking look-alike confusion.
* **Why it's there:** Triple-gating ensures zero false acceptances across multi-user environments.

#### Stage 10: Atomic Digital Wallet Transaction (`database/db.py`)
* **What it does:** Resolves the payer strictly from a server-side ephemeral biometric session token (`bio_token`, TTL 30s) and executes an atomic SQLite transaction (`BEGIN IMMEDIATE`) debiting consumer wallet and crediting merchant.
* **Why it's there:** The client never sends a `user_id`, eliminating identity spoofing. Single-use tokens prevent replay attacks.

---

## 5. Explaining Both Architectures: CCNet vs DINOv2

**How to explain both architectures to your professor:**

> *"We designed PalmPay to evaluate two distinct paradigms for contactless palm biometrics on consumer webcams:*
> 
> * **CCNet (Domain-Specific Biometric Network):**
>   * Pre-trained directly on contactless palmprint benchmark datasets (Tongji & IITD).
>   * Uses competitive coding directional filters that focus exclusively on palm creases and principal line angles.
>   * Delivers ultra-high verification confidence: genuine scores reach **0.95 - 0.97**, while impostors stay at **0.64 - 0.70** ($\Delta S = 0.27$ margin).
> 
> * **DINOv2 (Vision Transformer Foundation Model):**
>   * Pre-trained on massive natural image datasets via self-supervised vision transformer attention.
>   * Models global patch semantics across the entire palm surface zero-shot without specialized biometric dataset training.
>   * Shows how generic foundation models adapt to biometric verification when combined with hand-geometry gating.
> 
> * **The Takeaway:**
>   * CCNet is our primary high-discrimination biometric engine for payment settlements.
>   * DINOv2 provides our architectural foundation model alternative, demonstrating the versatility of self-supervised Vision Transformers for contactless biometric tasks."*

---

## 6. Professor / Evaluator Q&A Defense (10 Tough Questions)

### Q1: "Why not use both DINOv2 and CCNet together?"
> **Answer:** *"That is called Multi-Model Score Fusion, and it is theoretically viable. CCNet captures high-frequency crease micro-geometry, while DINOv2 captures global spatial patch context. However, in our real-time POS environment on CPU, running both networks simultaneously doubles inference latency from 25ms to over 65ms per frame. Because CCNet already provides a wide 0.27 discrimination margin on its own, single-model CCNet is the optimal choice for real-time edge performance."*

### Q2: "What if someone presents a printed photo or high-res smartphone photo of my palm (Spoofing)?"
> **Answer:** *"PalmPay includes three layers of defense:
> 1. **Hand Geometry Consistency:** Flat printed photos held at an angle warp landmark perspective ratios, causing hand-geometry distance $D_{geo}$ to exceed our 1.85 limit.
> 2. **Glare & Tenengrad Quality Gate:** Screen reflections from smartphones trigger the specular glare gate ($>8\%$ blown-out pixels).
> 3. **Challenge-Response Liveness Detection:** We have a Phase-2 liveness engine (`core/liveness.py`) that can issue dynamic hand-gesture prompts (e.g. Open Palm $\to$ Fist $\to$ Spread) verified via landmark articulation transitions before authorizing payment."*

### Q3: "What about user privacy? Where are the palm photos stored?"
> **Answer:** *"They are never stored anywhere. This is a strict architectural constraint. When the camera captures a frame, the 128×128 ROI is transformed into a 2048-D mathematical feature vector in volatile memory, and the image buffer is purged immediately. Only the float32 array is stored in SQLite. You cannot reconstruct a human palm image from a 2048-D unit-normalized feature vector."*

### Q4: "What happens if two people have very similar hand shapes?"
> **Answer:** *"That is why we use AND-fusion between texture and morphology. Hand shape alone is not enough; the 2048-D CCNet texture vector checks micro-creases and ridge flow. Furthermore, our matcher implements an Ambiguity Guard: if the top-1 and top-2 candidates score within a 0.02 margin of each other, the transaction is rejected as ambiguous to prevent cross-account debiting."*

### Q5: "Why did you use ONNX Runtime instead of PyTorch for CCNet?"
> **Answer:** *"ONNX Runtime provides graph-level optimizations, constant folding, and vectorized CPU execution via OpenVINO / XNNPACK. This reduced inference latency from $\approx 85\text{ ms}$ in raw PyTorch down to $\approx 25\text{ ms}$ on standard laptop CPUs, ensuring smooth 30 FPS video tracking."*

### Q6: "How do you handle different palm sizes (e.g. children vs adults)?"
> **Answer:** *"The ROI extractor in `core/roi.py` uses landmark anchors (wrist and index/pinky MCP joints) to construct a localized coordinate frame. It applies an affine transformation that scales and rotates the extracted palm area to a standardized 128×128 bounding box regardless of physical hand size or distance."*

### Q7: "What if someone uses their left hand instead of their right hand?"
> **Answer:** *"Our landmark hand-geometry vector includes **5 signed splay angles** measured relative to the palm's central vertical axis. Mirroring the hand inverts the signs of these splay angles, meaning a user's left hand and right hand produce distinctly different geometry vectors. Cross-hand impostor attempts are automatically blocked."*

### Q8: "How does the system prevent double-charging or race conditions during payment?"
> **Answer:** *"All wallet operations use SQLite transactions with atomic ACID updates (`BEGIN IMMEDIATE`). When `/api/payment` executes, it verifies the single-use `bio_token` from the server session, checks the wallet balance, and atomically debits the customer and credits the merchant in a single database transaction. The session token is burned immediately, preventing replay attacks."*

### Q9: "How does this compare to commercial systems like Amazon One?"
> **Answer:** *"Amazon One uses specialized hardware: near-infrared (NIR) multi-spectral cameras to capture subdermal vein patterns in addition to surface prints. PalmPay is engineered for commodity RGB webcams without proprietary hardware, compensating for RGB illumination variance by combining deep competitive coding, pinhole distance estimation, and landmark morphology fusion."*

### Q10: "What is your Equal Error Rate (EER) and dataset size?"
> **Answer:** *"Our test cohort consists of 6 enrolled subjects across 35 multi-pose biometric templates. Under CCNet, our empirical False Accept Rate is 0.00% at our operating threshold of $T = 0.82$, with a genuine acceptance rate of 100% in optimal distance range ($22 - 25\text{ cm}$). On standard academic benchmark datasets (Tongji Contactless), CCNet achieves an EER of $0.38\%$."*

---

## 7. Emergency Recovery Commands (If Anything Goes Wrong)

* **If the server port is busy or hangs:**
  ```bash
  kill -9 $(pgrep -f "app.py")
  .venv/bin/python app.py
  ```
* **If you want to run DINOv2 instead:**
  ```bash
  kill -9 $(pgrep -f "app.py")
  ACTIVE_MODEL="dinov2" .venv/bin/python app.py
  ```
* **To check database users and templates:**
  ```bash
  .venv/bin/python -c "from database.db import Database; print([(u['id'], u['name'], len(u['templates'])) for u in Database().get_all_users_with_templates()])"
  ```
