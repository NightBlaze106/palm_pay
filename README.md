# PalmPay — Contactless Palm Biometric Payment System

**Project:** PalmPay — Pay Without Your Phone (Contactless Palm Biometrics on RGB Camera)
**Course Context:** 6th-Semester CSE Major Project · **Status:** Demo prototype (v2-fm accuracy sprint shipped, real-palm calibration pending)

> **What it does:** a merchant enters an amount at a POS terminal, the customer shows their
> open palm to an ordinary webcam, and the system recognizes them and debits their wallet —
> automatically, in a few seconds, with no phone, card, or touch.

Companion docs:

| Doc | Contents |
|---|---|
| `PROJECT_CONTEXT.md` | Session-continuation context: architecture decisions, decision log, current state |
| `ROADMAP.md` | Product phases (Demo → Pilot → Robustness → Productization) and gates |
| `AGENTS.md` | Dev-environment rules (venv discipline, node version manager, test policy) |

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
   │  2. Palm ROI + Affine Normalization (core/roi.py)                │
   │     wrist/MCP anchors → rotation+scale-normalized 128×128 crop   │
   │     (+ knuckle-line projection into ROI coordinates)             │
   │  3. Preprocessing                (core/preprocessing.py)         │
   │     grayscale → CLAHE(2.0,(8,8)) — Gaussian blur DISABLED        │
   │  4. Capture Quality Gate         (core/pipeline.py)              │
   │     rejects blur/glare/dark/too-far BEFORE scoring               │
   │  5. Finger Masking               (core/pipeline.py)              │
   │     rows above MCP knuckle line → feathered neutral fill         │
   │  6. Feature Extraction           (core/features.py)              │
   │     multi-scale uniform LBP, radii (1,2), 4×4 grid → 1888-D      │
   │  7. Hardened Matching            (core/matcher.py)               │
   │     dual mean-template AND best-sample pass + ambiguity guard    │
   │  8. Multi-Frame Fusion           (app.py /api/recognize)         │
   │     ≥3 of 4 frames must individually agree                       │
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
| Vision | OpenCV 4.x, NumPy | Affine warps, CLAHE, vectorized LBP |
| Storage | SQLite (single file, WAL-free demo mode) | Zero-config deployment on kiosk laptops |
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

### Stage 4 — Capture quality gate (`FrameQualityGate`)
Frames are rejected *before feature extraction*, with human-readable coaching messages
shown live in the UI:

| Check | Threshold | Rejection message |
|---|---|---|
| Blur (Laplacian variance) | ≥ 40 | "Image too blurry — hold palm steady" |
| Glare (fraction of pixels >245) | ≤ 8% | "Glare detected — tilt palm away" |
| Mean brightness | ∈ [40, 232] | "Too dark" / "Over-exposed" |
| Palm span | ≥ 60 px | "Palm too far — move closer" |

### Stage 5 — Finger masking (`apply_finger_mask`)
Rows **above** the projected knuckle line are replaced with a neutral fill equal to the mean
brightness of the kept palm area, using a Gaussian-feathered mask edge (σ=3) and byte-exact
rounding below the cut. Purpose: finger spread / pitch stops leaking into block histograms —
a measured cause of room→hall recognition drift. Applied identically at enrollment and
recognition (single code path ⇒ consistent by construction). Config: `FINGER_MASK_ENABLED`,
`FINGER_MASK_OFFSET_PX`.

### Stage 6 — Feature extraction: multi-block multi-scale uniform LBP (`core/features.py`)
For each radius $R \in \{1, 2\}$ the enhanced ROI is coded with the standard operator

$$LBP_{8,R}^{u2}(c) = \sum_{p=0}^{7} s(g_p - g_c)\,2^{\,p}, \qquad s(x)=\begin{cases}1 & x \ge 0\\ 0 & \text{else}\end{cases}$$

then mapped through the 256→59 uniform-pattern lookup table (58 rotation-consistent uniform
patterns + 1 non-uniform bin). The ROI is split into a **4×4 grid**; per block, per scale, a
59-bin histogram is extracted, block-$L_2$ normalized, concatenated, then globally $L_2$
normalized:

$$\mathbf{v} \in \mathbb{R}^{1888},\; \lVert\mathbf{v}\rVert_2 = 1 \quad (4\times4\ \text{blocks} \times 59\ \text{bins} \times 2\ \text{scales})$$

R=1 captures fine ridge micro-texture; R=2 adds coarser contextual texture — the classic
multiresolution LBP accuracy lever. Layout order is fixed: block-major, R1 before R2.

### Stage 7 — Hardened 1:N matching (`core/matcher.py`)
Cosine similarity (unit vectors ⇒ dot product), clamped to [-1,1], dimension-safe (stale-model
templates score 0.0 rather than crash). Per candidate identity, **two independent signals**
must clear the operating threshold T:

- `mean_score` — query vs the user's normalized **mean template** (primary, robust), and
- `best_sample_score` — query vs the user's best individual enrollment sample (sanity check).

Top-1 qualifies only if **both** pass; ranking uses the mean score; and if top-1 and top-2 both
qualify within `MATCHER_AMBIGUITY_MARGIN` (0.02), the result is **rejected as ambiguous**
rather than guessed. These rules remove the max-of-noisy-scores inflation that originally let
unregistered palms slip through at 0.80.

### Stage 8 — Multi-frame fusion (`/api/recognize`)
POS sends a burst of up to `FUSION_WINDOW`=4 frames (~220 ms apart). Decision requires:

1. the fused **mean query embedding** matches per Stage 7, **and**
2. at least `FUSION_MIN_PASS_FRAMES`=3 of 4 frames individually clear T.

Otherwise the response carries `match_reason: unstable_capture` and the POS retries
automatically. Legacy single-image requests degrade gracefully (require 1/1).

### Threshold calibration policy (`core/evaluation.py`, `/test` page)
T is **never guessed**. `/test` cross-matches all enrolled samples (genuine pairs vs impostor
pairs), reports distributions, a 0→1 sweep (step 0.02), EER estimate, and suggests the
operating point under the strict payment rule:

> **Highest T with ZERO observed impostor accepts while GAR ≥ 90%.**

Target margins before demo sign-off: `T − impostor_max ≥ 0.05` and `genuine_min − T ≥ 0.03`.
The current default **0.78 is an uncalibrated placeholder**.

---

## 4. Automated Enrollment (`/enroll`)

Fingerprint-setup style — the user never clicks per-sample:

1. Enter name + USN → **🚀 Start Enrollment Scan** (camera opens only for the scan).
2. A 400 ms probe loop sends low-quality preview frames to `/api/process_frame`; the quality
   gate's messages coach positioning live ("move closer", "too blurry").
3. When aligned + quality-passed, a high-quality (JPEG q=0.92) frame is auto-captured;
   **1.3 s cooldown**, then the next prompt appears.
4. **Condition script** (6 samples): neutral baseline → tilt LEFT → tilt RIGHT → move closer →
   turn palm **toward the light** → tip top of hand **away from camera**. Lighting and pitch
   entries directly target the room→hall generalization failure.
5. **Near-duplicate guard:** each candidate is diffed against the previous sample on a 32×32
   grayscale snapshot (< 1.5% mean-abs difference ⇒ skipped with "vary a bit more").
6. Auto-submit to `/api/enroll` → wallet auto-created → done.

Config: `DEFAULT_ENROLLMENT_SAMPLES=6` (min 3 / max 8), `ENROLL_CAPTURE_COOLDOWN_MS`,
`ENROLL_POSE_PROMPTS`, `ENROLL_DUPLICATE_DIFF_PCT`.

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

### Why not deep learning yet?
Public palmprint models (CCNet, PalmNet, CompCode variants, ViTs) ship without turnkey
pretrained embeddings for unconstrained webcam input, and would require training data we don't
have pre-pilot. The classical MB-uLBP baseline is literature-standard, fully explainable
(viva-friendly), runs in <100 ms on an i3, and the pilot's score logs become the training set
for the planned deep-embedding swap behind the same `BasePalmModel` interface (see ROADMAP
Phase 2). This is a deliberate sequencing decision, not a limitation of ambition.

---

## 7. Installation & Running

### Linux (dev machine — managed via mise)
```bash
mise exec python@3.11 -- python -m venv .venv      # create once (gitignored)
.venv/bin/pip install -r requirements.txt          # deps ONLY inside .venv
.venv/bin/python app.py                            # http://127.0.0.1:5000
```

### Windows (target demo laptop)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py                                      # http://127.0.0.1:5000
```

Server flags live in `config.py` (`SERVER_HOST`, `SERVER_PORT`, debug ON, reloader OFF —
restart manually after edits).

### Utilities
```bash
python -m unittest discover tests          # 54 tests — must stay green after ANY change
python scripts/verify_demo_script.py       # 13-step end-to-end demo rehearsal (pipeline mocked)
python scripts/reset_biometrics.py --yes   # wipe biometric users/templates, PRESERVES merchants
python scripts/seed_demo_dataset.py        # synthetic templates for offline evaluation testing
python scripts/demo_pipeline_test.py       # visual pipeline stage dump (needs webcam)
node --check static/js/enroll.js           # JS syntax gate after editing any controller
```

---

## 8. How-To: Every User Journey

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

## 9. Page & API Map

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

## 10. Verified vs Pending

**Verified (code-level):**
- [x] End-to-end pipeline on CPU: frame → landmarks → ROI → CLAHE → mask → 1888-D LBP → match
- [x] Quality-gate rejection paths with live coaching messages
- [x] Finger-mask geometry (pure-function unit tests incl. byte-exactness below the cut)
- [x] Hardened matcher semantics (dual-pass, ambiguity guard, dimension safety)
- [x] 4-frame fusion agreement counting + `unstable_capture` path
- [x] Payment ledger atomicity, session gating, IDOR-resistance (payer from server session)
- [x] 54/54 automated tests · 13-step demo script green

**Pending (empirical, on real palms):**
- [ ] Genuine/impostor distributions on the upgraded extractor — the motivating incidents
      (left palm accepted @0.80; room→hall failures; spread-finger sensitivity) have
      structural fixes shipped but **not yet validated with human subjects**
- [ ] Operating-threshold calibration + baked-in config value
- [ ] Cross-condition GAR/FAR across ≥10 subjects at the actual venue lighting

---

## 11. Known Limitations (state honestly — also viva material)

1. **Illumination sensitivity mitigated, not solved.** RGB + classical features remain
   environment-sensitive. Mitigations shipped: quality gate, diversity-scripted enrollment,
   finger masking. Definitive fixes (NIR sensor / learned invariant embeddings) are ROADMAP
   Phases 2–3. Interim policy: **enroll where you pay.**
2. **No liveness/anti-spoofing.** A printed photo may still authenticate. Challenge-response
   liveness is a Phase-2 item; treat terminals as attended in the demo.
3. **Small-cohort statistics.** Thresholds calibrated on few subjects generalize weakly;
   recalibrate as the cohort grows.
4. **Security floor deferred by agreement** (documented in ROADMAP Phase 1): merchant terminal
   auth, consumer PIN binding, rate limiting, HTTPS/secrets management. Acceptable only
   because the system holds no real money.
5. **Privacy posture:** raw images never stored; only unit-norm float32 embeddings — note
   embeddings are still biometric data and Phase 2 plans cancelable templates + encryption.
