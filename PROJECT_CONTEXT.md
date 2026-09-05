# PalmPay — Project Context (Session Continuation Doc)

> **Purpose:** everything a fresh session needs to continue this project without
> re-discovering history. Last updated: end of "accuracy sprint" session.
> Companion docs: `README.md` (Phase-1 biometric prototype), `ROADMAP.md` (PM phases/gates).

---

## 1. What PalmPay Is

Contactless palm biometric **payment demo system** ("Pay without your phone") built on a
Phase-1 biometric prototype (Flask + MediaPipe + OpenCV + SQLite). Consumers never touch a
device to pay: merchant enters amount at a POS terminal, customer presents palm, system
recognizes and debits automatically. Everything labeled **Demo / Prototype** — no real money.

- Product thesis & phased gates: see `ROADMAP.md`
- Biometric research basis & architecture diagram: see `README.md`

## 2. How To Run / Verify (environment notes)

```bash
# Environment: system python3 is 3.14 (NO deps). Use Python 3.11 via mise in .venv at project root:
# mise exec python@3.11 -- python -m venv .venv
# .venv/bin/pip install -r requirements.txt   # flask, opencv, numpy, pillow, mediapipe
# (Optional training: .venv/bin/pip install -r requirements-train.txt for PyTorch)

# Run server (port 5001, debug on, reloader OFF → restart manually after edits):
.venv/bin/python app.py                      # http://127.0.0.1:5001

# Tests (101 total Python tests + 4 C++ tests, must always be green):
.venv/bin/python -m unittest discover tests
ctest --test-dir cpp/build --output-on-failure

# End-to-end demo verification (13 scripted steps, pipeline mocked):
.venv/bin/python scripts/verify_demo_script.py

# Controlled deep model smoke test (enrollment, DB schema, matching, rejection):
.venv/bin/python scripts/smoke_test_deep.py

# Wipe biometric users/templates (PRESERVES merchant accounts m1–m4):
.venv/bin/python scripts/reset_biometrics.py --yes
```

Shell quirk: killing the server can hang the bash tool past its timeout — issue
`kill -9 $(pgrep -f "...app.py")`, ignore the timeout error, then relaunch with
`setsid <venv-python> app.py > log 2>&1 < /dev/null & disown; exit 0`.

**Admin creds:** user `admin` / password `palmpay2025` (`config.ADMIN_PASSWORD`).
**Merchants:** hardcoded in `config.MERCHANTS` (m1 Canteen/Food, m2 Photocopy/Services,
m3 Stationery/Retail, m4 Library/Services); seeded as `MERCH-*` users with MERCHANT wallets.
They are excluded from biometric user listings.

## 3. Current Biometric Approach (v4 — geometry fusion)

Model string: **`HogLbpFusion-v1-fm`** (texture, **3652-D**) + **`handgeo-v1`** (geometry, 20-D).
Alternative baselines available in `config.py`: CompCode (`CompCode-v1-fm`, 1536-D) and legacy MB-uLBP (`MB-uLBP2R-CLAHE-v2-fm`, 1888-D).
Phase-2 deep embedding model (`deep-palm-embedder-v1`, 512-D) is implemented and flag-gated (`DEEP_ENABLED=False`).

Pipeline (all inside `BiometricPipeline.process_frame`, `core/pipeline.py`):

1. **Hand detection** — MediaPipe Tasks HandLandmarker (`core/hand_detector.py`),
   model file committed at `models/hand_landmarker.task`.
2. **Hand geometry** (`core/geometry.py`, Stage 1.5) — 20-dim shape descriptor from raw
   landmarks: palm aspect, 5 finger chain lengths, 3 knuckle widths, 6 inter-tip distances
   (all ratios vs wrist→middle-MCP reference ⇒ scale/rotation/translation invariant) +
   **5 signed splay angles** vs the palm axis (mirror-sensitive ⇒ left ≠ right hand).
   Computed from landmarks ONLY — independent of ROI/masking/CLAHE.
3. **ROI extraction** (`core/roi.py`) — wrist/MCP anchors → affine warp to 128×128
   (scale = max(breadth, height·0.75)·1.05, BORDER_REPLICATE). Also projects MCPs
   through the matrix and returns `knuckle_y_roi`.
4. **Preprocessing** (`core/preprocessing.py`) — grayscale → CLAHE(2.0, (8,8)).
   **Gaussian blur DISABLED** (`GAUSSIAN_BLUR_KERNEL=(0,0)`).
5. **Capture quality gate** (`FrameQualityGate`) — rejects frames *before* scoring:
   Laplacian blur var <40, glare >8% pixels >245, mean brightness outside [40, 232],
   palm span <60px. Returns human guidance messages shown live in the UI.
6. **Finger masking** (`apply_finger_mask`) — rows ABOVE the projected MCP knuckle line
   replaced with neutral fill (mean of kept area), feathered edge (σ3), byte-exact below.
   Texture-channel only — geometry never reads pixels.
7. **Texture features** (`core/features.py`) — HOG + MB-uLBP Fusion:
   HOG block histograms (1764-D) + Multi-scale uniform LBP (1888-D) → **3652-D** unit-norm vector.
8. **Hardened matching** (`core/matcher.py`) — candidate qualifies only if
   (a) mean-template score ≥ T **AND** best-sample score ≥ T (dual texture pass),
   (b) **geometry distance ≤ `GEO_MAX_DIST`** (RMS deviation vs per-user enrollment
   mean/std, floored), and (c) no ambiguity (top1-top2 texture gap ≥ 0.02 among fully
   qualifying candidates). Explicit reasons: `below_threshold` / `geometry_mismatch` /
   `ambiguous`. Matcher dynamically evaluates `active_threshold` (0.85 when `DEEP_ENABLED=True`).
9. **Multi-frame fusion** (`/api/recognize`) — `{images:[...]}` burst (legacy `{image}`
   works). Mean query embedding + mean query geometry across quality-passed frames;
   decision needs matcher qualification AND ≥3-of-4 frame texture agreement
   (`unstable_capture` otherwise). Response fields additive incl. `geo_distance`,
   `geo_threshold`, `match_reason`.
10. **Thresholds** — **T = 0.95**, **GEO_MAX_DIST = 1.5** (calibrated on cohort).
    `/api/evaluate` returns `geometry_eval` (leave-one-out genuine vs cross-user
    impostor geometry distances) and suggests `suggested_geo_max_dist` under strict
    zero-FAR @ GAR≥0.90 policy.

**Measured motivation (why geometry exists):** LBP texture impostor scores on a 9-user
RGB-webcam cohort spanned 0.877–0.989 (mean 0.951) — texture alone can NEVER be safe here.
Geometry means separate ~3× (genuine 0.94 vs impostor 2.86) with mirror-sensitivity closing
the same-person-cross-hand hole.

### Enrollment (automated, fingerprint-style, pose-constrained)

`/enroll` + `static/js/enroll.js`: ONE button "🚀 Start Enrollment Scan". Camera starts,
probes `/api/process_frame` every 400 ms, auto-captures only quality-passed aligned frames,
1.3 s cooldown, condition script (neutral → tilt LEFT → tilt RIGHT → closer → same-spread →
flat-palm; **pitch prompt REMOVED** — foreshortening destroys geometry consistency),
near-duplicate skip (32×32 gray diff < 1.5%), auto-submits. Default **6 samples**.
**Pose-consistency gate (server, `/api/enroll`)**: samples with geometry deviation >
`ENROLL_GEO_CONSISTENCY_MAX_DIST` (1.5, absolute physical units vs per-dim median —
`enrollment_consistency_distances`) are dropped; enrollment fails with coaching if fewer
than `MIN_ENROLLMENT_SAMPLES` survive. Wallet auto-created CONSUMER.

### Payment authentication security model

- `/api/recognize` success ⇒ server-side `BiometricSessionStore` entry
  (`bio_session.py`; opaque token in Flask session cookie). TTL 30 s, single-use
  (burned on successful payment), invalidated by any failed scan or logout.
- `/api/payment {merchant_id, amount}` resolves the payer ONLY from that session —
  frontend-supplied `user_id` is ignored (test-proven). Amount validated (>0, ≤5000);
  atomic debit/credit + transaction row in one SQLite txn (`Database.process_payment`);
  `receipt_url` returned. Failure paths log FAILED_AMOUNT / FAILED_BALANCE audit rows;
  FAILED_AUTH logs nothing (identity unknown).
- Consumer dashboard is READ-ONLY (balance/history/top-up-request). No Pay button/tab.
- POS camera lifecycle: OFF until "Start Payment", ON during the sale, OFF on
  success/decline/cancel/POS-tab-leave.

## 4. Repository Map (payment-era & Phase-2 additions ★)

```
app.py                    Flask routes: legacy pages + payment pages/APIs + dataset capture
config.py                 ALL tunables (quality gate, fusion, masking, enrollment, deep, liveness, C++)
bio_session.py ★          Server-side biometric session store (TTL/single-use)
database/db.py            SQLite layer: users/templates/wallets/transactions/
                          cashout_requests/topup_requests + reset-safe loader filters
core/                     Biometric pipeline:
  hand_detector.py        MediaPipe Tasks HandLandmarker wrapper
  geometry.py ★           Hand geometry extractor (20-D invariant descriptor)
  roi.py                  Palm ROI extraction & affine normalization
  preprocessing.py        CLAHE & grayscale enhancement
  features.py             HOG (1764-D), MB-uLBP (1888-D), and CompCode (1536-D)
  palm_model.py ★         BasePalmModel, ClassicalPalmModel, DeepPalmModel (ONNX 512-D)
  matcher.py              Hardened 1:N matcher + geometry gate + active_threshold routing
  liveness.py ★           Landmark challenge-response liveness detector & session store
  cpp_bridge.py ★         C++ native subprocess bridge fallback
  pipeline.py             BiometricPipeline with dynamic model sync & quality gate
  evaluation.py           Cross-matching ROC/EER & zero-FAR threshold calibration
templates/                Legacy dark-theme pages + fintech UI:
                          pay_base/admin_base/partials/pos_terminal, consumer, pay,
                          merchant, transactions, receipt, admin_*
static/css/style.css      Legacy theme + fintech design system scoped under body.pay
static/js/                camera.js (PalmCamera), enroll.js (auto-scan + consent),
                          pay.js (merchant POS state machine), consumer.js,
                          merchant.js (tabs/chart/pagination/cashout), liveness.js ★, recognize/test
cpp/ ★                    Native C++ core (CMake, matcher, liveness, preprocess, CLI)
requirements.txt          Core runtime dependencies (CPU-only, no PyTorch)
requirements-train.txt ★  Isolated training dependencies (PyTorch CPU-only)
training/ ★               DeepPalm training pipeline:
  dataset.py              Disjoint subject splitter, ROI loader, data augmentation
  model.py                MobileNetV3-Small deep embedder (512-D)
  train.py                ArcFace loss + disjoint pairwise verification (EER/Zero-FAR)
tests/                    101 automated Python unit tests across 13 test modules:
                          test_accuracy, test_cpp_bridge, test_deep_model, test_geometry,
                          test_liveness, test_payment, test_payment_liveness, test_pipeline,
                          test_quality_gate, test_routes, test_templates, test_training_pipeline
scripts/
  reset_biometrics.py     wipe biometric users (keeps merchants)
  verify_demo_script.py   13-step end-to-end check (pipeline mocked)
  smoke_test_deep.py ★    Controlled 8-step DeepPalm end-to-end integration test
  eval_deep_model.py ★    Head-to-head Classical vs Deep evaluation benchmark
  export_deep_model.py ★  Export dummy/trained ONNX model (3-channel normalized)
  purge_dataset.py ★      Standing privacy tool to wipe ephemeral pilot ROIs
  compare_models.py       Comparison harness between feature extractors
ROADMAP.md                PM phases, pilot gates, parked ideas
PROJECT_CONTEXT.md        this file
models/hand_landmarker.task MediaPipe model (committed)
```

## 5. Pages & APIs (payment era)

Pages: `/` legacy hub · `/enroll` auto-scan · `/recognize` legacy live recognition ·
`/users` · `/test` calibration suite · `/consumer?user_id=` mobile wallet ·
`/pay` standalone POS (same terminal also embedded as merchant tab) ·
`/merchant?merchant_id=m1&view=dashboard|pos|transactions|cashout` ·
`/transactions?role=consumer|merchant&user_id=` · `/receipt/<txn_uuid>` ·
`/admin(/login|/dashboard|/topup|/cashouts)`.

APIs: `/api/payment` · `/api/topup` (admin direct) · `/api/topup/request` +
`/api/topup_request/<id>/approve` (admin) · `/api/cashout` +
`/api/cashout/<id>/approve` (admin) · `/api/wallet/<uid|m1..m4>` ·
`/api/transactions[?filters]` + `/api/transactions/export` (CSV) ·
`/api/merchant/stats/<mid>` · legacy: `/api/process_frame`, `/api/enroll`,
`/api/recognize`, `/api/users…`, `/api/config`, `/api/evaluate`, `/api/export_csv`.

DB schema additions: `wallets(user_id UNIQUE, role, balance)` ·
`transactions(id UUID PK, sender_user_id, merchant_id, merchant_name, amount,
status SUCCESS/FAILED_BALANCE/FAILED_AUTH/FAILED_AMOUNT, biometric_score, note)` —
note encodes type: `TOPUP` / `CASHOUT*` / else PAYMENT · `cashout_requests(PENDING/APPROVED)` ·
`topup_requests(PENDING/APPROVED)` (consumer-initiated, admin-approved).

## 6. Decision Log (why things are the way they are)

1. **Merchant-side POS only.** Consumers never initiate payments; payer identity comes
   exclusively from the server-side biometric session. Frontend `user_id` ignored.
2. **Consumer dashboard read-only** — no Pay button/tab anywhere (regression-tested).
3. **Camera privacy lifecycle** — webcam opens only when a payment starts; closes on
   completion/cancel/tab switch; idle mode makes zero API calls (no stale sessions).
4. **Everything-in-one merchant console** — sidebar tabs Dashboard/Pay·POS/Transactions/
   Cash Out; POS shared via Jinja partial `partials/pos_terminal.html`.
5. **Accuracy Sprint v1** (after real false-accept): capture quality gate; hardened matcher
   (dual-pass + ambiguity); strict eval policy zero-FAR@GAR≥90% (replaced FAR≤5% rule);
   4-frame fusion requiring ≥3 agreeing frames (`unstable_capture` otherwise).
6. **Feature upgrade v2** (left hand STILL matched @0.80 ⇒ feature space, not logic):
   Gaussian blur removed; LBP radii (1,2)=1888-D; model-name bump invalidates old
   templates via loader filter (`feature_dim`+`model_name` WHERE clause) — stale rows can
   never crash or contaminate matching.
7. **Finger masking** above MCP knuckle line (default variant; neutral fill + feather).
8. **Automated enrollment**: one-button scan, quality-gated auto-capture, cooldown,
   condition prompts, near-duplicate skip, auto-submit, 6 samples.
9. **Diversity prompts target cross-room failure**: explicit light-angle and pitch entries.
10. **Threshold discipline**: never guess (a blind 0.90 was discussed and rejected) —
    calibrate empirically via `/test` strict policy; bake measured value into config after.
11. **Pilot gates LOCKED**: 0 observed false-accepts across ~2,000 transactions, FRR <5%,
    median pay <5s (see ROADMAP.md Phase 1→2 gate).
12. **Back-of-hand / knuckle-print idea PARKED** as Phase-3 multi-modal item (real
    modality, wrong timing; doesn't fix illumination anyway).
13. **Never store raw palm images** (privacy + attack surface) — upheld throughout.
14. Single CSS file `style.css`: legacy theme untouched; fintech system scoped `body.pay`.
15. **Geometry fusion (Sprint v4)**: LBP texture measured unsafe on RGB cohorts (impostor
    0.88–0.99) → added 20-dim landmark-geometry channel (Sanchez-Reillo TPAMI 2000 basis);
    AND-fusion with texture; signed splay angles make left≠right hand distinguishable.
16. **Pitch prompt removed + pose-consistency gate**: calibration showed session-to-session
    pose drift (esp. foreshortening) dominates geometry error; enrollment now enforces a
    flat-pose envelope server-side (absolute-unit outlier rejection) instead of trusting
    user discipline. MAD/relative scales rejected — they cannot detect globally-wild
    submissions (caught by unit test).
17. **Calibration discipline upheld**: interim T=0.95 / GEO_MAX_DIST=1.5 baked with measured
    FAR/GAR in comments; final numbers pending flat-palm cohort re-enrollment.
18. **Phase-2 Flag-Gated Architecture**: Deep palm inference, landmark challenge-response liveness,
    native C++ core, and dataset capture are disabled by default (`DEEP_ENABLED=False`,
    `LIVENESS_ENABLED=False`, `NATIVE_CPP_ENABLED=False`, `PAYMENT_REQUIRES_LIVENESS=False`,
    `DATASET_CAPTURE_ENABLED=False`). Phase-0 demo remains default, isolated, and 100% stable.
19. **Dynamic Model Derivation for Enrollment**: Root cause of deep recognition bug resolved:
    template enrollment dynamically queries `pipeline.model.get_model_name()` and
    `pipeline.model.get_feature_dim()` rather than hardcoding classical constants. Database
    template loader safely isolates models by `model_name` and `feature_dim`.
20. **Disjoint Validation via Pairwise Verification**: ArcFace classification loss cannot compute
    cross-entropy on unseen validation identities. In `training/train.py`, validation split is evaluated
    via genuine/impostor pairwise cosine similarity distributions, EER, and separation metrics.
21. **Channel & Normalization Parity**: MobileNetV3-Small standard 3-channel input matched across
    training, ONNX export, and serving runtime (`DeepPalmModel.extract_embedding` dynamically checks
    ONNX input channels and maps grayscale to normalized $[-1.0, 1.0]$).
22. **Active Threshold Routing & Dead Config Audit**: `PalmMatcher.active_threshold` dynamically
    evaluates `DEEP_MATCH_THRESHOLD` (0.85) when `DEEP_ENABLED=True` and falls back to classical
    threshold (0.64). `LIVENESS_MIN_SCORE` (0.50) enforces composite sequence confidence.
    `DEEP_FALLBACK_TO_CLASSIC` is documented as reserved multi-model scaffolding.
23. **Privacy-Preserving Ephemeral Data Flywheel**: Raw palm images are never stored. Pilot training
    data is collected as $128\times128$ CLAHE+masked ROI crops strictly when `DATASET_CAPTURE_ENABLED=True`
    and user grants consent. `scripts/purge_dataset.py` wipes ephemeral files on demand.

## 7. Current State & Evaluation Results

- **Tests: 101/101 passing** (Python test suite across classical recognition, payment system,
  accuracy suite, finger-masking, geometry, deep model abstraction, liveness verification,
  native C++ bridge, and Phase-2 DeepPalm training/evaluation/integration suite). Run after every change; keep green.
- **C++ Tests: 4/4 passing** (`test_matcher`, `test_liveness`, `test_preprocess`, `test_cli`).
- **Demo script: all 13 steps verified** (multi-frame path included).
- **Deep Smoke Test Verified:** 8-step controlled integration script (`scripts/smoke_test_deep.py`) passes end-to-end (enrollment, DB schema, template loading, genuine match, impostor rejection with `deep_below_threshold`, test cleanup).
- **Phase-2 Spike Implemented (Flag-Gated):**
  - Deep learning palm embedding pathway (`core/palm_model.py`, `DeepPalmModel`, ONNX runtime support + deterministic fallback).
  - Landmark challenge-response liveness detection (`core/liveness.py`, `LandmarkLivenessDetector`, `LivenessSessionStore`, gesture sequences: OPEN_PALM, FIST, SPREAD).
  - Optional C++ acceleration core (`cpp/`, `palmpay_native` CLI, `core/cpp_bridge.py`).
  - Strict server-side security: liveness tokens and payer identities resolved strictly on the server; raw images never stored.
- **Real-palm calibration (2026-08-23, 4-user cohort):** texture impostor 0.877–0.989 ⇒
  texture alone unsafe (motivates geometry); joint texture⊕geometry at T=0.95/D=1.5 gave
  FAR 1.4% / GAR 67%, with ALL residual FAR from one similar-shape user pair + one bad
  enrollment sample. Pose-consistency gate + prompt fix shipped in response; final
  calibration pending flat-palm cohort re-enrollment.
- **mediapipe pinned `<1.0`** — 1.x TF-Lite runtime is SIGKILLed by the kernel during
  XNNPACK init on current Arch kernels (Tiger Lake). 0.10.35 verified.
- **DB state:** cohort re-enrollment in progress under the new pose-consistency protocol.
- **Env:** `.venv/` at project root (Python 3.11 via mise); see AGENTS.md rules.

## 8. Exact Next Steps (in order)

1. **Re-enroll cohort (one sitting, flat palms)** — all users under the pose-consistency
   protocol; include the operator's LEFT hand as decoy impostor.
2. **Recalibrate:** `/api/evaluate` → `geometry_eval` → set `GEO_MAX_DIST` (and confirm T)
   by the strict zero-FAR rule; bake values + measured FAR/GAR into config comments.
3. **Acceptance test:** decoy left hand rejected 10/10 (mirror-sensitive angles); strangers
   rejected; each registered palm pays reliably; low-balance path shows orange screen.
4. **Rehearse demo:** fresh snapshot → scripted flow (top-up → POS pay → receipt →
   dashboards → wrong-palm ✗ → cash-out approval).
5. **Phase-2 Data Flywheel (Pilot):** Enable `DATASET_CAPTURE_ENABLED=True` with volunteer consent
   to collect real $128\times128$ ROIs, train MobileNetV3 with `training/train.py`, export with
   `scripts/export_deep_model.py`, and evaluate on real palms before toggling `DEEP_ENABLED=True`.
6. **After demo:** ROADMAP Phase 1 (merchant terminal auth, consumer PIN, rate limiting,
   HTTPS + secrets, backup drill).

## 9. Open Risks / Known Limitations (do not forget in report/viva)

- Illumination sensitivity is mitigated, not solved: RGB + classical features remain
  environment-sensitive → enroll-at-venue policy for demo; NIR/deep embeddings later.
- Threshold calibrated on few subjects generalizes weakly → recalibrate as cohort grows
  (~every 10× users).
- Liveness & Anti-Spoofing posture: Challenge-response landmark liveness is implemented
  and flag-gated (`LIVENESS_ENABLED=False` for demo stability); optical NIR anti-spoofing is Phase 2/3.
- Security floor items deferred by agreement: merchant terminal auth, consumer PIN,
  rate limiting, HTTPS, secrets management (Phase 1 list, ROADMAP.md).
- Virtual environment discipline: always run via `.venv/bin/python` (Python 3.11 managed via mise).
  System python is 3.14 with no dependencies.

