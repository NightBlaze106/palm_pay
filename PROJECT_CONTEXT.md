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
# Environment: system python3 is 3.14 (NO deps). A Python 3.11 venv was created at
# /tmp/opencode/palmpay_venv — /tmp MAY BE WIPED between sessions. Recreate if missing:
python3.11 -m venv /tmp/opencode/palmpay_venv
/tmp/opencode/palmpay_venv/bin/pip install -r requirements.txt   # flask, opencv, numpy, pillow, mediapipe

# Run server (port 5000, debug on, reloader OFF → restart manually after edits):
python app.py                      # http://127.0.0.1:5000

# Tests (54 total, must always be green):
python -m unittest discover tests

# End-to-end demo verification (13 scripted steps, pipeline mocked):
python scripts/verify_demo_script.py

# Wipe biometric users/templates (PRESERVES merchant accounts m1–m4):
python scripts/reset_biometrics.py --yes
```

Shell quirk: killing the server can hang the bash tool past its timeout — issue
`kill -9 $(pgrep -f "...app.py")`, ignore the timeout error, then relaunch with
`setsid <venv-python> app.py > log 2>&1 < /dev/null & disown; exit 0`.

**Admin creds:** user `admin` / password `palmpay2025` (`config.ADMIN_PASSWORD`).
**Merchants:** hardcoded in `config.MERCHANTS` (m1 Canteen/Food, m2 Photocopy/Services,
m3 Stationery/Retail, m4 Library/Services); seeded as `MERCH-*` users with MERCHANT wallets.
They are excluded from biometric user listings.

## 3. Current Biometric Approach (v2-fm)

Model string: **`MB-uLBP2R-CLAHE-v2-fm`**, embedding dim **1888** float32 unit-norm.

Pipeline (all inside `BiometricPipeline.process_frame`, `core/pipeline.py`):

1. **Hand detection** — MediaPipe Tasks HandLandmarker (`core/hand_detector.py`),
   model file committed at `models/hand_landmarker.task`.
2. **ROI extraction** (`core/roi.py`) — wrist/MCP anchors → affine warp to 128×128
   (scale = max(breadth, height·0.75)·1.05, BORDER_REPLICATE). Now also projects MCPs
   through the matrix and returns `knuckle_y_roi`.
3. **Preprocessing** (`core/preprocessing.py`) — grayscale → CLAHE(2.0, (8,8)).
   **Gaussian blur DISABLED** (`GAUSSIAN_BLUR_KERNEL=(0,0)`): it erased discriminative
   micro-texture; noise is handled by the quality gate + fusion instead.
4. **Capture quality gate** (`FrameQualityGate`) — rejects frames *before* scoring:
   Laplacian blur var <40, glare >8% pixels >245, mean brightness outside [40, 232],
   palm span <60px. Returns human guidance messages shown live in the UI.
5. **Finger masking** (`apply_finger_mask`) — rows ABOVE the projected MCP knuckle line
   replaced with neutral fill (mean of kept area), feathered edge (σ3), byte-exact below
   (`np.rint`). Config: `FINGER_MASK_ENABLED/OFFSET_PX`. Purpose: finger spread/pose stops
   leaking into histograms.
6. **Features** (`core/features.py`) — multi-scale uniform LBP: radii **(1,2)**, 4×4 grid,
   59 uniform bins each → 16 blocks × 59 bins × 2 scales = **1888-D**, block-L2 + global-L2.
7. **Hardened matching** (`core/matcher.py`) — per-user **mean-template score** AND
   **best-sample score** must BOTH clear threshold (kills max-of-noisy-scores inflation);
   ambiguity guard rejects when top-1/top-2 both qualify within 0.02 gap;
   `frame_pass_score()` helper for per-frame checks; cosine clamped [-1,1], dimension-safe.
8. **Multi-frame fusion** (`/api/recognize`) — accepts `{images:[...]}` burst (legacy
   `{image}` still works identically). Decision = match on MEAN query embedding **AND**
   ≥ `FUSION_MIN_PASS_FRAMES`(3) of up-to-4 frames individually clearing threshold,
   else reason `unstable_capture`. Response contract preserved; fusion fields additive.
9. **Threshold** — still placeholder **0.78** until real-palm calibration. `/test` page's
   evaluator suggests the operating point under strict policy: highest T with ZERO observed
   impostor accepts while GAR ≥ `EVAL_SUGGESTION_MIN_GAR`(0.90)
   (`core/evaluation.py`, response field `suggestion_policy`).

### Enrollment (automated, fingerprint-style)

`/enroll` + `static/js/enroll.js`: ONE button "🚀 Start Enrollment Scan". Camera starts,
probes `/api/process_frame` every 400 ms, auto-captures only quality-passed aligned frames,
1.3 s cooldown between samples, rotates a 6-step condition script (neutral → tilt LEFT →
tilt RIGHT → closer → toward-light → pitch-away; `config.ENROLL_POSE_PROMPTS`), skips
near-duplicates (32×32 gray diff < `ENROLL_DUPLICATE_DIFF_PCT`=1.5%), auto-submits.
Default **6 samples** (`config.DEFAULT_ENROLLMENT_SAMPLES`). Wallet auto-created CONSUMER.

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

## 4. Repository Map (payment-era additions ★)

```
app.py                    Flask routes: legacy pages + payment pages/APIs (see §5)
config.py                 ALL tunables (quality gate, fusion, masking, enrollment, merchants, admin)
bio_session.py ★          Server-side biometric session store (TTL/single-use)
database/db.py            SQLite layer: users/templates/wallets/transactions/
                          cashout_requests/topup_requests + reset-safe loader filters
core/                     Biometric pipeline (hand_detector, roi, preprocessing,
                          features, palm_model, pipeline, matcher, evaluation)
templates/                Legacy dark-theme pages + fintech UI:
                          pay_base/admin_base/partials/pos_terminal, consumer, pay,
                          merchant, transactions, receipt, admin_*
static/css/style.css      Legacy theme + fintech design system scoped under body.pay
static/js/                camera.js (PalmCamera), enroll.js (auto-scan),
                          pay.js (merchant POS state machine), consumer.js,
                          merchant.js (tabs/chart/pagination/cashout), recognize/test (legacy)
tests/                    test_pipeline/test_api/test_evaluation (legacy 13),
                          test_payment.py (22), test_accuracy.py (19 incl. finger-mask)
scripts/reset_biometrics.py      wipe biometric users (keeps merchants)
scripts/verify_demo_script.py    13-step end-to-end check (pipeline mocked)
ROADMAP.md                PM phases, pilot gates, parked ideas
PROJECT_CONTEXT.md        this file
models/hand_landmarker.task      MediaPipe model (committed)
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

## 7. Current State & Evaluation Results

- **Tests: 54/54 passing** (13 legacy recognition/api/evaluation + 22 payment-system +
  19 accuracy-suite incl. 6 finger-mask tests). Run after every change; keep green.
- **Demo script: all 13 steps verified** (multi-frame path included; step 6 shows
  “N/N frames agree”).
- **Live server smoke-tested** after every milestone (pages 200, JS syntax-checked with
  `node --check`, inline configs render valid values).
- **Real-palm evaluation: NOT DONE YET.** All accuracy evidence is synthetic/unit-level.
  The motivating incidents were real: left palm accepted @0.80 (pre-v2 extractor);
  room→hall recognition failures; finger-spread sensitivity — structural fixes shipped,
  **empirical validation pending** (calibration session below).
- **DB state:** clean — only 4 MERCH-* users with wallets, 0 biometric users,
  0 templates. Threshold 0.78 (uncalibrated placeholder).

## 8. Exact Next Steps (in order)

1. **Venue setup:** mount laptop + small LED light at demo spot; open `/enroll`.
2. **Enroll:** yourself (right palm), your LEFT palm as decoy user, then 5–10 classmates.
   Follow on-screen prompts; quality gate will coach (“move closer”, “too blurry”).
3. **Calibrate:** `/test` → Run Evaluation → note genuine_min vs impostor_max → click
   Use Suggested → Apply Threshold. Target margin: `T − impostor_max ≥ 0.05` and
   `genuine_min − T ≥ 0.03`. If margin impossible → report numbers back before touching
   the extractor again (next levers documented: raise span floor; multi-scale variants).
4. **Acceptance test:** decoy left palm rejected 10/10 at `/pay`; each registered palm
   pays reliably; wrong-person attempts fail; low-balance path shows orange screen.
5. **Rehearse demo:** fresh snapshot (`reset_biometrics.py --yes` then re-seed 2–3 users),
   scripted flow: top-up → POS pay → receipt → dashboards → wrong-palm ✗ → cash-out approval.
6. **After demo:** start ROADMAP Phase 1 (merchant terminal auth first, consumer PIN,
   rate limiting, HTTPS + secrets out of source, backup drill).

## 9. Open Risks / Known Limitations (do not forget in report/viva)

- Illumination sensitivity is mitigated, not solved: RGB + classical features remain
  environment-sensitive → enroll-at-venue policy for demo; NIR/deep embeddings later.
- Threshold calibrated on few subjects generalizes weakly → recalibrate as cohort grows
  (~every 10× users).
- No liveness/anti-spoofing yet (printed palm photo may fool it) — Phase 2.
- Security floor items deferred by agreement: merchant terminal auth, consumer PIN,
  rate limiting, HTTPS, secrets management (Phase 1 list, ROADMAP.md).
- `/tmp` venv volatility (see §2) — recreate venv if imports fail in a fresh session.
