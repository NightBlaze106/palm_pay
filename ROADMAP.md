# PalmPay — Product Roadmap

**Thesis:** *"Pay without your phone."* Reliability at the counter beats algorithmic
sophistication. Trust compounds; demos don't.

**Strategic asset:** every pilot transaction logs a genuine/impostor score pair under
real conditions. That dataset funds the deep-embedding upgrade, per-site calibration,
and fraud analytics later. Getting a small pilot live fast > perfecting the model first.

---

## Phase 0 — Demo (current)
Goal: protect against catastrophic failure in front of an audience.

- [x] Zero-FAR decision policy: dual mean/best-sample match, ambiguity guard, 3/4-frame fusion
- [x] Capture quality gate (blur / glare / brightness / palm-span)
- [x] Finger masking above knuckle line (`MB-uLBP2R-CLAHE-v2-fm`)
- [x] Condition-scripted auto-enrollment (6 samples: neutral → tilt L/R → closer → light → pitch)
- [ ] Enroll AT the venue + fixed clip-on light at demo station
- [ ] Calibrate threshold on real palms via `/test` (strict `zero_far_gar_ge_90pct` policy), apply
- [ ] Acceptance: unregistered palms rejected 10/10; registered users ≥90% pass; full payment loop <8s

**Explicitly not building:** dorsal/knuckle surface, deep learning, liveness (→ Phase 2/3).

## Phase 1 — Campus pilot (weeks 1–6 post-demo): earn trust
1. Merchant terminal authentication (today anyone can open `/pay` as any merchant)
2. Consumer identity binding (enrollment-time PIN; kill `/consumer?user_id=` IDOR)
3. Rate limiting on `/api/recognize`; audit logging of admin actions; HTTPS + secrets out of source
4. Pilot protocol: 20–50 users, ONE counter, fixed light; weekly GAR/FAR review from logged scores
5. Ops: nightly DB backup + tested restore; runbook (restart command, manual IOU fallback policy)

**Gate to Phase 2:** 0 observed false-accepts across ~2,000 transactions · FRR <5% · median pay <5s.

## Phase 2 — Robustness engineering (next semester)
- Deep embeddings trained on pilot data (lighting/pose augmentation) via existing `BasePalmModel` swap
- Challenge-response liveness → cancelable biometric templates → encrypted storage (SQLCipher)
- Postgres migration; double-entry ledger refactor; automated recalibration cadence (~every 10× users)
- NIR camera + IR illuminator pilot at flagship counter

## Phase 3 — Productization
- Multi-modal fusion: palmprint + finger-knuckle-print (RGB-friendly; dorsal-vein needs NIR) — includes
  surface-tagged template groups and per-surface thresholds
- Offline-tolerant POS mode; DPDP compliance pack (consent flow, retention policy); POS enclosure design

## Standing principles
1. False accepts are existential; false rejects are UX.
2. Thresholds by rule (`zero-FAR @ ≥90% GAR`), never intuition.
3. Never store raw biometric images.
4. One great counter beats ten mediocre ones — scale only after one site proves stable.
