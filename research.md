# PalmPay — Research Paper Blueprint (`research.md`)

> **Purpose:** everything needed to turn this project into a proper college research paper.
> Recommended paper structure, pre-drafted sections you can adapt, the exact experiments to
> run and numbers to report, figure/table checklist, full bibliography, and future-work
> directions. Items marked **[CALIBRATE]** need your real `/test` numbers; everything else is
> ready to use.
>
> Format suggestion: IEEE conference double-column template (or your department's format),
> 6–10 pages. Every claim below is traceable to code/docs in this repo (`README.md`,
> `PROJECT_CONTEXT.md`, `ROADMAP.md`) — never invent results.

---

## 0. Title Options (pick one)

1. *PalmPay: A Contactless Palmprint-Based Payment System Using Multiscale Uniform Local Binary Patterns on Commodity RGB Cameras*
2. *Reliability at the Counter: An Explainable Palmprint Payment Prototype with Capture-Quality Gating and Finger Masking*
3. *Phone-Free Campus Payments: Landmark-Guided Multiscale MB-uLBP Recognition with Multi-Frame Fusion*

**Keywords:** palmprint recognition · local binary patterns · contactless biometrics · CLAHE · multi-frame score fusion · biometric payments · MediaPipe landmarks · threshold calibration

---

## 1. Abstract (draft — finalize after calibration)

> Contactless biometric payments typically require specialized near-infrared hardware,
> limiting adoption in cost-sensitive campus deployments. We present **PalmPay**, a phone-free
> payment prototype that authenticates customers from an ordinary RGB webcam using a fully
> classical, explainable pipeline: MediaPipe hand landmarks guide an affine-normalized
> 128×128 palm ROI; CLAHE enhances flexion creases; a capture-quality gate rejects degraded
> frames before scoring; rows above the projected metacarpal knuckle line are masked out to
> remove pose leakage; and multiscale uniform local binary pattern histograms (radii 1 and 2,
> 4×4 grid, 1888 dimensions) are matched by cosine similarity under a hardened decision rule
> requiring agreement between mean-template and best-sample scores plus ≥3-of-4 frame
> consensus. Operating thresholds are chosen exclusively by an evaluation policy enforcing
> zero observed false accepts at ≥90% genuine acceptance. On [N] subjects enrolled under
> venue lighting, the system achieves GAR of [x]% at FAR = 0 with median payment latency of
> [y] s on an Intel i3 CPU. We document three empirically-driven design iterations — quality
> gating, feature-space widening, and geometric masking — each motivated by observed failure
> modes, and present a threat model and deployment roadmap. **[CALIBRATE]**

---

## 2. Recommended Paper Structure

| § | Section | Source material |
|---|---|---|
| I | Introduction | §3 below |
| II | Literature Survey | §4 below + bibliography group A–D |
| III | System Design & Methodology | README §1–§5 (pipeline stages 1–8, enrollment, payment security model) |
| IV | Empirical Design Iterations | §5 below — the v1→v2→v3 ablation narrative (your strongest original content) |
| V | Experimental Setup & Results | §6 below — run protocol, fill tables |
| VI | Security & Privacy Analysis | §7 below |
| VII | Limitations | README §11, verbatim-adaptable |
| VIII | Future Work | §8 below |
| IX | Conclusion | short; restate thesis + headline numbers |

---

## 3. Introduction — talking points (draft)

- **Problem:** campus micro-payments fail when phones are dead/lost/queued; contact hygiene persists.
- **Constraint:** target hardware is an i3 laptop + commodity webcam; no GPU, no NIR sensor.
- **Thesis:** reliability at the counter beats algorithmic sophistication — an explainable classical pipeline, disciplined threshold governance, and environment-aware enrollment deliver demo-safe accuracy today while logging data that funds learned embeddings later.
- **Contributions (number them in the paper):**
  1. End-to-end open pipeline: landmark-guided ROI → CLAHE → finger-masked multiscale MB-uLBP → dual-signal hardened matching → multi-frame fusion.
  2. Three documented failure-driven iterations (quality gate / feature widening / geometric masking) with reasoning preserved — reproducible engineering evidence.
  3. Zero-FAR threshold governance policy suitable for payment contexts.
  4. Server-side session-based payment authorization resistant to client-side identity spoofing.

---

## 4. Literature Survey — organization map

Four threads; cite 2–4 works per thread (full list in §10):

1. **Palmprint as a modality** — Zhang et al. 2003 (foundational TPAMI); Kong et al. 2009 survey.
2. **Texture descriptors** — Ojala et al. 2002 ($LBP^{u2}$, multiresolution); Zhang L. et al. 2007 (Multi-Block LBP histograms).
3. **Contactless normalization & preprocessing** — Sánchez-Brizuela et al. 2023 (landmark-based affine normalization); Amrouni et al. 2022 (CLAHE for contactless ROIs); Wang et al. 2004 (Self-Quotient Images — cite as *future* illumination lever).
4. **Systems, fusion & trust** — Ross & Jain 2003 (information-fusion framing for our mean+best-sample and frame-level fusion); Zhang F. et al. 2020 (MediaPipe Hands); Zhang D. et al. 2010 (finger-knuckle-print — future multimodal).

Close with the gap statement: *"No published turnkey pipeline combines commodity-RGB palmprint authentication with payment-grade zero-FAR decision governance on CPU-only hardware."*

---

## 5. Empirical Design Iterations (Section IV gold)

Your most original material: three iterations, each triggered by a *measured* failure.

| Iteration | Observed failure | Root cause | Fix shipped | Verification |
|---|---|---|---|---|
| **v1 — Decision hardening** | Unregistered left palm accepted at score ≈ 0.80 | Max-of-noisy-scores inflation; single-frame flukes; old policy tolerated FAR ≤ 5% | Dual mean+best-sample pass; 0.02 ambiguity guard; 3-of-4 fusion; strict zero-FAR@GAR≥90% suggestion policy | Unit + endpoint tests; fusion counters in API response |
| **v2 — Feature-space widening** | Impostor distribution still reached ~0.80 ⇒ distributions overlapped | Gaussian σ=0.5 blur erased ridge micro-texture; single-scale LBP too coarse | Blur removed; radii (1,2) concatenated → 1888-D; model-versioned templates; stale-model loader filtering | Dimension/norm unit tests; loader exclusion test |
| **v3 — Geometric & environmental robustness** | Palm enrolled in one room failed in another hall; finger spread shifted scores | Finger area leaked above knuckles into block histograms; single-condition enrollment overfit | Knuckle-line finger masking (feathered neutral fill); condition-scripted auto-enrollment (light/pitch prompts); near-duplicate capture rejection | Mask geometry unit tests incl. byte-exactness; live smoke |

Paper framing: each row = hypothesis → intervention → measurable outcome. **[CALIBRATE]** the outcome column with before/after genuine-min vs impostor-max from `/test`.

---

## 6. Experimental Protocol (run this, then fill every table)

### Setup to describe
- Hardware: Intel i3-1115G4, [RAM], USB webcam 640×480 @ [fps]; OS.
- Cohort: **≥10 subjects**, right palms registered; each subject's other hand enrolled as decoy impostor.
- Conditions: venue lighting (+ optionally one alternate room for cross-condition test).

### Metrics (all produced by built-in tooling — say so)
| Metric | Source |
|---|---|
| Genuine/impostor distributions (min, max, μ, σ, count) | `/api/evaluate` JSON |
| Sweep GAR/FAR/FRR vs T (0→1 step .02); EER estimate | same response |
| Selected operating point T*, policy string | same, `suggestion_policy` |
| Latency (extraction / matching / total ms) | `/api/recognize` fields; report median + p95 |
| Payment success rate & failure-reason breakdown | `transactions` ledger statuses |

### Tables/figures checklist
1. **Table I — Score separation:** genuine_min / impostor_max / margins at T* (targets: T*−impostor_max ≥ 0.05, genuine_min−T* ≥ 0.03). **[CALIBRATE]**
2. **Fig. 1 — Score histograms**, genuine vs impostor overlay (CSV export button on `/test`).
3. **Fig. 2 — Threshold sweep**, GAR & FAR vs T, T* marked.
4. **Fig. 3 — Pipeline diagram** (redraw README §1 for print).
5. **Fig. 4 — Finger-mask visual proof**: raw ROI vs masked ROI thumbnails.
6. **Table II — Ablation** v1 / +v2 / +v3 on impostor_max & cross-room genuine_min *(or narrate qualitatively if re-testing is impractical)*.
7. **Table III — Latency** median/p95 across N recognitions.
8. **Decoy result:** X/10 wrong-hand attempts rejected at POS. **[CALIBRATE]**

### Reporting integrity rules (state explicitly)
All scores computed by the shipped evaluator; no synthetic augmentation; subjects consented; raw images never stored (embeddings only).

---

## 7. Security & Privacy Analysis (Section VI skeleton)

- **Attack surface:** frame replay; client-side identity injection; merchant impersonation; session theft; template theft from DB.
- **Controls implemented:** server-side opaque-token sessions (30 s TTL, single-use, burned on success, invalidated by failed scan); payer identity never read from request body (regression-tested); atomic SQLite transactions (`BEGIN IMMEDIATE`) with audit rows for FAILED_AMOUNT/FAILED_BALANCE; server-side amount validation; ambiguity rejection when two identities tie within 0.02.
- **Accepted gaps (be honest):** no liveness detection (printed-photo attack possible — mitigated by attended terminals in demo scope); merchant terminal auth deferred; HTTP + in-source secrets acceptable only because ledger holds no real money. Map each gap to its ROADMAP phase.
- **Privacy:** raw palm images never persisted; templates are unit-norm float32 vectors tagged with model name/dimension; stale models excluded at load time; cancelable templates + encryption cited for future work (Ratha et al.).

---

## 8. Future Work (Section VIII — pick 4–5, ordered)

1. **Deep invariant embeddings** trained on pilot-collected pairs with lighting+pose augmentation, swapped behind the existing `BasePalmModel` interface (zero API change).
2. **NIR acquisition** at flagship counters — removes ambient-light dependence; enables dorsal-vein modality.
3. **Multimodal fusion: palmprint + finger-knuckle-print** (RGB-friendly; Zhang D. et al. 2010) — surface-tagged template groups, per-surface thresholds.
4. **Challenge-response liveness** (pose prompts as implicit challenge; micro-motion / PPG signals).
5. **Cancelable templates + encrypted storage** (Ratha et al.; SQLCipher).
6. **Illumination normalization pre-LBP:** Self-Quotient Image / DoG high-pass (Wang et al. 2004) as drop-in Stage-3 upgrade.
7. **Pilot hardening:** merchant terminal auth, consumer PIN binding, rate limiting, HTTPS, backup drills (ROADMAP Phase 1 gates: 0 FA across ~2k txns, FRR < 5%, median pay < 5 s).
8. **Regulatory alignment:** India DPDP Act 2023 consent + retention workflow.

---

## 9. Reusable Draft Sentences

- Motivation: *"A payment biometric must err toward rejection: a false accept is an existential trust failure, while a false reject is a recoverable UX event."*
- Governance: *"Operating thresholds were selected exclusively by rule — the highest threshold exhibiting zero observed impostor accepts while retaining ≥90% genuine acceptance — never by intuition."*
- Explainability: *"Every stage is inspectable: landmarks can be visualized, the masked ROI displayed, histograms exported, and every payment carries its biometric score in the audit ledger."*
- Failure narrative (methods intro): *"Three production incidents — a false acceptance at similarity 0.80, cross-room enrollment drift, and finger-pose sensitivity — drove the pipeline's evolution and are reported as measured design iterations rather than hidden."*
- Data flywheel: *"The system logs genuine/impostor behavior under real conditions from day one; this corpus, not an external dataset, funds the future learned-embedding upgrade."*

---

## 10. Full Bibliography (IEEE style; renumber on use)

### A. Palmprint modality
[1] D. Zhang, W.-K. Kong, J. You, and M. Wong, "Online palmprint identification," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 25, no. 9, pp. 1041–1050, 2003.
[2] A. Kong, D. Zhang, and M. Kamel, "A survey of palmprint recognition," *Pattern Recognition*, vol. 42, no. 7, pp. 1408–1418, 2009.

### B. Texture descriptors
[3] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE TPAMI*, vol. 24, no. 7, pp. 971–987, 2002.
[4] L. Zhang, R. Chu, S. Xiang, S. Liao, and S. Z. Li, "Face detection based on multi-block LBP representation," in *Proc. Int. Conf. Biometrics (ICB)*, 2007, pp. 11–18.

### C. Contactless normalization & preprocessing
[5] G. Sánchez-Brizuela et al., "Landmark-based geometric normalization for contactless hand/palm biometric systems," *Sensors*, 2023.
[6] M. Amrouni et al., "Contactless palmprint preprocessing and recognition," *Journal of Biometrics*, 2022.
[7] H. Wang, S. Z. Li, and Y. Wang, "Face recognition under varying lighting using self-quotient images," in *Proc. ICPR*, 2004.

### D. Systems, fusion & security
[8] F. Zhang et al., "MediaPipe Hands: On-device real-time hand tracking," *arXiv:2006.10214*, 2020.
[9] A. Ross and A. K. Jain, "Information fusion in biometrics," *Pattern Recognition Letters*, vol. 24, no. 13, pp. 2115–2125, 2003.
[10] N. K. Ratha, S. Chikkerrur, J. H. Connell, and R. M. Bolle, "Generating cancelable fingerprint templates," *IEEE TPAMI*, vol. 29, no. 4, pp. 561–572, 2007.
[11] D. Zhang, Z. Guo, G. Lu, L. Zhang, and Y. Wang, "An online system of multiscale palmprint verification," *Pattern Recognition*, vol. 43, no. 4 — see also finger-knuckle-print verification line of work, 2010.
[12] Government of India, *Digital Personal Data Protection Act, 2023* — compliance framing for §VIII.

> Verify page numbers/volume details against Google Scholar before submission; adjust author lists to the exact papers you retrieve. [11] covers both the multiscale verification system and the knuckle-print modality thread — pick whichever exact paper you cite.

---

## 11. Submission Checklist

- [ ] Run calibration session → replace every **[CALIBRATE]** marker with real numbers.
- [ ] Export evaluation CSV (`/test` → Download) as the paper's data artifact; keep with repo.
- [ ] Redraw Fig. 3 pipeline diagram for print (draw.io/inkscape), grayscale-safe.
- [ ] Capture Fig. 4 screenshots during a live enrollment (masked band visible).
- [ ] Anonymize subject names (USNs) in any table — use S1…Sn.
- [ ] Verify all bibliography entries against actual sources (Google Scholar).
- [ ] State hardware/software versions exactly (Python 3.11.x, MediaPipe 0.10.35, OpenCV x).
- [ ] Include repo structure + test count (54) as reproducibility evidence.
- [ ] Plagiarism-safe: paraphrase §3–§9 into your own voice; these are scaffolds, not final prose.

