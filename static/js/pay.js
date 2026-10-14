/**
 * PalmPay POS — MERCHANT-SIDE terminal (Demo / Prototype).
 *
 * Architecture:
 *   - The CUSTOMER does nothing on any device. They only present their palm.
 *   - The MERCHANT selects the store, enters the amount, and clicks "Start Payment".
 *   - The customer-zone camera is ALWAYS live; during an active sale it
 *     auto-detects a palm, runs recognition, and debits the matched wallet
 *     automatically — no button press by anyone after "Start Payment".
 *   - The payer identity is resolved SERVER-SIDE from the biometric session
 *     created by /api/recognize. This file never sends a user_id.
 *
 * States: IDLE (setup) → SALE (waiting/scanning/processing) → RESULT.
 */

document.addEventListener('DOMContentLoaded', () => {
    const cfg = window.PALMPAY_POS || {};
    const camera = new PalmCamera('webcam-video', 'camera-canvas');

    // ---- DOM ----
    const el = {
        setupPanel: document.getElementById('sale-setup'),
        activePanel: document.getElementById('sale-active'),
        merchantSelect: document.getElementById('merchant-select'),
        amountInput: document.getElementById('pos-amount'),
        amountError: document.getElementById('amount-error'),
        chips: document.querySelectorAll('#sale-setup .chip'),
        btnStart: document.getElementById('btn-start-sale'),
        btnCancelSale: document.getElementById('btn-cancel-sale'),
        activeAmount: document.getElementById('active-amount'),
        activeMerchant: document.getElementById('active-merchant'),
        statusLine: document.getElementById('sale-status-line'),
        detectStatus: document.getElementById('detect-status'),
        instruction: document.getElementById('camera-instruction'),
        scoreFill: document.getElementById('score-fill'),
        scoreText: document.getElementById('score-text'),
        roiThumb: document.getElementById('roi-thumb'),
        resultPanel: document.getElementById('result-panel'),
        distContainer: document.getElementById('distance-meter-container'),
        distVal: document.getElementById('distance-val-text'),
        distNeedle: document.getElementById('distance-needle'),
        distStatus: document.getElementById('distance-status-text'),
    };

    // ---- State ----
    let mode = 'idle';            // 'idle' | 'sale' | 'result'
    let sale = null;              // { merchantId, merchantName, amount }
    let probeTimer = null;
    let probeBusy = false;
    let processing = false;
    let resumeTimer = null;
    let sweetSpotDwellStart = 0;

    const PROBE_INTERVAL_MS = 650;
    const RETRY_COOLDOWN_MS = 1500;

    function esc(s) {
        return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    // ================= CAMERA (always live) =================
    // ================= CAMERA (payment-scoped) =================
    // The camera is OFF while idle. It turns ON only when the merchant starts a
    // payment and turns OFF as soon as the sale concludes (success, declined for
    // balance, or cancelled).
    async function startCamera() {
        try {
            await camera.start();
            setDetectStatus('WAITING FOR CUSTOMER', false);
            setInstruction('Place palm in frame');
            return true;
        } catch (err) {
            setDetectStatus('CAMERA OFF', false);
            setInstruction('Camera unavailable — check permissions');
            return false;
        }
    }

    function stopCamera() {
        camera.stop();
        camera.clearCanvas();
    }

    function setIdleCameraUi() {
        setDetectStatus('CAMERA OFF', false);
        setInstruction('Camera starts automatically when payment begins');
        hideRoiThumb();
        resetScoreBar();
        updateDistanceMeter(null);
    }

    function updateDistanceMeter(data) {
        if (!el.distContainer || !el.distVal || !el.distNeedle || !el.distStatus) return;
        if (!data || !data.hand_detected || data.distance_cm === null || data.distance_cm === undefined) {
            el.distVal.textContent = '-- cm';
            el.distNeedle.style.left = '50%';
            el.distNeedle.className = 'distance-needle';
            el.distStatus.textContent = 'Position palm in front of camera (20–33 cm)';
            return;
        }
        const dist = Number(data.distance_cm);
        el.distVal.textContent = `${dist.toFixed(1)} cm`;
        // Scale 10cm (left = 0%) to 45cm (right = 100%)
        const clamped = Math.max(10, Math.min(45, dist));
        const pct = ((clamped - 10) / (45 - 10)) * 100;
        el.distNeedle.style.left = `${pct.toFixed(1)}%`;

        if (data.distance_zone === 'optimal' || data.in_sweet_spot) {
            el.distNeedle.className = 'distance-needle optimal';
            el.distStatus.textContent = '🎯 Ideal distance (20–33 cm) — hold steady';
        } else if (data.distance_zone === 'too_close') {
            el.distNeedle.className = 'distance-needle danger';
            el.distStatus.textContent = '⚠️ Too close — move palm slightly back';
        } else {
            el.distNeedle.className = 'distance-needle warning';
            el.distStatus.textContent = '⚠️ Too far — move palm closer to camera';
        }
    }

    function setDetectStatus(text, found) {
        el.detectStatus.textContent = text;
        el.detectStatus.classList.toggle('found', !!found);
    }

    function setInstruction(text) {
        el.instruction.textContent = text;
    }

    function setStatusLine(html) {
        el.statusLine.innerHTML = html;
    }

    // ================= SETUP / VALIDATION =================
    function validateAmount() {
        const v = String(el.amountInput.value).trim();
        const ok = /^\d+(\.\d{1,2})?$/.test(v) && parseFloat(v) > 0 && parseFloat(v) <= (cfg.maxAmount || 5000);
        el.btnStart.disabled = !ok;
        return ok ? parseFloat(v) : null;
    }

    el.amountInput.addEventListener('input', () => {
        el.amountError.style.display = 'none';
        validateAmount();
    });
    el.chips.forEach(chip => chip.addEventListener('click', () => {
        if (mode === 'idle') { el.amountInput.value = chip.dataset.amt; validateAmount(); }
    }));

    // ================= SALE LIFECYCLE =================
    el.btnStart.addEventListener('click', startSale);
    el.btnCancelSale.addEventListener('click', endSaleToIdle);

    async function startSale() {
        if (mode !== 'idle') return;
        const amount = validateAmount();
        if (amount === null) {
            el.amountError.textContent = `Enter a numeric amount between ₹1 and ₹${cfg.maxAmount}.`;
            el.amountError.style.display = 'block';
            return;
        }

        const opt = el.merchantSelect.selectedOptions[0];
        sale = { merchantId: opt.value, merchantName: opt.dataset.name, amount };

        // Lock merchant controls
        mode = 'sale';
        el.setupPanel.style.display = 'none';
        el.activePanel.style.display = 'block';
        el.amountInput.disabled = true;
        el.btnStart.disabled = true;
        el.chips.forEach(c => c.disabled = true);
        el.resultPanel.style.display = 'none';
        el.activeAmount.textContent = sale.amount.toFixed(2);
        el.activeMerchant.textContent = sale.merchantName;

        resetScoreBar();
        hideRoiThumb();
        sweetSpotDwellStart = 0;
        updateDistanceMeter(null);
        if (el.distContainer) el.distContainer.style.display = 'block';

        setDetectStatus('CAMERA STARTING…', false);
        setInstruction('Waiting for customer — place palm in frame');
        setStatusLine('<span class="pulse-dot"></span> Starting camera…');

        // Camera turns ON only now — payment initiated.
        const camOk = await startCamera();
        if (!camOk) {
            alert('Camera could not start. Please allow webcam access and try again.');
            endSaleToIdle();
            return;
        }

        setStatusLine('<span class="pulse-dot"></span> Waiting for customer…');
        startProbing();
    }

    function endSaleToIdle() {
        mode = 'idle';
        sale = null;
        sweetSpotDwellStart = 0;
        stopProbing();
        clearTimers();
        stopCamera();  // sale over → camera off

        el.setupPanel.style.display = 'block';
        el.activePanel.style.display = 'none';
        el.amountInput.disabled = false;
        el.btnStart.disabled = false;
        el.chips.forEach(c => c.disabled = false);
        validateAmount();
        el.resultPanel.style.display = 'none';

        if (el.distContainer) el.distContainer.style.display = 'none';
        setIdleCameraUi();
        setStatusLine('<span class="pulse-dot"></span> No active sale');
    }

    // ================= AUTO-CAPTURE LOOP =================
    function startProbing() {
        stopProbing();
        probeTimer = setInterval(probeFrame, PROBE_INTERVAL_MS);
    }

    function stopProbing() {
        if (probeTimer) { clearInterval(probeTimer); probeTimer = null; }
    }

    function clearTimers() {
        if (resumeTimer) { clearTimeout(resumeTimer); resumeTimer = null; }
    }

    async function probeFrame() {
        if (!camera.isActive || probeBusy || processing || mode !== 'sale') return;
        probeBusy = true;
        try {
            const frameB64 = camera.captureFrameAsBase64(0.85);
            if (!frameB64) return;
            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });
            const data = await resp.json();
            if (!(mode === 'sale')) return;

            updateDistanceMeter(data);

            const inSweetSpot = Boolean(data.success && data.hand_detected && data.roi_extracted && data.in_sweet_spot);

            if (data.hand_detected) {
                if (inSweetSpot) {
                    const now = Date.now();
                    if (!sweetSpotDwellStart) {
                        sweetSpotDwellStart = now;
                    }
                    const dwellMs = now - sweetSpotDwellStart;
                    setDetectStatus('SWEET SPOT LOCKED', true);
                    setInstruction(`Ideal distance (${data.distance_cm} cm) 🎯 — hold steady…`);
                    showRoiThumb(data.preview_roi_b64);
                    camera.drawPolygon(data.roi_polygon || [], '#10b981', 3);

                    // Strictly gated: require hand to settle in sweet spot for >= 250ms
                    if (dwellMs >= 250) {
                        attemptPayment();
                    }
                } else {
                    sweetSpotDwellStart = 0;
                    showRoiThumb(data.preview_roi_b64);
                    if (data.distance_zone === 'too_close') {
                        setDetectStatus('TOO CLOSE', false);
                        setInstruction(`⚠️ Too close (${data.distance_cm} cm) — move palm back into sweet spot`);
                        camera.drawPolygon(data.roi_polygon || [], '#ef4444', 2);
                    } else if (data.distance_zone === 'too_far') {
                        setDetectStatus('TOO FAR', false);
                        setInstruction(`⚠️ Too far (${data.distance_cm} cm) — move palm closer into sweet spot`);
                        camera.drawPolygon(data.roi_polygon || [], '#f59e0b', 2);
                    } else {
                        setDetectStatus('ADJUST PALM', false);
                        setInstruction(data.message || 'Position palm flat in sweet spot (20–33 cm)');
                        camera.drawPolygon(data.roi_polygon || [], '#38bdf8', 2);
                    }
                }
            } else {
                sweetSpotDwellStart = 0;
                setDetectStatus('WAITING FOR CUSTOMER', false);
                setInstruction('Place palm in frame (20–33 cm)');
                camera.clearCanvas();
                hideRoiThumb();
            }
        } catch (e) {
            /* transient probe errors ignored */
        } finally {
            probeBusy = false;
        }
    }

    // Fully automatic: burst-capture → recognize → pay → render outcome.
    // Nobody presses anything.
    async function attemptPayment() {
        if (processing) return;
        processing = true;
        stopProbing();                       // one attempt at a time
        setDetectStatus('PROCESSING…', true);
        setInstruction('Verifying palm…');
        setStatusLine('<span class="pulse-dot"></span> Verifying palm…');

        try {
            // 0) Burst capture: several frames ~fusionIntervalMs apart so the
            //    server can fuse them and require frame agreement.
            const burst = [];
            const windowN = Math.max(1, cfg.fusionWindow || 4);
            const intervalMs = Math.max(80, cfg.fusionIntervalMs || 220);
            for (let i = 0; i < windowN; i++) {
                const f = camera.captureFrameAsBase64(0.9);
                if (f) burst.push(f);
                if (i < windowN - 1) {
                    await new Promise(r => setTimeout(r, intervalMs));
                    if (!camera.isActive) break;
                }
            }
            if (!burst.length) {
                showAuthFail('Could not capture palm frames.');
                scheduleRetry();
                return;
            }

            // 1) Biometric recognition (server opens the biometric session on match)
            const recResp = await fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ images: burst })
            });
            const rec = await recResp.json();

            const scorePct = Math.max(0, Math.min(100, (rec.similarity_score || 0) * 100));
            el.scoreFill.style.width = `${scorePct}%`;
            el.scoreText.textContent =
                rec.success && rec.recognized
                    ? `Score ${(rec.similarity_score || 0).toFixed(4)} · Frames ${rec.frames_above_threshold ?? '?'}/${rec.frames_quality_passed ?? windowN} ✓`
                    : `Score ${(rec.similarity_score || 0).toFixed(4)} — not verified`;

            // 2) Not recognized → no transaction; keep scanning for the next attempt.
            if (!rec.success || !rec.recognized) {
                const unstable = rec.match_reason === 'unstable_capture';
                showAuthFail(
                    unstable
                        ? 'Capture unstable — hold palm steady and try again'
                        : (rec.message || 'Palm did not match any enrolled account.')
                );
                scheduleRetry();
                return;
            }

            // 3) Matched → debit automatically via server-side biometric session.
            const payResp = await fetch('/api/payment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ merchant_id: sale.merchantId, amount: sale.amount })
            });
            const pay = await payResp.json();

            if (pay.success) {
                finishSale('success', { pay, name: rec.matched_user.name, score: rec.similarity_score });
            } else if (pay.reason === 'FAILED_BALANCE') {
                finishSale('balance_fail', {
                    message: pay.message,
                    balance: pay.balance,
                    name: pay.consumer_name || (rec.matched_user && rec.matched_user.name) || 'Customer',
                });
            } else {
                showAuthFail(pay.message || 'Payment could not be authorized.');
                scheduleRetry();
            }
        } catch (err) {
            showAuthFail('Network error: ' + err.message);
            scheduleRetry();
        } finally {
            processing = false;
        }
    }

    function scheduleRetry() {
        // Stay in this sale; rescan automatically after a short cooldown.
        sweetSpotDwellStart = 0;
        setDetectStatus('NOT RECOGNIZED', false);
        setInstruction('Try again — place palm in frame');
        setStatusLine('<span class="pulse-dot warn"></span> Palm not recognized — try again');
        camera.clearCanvas();
        resumeTimer = setTimeout(() => {
            resumeTimer = null;
            if (mode === 'sale' && !processing) {
                setDetectStatus('WAITING FOR CUSTOMER', false);
                setInstruction('Place palm in frame');
                startProbing();
            }
        }, RETRY_COOLDOWN_MS);
    }

    // ================= RESULTS =================
    function finishSale(kind, info) {
        mode = 'result';
        sweetSpotDwellStart = 0;
        stopProbing();
        clearTimers();
        stopCamera();  // sale concluded (success or declined) → camera off
        hideRoiThumb();
        if (el.distContainer) el.distContainer.style.display = 'none';
        updateDistanceMeter(null);

        if (kind === 'success') {
            const d = info.pay;
            setDetectStatus('VERIFIED ✓', true);
            setInstruction(`✓ ${info.name} verified`);
            setStatusLine(`<span class="pulse-dot ok"></span> Payment complete — ₹${d.amount.toFixed(2)} received`);
            el.scoreFill.style.width = `${Math.min(100, (info.score || 0) * 100)}%`;
            el.scoreText.textContent = `Score ${(info.score ?? 0).toFixed(4)} · Palm Verified`;
            el.resultPanel.innerHTML = `
                <div class="result-screen result-success pos-result">
                    <div class="rs-icon">✓</div>
                    <h2 class="rs-title">${esc(info.name)} Verified</h2>
                    <p class="rs-sub">₹${d.amount.toFixed(2)} charged to their wallet automatically.</p>
                    <div class="rs-details">
                        <div class="rs-row"><span>Customer</span><strong>${esc(d.consumer.name)}</strong></div>
                        <div class="rs-row"><span>Merchant</span><strong>${esc(d.merchant.name)}</strong></div>
                        <div class="rs-row"><span>Amount</span><strong>₹${d.amount.toFixed(2)}</strong></div>
                        <div class="rs-row"><span>Biometric Score</span><strong>${(info.score ?? d.biometric_score).toFixed(4)}</strong></div>
                        <div class="rs-row"><span>New Balance</span><strong>₹${d.new_balance.toFixed(2)}</strong></div>
                        <div class="rs-row"><span>Txn ID</span><strong>${esc(d.transaction_id.slice(0, 8))}…</strong></div>
                    </div>
                    <div class="rs-actions">
                        <a href="${esc(d.receipt_url)}" target="_blank" class="btn-outline" style="background:#fff;border-color:#fff;color:var(--green);font-weight:700;">View Receipt</a>
                        <button class="btn-primary" style="background:#fff;color:var(--green);font-weight:700;" id="btn-next-sale">Next Sale</button>
                    </div>
                </div>`;
            document.getElementById('btn-next-sale').addEventListener('click', endSaleToIdle);
        } else {
            // Insufficient balance — scanning stops; merchant decides what to do.
            setDetectStatus('LOW BALANCE', false);
            setInstruction(`${info.name} has insufficient balance`);
            setStatusLine(`<span class="pulse-dot warn"></span> Payment declined — insufficient balance`);
            el.resultPanel.innerHTML = `
                <div class="result-screen result-warn pos-result">
                    <div class="rs-icon">⚠</div>
                    <h2 class="rs-title">Insufficient Balance</h2>
                    <p class="rs-sub" style="color:var(--orange);">${esc(info.name)} has ₹${Number(info.balance ?? 0).toFixed(2)} available.</p>
                    <div class="rs-details">
                        <div class="rs-row"><span>Attempted charge</span><strong>₹${sale.amount.toFixed(2)}</strong></div>
                        <div class="rs-row"><span>Wallet balance</span><strong>₹${Number(info.balance ?? 0).toFixed(2)}</strong></div>
                    </div>
                    <p class="t-xs dim" style="margin-top:12px;">No money was deducted. Customer can top up from the admin console.</p>
                    <div class="rs-actions">
                        <button class="btn-primary" style="background:var(--orange);" id="btn-next-sale-low">Done</button>
                    </div>
                </div>`;
            document.getElementById('btn-next-sale-low').addEventListener('click', endSaleToIdle);
        }
        el.resultPanel.style.display = 'block';
    }

    function showAuthFail(message) {
        const text = esc(message || 'Palm not recognized — try again');
        el.resultPanel.innerHTML = `
            <div class="auth-fail-banner">✗ ${text}</div>`;
        el.resultPanel.style.display = 'block';
        setTimeout(() => { if (mode === 'sale') { el.resultPanel.style.display = 'none'; } }, 3200);
    }

    // ---- helpers ----
    function resetScoreBar() {
        el.scoreFill.style.width = '0%';
        el.scoreText.textContent = '—';
    }
    function showRoiThumb(src) {
        if (src) { el.roiThumb.src = src; el.roiThumb.style.display = 'block'; }
    }
    function hideRoiThumb() { el.roiThumb.style.display = 'none'; }

    // ---- Boot ----
    // The camera is NOT started on page load. It turns on only inside
    // startSale() (when the merchant clicks "Start Payment") and turns off in
    // finishSale()/endSaleToIdle() when the sale concludes or is cancelled.
    if (cfg.embedded) {
        // Merchant console: leaving the POS tab cancels any pending sale
        // cleanly, which also stops the camera.
        window.PalmPayPOSDeactivate = () => {
            if (mode !== 'idle') endSaleToIdle();
        };
    }
    setIdleCameraUi();
});
