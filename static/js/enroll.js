/**
 * PalmPay — Automated Enrollment Controller (fingerprint-registration style).
 *
 * One button: "Start Enrollment Scan". The system then:
 *   - polls /api/process_frame continuously,
 *   - auto-captures a sample whenever the palm is aligned AND passes the
 *     server-side capture quality gate,
 *   - enforces a short cooldown between samples and rotates pose prompts
 *     ("Tilt slightly LEFT"…) so the samples are varied,
 *   - auto-submits to /api/enroll once all samples are collected.
 *
 * The user never clicks per-sample. Cancel is available mid-scan.
 */

document.addEventListener('DOMContentLoaded', () => {
    const cfg = window.PALMPAY_ENROLL || {};
    const REQUIRED = Number(cfg.requiredSamples) || 6;
    const COOLDOWN_MS = Number(cfg.cooldownMs) || 1300;
    const PROMPTS = Array.isArray(cfg.posePrompts) && cfg.posePrompts.length
        ? cfg.posePrompts
        : ['Tilt your palm slightly LEFT', 'Tilt slightly RIGHT', 'Move a bit CLOSER'];

    const PROBE_INTERVAL_MS = 400;
    const DUPLICATE_DIFF_PCT = Number(cfg.duplicateDiffPct) || 1.5;

    // Tiny offscreen canvas for the near-duplicate check (32x32 grayscale diff
    // between consecutive auto-captures).
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = 32; snapCanvas.height = 32;
    const snapCtx = snapCanvas.getContext('2d', { willReadFrequently: true });
    let prevSnapshot = null;

    function snapshotVideo() {
        if (!camera.video || !camera.video.videoWidth) return null;
        snapCtx.drawImage(camera.video, 0, 0, 32, 32);
        return snapCtx.getImageData(0, 0, 32, 32).data;
    }

    function meanAbsDiffPct(a, b) {
        let sum = 0;
        const px = a.length / 4;
        for (let i = 0; i < a.length; i += 4) {
            const ga = a[i] * 0.299 + a[i + 1] * 0.587 + a[i + 2] * 0.114;
            const gb = b[i] * 0.299 + b[i + 1] * 0.587 + b[i + 2] * 0.114;
            sum += Math.abs(ga - gb);
        }
        return (sum / px) / 255 * 100;
    }

    const camera = new PalmCamera('webcam-video', 'camera-canvas');

    // ---- DOM ----
    const el = {
        statusPill: document.getElementById('camera-status-pill'),
        guide: document.getElementById('camera-guide'),
        progressPill: document.getElementById('scan-progress-pill'),
        feedback: document.getElementById('live-feedback-text'),

        startBtn: document.getElementById('btn-start-scan'),
        cancelBtn: document.getElementById('btn-cancel-scan'),
        resetBtn: document.getElementById('btn-reset-samples'),
        resultBox: document.getElementById('enrollment-result-box'),

        inputName: document.getElementById('input-name'),
        inputUsn: document.getElementById('input-usn'),
        progressBar: document.getElementById('progress-bar'),
        progressText: document.getElementById('progress-text'),
    };

    // ---- State ----
    let state = 'idle';            // idle | scanning | submitting | done
    let captured = [];             // base64 frames
    let probeTimer = null;
    let probing = false;
    let cooldownUntil = 0;

    function setFeedback(text) { el.feedback.textContent = text; }

    function updateProgress() {
        const count = captured.length;
        el.progressText.textContent = `${count} of ${REQUIRED} captured`;
        el.progressBar.style.width = `${(count / REQUIRED) * 100}%`;
        el.progressPill.textContent = `${count} / ${REQUIRED}`;
        el.resetBtn.disabled = count === 0 && state !== 'scanning';
        if (count >= REQUIRED) el.progressBar.classList.add('success');
        else el.progressBar.classList.remove('success');
    }

    function renderSlot(idx, imgSrc, labelExtra) {
        const slot = document.getElementById(`slot-${idx}`);
        if (!slot) return;
        slot.innerHTML = `
            <img src="${imgSrc}" alt="Sample ${idx + 1}">
            <span class="slot-label">Sample ${idx + 1}${labelExtra ? ' · ' + labelExtra : ''}</span>
        `;
    }

    function clearSlots() {
        for (let i = 0; i < REQUIRED; i++) {
            const slot = document.getElementById(`slot-${i}`);
            if (!slot) continue;
            slot.innerHTML = `
                <span style="font-size: 1.25rem; color: var(--text-dim);">✋</span>
                <span class="slot-label">Sample ${i + 1}</span>
            `;
        }
    }

    // ================= STATE TRANSITIONS =================
    async function startScan() {
        if (state !== 'idle') return;

        const name = el.inputName.value.trim();
        const usn = el.inputUsn.value.trim();
        if (!name) { setFeedback('Please enter the full name first.'); el.inputName.focus(); return; }
        if (!usn) { setFeedback('Please enter the USN / Student ID first.'); el.inputUsn.focus(); return; }

        el.startBtn.disabled = true;
        el.startBtn.textContent = '⏳ Starting camera…';
        el.statusPill.className = 'status-pill warning';
        el.statusPill.textContent = 'Starting…';

        try {
            await camera.start();
        } catch (err) {
            el.statusPill.className = 'status-pill danger';
            el.statusPill.textContent = 'Camera Error';
            setFeedback(err.message);
            el.startBtn.disabled = false;
            el.startBtn.textContent = '🚀 Start Enrollment Scan';
            return;
        }

        // Entering scanning mode
        state = 'scanning';
        captured = [];
        prevSnapshot = null;
        cooldownUntil = 0;
        clearSlots();
        updateProgress();
        el.resultBox.style.display = 'none';
        el.resultBox.innerHTML = '';

        el.startBtn.style.display = 'none';
        el.cancelBtn.style.display = 'inline-flex';
        el.progressPill.style.display = 'inline-flex';
        el.statusPill.className = 'status-pill success';
        el.statusPill.textContent = 'Scanning — auto-capturing';

        setFeedback('Hold your palm steady in the frame…');
        startProbe();
    }

    function cancelScan(msg) {
        if (state === 'submitting') return;
        stopProbe();
        camera.stop();
        camera.clearCanvas();
        resetToIdle(msg || 'Scan cancelled.');
    }

    function resetToIdle(message) {
        state = 'idle';
        captured = [];
        prevSnapshot = null;
        stopProbe();
        camera.stop();
        camera.clearCanvas();

        clearSlots();
        updateProgress();
        el.resultBox.style.display = 'none';
        el.resultBox.innerHTML = '';
        el.guide.classList.remove('active');

        el.startBtn.style.display = 'inline-flex';
        el.startBtn.disabled = false;
        el.startBtn.textContent = '🚀 Start Enrollment Scan';
        el.cancelBtn.style.display = 'none';
        el.progressPill.style.display = 'none';
        el.statusPill.className = 'status-pill neutral';
        el.statusPill.textContent = 'Camera Offline';
        el.inputName.disabled = false;
        el.inputUsn.disabled = false;
        setFeedback(message || 'Enter your name and USN, then start the scan.');
    }

    // ================= PROBE LOOP (auto-capture) =================
    function startProbe() {
        stopProbe();
        probeTimer = setInterval(probeOnce, PROBE_INTERVAL_MS);
    }

    function stopProbe() {
        if (probeTimer) { clearInterval(probeTimer); probeTimer = null; }
    }

    async function probeOnce() {
        if (state !== 'scanning' || probing) return;
        probing = true;
        try {
            const frameB64 = camera.captureFrameAsBase64(0.7);
            if (!frameB64) return;

            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });
            const data = await resp.json();
            if (state !== 'scanning') return;

            if (data.success && data.hand_detected && data.roi_extracted) {
                // Aligned + quality passed.
                el.guide.classList.add('active');
                el.guide.textContent = 'Hold still…';
                camera.drawPolygon(data.roi_polygon, '#10b981', 3);

                if (Date.now() >= cooldownUntil) {
                    await captureSample(data);
                } else {
                    const secs = Math.max(1, Math.ceil((cooldownUntil - Date.now()) / 1000));
                    setFeedback(`Next sample in ${secs}s — ${currentPrompt()}`);
                }
            } else {
                // Not detected, or quality-gate refusal (too far / blurry / glare…).
                el.guide.classList.remove('active');
                el.guide.textContent = 'Position your open palm facing the camera';
                camera.clearCanvas();
                setFeedback(data.message || 'Waiting for a clear palm view…');
            }
        } catch (e) {
            /* transient probe errors ignored */
        } finally {
            probing = false;
        }
    }

    function currentPrompt() {
        if (captured.length === 0) return 'hold steady';
        return PROMPTS[(captured.length - 1) % PROMPTS.length];
    }

    async function captureSample(probeData) {
        // Near-duplicate guard: if the frame barely differs from the previous
        // sample, skip it and ask the user to vary more (keeps the template
        // set diverse without wasting a slot).
        const snap = snapshotVideo();
        if (snap && prevSnapshot) {
            const diff = meanAbsDiffPct(snap, prevSnapshot);
            if (diff < DUPLICATE_DIFF_PCT) {
                cooldownUntil = Date.now() + 800;  // retry sooner than full cooldown
                setFeedback(`Too similar to the last sample — vary a bit more: ${currentPrompt()}`);
                return;
            }
        }

        // Re-capture at higher JPEG quality for storage.
        const hiQuality = camera.captureFrameAsBase64(0.92);
        if (!hiQuality) return;

        if (snap) prevSnapshot = snap;
        captured.push(hiQuality);
        cooldownUntil = Date.now() + COOLDOWN_MS;

        const idx = captured.length - 1;
        const thumb = (probeData && (probeData.preview_enhanced_b64 || probeData.preview_roi_b64)) || hiQuality;
        renderSlot(idx, thumb);

        updateProgress();

        if (captured.length >= REQUIRED) {
            finishCapture();
        } else {
            setFeedback(`Sample ${captured.length}/${REQUIRED} captured ✓ — Now: ${currentPrompt()}`);
        }
    }

    function finishCapture() {
        state = 'submitting';
        stopProbe();
        camera.drawPolygon(null);
        el.guide.textContent = 'All samples captured';
        setFeedback(`All ${REQUIRED} samples captured — saving enrollment…`);
        el.cancelBtn.style.display = 'none';
        submitEnrollment();
    }

    // ================= SUBMIT =================
    async function submitEnrollment() {
        const name = el.inputName.value.trim();
        const usn = el.inputUsn.value.trim();
        const consentCheckbox = document.getElementById('check-dataset-consent');
        const datasetConsent = consentCheckbox ? consentCheckbox.checked : false;

        try {
            const resp = await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    usn: usn,
                    samples: captured,
                    dataset_consent: datasetConsent
                })
            });
            const result = await resp.json();
            camera.stop();
            el.statusPill.className = 'status-pill neutral';
            el.statusPill.textContent = 'Camera Offline';

            el.resultBox.style.display = 'block';

            if (result.success) {
                state = 'done';
                el.resultBox.innerHTML = `
                    <div style="background: var(--success-bg); border: 1px solid var(--success); padding: 1rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--success); margin-bottom: 0.25rem;">✅ Enrollment Successful</h3>
                        <p style="font-size: 0.9rem; color: var(--text-main); margin-bottom: 0.5rem;">${result.message}</p>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">
                            User ID: #${result.user_id} &bull; Model: ${result.model_name} &bull; Dim: ${result.feature_dim}-D &bull; Avg Extraction: ${result.avg_extraction_latency_ms} ms
                        </div>
                        <div style="margin-top: 0.75rem;">
                            <a href="/recognize" class="btn btn-sm">Test in Recognition</a>
                            <a href="/consumer?user_id=${result.user_id}" class="btn btn-secondary btn-sm" style="margin-left: 0.5rem;">Open Wallet</a>
                            <button class="btn btn-secondary btn-sm" id="btn-enroll-another" style="margin-left: 0.5rem;">Enroll Another</button>
                        </div>
                    </div>`;
                const anotherBtn = document.getElementById('btn-enroll-another');
                if (anotherBtn) anotherBtn.addEventListener('click', () => {
                    el.inputUsn.value = '';
                    resetToIdle('Enter details for the next person.');
                });
                setFeedback('Enrollment complete.');
            } else {
                state = 'idle';
                el.resultBox.innerHTML = `
                    <div style="background: var(--danger-bg); border: 1px solid var(--danger); padding: 1rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--danger); margin-bottom: 0.25rem;">❌ Enrollment Failed</h3>
                        <p style="font-size: 0.9rem; color: var(--text-main);">${result.message}</p>
                        <button class="btn btn-sm" id="btn-retry-scan" style="margin-top: 0.5rem;">🔄 Retry Scan</button>
                    </div>`;
                const retryBtn = document.getElementById('btn-retry-scan');
                if (retryBtn) retryBtn.addEventListener('click', () => resetToIdle('Adjust details and scan again.'));
                setFeedback(result.message || 'Enrollment failed — press Retry Scan.');
            }
        } catch (err) {
            camera.stop();
            state = 'idle';
            alert('Network error during enrollment: ' + err.message);
            resetToIdle('Network error — scan again.');
        }
    }

    // ---- Wiring ----
    el.startBtn.addEventListener('click', startScan);
    el.cancelBtn.addEventListener('click', () => cancelScan());
    el.resetBtn.addEventListener('click', () => resetToIdle());
});
