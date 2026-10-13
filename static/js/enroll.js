/**
 * PalmPay — Modern Guided Palm Biometric Enrollment Controller
 *
 * Implements a high-precision biometric enrollment UX (Face-Unlock / Amazon One style):
 *   - Live Canvas Reticle: 6-segment outer progress ring animating sample-by-sample
 *   - Dynamic Tracking Puck: floats over palm center, size scales with distance
 *   - Monocular Distance Coaching: live needle gauge & coaching (20–33 cm sweet spot)
 *   - Web Audio Synthesizer: lock-on, capture, and completion chimes
 *   - Safe Pose Coaching: strictly in-plane rotations and distance steps (no foreshortening pitch)
 *   - Near-duplicate guard (32x32 grayscale difference >= 1.5%)
 */

document.addEventListener('DOMContentLoaded', () => {
    const cfg = window.PALMPAY_ENROLL || {};
    const REQUIRED = Number(cfg.requiredSamples) || 6;
    const COOLDOWN_MS = Number(cfg.cooldownMs) || 1200;
    const PROMPTS = Array.isArray(cfg.posePrompts) && cfg.posePrompts.length
        ? cfg.posePrompts
        : [
            'Rotate palm slightly LEFT (⟲ -15°)',
            'Rotate palm slightly RIGHT (⟳ +15°)',
            'Spread fingers naturally',
            'Hold in sweet spot (20–30 cm)',
            'Keep palm flat facing camera'
        ];

    const PROBE_INTERVAL_MS = 320;
    const DUPLICATE_DIFF_PCT = Number(cfg.duplicateDiffPct) || 1.5;

    // =========================================================================
    // 1. Web Audio Synthesizer (Audio Cues)
    // =========================================================================
    class SoundFx {
        constructor() {
            this.ctx = null;
            this.enabled = localStorage.getItem('palmpay_sound_enabled') !== 'false';
        }

        init() {
            if (!this.ctx) {
                const AudioCtx = window.AudioContext || window.webkitAudioContext;
                if (AudioCtx) this.ctx = new AudioCtx();
            }
            if (this.ctx && this.ctx.state === 'suspended') {
                this.ctx.resume();
            }
        }

        toggle() {
            this.enabled = !this.enabled;
            localStorage.setItem('palmpay_sound_enabled', this.enabled ? 'true' : 'false');
            return this.enabled;
        }

        playTone(freq, duration = 0.08, type = 'sine', gainVal = 0.12) {
            if (!this.enabled) return;
            try {
                this.init();
                if (!this.ctx) return;
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = type;
                osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
                gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start();
                osc.stop(this.ctx.currentTime + duration);
            } catch (_) {}
        }

        playLock() {
            this.playTone(880, 0.06, 'sine', 0.10);
        }

        playCapture() {
            if (!this.enabled) return;
            try {
                this.init();
                if (!this.ctx) return;
                // Dual-tone chime chord: C5 (523Hz) and E5 (659Hz)
                const now = this.ctx.currentTime;
                [523.25, 659.25].forEach(freq => {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.setValueAtTime(freq, now);
                    gain.gain.setValueAtTime(0.15, now);
                    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
                    osc.connect(gain);
                    gain.connect(this.ctx.destination);
                    osc.start(now);
                    osc.stop(now + 0.35);
                });
            } catch (_) {}
        }

        playSuccess() {
            if (!this.enabled) return;
            try {
                this.init();
                if (!this.ctx) return;
                const now = this.ctx.currentTime;
                const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
                notes.forEach((freq, idx) => {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();
                    const t = now + idx * 0.08;
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(freq, t);
                    gain.gain.setValueAtTime(0.18, t);
                    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
                    osc.connect(gain);
                    gain.connect(this.ctx.destination);
                    osc.start(t);
                    osc.stop(t + 0.4);
                });
            } catch (_) {}
        }
    }

    const sfx = new SoundFx();

    // =========================================================================
    // 2. Offscreen Canvas for Duplicate Guard (32x32 downscale)
    // =========================================================================
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = 32;
    snapCanvas.height = 32;
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

    // =========================================================================
    // 3. DOM Elements
    // =========================================================================
    const el = {
        statusPill: document.getElementById('camera-status-pill'),
        guide: document.getElementById('camera-guide'),
        progressPill: document.getElementById('scan-progress-pill'),
        feedback: document.getElementById('live-feedback-text'),

        soundBtn: document.getElementById('btn-toggle-sound'),
        poseBanner: document.getElementById('hud-pose-banner'),
        poseIcon: document.getElementById('hud-pose-icon'),
        poseText: document.getElementById('hud-pose-text'),

        distContainer: document.getElementById('distance-meter-container'),
        distVal: document.getElementById('distance-val-text'),
        distNeedle: document.getElementById('distance-needle'),
        distStatus: document.getElementById('distance-status-text'),

        startBtn: document.getElementById('btn-start-scan'),
        cancelBtn: document.getElementById('btn-cancel-scan'),
        resetBtn: document.getElementById('btn-reset-samples'),
        resultBox: document.getElementById('enrollment-result-box'),

        inputName: document.getElementById('input-name'),
        inputUsn: document.getElementById('input-usn'),
        progressBar: document.getElementById('progress-bar'),
        progressText: document.getElementById('progress-text'),
    };

    // Update sound toggle UI on load
    if (el.soundBtn) {
        el.soundBtn.textContent = sfx.enabled ? '🔊 Sound' : '🔇 Muted';
        el.soundBtn.addEventListener('click', () => {
            const active = sfx.toggle();
            el.soundBtn.textContent = active ? '🔊 Sound' : '🔇 Muted';
        });
    }

    // =========================================================================
    // 4. State Management
    // =========================================================================
    let state = 'idle';            // idle | scanning | submitting | done
    let captured = [];             // base64 frames
    let probeTimer = null;
    let probing = false;
    let cooldownUntil = 0;
    let sweetSpotDwellStart = 0;
    let hadSweetSpotLock = false;

    // Smoothed puck position for fluid Canvas rendering
    let puckX = null;
    let puckY = null;
    let puckTargetRadius = 45;
    let puckRadius = 45;
    let puckColor = 'rgba(56, 189, 248, 0.85)'; // Default cyan
    let puckGlow = false;

    function setFeedback(text) {
        if (el.feedback) el.feedback.textContent = text;
    }

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

    function currentPrompt() {
        if (captured.length === 0) return 'Hold palm flat, centered (baseline)';
        return PROMPTS[(captured.length - 1) % PROMPTS.length];
    }

    function currentPoseIcon() {
        if (captured.length === 0) return '✋';
        const p = currentPrompt().toLowerCase();
        if (p.includes('left')) return '⟲';
        if (p.includes('right')) return '⟳';
        if (p.includes('spread')) return '🖐';
        if (p.includes('sweet') || p.includes('closer')) return '🎯';
        return '✋';
    }

    function updatePoseHUD(isSweetSpot = false) {
        if (!el.poseBanner) return;
        if (state !== 'scanning') {
            el.poseBanner.style.display = 'none';
            return;
        }
        el.poseBanner.style.display = 'inline-flex';
        el.poseIcon.textContent = currentPoseIcon();
        el.poseText.textContent = `Sample ${captured.length + 1}/${REQUIRED}: ${currentPrompt()}`;

        if (isSweetSpot) el.poseBanner.classList.add('sweet-spot');
        else el.poseBanner.classList.remove('sweet-spot');
    }

    // =========================================================================
    // 5. Canvas HUD & Reticle Renderer
    // =========================================================================
    function drawHUD(data) {
        if (!camera.ctx || !camera.canvas || state !== 'scanning') return;
        const ctx = camera.ctx;
        const w = camera.canvas.width;
        const h = camera.canvas.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2;
        const reticleRadius = Math.min(w, h) * 0.35;

        // 1. Draw 6 Segmented Outer Progress Arcs (Phone Biometric Style)
        const segments = REQUIRED;
        const gap = 0.08; // angular gap between sectors
        const arcStep = (2 * Math.PI) / segments;

        ctx.save();
        for (let i = 0; i < segments; i++) {
            const startAngle = i * arcStep + gap / 2 - Math.PI / 2;
            const endAngle = (i + 1) * arcStep - gap / 2 - Math.PI / 2;

            ctx.beginPath();
            ctx.arc(cx, cy, reticleRadius, startAngle, endAngle);

            if (i < captured.length) {
                // Completed segment: glowing emerald
                ctx.strokeStyle = '#10b981';
                ctx.lineWidth = 6;
                ctx.shadowColor = '#10b981';
                ctx.shadowBlur = 10;
            } else if (i === captured.length) {
                // Current active sector: highlighted cyan/amber
                ctx.strokeStyle = data && data.in_sweet_spot ? '#10b981' : '#38bdf8';
                ctx.lineWidth = 5;
                ctx.shadowColor = ctx.strokeStyle;
                ctx.shadowBlur = 8;
            } else {
                // Pending sector: subtle slate track
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
                ctx.lineWidth = 3;
                ctx.shadowBlur = 0;
            }
            ctx.stroke();
        }
        ctx.restore();

        // 2. Draw Subtle Central Alignment Target (Dashed crosshair or circle)
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, 32, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();

        // 3. Draw Dynamic Tracking Puck
        if (data && data.hand_detected && data.palm_center) {
            const targetX = data.palm_center[0];
            const targetY = data.palm_center[1];

            // Smooth position lerp
            puckX = puckX === null ? targetX : puckX + (targetX - puckX) * 0.45;
            puckY = puckY === null ? targetY : puckY + (targetY - puckY) * 0.45;

            // Radius scales inversely with estimated distance
            if (data.distance_cm && data.distance_cm > 10) {
                puckTargetRadius = Math.max(26, Math.min(65, 42 * (26 / data.distance_cm)));
            } else {
                puckTargetRadius = 42;
            }
            puckRadius += (puckTargetRadius - puckRadius) * 0.35;

            // Determine puck state color
            if (data.in_sweet_spot) {
                puckColor = '#10b981'; // Green (Aligned + Quality Passed)
                puckGlow = true;
            } else if (data.distance_zone === 'too_close' || data.quality_reason === 'glare' || data.quality_reason === 'blurry') {
                puckColor = '#ef4444'; // Red (Warning / Defect)
                puckGlow = false;
            } else {
                puckColor = '#f59e0b'; // Amber (Guiding into position)
                puckGlow = false;
            }

            ctx.save();
            ctx.beginPath();
            ctx.arc(puckX, puckY, puckRadius, 0, 2 * Math.PI);
            ctx.strokeStyle = puckColor;
            ctx.lineWidth = puckGlow ? 4 : 2.5;
            if (puckGlow) {
                ctx.shadowColor = puckColor;
                ctx.shadowBlur = 16;
                ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
                ctx.fill();
            }
            ctx.stroke();

            // Center target pip
            ctx.beginPath();
            ctx.arc(puckX, puckY, 4, 0, 2 * Math.PI);
            ctx.fillStyle = puckColor;
            ctx.fill();

            // Draw palm ROI polygon outline if present
            if (data.roi_polygon && data.roi_polygon.length === 4) {
                ctx.beginPath();
                ctx.moveTo(data.roi_polygon[0][0], data.roi_polygon[0][1]);
                for (let j = 1; j < 4; j++) {
                    ctx.lineTo(data.roi_polygon[j][0], data.roi_polygon[j][1]);
                }
                ctx.closePath();
                ctx.strokeStyle = puckGlow ? 'rgba(16, 185, 129, 0.45)' : 'rgba(255, 255, 255, 0.2)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
            ctx.restore();
        } else {
            puckX = null;
            puckY = null;
        }
    }

    // =========================================================================
    // 6. Live Distance Gauge Updates
    // =========================================================================
    function updateDistanceMeter(data) {
        if (!el.distContainer) return;
        if (state !== 'scanning') {
            el.distContainer.style.display = 'none';
            return;
        }
        el.distContainer.style.display = 'block';

        if (!data || !data.hand_detected || data.distance_cm === null || data.distance_cm === undefined) {
            el.distVal.textContent = '-- cm';
            el.distStatus.textContent = 'Position open palm facing camera';
            el.distNeedle.className = 'distance-needle';
            el.distNeedle.style.left = '50%';
            return;
        }

        const dist = Number(data.distance_cm);
        el.distVal.textContent = `${dist.toFixed(1)} cm`;

        // Map 15 cm -> 45 cm to track width (0% to 100%)
        // Closer hand (low cm) maps to left (too close); farther hand maps to right (too far)
        const minRange = 14;
        const maxRange = 44;
        let pct = ((dist - minRange) / (maxRange - minRange)) * 100;
        pct = Math.max(6, Math.min(94, pct));
        el.distNeedle.style.left = `${pct}%`;

        if (data.distance_zone === 'optimal') {
            el.distNeedle.className = 'distance-needle optimal';
            el.distStatus.textContent = '🎯 Ideal distance — hold steady';
            el.distStatus.style.color = 'var(--success)';
        } else if (data.distance_zone === 'too_close') {
            el.distNeedle.className = 'distance-needle danger';
            el.distStatus.textContent = '⚠️ Too close — move palm back slightly';
            el.distStatus.style.color = 'var(--danger)';
        } else {
            el.distNeedle.className = 'distance-needle warning';
            el.distStatus.textContent = '📏 Move palm closer to camera';
            el.distStatus.style.color = '#f59e0b';
        }
    }

    // =========================================================================
    // 7. Scan Lifecycle
    // =========================================================================
    async function startScan() {
        if (state !== 'idle') return;

        const name = el.inputName.value.trim();
        const usn = el.inputUsn.value.trim();
        if (!name) { setFeedback('Please enter the full name first.'); el.inputName.focus(); return; }
        if (!usn) { setFeedback('Please enter the USN / Student ID first.'); el.inputUsn.focus(); return; }

        sfx.init();

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

        state = 'scanning';
        captured = [];
        prevSnapshot = null;
        cooldownUntil = 0;
        sweetSpotDwellStart = 0;
        hadSweetSpotLock = false;
        clearSlots();
        updateProgress();
        updatePoseHUD(false);

        el.resultBox.style.display = 'none';
        el.resultBox.innerHTML = '';
        el.guide.style.display = 'none'; // Replaced by dynamic canvas reticle

        el.startBtn.style.display = 'none';
        el.cancelBtn.style.display = 'inline-flex';
        el.progressPill.style.display = 'inline-flex';
        el.statusPill.className = 'status-pill success';
        el.statusPill.textContent = 'Scanning — auto-capturing';

        setFeedback('Center your palm inside the circular guide…');
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
        updatePoseHUD(false);
        updateDistanceMeter(null);

        el.resultBox.style.display = 'none';
        el.resultBox.innerHTML = '';
        el.guide.style.display = 'flex';

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

    // =========================================================================
    // 8. Probe Loop (Real-Time HUD & Auto-Capture)
    // =========================================================================
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
            // High quality probe (0.80) to preserve edge definition
            const frameB64 = camera.captureFrameAsBase64(0.80);
            if (!frameB64) return;

            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });
            const data = await resp.json();
            if (state !== 'scanning') return;

            // Render modern Canvas HUD & Distance meter
            drawHUD(data);
            updateDistanceMeter(data);

            const now = Date.now();
            const inSweetSpot = Boolean(data.success && data.hand_detected && data.roi_extracted && data.in_sweet_spot);

            updatePoseHUD(inSweetSpot);

            if (inSweetSpot) {
                // Play gentle lock-on chime when first entering sweet spot
                if (!hadSweetSpotLock) {
                    hadSweetSpotLock = true;
                    sweetSpotDwellStart = now;
                    sfx.playLock();
                }

                // Dwell verification: palm must be steady in sweet spot for >= 300ms
                const dwellMs = now - sweetSpotDwellStart;
                if (now >= cooldownUntil && dwellMs >= 300) {
                    await captureSample(data);
                } else if (now < cooldownUntil) {
                    const secs = Math.max(1, Math.ceil((cooldownUntil - now) / 1000));
                    setFeedback(`Hold steady — next sample in ${secs}s (${currentPrompt()})`);
                } else {
                    setFeedback(`Sweet spot locked 🎯 — hold still (${currentPrompt()})`);
                }
            } else {
                hadSweetSpotLock = false;
                sweetSpotDwellStart = 0;
                setFeedback(data.message || 'Center your open palm facing the camera…');
            }
        } catch (_) {
            /* Transient probe errors ignored */
        } finally {
            probing = false;
        }
    }

    // =========================================================================
    // 9. Sample Capture & Verification
    // =========================================================================
    async function captureSample(probeData) {
        // Near-duplicate guard: require variation across frames
        const snap = snapshotVideo();
        if (snap && prevSnapshot) {
            const diff = meanAbsDiffPct(snap, prevSnapshot);
            if (diff < DUPLICATE_DIFF_PCT) {
                cooldownUntil = Date.now() + 600;
                setFeedback(`Too similar to previous sample — vary slightly: ${currentPrompt()}`);
                return;
            }
        }

        // High fidelity capture (0.92) for biometric template generation
        const hiQuality = camera.captureFrameAsBase64(0.92);
        if (!hiQuality) return;

        if (snap) prevSnapshot = snap;
        captured.push(hiQuality);
        cooldownUntil = Date.now() + COOLDOWN_MS;
        hadSweetSpotLock = false;

        // Sound effect
        sfx.playCapture();

        const idx = captured.length - 1;
        const thumb = (probeData && (probeData.preview_enhanced_b64 || probeData.preview_roi_b64)) || hiQuality;
        renderSlot(idx, thumb);

        updateProgress();
        drawHUD(probeData);

        if (captured.length >= REQUIRED) {
            finishCapture();
        } else {
            updatePoseHUD(false);
            setFeedback(`Sample ${captured.length}/${REQUIRED} captured ✓ — Now: ${currentPrompt()}`);
        }
    }

    function finishCapture() {
        state = 'submitting';
        stopProbe();
        camera.clearCanvas();
        updatePoseHUD(false);
        updateDistanceMeter(null);

        sfx.playSuccess();
        setFeedback(`All ${REQUIRED} samples captured! Saving biometric templates…`);
        el.cancelBtn.style.display = 'none';
        submitEnrollment();
    }

    // =========================================================================
    // 10. Enrollment Submission
    // =========================================================================
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
                    <div style="background: var(--success-bg); border: 1px solid var(--success); padding: 1.25rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--success); margin-bottom: 0.35rem;">✅ Enrollment Successful</h3>
                        <p style="font-size: 0.95rem; color: var(--text-main); margin-bottom: 0.65rem;">${result.message}</p>
                        <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
                            User ID: #${result.user_id} &bull; Model: <strong>${result.model_name}</strong> &bull; Dim: ${result.feature_dim}-D &bull; Avg Latency: ${result.avg_extraction_latency_ms} ms
                        </div>
                        <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                            <a href="/recognize" class="btn btn-sm">Test in Recognition</a>
                            <a href="/consumer?user_id=${result.user_id}" class="btn btn-secondary btn-sm">Open Wallet</a>
                            <button class="btn btn-secondary btn-sm" id="btn-enroll-another">Enroll Another</button>
                        </div>
                    </div>`;
                const anotherBtn = document.getElementById('btn-enroll-another');
                if (anotherBtn) anotherBtn.addEventListener('click', () => {
                    el.inputUsn.value = '';
                    resetToIdle('Enter details for the next user.');
                });
                setFeedback('Enrollment complete.');
            } else {
                state = 'idle';
                el.resultBox.innerHTML = `
                    <div style="background: var(--danger-bg); border: 1px solid var(--danger); padding: 1.25rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--danger); margin-bottom: 0.35rem;">❌ Enrollment Failed</h3>
                        <p style="font-size: 0.95rem; color: var(--text-main);">${result.message}</p>
                        <button class="btn btn-sm" id="btn-retry-scan" style="margin-top: 0.75rem;">🔄 Retry Scan</button>
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

    // Wiring
    el.startBtn.addEventListener('click', startScan);
    el.cancelBtn.addEventListener('click', () => cancelScan());
    el.resetBtn.addEventListener('click', () => resetToIdle());
});
