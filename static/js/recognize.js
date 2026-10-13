/**
 * PalmPay — Real-Time 1:N Recognition Controller.
 * Features:
 *   - Continuous live camera inference
 *   - Canvas HUD: Centered guidance reticle & dynamic tracking puck
 *   - Monocular distance coaching meter (20–33 cm sweet spot)
 *   - Web Audio feedback (chimes on verified biometric match)
 *   - Biometric ROI inset preview & latency breakdown
 */

document.addEventListener('DOMContentLoaded', () => {
    // =========================================================================
    // 1. Web Audio Synthesizer
    // =========================================================================
    class SoundFx {
        constructor() {
            this.ctx = null;
            this.enabled = localStorage.getItem('palmpay_sound_enabled') !== 'false';
            this.lastMatchSoundTime = 0;
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

        playMatch() {
            if (!this.enabled) return;
            const now = Date.now();
            if (now - this.lastMatchSoundTime < 1800) return; // Prevent audio stutter
            this.lastMatchSoundTime = now;

            try {
                this.init();
                if (!this.ctx) return;
                const audioTime = this.ctx.currentTime;
                // Harmonious chord: C5 (523Hz) and G5 (784Hz)
                [523.25, 783.99].forEach(freq => {
                    const osc = this.ctx.createOscillator();
                    const gain = this.ctx.createGain();
                    osc.type = 'triangle';
                    osc.frequency.setValueAtTime(freq, audioTime);
                    gain.gain.setValueAtTime(0.16, audioTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, audioTime + 0.4);
                    osc.connect(gain);
                    gain.connect(this.ctx.destination);
                    osc.start(audioTime);
                    osc.stop(audioTime + 0.4);
                });
            } catch (_) {}
        }
    }

    const sfx = new SoundFx();

    const camera = new PalmCamera('webcam-video', 'camera-canvas');

    // DOM Elements
    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnScanOnce = document.getElementById('btn-scan-once');
    const chkContinuous = document.getElementById('chk-continuous-mode');
    const btnToggleSound = document.getElementById('btn-toggle-sound');
    const fpsCounter = document.getElementById('fps-counter');
    const recognizeStatusPill = document.getElementById('recognize-status-pill');
    const cameraGuide = document.getElementById('camera-guide');

    const hudPoseBanner = document.getElementById('hud-pose-banner');
    const hudPoseIcon = document.getElementById('hud-pose-icon');
    const hudPoseText = document.getElementById('hud-pose-text');

    const distContainer = document.getElementById('distance-meter-container');
    const distVal = document.getElementById('distance-val-text');
    const distNeedle = document.getElementById('distance-needle');
    const distStatus = document.getElementById('distance-status-text');

    const imgPreviewRoi = document.getElementById('img-preview-roi');
    const textNoRoi = document.getElementById('text-no-roi');
    const imgPreviewEnhanced = document.getElementById('img-preview-enhanced');
    const textNoEnhanced = document.getElementById('text-no-enhanced');

    const matchResultContainer = document.getElementById('match-result-container');
    const matchStatusIcon = document.getElementById('match-status-icon');
    const matchPrimaryTitle = document.getElementById('match-primary-title');
    const matchSubDetails = document.getElementById('match-sub-details');
    const scoreTextDisplay = document.getElementById('score-text-display');
    const scoreMeterFill = document.getElementById('score-meter-fill');
    const activeThresholdDisplay = document.getElementById('active-threshold-display');

    const latencyTotalBadge = document.getElementById('latency-total-badge');
    const latencyExtraction = document.getElementById('latency-extraction');
    const latencyMatching = document.getElementById('latency-matching');
    const candidateRankingsTbody = document.getElementById('candidate-rankings-tbody');

    // Sound toggle button wiring
    if (btnToggleSound) {
        btnToggleSound.textContent = sfx.enabled ? '🔊 Sound' : '🔇 Muted';
        btnToggleSound.addEventListener('click', () => {
            const active = sfx.toggle();
            btnToggleSound.textContent = active ? '🔊 Sound' : '🔇 Muted';
        });
    }

    let isScanning = false;
    let scanLoopTimer = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Smoothed puck coordinates for Canvas rendering
    let puckX = null;
    let puckY = null;
    let puckRadius = 45;

    // Toggle Camera
    btnToggleCamera.addEventListener('click', async () => {
        if (!camera.isActive) {
            try {
                btnToggleCamera.disabled = true;
                btnToggleCamera.textContent = '⏳ Starting...';
                recognizeStatusPill.className = 'status-pill warning';
                recognizeStatusPill.textContent = 'Connecting...';

                sfx.init();
                await camera.start();

                recognizeStatusPill.className = 'status-pill success';
                recognizeStatusPill.textContent = 'Live Stream Active';
                btnToggleCamera.textContent = '🛑 Stop Camera';
                btnToggleCamera.disabled = false;
                btnScanOnce.disabled = false;

                if (cameraGuide) cameraGuide.style.display = 'none'; // Replaced by dynamic canvas reticle
                if (distContainer) distContainer.style.display = 'block';
                if (hudPoseBanner) hudPoseBanner.style.display = 'inline-flex';

                if (chkContinuous.checked) {
                    startScanLoop();
                }
            } catch (err) {
                recognizeStatusPill.className = 'status-pill danger';
                recognizeStatusPill.textContent = 'Camera Error';
                btnToggleCamera.disabled = false;
                btnToggleCamera.textContent = '📷 Start Camera';
                alert(err.message);
            }
        } else {
            stopScanLoop();
            camera.stop();
            camera.clearCanvas();
            recognizeStatusPill.className = 'status-pill neutral';
            recognizeStatusPill.textContent = 'Camera Offline';
            btnToggleCamera.textContent = '📷 Start Camera';
            btnScanOnce.disabled = true;
            fpsCounter.textContent = '0 FPS';

            if (cameraGuide) cameraGuide.style.display = 'flex';
            if (distContainer) distContainer.style.display = 'none';
            if (hudPoseBanner) hudPoseBanner.style.display = 'none';
        }
    });

    chkContinuous.addEventListener('change', () => {
        if (camera.isActive) {
            if (chkContinuous.checked) {
                startScanLoop();
            } else {
                stopScanLoop();
            }
        }
    });

    btnScanOnce.addEventListener('click', () => {
        performRecognitionScan();
    });

    function startScanLoop() {
        if (scanLoopTimer) clearInterval(scanLoopTimer);

        scanLoopTimer = setInterval(() => {
            if (camera.isActive && !isScanning && chkContinuous.checked) {
                performRecognitionScan();
            }
        }, 320); // ~3 scans per second on CPU
    }

    function stopScanLoop() {
        if (scanLoopTimer) {
            clearInterval(scanLoopTimer);
            scanLoopTimer = null;
        }
    }

    async function performRecognitionScan() {
        if (!camera.isActive || isScanning) return;
        isScanning = true;

        const frameB64 = camera.captureFrameAsBase64(0.80);
        if (!frameB64) {
            isScanning = false;
            return;
        }

        try {
            const resp = await fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });

            const data = await resp.json();
            updateHUD(data);
            drawRecognitionHUD(data);
            updateDistanceMeter(data);

            // FPS measurement
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                const fps = ((frameCount * 1000) / (now - lastFpsTime)).toFixed(1);
                fpsCounter.textContent = `${fps} FPS`;
                frameCount = 0;
                lastFpsTime = now;
            }
        } catch (err) {
            console.error('Recognition error:', err);
        } finally {
            isScanning = false;
        }
    }

    // =========================================================================
    // 2. Canvas Reticle & Tracking Puck
    // =========================================================================
    function drawRecognitionHUD(data) {
        if (!camera.ctx || !camera.canvas || !camera.isActive) return;
        const ctx = camera.ctx;
        const w = camera.canvas.width;
        const h = camera.canvas.height;
        ctx.clearRect(0, 0, w, h);

        const cx = w / 2;
        const cy = h / 2;
        const reticleRadius = Math.min(w, h) * 0.35;

        // 1. Static Centered Target Ring
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, reticleRadius, 0, 2 * Math.PI);
        const ringColor = data && data.recognized
            ? '#10b981'
            : (data && data.in_sweet_spot ? '#38bdf8' : 'rgba(255, 255, 255, 0.2)');
        ctx.strokeStyle = ringColor;
        ctx.lineWidth = data && data.recognized ? 5 : 2.5;
        if (data && data.recognized) {
            ctx.shadowColor = '#10b981';
            ctx.shadowBlur = 14;
        }
        ctx.stroke();

        // Crosshairs in target zone
        ctx.beginPath();
        ctx.arc(cx, cy, 28, 0, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();

        // 2. Dynamic Tracking Puck
        if (data && data.hand_detected && data.palm_center) {
            const targetX = data.palm_center[0];
            const targetY = data.palm_center[1];

            // Smooth position interpolation
            puckX = puckX === null ? targetX : puckX + (targetX - puckX) * 0.45;
            puckY = puckY === null ? targetY : puckY + (targetY - puckY) * 0.45;

            let targetRadius = 42;
            if (data.distance_cm && data.distance_cm > 10) {
                targetRadius = Math.max(26, Math.min(65, 42 * (26 / data.distance_cm)));
            }
            puckRadius += (targetRadius - puckRadius) * 0.35;

            let puckColor = '#f59e0b'; // Amber (aligning)
            let isGlowing = false;

            if (data.recognized) {
                puckColor = '#10b981'; // Green (Recognized!)
                isGlowing = true;
            } else if (data.in_sweet_spot) {
                puckColor = '#38bdf8'; // Cyan (In sweet spot)
                isGlowing = true;
            } else if (data.distance_zone === 'too_close') {
                puckColor = '#ef4444'; // Red (Too close)
            }

            ctx.save();
            ctx.beginPath();
            ctx.arc(puckX, puckY, puckRadius, 0, 2 * Math.PI);
            ctx.strokeStyle = puckColor;
            ctx.lineWidth = isGlowing ? 4 : 2.5;
            if (isGlowing) {
                ctx.shadowColor = puckColor;
                ctx.shadowBlur = 16;
                ctx.fillStyle = data.recognized ? 'rgba(16, 185, 129, 0.2)' : 'rgba(56, 189, 248, 0.15)';
                ctx.fill();
            }
            ctx.stroke();

            // Center target pip
            ctx.beginPath();
            ctx.arc(puckX, puckY, 4, 0, 2 * Math.PI);
            ctx.fillStyle = puckColor;
            ctx.fill();

            // Draw ROI polygon box if available
            if (data.roi_polygon && data.roi_polygon.length === 4) {
                ctx.beginPath();
                ctx.moveTo(data.roi_polygon[0][0], data.roi_polygon[0][1]);
                for (let j = 1; j < 4; j++) {
                    ctx.lineTo(data.roi_polygon[j][0], data.roi_polygon[j][1]);
                }
                ctx.closePath();
                ctx.strokeStyle = data.recognized ? 'rgba(16, 185, 129, 0.5)' : 'rgba(255, 255, 255, 0.25)';
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
    // 3. Distance Coaching Gauge Updates
    // =========================================================================
    function updateDistanceMeter(data) {
        if (!distContainer || !camera.isActive) return;

        if (!data || !data.hand_detected || data.distance_cm === null || data.distance_cm === undefined) {
            distVal.textContent = '-- cm';
            distStatus.textContent = 'Position open palm in front of camera';
            distNeedle.className = 'distance-needle';
            distNeedle.style.left = '50%';
            return;
        }

        const dist = Number(data.distance_cm);
        distVal.textContent = `${dist.toFixed(1)} cm`;

        const minRange = 14;
        const maxRange = 44;
        let pct = ((dist - minRange) / (maxRange - minRange)) * 100;
        pct = Math.max(6, Math.min(94, pct));
        distNeedle.style.left = `${pct}%`;

        if (data.distance_zone === 'optimal') {
            distNeedle.className = 'distance-needle optimal';
            distStatus.textContent = '🎯 Ideal distance — hold steady';
            distStatus.style.color = 'var(--success)';
        } else if (data.distance_zone === 'too_close') {
            distNeedle.className = 'distance-needle danger';
            distStatus.textContent = '⚠️ Too close — move palm slightly back';
            distStatus.style.color = 'var(--danger)';
        } else {
            distNeedle.className = 'distance-needle warning';
            distStatus.textContent = '📏 Move palm closer to camera';
            distStatus.style.color = '#f59e0b';
        }
    }

    // =========================================================================
    // 4. Update Main HUD & Match Reporting
    // =========================================================================
    function updateHUD(data) {
        if (!data) return;

        // Update threshold display
        if (data.threshold_used !== undefined) {
            activeThresholdDisplay.textContent = Number(data.threshold_used).toFixed(2);
        }

        // Update ROI Insets
        if (data.preview_roi_b64) {
            imgPreviewRoi.src = data.preview_roi_b64;
            imgPreviewRoi.style.display = 'block';
            textNoRoi.style.display = 'none';
        } else {
            imgPreviewRoi.style.display = 'none';
            textNoRoi.style.display = 'block';
        }

        if (data.preview_enhanced_b64) {
            imgPreviewEnhanced.src = data.preview_enhanced_b64;
            imgPreviewEnhanced.style.display = 'block';
            textNoEnhanced.style.display = 'none';
        } else {
            imgPreviewEnhanced.style.display = 'none';
            textNoEnhanced.style.display = 'block';
        }

        // Update Latency Breakdown
        latencyTotalBadge.textContent = `${(data.total_latency_ms || 0).toFixed(1)} ms`;
        latencyExtraction.textContent = `${(data.feature_extraction_latency_ms || 0).toFixed(1)} ms`;
        latencyMatching.textContent = `${(data.matching_latency_ms || 0).toFixed(1)} ms`;

        // Update Score Display
        const score = data.similarity_score !== undefined ? Number(data.similarity_score) : 0.0;
        scoreTextDisplay.textContent = score.toFixed(4);
        const scorePct = Math.min(Math.max(score * 100, 0), 100);
        scoreMeterFill.style.width = `${scorePct}%`;

        // Update Top HUD Pose Banner
        if (hudPoseBanner) {
            if (!data.hand_detected) {
                hudPoseIcon.textContent = '✋';
                hudPoseText.textContent = 'Present open palm to camera';
                hudPoseBanner.classList.remove('sweet-spot');
            } else if (data.recognized) {
                hudPoseIcon.textContent = '✅';
                hudPoseText.textContent = `Recognized: ${data.matched_user.name}`;
                hudPoseBanner.classList.add('sweet-spot');
            } else if (data.in_sweet_spot) {
                hudPoseIcon.textContent = '🎯';
                hudPoseText.textContent = 'Sweet spot locked — authenticating…';
                hudPoseBanner.classList.add('sweet-spot');
            } else if (data.distance_zone === 'too_close') {
                hudPoseIcon.textContent = '⚠️';
                hudPoseText.textContent = 'Too close — move back slightly';
                hudPoseBanner.classList.remove('sweet-spot');
            } else {
                hudPoseIcon.textContent = '📏';
                hudPoseText.textContent = 'Move palm closer to camera';
                hudPoseBanner.classList.remove('sweet-spot');
            }
        }

        // Decision Handling
        if (!data.hand_detected) {
            matchResultContainer.className = 'match-result-card';
            matchStatusIcon.textContent = '✋';
            matchPrimaryTitle.textContent = 'No Hand Detected';
            matchSubDetails.textContent = 'Please position your palm facing the camera.';
            scoreMeterFill.className = 'progress-bar-fill';
        } else if (!data.roi_extracted) {
            matchResultContainer.className = 'match-result-card';
            matchStatusIcon.textContent = '🔍';
            matchPrimaryTitle.textContent = 'Hand Detected (Aligning Palm)';
            matchSubDetails.textContent = data.message || 'Adjust hand distance.';
        } else {
            if (data.recognized && data.matched_user) {
                // Verified Match
                matchResultContainer.className = 'match-result-card match-success';
                matchStatusIcon.textContent = '✅';
                matchPrimaryTitle.textContent = `Recognized: ${data.matched_user.name}`;
                matchSubDetails.textContent = `USN: ${data.matched_user.usn} • Similarity: ${score.toFixed(4)} ≥ Threshold ${Number(data.threshold_used).toFixed(2)}`;
                scoreMeterFill.className = 'progress-bar-fill success';

                // Play sound chime
                sfx.playMatch();
            } else {
                // Impostor / Unknown / Sub-threshold
                matchResultContainer.className = 'match-result-card match-fail';
                matchStatusIcon.textContent = '❌';
                matchPrimaryTitle.textContent = 'Palm Not Recognized';

                if (data.best_candidate) {
                    matchSubDetails.textContent = `Best candidate: ${data.best_candidate.name} (Score: ${score.toFixed(4)} < Threshold ${Number(data.threshold_used).toFixed(2)})`;
                } else {
                    matchSubDetails.textContent = `Similarity score: ${score.toFixed(4)} is below threshold.`;
                }
                scoreMeterFill.className = 'progress-bar-fill';
            }
        }

        // Update Candidate Rankings Table
        if (data.all_scores && data.all_scores.length > 0) {
            let rowsHtml = '';
            data.all_scores.forEach((cand, idx) => {
                const isMatch = cand.score >= data.threshold_used;
                rowsHtml += `
                    <tr>
                        <td>#${idx + 1}</td>
                        <td><strong>${cand.name}</strong></td>
                        <td><code>${cand.usn}</code></td>
                        <td>
                            <span class="status-pill ${isMatch ? 'success' : 'neutral'}" style="font-size: 0.75rem;">
                                ${cand.score.toFixed(4)}
                            </span>
                        </td>
                    </tr>
                `;
            });
            candidateRankingsTbody.innerHTML = rowsHtml;
        } else {
            candidateRankingsTbody.innerHTML = `
                <tr>
                    <td colspan="4" style="text-align: center; color: var(--text-dim);">No enrolled candidate matches.</td>
                </tr>
            `;
        }
    }
});
