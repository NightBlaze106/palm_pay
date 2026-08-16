/**
 * PalmPay — Real-Time 1:N Recognition Controller.
 * Manages live stream inference, biometric ROI HUD updates, and match reporting.
 */

document.addEventListener('DOMContentLoaded', () => {
    const camera = new PalmCamera('webcam-video', 'camera-canvas');

    const btnToggleCamera = document.getElementById('btn-toggle-camera');
    const btnScanOnce = document.getElementById('btn-scan-once');
    const chkContinuous = document.getElementById('chk-continuous-mode');
    const fpsCounter = document.getElementById('fps-counter');
    const recognizeStatusPill = document.getElementById('recognize-status-pill');
    const cameraGuide = document.getElementById('camera-guide');

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

    let isScanning = false;
    let scanLoopTimer = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Toggle Camera
    btnToggleCamera.addEventListener('click', async () => {
        if (!camera.isActive) {
            try {
                btnToggleCamera.disabled = true;
                btnToggleCamera.textContent = '⏳ Starting...';
                recognizeStatusPill.className = 'status-pill warning';
                recognizeStatusPill.textContent = 'Connecting...';

                await camera.start();

                recognizeStatusPill.className = 'status-pill success';
                recognizeStatusPill.textContent = 'Live Stream Active';
                btnToggleCamera.textContent = '🛑 Stop Camera';
                btnToggleCamera.disabled = false;
                btnScanOnce.disabled = false;

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
            recognizeStatusPill.className = 'status-pill neutral';
            recognizeStatusPill.textContent = 'Camera Offline';
            btnToggleCamera.textContent = '📷 Start Camera';
            btnScanOnce.disabled = true;
            fpsCounter.textContent = '0 FPS';
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

        const frameB64 = camera.captureFrameAsBase64(0.75);
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

        // Decision Handling
        if (!data.hand_detected) {
            cameraGuide.classList.remove('active');
            cameraGuide.textContent = 'Present palm clearly in view';
            matchResultContainer.className = 'match-result-card';
            matchStatusIcon.textContent = '✋';
            matchPrimaryTitle.textContent = 'No Hand Detected';
            matchSubDetails.textContent = 'Please position your palm facing the camera.';
            scoreMeterFill.className = 'progress-bar-fill';
            camera.clearCanvas();
        } else if (!data.roi_extracted) {
            cameraGuide.classList.remove('active');
            matchResultContainer.className = 'match-result-card';
            matchStatusIcon.textContent = '🔍';
            matchPrimaryTitle.textContent = 'Hand Detected (Aligning Palm)';
            matchSubDetails.textContent = data.message || 'Adjust hand distance.';
            camera.clearCanvas();
        } else {
            cameraGuide.classList.add('active');
            cameraGuide.textContent = 'Palm Tracked';

            if (data.recognized && data.matched_user) {
                // Verified Match
                matchResultContainer.className = 'match-result-card match-success';
                matchStatusIcon.textContent = '✅';
                matchPrimaryTitle.textContent = `Recognized: ${data.matched_user.name}`;
                matchSubDetails.textContent = `USN: ${data.matched_user.usn} • Similarity: ${score.toFixed(4)} ≥ Threshold ${Number(data.threshold_used).toFixed(2)}`;
                scoreMeterFill.className = 'progress-bar-fill success';
            } else {
                // Impostor / Unknown / Sub-threshold
                matchResultContainer.className = 'match-result-card match-fail';
                matchStatusIcon.textContent = '❌';
                matchPrimaryTitle.textContent = 'Palm Not Recognized';

                if (data.best_candidate) {
                    matchSubDetails.textContent = `Best match: ${data.best_candidate.name} (Score: ${score.toFixed(4)} < Threshold ${Number(data.threshold_used).toFixed(2)})`;
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
