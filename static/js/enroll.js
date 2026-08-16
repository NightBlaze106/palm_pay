/**
 * PalmPay — Enrollment Controller.
 * Manages multi-sample capture, validation, and submission to the backend.
 */

document.addEventListener('DOMContentLoaded', () => {
    const REQUIRED_SAMPLES = 4;
    const camera = new PalmCamera('webcam-video', 'camera-canvas');

    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnCaptureSample = document.getElementById('btn-capture-sample');
    const btnResetSamples = document.getElementById('btn-reset-samples');
    const btnSubmitEnrollment = document.getElementById('btn-submit-enrollment');

    const inputName = document.getElementById('input-name');
    const inputUsn = document.getElementById('input-usn');
    const cameraStatusPill = document.getElementById('camera-status-pill');
    const cameraGuide = document.getElementById('camera-guide');
    const liveFeedbackText = document.getElementById('live-feedback-text');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const sampleCounterBtn = document.getElementById('sample-counter-btn');
    const enrollmentResultBox = document.getElementById('enrollment-result-box');

    let capturedSamples = []; // Array of Base64 strings
    let isProbeRunning = false;
    let probeInterval = null;

    // Start Camera
    btnStartCamera.addEventListener('click', async () => {
        try {
            btnStartCamera.disabled = true;
            btnStartCamera.textContent = '⏳ Initializing...';
            cameraStatusPill.className = 'status-pill warning';
            cameraStatusPill.textContent = 'Connecting Camera...';

            await camera.start();

            cameraStatusPill.className = 'status-pill success';
            cameraStatusPill.textContent = 'Camera Live';
            btnStartCamera.textContent = '✅ Camera Active';
            btnCaptureSample.disabled = false;
            liveFeedbackText.textContent = 'Position your open palm steadily inside the guide box.';

            startAlignmentProbe();
        } catch (err) {
            cameraStatusPill.className = 'status-pill danger';
            cameraStatusPill.textContent = 'Camera Error';
            btnStartCamera.disabled = false;
            btnStartCamera.textContent = '📷 Retry Camera';
            liveFeedbackText.textContent = err.message;
        }
    });

    // Frame alignment probe
    function startAlignmentProbe() {
        if (probeInterval) clearInterval(probeInterval);

        probeInterval = setInterval(async () => {
            if (!camera.isActive || isProbeRunning) return;
            isProbeRunning = true;

            try {
                const frameB64 = camera.captureFrameAsBase64(0.7);
                if (!frameB64) return;

                const resp = await fetch('/api/process_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: frameB64 })
                });

                const data = await resp.json();
                if (data.success && data.hand_detected && data.roi_extracted) {
                    cameraGuide.classList.add('active');
                    cameraGuide.textContent = 'Palm Detected & Aligned';
                    liveFeedbackText.textContent = `Palm Aligned (${data.latency_ms.toFixed(1)} ms). Click "Capture Sample".`;
                    camera.drawPolygon(data.roi_polygon, '#10b981', 3);
                } else {
                    cameraGuide.classList.remove('active');
                    cameraGuide.textContent = 'Position open palm facing camera';
                    liveFeedbackText.textContent = data.message || 'Waiting for hand...';
                    camera.clearCanvas();
                }
            } catch (e) {
                // Ignore transient frame probe errors
            } finally {
                isProbeRunning = false;
            }
        }, 350);
    }

    // Capture Sample
    btnCaptureSample.addEventListener('click', async () => {
        if (!camera.isActive) return;
        if (capturedSamples.length >= REQUIRED_SAMPLES) return;

        btnCaptureSample.disabled = true;
        liveFeedbackText.textContent = 'Validating biometric sample...';

        const frameB64 = camera.captureFrameAsBase64(0.92);
        if (!frameB64) {
            btnCaptureSample.disabled = false;
            return;
        }

        try {
            const resp = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameB64 })
            });

            const data = await resp.json();
            if (!data.success || !data.roi_extracted) {
                alert(`Sample capture failed: ${data.message}. Please hold your palm steady and centered.`);
                btnCaptureSample.disabled = false;
                return;
            }

            // Valid sample captured
            const sampleIdx = capturedSamples.length;
            capturedSamples.push(frameB64);

            // Update Thumbnail Slot
            const slot = document.getElementById(`slot-${sampleIdx}`);
            if (slot) {
                slot.innerHTML = `
                    <img src="${data.preview_enhanced_b64 || data.preview_roi_b64 || frameB64}" alt="Sample ${sampleIdx + 1}">
                    <span class="slot-label">Sample ${sampleIdx + 1} (${data.latency_ms.toFixed(0)}ms)</span>
                `;
            }

            updateProgress();
        } catch (err) {
            alert(`Error processing sample: ${err.message}`);
        } finally {
            btnCaptureSample.disabled = capturedSamples.length >= REQUIRED_SAMPLES;
        }
    });

    function updateProgress() {
        const count = capturedSamples.length;
        sampleCounterBtn.textContent = count;
        progressText.textContent = `${count} of ${REQUIRED_SAMPLES} captured`;
        const pct = (count / REQUIRED_SAMPLES) * 100;
        progressBar.style.width = `${pct}%`;

        btnResetSamples.disabled = count === 0;

        if (count >= REQUIRED_SAMPLES) {
            progressBar.classList.add('success');
            btnCaptureSample.disabled = true;
            btnSubmitEnrollment.disabled = false;
            liveFeedbackText.textContent = 'All samples captured! Enter user details and click "Save Enrollment".';
        } else {
            progressBar.classList.remove('success');
            btnSubmitEnrollment.disabled = true;
        }
    }

    // Reset Samples
    btnResetSamples.addEventListener('click', () => {
        capturedSamples = [];
        for (let i = 0; i < REQUIRED_SAMPLES; i++) {
            const slot = document.getElementById(`slot-${i}`);
            if (slot) {
                slot.innerHTML = `
                    <span style="font-size: 1.25rem; color: var(--text-dim);">✋</span>
                    <span class="slot-label">Sample ${i + 1}</span>
                `;
            }
        }
        updateProgress();
        enrollmentResultBox.style.display = 'none';
        if (camera.isActive) btnCaptureSample.disabled = false;
    });

    // Submit Enrollment
    btnSubmitEnrollment.addEventListener('click', async () => {
        const name = inputName.value.trim();
        const usn = inputUsn.value.trim();

        if (!name) {
            alert('Please enter the user full name.');
            inputName.focus();
            return;
        }
        if (!usn) {
            alert('Please enter the USN / Student ID.');
            inputUsn.focus();
            return;
        }
        if (capturedSamples.length < REQUIRED_SAMPLES) {
            alert(`Please capture all ${REQUIRED_SAMPLES} samples.`);
            return;
        }

        btnSubmitEnrollment.disabled = true;
        btnSubmitEnrollment.textContent = '⏳ Processing & Enrolling...';

        try {
            const resp = await fetch('/api/enroll', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    usn: usn,
                    samples: capturedSamples
                })
            });

            const result = await resp.json();
            enrollmentResultBox.style.display = 'block';

            if (result.success) {
                enrollmentResultBox.innerHTML = `
                    <div style="background: var(--success-bg); border: 1px solid var(--success); padding: 1rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--success); margin-bottom: 0.25rem;">✅ Enrollment Successful</h3>
                        <p style="font-size: 0.9rem; color: var(--text-main); margin-bottom: 0.5rem;">${result.message}</p>
                        <div style="font-size: 0.8rem; color: var(--text-muted);">
                            User ID: #${result.user_id} &bull; Model: ${result.model_name} &bull; Dim: ${result.feature_dim}-D &bull; Avg Extraction: ${result.avg_extraction_latency_ms} ms
                        </div>
                        <div style="margin-top: 0.75rem;">
                            <a href="/recognize" class="btn btn-sm">Test in Recognition</a>
                            <a href="/users" class="btn btn-secondary btn-sm" style="margin-left: 0.5rem;">View Users</a>
                        </div>
                    </div>
                `;
                btnSubmitEnrollment.textContent = '✅ Enrolled';
            } else {
                enrollmentResultBox.innerHTML = `
                    <div style="background: var(--danger-bg); border: 1px solid var(--danger); padding: 1rem; border-radius: var(--radius-sm);">
                        <h3 style="color: var(--danger); margin-bottom: 0.25rem;">❌ Enrollment Failed</h3>
                        <p style="font-size: 0.9rem; color: var(--text-main);">${result.message}</p>
                    </div>
                `;
                btnSubmitEnrollment.disabled = false;
                btnSubmitEnrollment.textContent = 'Save Enrollment';
            }
        } catch (err) {
            alert(`Network error during enrollment: ${err.message}`);
            btnSubmitEnrollment.disabled = false;
            btnSubmitEnrollment.textContent = 'Save Enrollment';
        }
    });
});
