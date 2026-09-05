/**
 * PalmPay Liveness Detection UI Module (Phase 2 Challenge-Response).
 *
 * Orchestrates:
 * 1. Requesting a random gesture challenge sequence from POST /api/liveness/start
 * 2. Showing step-by-step coaching guides to the user in clean plain English
 * 3. Capturing a short burst of landmark-rich frames (5-8 fps)
 * 4. Submitting frames to POST /api/liveness/verify
 * 5. Yielding a single-use server-issued liveness token
 */

class PalmPayLiveness {
    constructor(camera, options = {}) {
        this.camera = camera;
        this.onSuccess = options.onSuccess || (() => {});
        this.onError = options.onError || (() => {});
        this.onProgress = options.onProgress || (() => {});

        this.challenge = null;
        this.capturing = false;
        this.token = null;
    }

    async startChallenge() {
        try {
            const resp = await fetch('/api/liveness/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await resp.json();
            if (!data.success) {
                throw new Error(data.message || 'Failed to start liveness challenge');
            }

            this.challenge = data;
            this.onProgress({
                stage: 'started',
                sequence: data.sequence,
                instructions: data.instructions,
                expires_at: data.expires_at
            });
            return data;
        } catch (err) {
            this.onError(err.message);
            throw err;
        }
    }

    async captureAndVerify(durationMs = 3500, fps = 6) {
        if (!this.challenge) {
            await this.startChallenge();
        }

        this.capturing = true;
        const frames = [];
        const intervalMs = Math.round(1000 / fps);
        const startTime = Date.now();

        try {
            while (Date.now() - startTime < durationMs && this.capturing) {
                const b64 = this.camera.captureFrameAsBase64(0.85);
                if (b64) {
                    // We send the image base64, and the server runs the detector pipeline
                    // to extract landmarks server-side safely.
                    // If client already computed landmarks, it can attach them too.
                    frames.push({
                        ts_ms: Date.now() - startTime,
                        image: b64
                    });
                }
                await new Promise(r => setTimeout(r, intervalMs));
            }

            this.onProgress({ stage: 'verifying', frameCount: frames.length });

            // In server-side landmark extraction, the server can accept frame images or landmarks
            const verifyResp = await fetch('/api/liveness/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    challenge_id: this.challenge.challenge_id,
                    frames: frames
                })
            });

            const result = await verifyResp.json();
            if (result.success && result.status === 'passed') {
                this.token = result.liveness_token;
                this.onSuccess(result);
                return result;
            } else {
                const msg = result.message || 'Liveness check failed. Please repeat the gestures.';
                this.onError(msg, result.reasons);
                return result;
            }
        } catch (err) {
            this.onError(err.message);
            throw err;
        } finally {
            this.capturing = false;
        }
    }

    stop() {
        this.capturing = false;
    }
}

if (typeof window !== 'undefined') {
    window.PalmPayLiveness = PalmPayLiveness;
}
