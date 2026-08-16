/**
 * Camera & WebRTC Helper for PalmPay.
 * Handles webcam access, live video rendering, and frame capture to Base64 JPEG.
 */

class PalmCamera {
    constructor(videoElementId, canvasElementId) {
        this.video = document.getElementById(videoElementId);
        this.canvas = document.getElementById(canvasElementId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.stream = null;
        this.isActive = false;
    }

    async start() {
        if (this.isActive) return true;

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: false
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    if (this.canvas) {
                        this.canvas.width = this.video.videoWidth || 640;
                        this.canvas.height = this.video.videoHeight || 480;
                    }
                    this.isActive = true;
                    resolve(true);
                };
            });

            return true;
        } catch (err) {
            console.error('Failed to access webcam:', err);
            throw new Error(`Webcam access error: ${err.message}. Please verify camera permissions.`);
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
        this.isActive = false;
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    captureFrameAsBase64(quality = 0.88) {
        if (!this.isActive || !this.video || this.video.videoWidth === 0) {
            return null;
        }

        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = this.video.videoWidth;
        captureCanvas.height = this.video.videoHeight;
        const captureCtx = captureCanvas.getContext('2d');

        // Draw current unmirrored frame
        captureCtx.drawImage(this.video, 0, 0, captureCanvas.width, captureCanvas.height);
        return captureCanvas.toDataURL('image/jpeg', quality);
    }

    clearCanvas() {
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    drawPolygon(polygonPoints, color = '#10b981', lineWidth = 3) {
        if (!this.ctx || !polygonPoints || polygonPoints.length < 3) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.beginPath();

        this.ctx.moveTo(polygonPoints[0][0], polygonPoints[0][1]);
        for (let i = 1; i < polygonPoints.length; i++) {
            this.ctx.lineTo(polygonPoints[i][0], polygonPoints[i][1]);
        }
        this.ctx.closePath();
        this.ctx.stroke();
    }
}
