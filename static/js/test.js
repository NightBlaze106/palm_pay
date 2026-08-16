/**
 * PalmPay — Biometric Evaluation and Threshold Calibration Controller.
 */

document.addEventListener('DOMContentLoaded', () => {
    const btnRunEval = document.getElementById('btn-run-eval');
    const btnApplyThreshold = document.getElementById('btn-apply-threshold');
    const btnUseSuggested = document.getElementById('btn-use-suggested');
    const thresholdSlider = document.getElementById('threshold-slider');
    const sliderValDisplay = document.getElementById('slider-val-display');
    const currentThresholdBadge = document.getElementById('current-threshold-badge');
    const thresholdFeedbackMsg = document.getElementById('threshold-feedback-msg');

    const statSubjects = document.getElementById('stat-subjects');
    const statSamples = document.getElementById('stat-samples');
    const statGenuinePairs = document.getElementById('stat-genuine-pairs');
    const statImpostorPairs = document.getElementById('stat-impostor-pairs');
    const evalPromptText = document.getElementById('eval-prompt-text');

    const distributionCard = document.getElementById('distribution-card');
    const sweepCard = document.getElementById('sweep-card');
    const eerBadge = document.getElementById('eer-badge');
    const sweepTbody = document.getElementById('sweep-tbody');

    const genMean = document.getElementById('gen-mean');
    const genStd = document.getElementById('gen-std');
    const genMin = document.getElementById('gen-min');
    const genMax = document.getElementById('gen-max');

    const impMean = document.getElementById('imp-mean');
    const impStd = document.getElementById('imp-std');
    const impMin = document.getElementById('imp-min');
    const impMax = document.getElementById('imp-max');

    let lastEvalResult = null;

    // Threshold Slider update
    thresholdSlider.addEventListener('input', () => {
        const val = parseFloat(thresholdSlider.value).toFixed(2);
        sliderValDisplay.textContent = val;
    });

    // Apply Threshold API
    btnApplyThreshold.addEventListener('click', async () => {
        const val = parseFloat(thresholdSlider.value);
        try {
            btnApplyThreshold.disabled = true;
            const resp = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: val })
            });
            const data = await resp.json();
            if (data.success) {
                currentThresholdBadge.textContent = `Active: ${data.threshold.toFixed(2)}`;
                thresholdFeedbackMsg.style.color = 'var(--success)';
                thresholdFeedbackMsg.textContent = `✅ Decision threshold successfully updated to ${data.threshold.toFixed(2)}.`;
            } else {
                thresholdFeedbackMsg.style.color = 'var(--danger)';
                thresholdFeedbackMsg.textContent = `❌ ${data.message}`;
            }
        } catch (err) {
            thresholdFeedbackMsg.style.color = 'var(--danger)';
            thresholdFeedbackMsg.textContent = `Error: ${err.message}`;
        } finally {
            btnApplyThreshold.disabled = false;
        }
    });

    // Use Suggested EER threshold
    btnUseSuggested.addEventListener('click', () => {
        if (lastEvalResult && lastEvalResult.suggested_threshold !== undefined) {
            thresholdSlider.value = lastEvalResult.suggested_threshold;
            sliderValDisplay.textContent = Number(lastEvalResult.suggested_threshold).toFixed(2);
            btnApplyThreshold.click();
        }
    });

    // Run Evaluation
    btnRunEval.addEventListener('click', async () => {
        btnRunEval.disabled = true;
        btnRunEval.textContent = '⏳ Evaluating Dataset...';
        evalPromptText.textContent = 'Computing cross-comparison matrices across enrolled templates...';

        try {
            const resp = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const result = await resp.json();
            lastEvalResult = result;

            if (result.status === 'insufficient_data') {
                alert(result.message);
                evalPromptText.textContent = result.message;
                return;
            }

            renderEvaluationResults(result);
        } catch (err) {
            alert(`Evaluation error: ${err.message}`);
        } finally {
            btnRunEval.disabled = false;
            btnRunEval.textContent = '🚀 Run Evaluation';
        }
    });

    function renderEvaluationResults(res) {
        statSubjects.textContent = res.total_users;
        statSamples.textContent = res.total_samples;
        statGenuinePairs.textContent = res.genuine_comparisons;
        statImpostorPairs.textContent = res.impostor_comparisons;
        evalPromptText.textContent = `Evaluation complete on ${res.genuine_comparisons} genuine and ${res.impostor_comparisons} impostor pairs.`;

        // Render Distribution Stats
        const gs = res.genuine_stats || {};
        genMean.textContent = gs.mean !== undefined ? gs.mean.toFixed(4) : '-';
        genStd.textContent = gs.std !== undefined ? gs.std.toFixed(4) : '-';
        genMin.textContent = gs.min !== undefined ? gs.min.toFixed(4) : '-';
        genMax.textContent = gs.max !== undefined ? gs.max.toFixed(4) : '-';

        const ims = res.impostor_stats || {};
        impMean.textContent = ims.mean !== undefined ? ims.mean.toFixed(4) : '-';
        impStd.textContent = ims.std !== undefined ? ims.std.toFixed(4) : '-';
        impMin.textContent = ims.min !== undefined ? ims.min.toFixed(4) : '-';
        impMax.textContent = ims.max !== undefined ? ims.max.toFixed(4) : '-';

        distributionCard.style.display = 'block';

        // EER & Suggested Threshold
        if (res.eer_estimate) {
            eerBadge.textContent = `EER: ${(res.eer_estimate.eer_value * 100).toFixed(2)}% @ T=${res.eer_estimate.threshold.toFixed(2)}`;
            eerBadge.className = 'status-pill warning';
        }
        if (res.suggested_threshold !== undefined) {
            btnUseSuggested.disabled = false;
            btnUseSuggested.textContent = `Use Suggested (T=${res.suggested_threshold.toFixed(2)})`;
        }

        // Render Threshold Sweep Table
        let rowsHtml = '';
        const sweep = res.threshold_sweep || [];
        sweep.forEach(pt => {
            const garPct = (pt.gar * 100).toFixed(1);
            const frrPct = (pt.frr * 100).toFixed(1);
            const farPct = (pt.far * 100).toFixed(1);
            const accPct = (pt.accuracy * 100).toFixed(1);

            rowsHtml += `
                <tr>
                    <td><strong>${pt.threshold.toFixed(2)}</strong></td>
                    <td style="color: var(--success);">${garPct}%</td>
                    <td style="color: var(--warning);">${frrPct}%</td>
                    <td style="color: ${pt.far > 0.05 ? 'var(--danger)' : 'var(--text-main)'};">${farPct}%</td>
                    <td><strong>${accPct}%</strong></td>
                    <td>${pt.gen_accept_count}</td>
                    <td>${pt.gen_reject_count}</td>
                    <td>${pt.imp_accept_count}</td>
                    <td>${pt.imp_reject_count}</td>
                </tr>
            `;
        });
        sweepTbody.innerHTML = rowsHtml;
        sweepCard.style.display = 'block';
    }
});
