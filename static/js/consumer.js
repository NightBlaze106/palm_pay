/**
 * PalmPay Consumer Dashboard (Demo / Prototype).
 * Handles the top-up request modal and demo user switching.
 */

document.addEventListener('DOMContentLoaded', () => {
    const userId = window.PALMPAY_USER_ID;

    // Demo user switcher
    const switcher = document.getElementById('user-switcher');
    if (switcher) {
        switcher.addEventListener('change', () => {
            window.location.href = `/consumer?user_id=${switcher.value}`;
        });
    }

    // Top-up modal controls
    const modal = document.getElementById('topup-modal');
    const btnTopup = document.getElementById('btn-topup');
    const btnClose = document.getElementById('btn-close-topup');
    const amountInput = document.getElementById('topup-amount');
    const btnSubmit = document.getElementById('btn-submit-topup');
    const feedback = document.getElementById('topup-feedback');

    if (!modal || !userId) return;

    function openModal() {
        modal.classList.add('open');
        feedback.style.display = 'none';
        setTimeout(() => amountInput && amountInput.focus(), 60);
    }
    function closeModal() {
        modal.classList.remove('open');
        amountInput.value = '';
    }

    btnTopup.addEventListener('click', openModal);
    btnClose.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });

    document.querySelectorAll('.chip[data-amt]').forEach(chip => {
        chip.addEventListener('click', () => { amountInput.value = chip.dataset.amt; });
    });

    btnSubmit.addEventListener('click', async () => {
        const amount = parseFloat(amountInput.value);
        feedback.style.display = 'block';
        if (!amount || amount <= 0) {
            feedback.style.color = 'var(--red)';
            feedback.textContent = 'Enter a valid amount greater than ₹0.';
            return;
        }
        btnSubmit.disabled = true;
        try {
            const resp = await fetch('/api/topup/request', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, amount: amount })
            });
            const data = await resp.json();
            if (data.success) {
                feedback.style.color = 'var(--green)';
                feedback.textContent = '✓ ' + data.message;
                setTimeout(() => { window.location.reload(); }, 1200);
            } else {
                feedback.style.color = 'var(--red)';
                feedback.textContent = '✗ ' + data.message;
                btnSubmit.disabled = false;
            }
        } catch (err) {
            feedback.style.color = 'var(--red)';
            feedback.textContent = 'Network error: ' + err.message;
            btnSubmit.disabled = false;
        }
    });
});
