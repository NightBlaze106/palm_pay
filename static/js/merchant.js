/**
 * PalmPay Merchant Dashboard (Demo / Prototype).
 * Chart.js revenue chart + paginated incoming transactions + cash out requests.
 */

document.addEventListener('DOMContentLoaded', () => {
    const cfg = window.PALMPAY_MERCHANT;
    if (!cfg) return;

    // Merchant switcher
    const switcher = document.getElementById('merchant-switcher');
    if (switcher) {
        switcher.addEventListener('change', () => {
            window.location.href = `/merchant?merchant_id=${switcher.value}`;
        });
    }

    // ---------- Revenue chart (Chart.js, lazy on first Dashboard render) ----------
    // NOTE: declared before showView() below, which calls it during init.
    let chartBooted = false;
    function ensureRevenueChart() {
        if (chartBooted) return;
        const canvas = document.getElementById('revenue-chart');
        if (!canvas || !window.Chart) return;
        chartBooted = true;

        const data = cfg.chartData || [];
        new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: data.map(d => d.label),
                datasets: [{
                    label: 'Revenue (₹)',
                    data: data.map(d => d.amount),
                    backgroundColor: '#1A73E8',
                    hoverBackgroundColor: '#1557B0',
                    borderRadius: 6,
                    maxBarThickness: 46
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { color: '#E2E6EA' }, ticks: { color: '#5F6368' } },
                    x: { grid: { display: false }, ticks: { color: '#9AA0A6' } }
                }
            }
        });
    }

    // ---------- Single-page view switching (Dashboard / POS / Transactions / Cash Out) ----------
    const VIEWS = ['dashboard', 'pos', 'transactions', 'cashout'];
    const viewSections = document.querySelectorAll('.merchant-view');
    const navLinks = document.querySelectorAll('#m-nav a[data-view]');

    function showView(name) {
        if (!VIEWS.includes(name)) name = 'dashboard';
        viewSections.forEach(v => v.classList.toggle('active', v.dataset.view === name));
        navLinks.forEach(a => a.classList.toggle('active', a.dataset.view === name));

        if (name === 'dashboard') ensureRevenueChart();
        // POS tab: camera is NOT started by viewing it — it only runs during an
        // active payment (started/stopped inside pay.js).
        if (name !== 'pos' && window.PalmPayPOSDeactivate) {
            window.PalmPayPOSDeactivate();  // cancel any pending sale when leaving the terminal
        }
        history.replaceState(null, '', `/merchant?merchant_id=${cfg.merchantId}&view=${name}`);
    }

    navLinks.forEach(a => a.addEventListener('click', (e) => {
        e.preventDefault();
        showView(a.dataset.view);
    }));
    showView(cfg.activeView || 'dashboard');

    // ---------- Paginated transactions (10 per page) ----------
    const PAGE_SIZE = 10;
    let page = 0;
    let total = 0;
    const bodyEl = document.getElementById('m-txn-body');
    const infoEl = document.getElementById('pg-info');
    const prevBtn = document.getElementById('pg-prev');
    const nextBtn = document.getElementById('pg-next');

    function esc(s) {
        return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function statusBadge(status) {
        if (status === 'SUCCESS') return '<span class="badge badge-success">SUCCESS</span>';
        return `<span class="badge badge-failed">${esc(status.replace('FAILED_', 'FAILED '))}</span>`;
    }

    async function loadTxns() {
        bodyEl.innerHTML = '<tr><td colspan="5" class="dim" style="text-align:center;padding:24px;">Loading…</td></tr>';
        try {
            const resp = await fetch(
                `/api/transactions?role=merchant&user_id=${cfg.userId}&limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`
            );
            const data = await resp.json();
            total = data.total || 0;
            const txns = data.transactions || [];

            if (!txns.length) {
                bodyEl.innerHTML = '<tr><td colspan="5" class="dim" style="text-align:center;padding:24px;">No transactions yet.</td></tr>';
            } else {
                bodyEl.innerHTML = txns.map(t => `
                    <tr>
                        <td>${esc((t.created_at || '').slice(11, 16))}</td>
                        <td><strong>${esc(t.sender_name || '—')}</strong></td>
                        <td class="${t.status === 'SUCCESS' ? 'amt-received' : ''}" style="font-weight:700;">+₹${Number(t.amount).toFixed(2)}</td>
                        <td>${t.biometric_score != null ? Number(t.biometric_score).toFixed(4) : '—'}</td>
                        <td>${statusBadge(t.status)}</td>
                    </tr>`).join('');
            }
            const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
            infoEl.textContent = `Page ${page + 1} of ${pages} (${total} txns)`;
            prevBtn.disabled = page === 0;
            nextBtn.disabled = (page + 1) >= pages;
        } catch (err) {
            bodyEl.innerHTML = `<tr><td colspan="5" style="color:var(--red);text-align:center;padding:24px;">Failed to load transactions.</td></tr>`;
        }
    }

    prevBtn.addEventListener('click', () => { if (page > 0) { page--; loadTxns(); } });
    nextBtn.addEventListener('click', () => { if ((page + 1) * PAGE_SIZE < total) { page++; loadTxns(); } });
    loadTxns();

    // ---------- Cash out ----------
    const form = document.getElementById('cashout-form');
    const feedback = document.getElementById('cashout-feedback');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const amount = parseFloat(document.getElementById('cashout-amount').value);
            feedback.style.display = 'block';
            if (!amount || amount <= 0) {
                feedback.style.color = 'var(--red)';
                feedback.textContent = 'Enter a valid amount.';
                return;
            }
            try {
                const resp = await fetch('/api/cashout', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ merchant_id: cfg.merchantId, amount: amount })
                });
                const data = await resp.json();
                if (data.success) {
                    feedback.style.color = 'var(--green)';
                    feedback.textContent = '✓ ' + data.message;
                    setTimeout(() => window.location.reload(), 1000);
                } else {
                    feedback.style.color = 'var(--red)';
                    feedback.textContent = '✗ ' + data.message;
                }
            } catch (err) {
                feedback.style.color = 'var(--red)';
                feedback.textContent = 'Network error: ' + err.message;
            }
        });
    }
});
