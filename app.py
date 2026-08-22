"""
PalmPay — Contactless Palm Biometric Authentication System (Phase 1 Prototype)
Flask Backend Application and REST API.
"""

from datetime import datetime
import csv
import io
import os
import time
from functools import wraps
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for

import config
from database.db import Database
from core.pipeline import BiometricPipeline
from core.matcher import PalmMatcher
from core.evaluation import EvaluationEngine
from bio_session import BiometricSessionStore

app = Flask(__name__)
app.config["SECRET_KEY"] = "palmpay-dev-secret-key"

# Initialize global components
db = Database(config.DATABASE_PATH)
pipeline = BiometricPipeline()
matcher = PalmMatcher(threshold=config.DEFAULT_RECOGNITION_THRESHOLD)
evaluation_engine = EvaluationEngine(matcher=matcher)

# Server-side biometric payment sessions (Demo / Prototype):
# a successful /api/recognize match creates a short-lived authenticated
# session here; /api/payment resolves the payer from it — never from the request body.
biometric_sessions = BiometricSessionStore(ttl_seconds=config.BIOMETRIC_SESSION_TTL_SECONDS)


def ensure_merchant_accounts():
    """Idempotently creates users rows + MERCHANT wallets for configured demo merchants."""
    try:
        db.seed_merchant_accounts()
    except Exception as exc:  # pragma: no cover - startup safety net
        print(f"[PalmPay] Warning: merchant account seeding failed: {exc}")


ensure_merchant_accounts()

# Resolved demo merchants: config id -> {"merchant_id", "name", "category", "user_id"}
MERCHANT_ACCOUNTS = {acc["merchant_id"]: acc for acc in db.seed_merchant_accounts()}
MERCHANT_BY_USER_ID = {acc["user_id"]: acc for acc in MERCHANT_ACCOUNTS.values()}


def resolve_merchant(merchant_key):
    """
    Resolves a merchant reference (config id like 'm1' or numeric users.id)
    to its account dict, or None.
    """
    if merchant_key is None:
        return None
    key = str(merchant_key).strip()
    if key in MERCHANT_ACCOUNTS:
        return MERCHANT_ACCOUNTS[key]
    try:
        return MERCHANT_BY_USER_ID.get(int(key))
    except ValueError:
        return None


@app.context_processor
def payment_template_helpers():
    """Template helpers for the Demo / Prototype payment system."""

    def merchant_config_id(user_id):
        acc = MERCHANT_BY_USER_ID.get(user_id)
        return acc["merchant_id"] if acc else None

    def merchant_category(user_id):
        acc = MERCHANT_BY_USER_ID.get(user_id)
        return acc["category"] if acc else ""

    def merchant_theme_class(user_or_config):
        """Icon-circle color class per design system (Library gets its own tint)."""
        acc = None
        if isinstance(user_or_config, dict):
            acc = user_or_config
        else:
            acc = resolve_merchant(user_or_config)
        if not acc:
            return "cat-services"
        if acc["name"] == "Library":
            return "cat-library"
        category = acc.get("category", "")
        return {
            "Food": "cat-food",
            "Services": "cat-services",
            "Retail": "cat-retail",
        }.get(category, "cat-services")

    def inr(amount):
        try:
            return f"{float(amount):,.2f}"
        except (TypeError, ValueError):
            return "0.00"

    return dict(
        merchant_config_id=merchant_config_id,
        merchant_category=merchant_category,
        merchant_theme_class=merchant_theme_class,
        merchant_list=list(MERCHANT_ACCOUNTS.values()),
        inr=inr,
        # Payment system constants available to all templates (POS partial needs them
        # everywhere it is included — standalone /pay AND the embedded merchant tab).
        max_amount=config.MAX_PAYMENT_AMOUNT,
        ttl_seconds=config.BIOMETRIC_SESSION_TTL_SECONDS,
        fusion_window=config.FUSION_WINDOW,
        fusion_interval_ms=config.FUSION_CAPTURE_INTERVAL_MS,
        # Automated enrollment (fingerprint-style scan)
        cooldown_ms=config.ENROLL_CAPTURE_COOLDOWN_MS,
        pose_prompts=list(config.ENROLL_POSE_PROMPTS),
        duplicate_diff_pct=config.ENROLL_DUPLICATE_DIFF_PCT,
    )


# ---------------------------------------------------------
# Page Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    """Home / Overview Page."""
    stats = db.get_stats()
    return render_template(
        "index.html",
        stats=stats,
        threshold=matcher.threshold,
        model_name=config.MODEL_NAME,
        feature_dim=config.FEATURE_VECTOR_DIM,
    )


@app.route("/enroll")
def enroll_page():
    """Palm Enrollment Page."""
    return render_template(
        "enroll.html",
        required_samples=config.DEFAULT_ENROLLMENT_SAMPLES,
        min_samples=config.MIN_ENROLLMENT_SAMPLES,
        max_samples=config.MAX_ENROLLMENT_SAMPLES,
    )


@app.route("/recognize")
def recognize_page():
    """Real-Time 1:N Palm Recognition Page."""
    stats = db.get_stats()
    return render_template(
        "recognize.html",
        threshold=matcher.threshold,
        user_count=stats["total_users"],
        model_name=config.MODEL_NAME,
    )


@app.route("/users")
def users_page():
    """Enrolled Users Management Page."""
    users = db.get_all_users_with_templates()
    stats = db.get_stats()
    return render_template(
        "users.html",
        users=users,
        stats=stats,
    )


@app.route("/test")
def test_page():
    """Evaluation & Threshold Calibration Page."""
    stats = db.get_stats()
    return render_template(
        "test.html",
        stats=stats,
        threshold=matcher.threshold,
    )


# ---------------------------------------------------------
# Payment System (Demo / Prototype) — Admin Session Helpers
# ---------------------------------------------------------

def is_admin() -> bool:
    return bool(session.get("is_admin"))


def admin_required(view_fn):
    @wraps(view_fn)
    def wrapper(*args, **kwargs):
        if not is_admin():
            return redirect(url_for("admin_login", next=request.path))
        return view_fn(*args, **kwargs)
    return wrapper


@app.route("/admin")
def admin_root():
    """Admin entry point — redirects to dashboard or login."""
    if is_admin():
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("admin_login"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Simple password-protected admin login (demo only)."""
    error = None
    if request.method == "POST":
        username = str(request.form.get("username", "")).strip()
        password = str(request.form.get("password", ""))
        if username == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD:
            session["is_admin"] = True
            next_url = request.args.get("next") or url_for("admin_dashboard")
            if not next_url.startswith("/"):
                next_url = url_for("admin_dashboard")
            return redirect(next_url)
        error = "Invalid admin credentials."

    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    session.pop("bio_token", None)
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """All users, wallet balances, and system-wide payment stats."""
    wallets = db.get_all_wallets_with_users()
    pay_stats = db.get_payment_system_stats()
    consumers = [w for w in wallets if w["role"] == "CONSUMER"]
    merchants = [w for w in wallets if w["role"] == "MERCHANT"]
    pending_topups = db.list_topup_requests(status="PENDING")
    return render_template(
        "admin_dashboard.html",
        consumers=consumers,
        merchants=merchants,
        pay_stats=pay_stats,
        pending_topups=pending_topups,
    )


@app.route("/admin/topup", methods=["GET"])
@admin_required
def admin_topup():
    """Direct top-up form (demo convenience)."""
    consumers = db.get_all_wallets_with_users()
    consumers = [w for w in consumers if w["role"] == "CONSUMER"]
    recent_topups = db.list_transactions(types=["TOPUP"], limit=10)["transactions"]
    return render_template(
        "admin_topup.html",
        consumers=consumers,
        recent_topups=recent_topups,
        max_amount=config.MAX_PAYMENT_AMOUNT,
    )


@app.route("/admin/cashouts", methods=["GET"])
@admin_required
def admin_cashouts():
    """Pending + historical merchant cashout requests (logged only, no real transfer)."""
    pending = db.list_cashout_requests(status="PENDING")
    history = db.list_cashout_requests()
    return render_template(
        "admin_cashouts.html",
        pending=pending,
        history=history,
    )


@app.route("/api/cashout/<int:request_id>/approve", methods=["POST"])
def api_cashout_approve(request_id: int):
    """Admin-only approval of a PENDING cashout request (simulated settlement)."""
    if not is_admin():
        return jsonify({"success": False, "message": "Admin authentication required."}), 401
    result = db.approve_cashout_request(request_id)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/api/topup", methods=["POST"])
def api_topup():
    """
    Admin-only wallet top-up. Validates the admin server-side session.
    Body: { user_id, amount } -> credits wallet + logs TOPUP transaction.
    """
    if not is_admin():
        return jsonify({"success": False, "message": "Admin authentication required."}), 401

    data = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get("user_id"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "A valid user must be selected."}), 400

    try:
        amount = float(data.get("amount"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Amount must be a number."}), 400

    if amount <= 0:
        return jsonify({"success": False, "message": "Top-up amount must be greater than zero."}), 400

    wallet = db.get_wallet(user_id)
    if not wallet:
        return jsonify({"success": False, "message": "Wallet not found for this user."}), 404
    if wallet["role"] != "CONSUMER":
        return jsonify({"success": False, "message": "Only CONSUMER wallets can be topped up."}), 400

    result = db.add_topup(user_id, amount, note="TOPUP")
    if not result.get("success"):
        return jsonify(result), 400

    user = db.get_user_by_id(user_id)
    return jsonify(
        {
            "success": True,
            "message": f"Added ₹{amount:.2f} to {user['name'] if user else 'user'}'s wallet.",
            "new_balance": result["new_balance"],
            "transaction_id": result["transaction_id"],
        }
    )


# ---------------------------------------------------------
# Payment System (Demo / Prototype) — Consumer Dashboard
# ---------------------------------------------------------

def _list_consumers():
    """All CONSUMER wallets joined with identity, oldest first."""
    consumers = [w for w in db.get_all_wallets_with_users() if w["role"] == "CONSUMER"]
    consumers.sort(key=lambda w: w["user_id"])
    return consumers


@app.route("/consumer")
def consumer_dashboard():
    """Mobile-first consumer wallet dashboard (Demo / Prototype)."""
    consumers = _list_consumers()
    if not consumers:
        return render_template("consumer.html", consumer=None)

    user_id = request.args.get("user_id", type=int)
    current = next((w for w in consumers if w["user_id"] == user_id), None)
    if current is None:
        current = consumers[-1]  # most recently enrolled consumer

    uid = current["user_id"]
    wallet = db.ensure_wallet(uid, "CONSUMER")
    stats = db.get_consumer_stats(uid)
    recent = db.list_transactions(user_id=uid, role="consumer", limit=5)["transactions"]
    my_requests = db.list_topup_requests(user_id=uid)[:5]

    return render_template(
        "consumer.html",
        consumer=current,
        balance=wallet["balance"],
        stats=stats,
        recent_txns=recent,
        consumers=consumers,
        my_topup_requests=my_requests,
    )


@app.route("/api/topup/request", methods=["POST"])
def api_topup_request():
    """
    Consumer submits a top-up request. Sits in PENDING until an admin approves it
    on /admin (Demo / Prototype — no real money).
    Body: { user_id, amount }
    """
    data = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get("user_id"))
        amount = float(data.get("amount"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Valid user and amount are required."}), 400

    if amount <= 0:
        return jsonify({"success": False, "message": "Top-up amount must be greater than zero."}), 400

    wallet = db.get_wallet(user_id)
    if not wallet or wallet["role"] != "CONSUMER":
        return jsonify({"success": False, "message": "Consumer wallet not found."}), 404

    request_id = db.create_topup_request(user_id, amount)
    return jsonify(
        {
            "success": True,
            "message": f"Top-up request for ₹{amount:.2f} submitted. Pending admin approval.",
            "request_id": request_id,
        }
    )


@app.route("/api/topup_request/<int:request_id>/approve", methods=["POST"])
def api_topup_request_approve(request_id: int):
    """Admin-only approval of a consumer top-up request."""
    if not is_admin():
        return jsonify({"success": False, "message": "Admin authentication required."}), 401
    result = db.approve_topup_request(request_id)
    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


# ---------------------------------------------------------
# Payment System (Demo / Prototype) — POS Payment API
# ---------------------------------------------------------

def _log_failed_amount(bio, merchant, raw_amount, reason):
    """Audit-logs a FAILED_AMOUNT attempt when the payer is known."""
    try:
        amount_val = float(raw_amount)
        if amount_val < 0:
            amount_val = 0.0
    except (TypeError, ValueError):
        amount_val = 0.0
    db.add_transaction(
        amount=amount_val,
        status="FAILED_AMOUNT",
        sender_user_id=bio["user_id"] if bio else None,
        merchant_id=merchant["user_id"] if merchant else None,
        merchant_name=merchant["name"] if merchant else None,
        biometric_score=bio["score"] if bio else None,
        note=f"FAILED_AMOUNT: {reason}",
    )


@app.route("/api/payment", methods=["POST"])
def api_payment():
    """
    Processes a palm-authenticated demo payment.

    Security model (Demo / Prototype):
      - The payer identity ALWAYS comes from the server-side biometric session
        created by /api/recognize within the last 30 seconds.
      - user_id sent by the frontend is never trusted / never accepted.

    Body: { merchant_id, amount }
    """
    data = request.get_json(silent=True) or {}

    # Resolve target merchant (hardcoded demo set).
    merchant = resolve_merchant(data.get("merchant_id"))
    if not merchant or merchant.get("user_id") is None:
        return jsonify(
            {
                "success": False,
                "reason": "INVALID_MERCHANT",
                "message": "Unknown merchant. Select one of the demo merchants.",
            }
        ), 400

    # 1. Validate a fresh (<30s) server-side biometric result exists.
    bio = biometric_sessions.get(session.get("bio_token"))
    if bio is None:
        return jsonify(
            {
                "success": False,
                "reason": "FAILED_AUTH",
                "message": "No fresh palm authentication. Please scan your palm again.",
            }
        ), 401

    # 3./4. Amount validation (>0 and <= MAX_PAYMENT_AMOUNT, numeric only).
    raw_amount = data.get("amount")
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        _log_failed_amount(bio, merchant, raw_amount, "non-numeric amount")
        return jsonify(
            {
                "success": False,
                "reason": "FAILED_AMOUNT",
                "message": "Amount must be a number.",
            }
        ), 400

    if amount <= 0:
        _log_failed_amount(bio, merchant, amount, "amount must be > 0")
        return jsonify(
            {"success": False, "reason": "FAILED_AMOUNT", "message": "Amount must be greater than ₹0."}
        ), 400

    if amount > config.MAX_PAYMENT_AMOUNT:
        _log_failed_amount(bio, merchant, amount, f"exceeds limit ₹{config.MAX_PAYMENT_AMOUNT}")
        return jsonify(
            {
                "success": False,
                "reason": "FAILED_AMOUNT",
                "message": f"Amount exceeds the demo limit of ₹{config.MAX_PAYMENT_AMOUNT}.",
            }
        ), 400

    # 2. Payer comes strictly from the server-side session (never request body).
    sender_user_id = bio["user_id"]

    # 5.-9. Balance fetch + atomic debit/credit + transaction record.
    pay_res = db.process_payment(
        sender_user_id=sender_user_id,
        merchant_user_id=merchant["user_id"],
        merchant_name=merchant["name"],
        amount=amount,
        biometric_score=bio["score"],
    )

    if pay_res["status"] == "SUCCESS":
        # Burn the biometric session: one scan authorizes exactly one payment.
        biometric_sessions.invalidate(session.get("bio_token"))
        session.pop("bio_token", None)

        consumer = db.get_user_by_id(sender_user_id)
        return jsonify(
            {
                "success": True,
                "transaction_id": pay_res["transaction_id"],
                "new_balance": pay_res["new_balance"],
                "receipt_url": f"/receipt/{pay_res['transaction_id']}",
                "consumer": {"id": sender_user_id, "name": bio["name"], "usn": bio["usn"]},
                "merchant": {"id": merchant["merchant_id"], "name": merchant["name"]},
                "amount": pay_res["amount"],
                "biometric_score": bio["score"],
            }
        )

    if pay_res["status"] == "FAILED_BALANCE":
        return jsonify(
            {
                "success": False,
                "reason": "FAILED_BALANCE",
                "transaction_id": pay_res.get("transaction_id"),
                "balance": pay_res.get("balance"),
                "consumer_name": bio["name"],
                "message": "Insufficient wallet balance for this payment.",
            }
        ), 402

    return jsonify({"success": False, "reason": pay_res["status"], "message": pay_res.get("message", "Payment failed.")}), 400


@app.route("/pay")
def pay_page():
    """POS terminal screen (Demo / Prototype)."""
    return render_template(
        "pay.html",
        merchants=list(MERCHANT_ACCOUNTS.values()),
        max_amount=config.MAX_PAYMENT_AMOUNT,
        ttl_seconds=config.BIOMETRIC_SESSION_TTL_SECONDS,
    )


# ---------------------------------------------------------
# Payment System (Demo / Prototype) — Merchant Dashboard
# ---------------------------------------------------------

@app.route("/merchant")
def merchant_dashboard():
    """Desktop-optimized merchant/POS dashboard (Demo / Prototype)."""
    accounts = list(MERCHANT_ACCOUNTS.values())
    if not accounts:
        return render_template("merchant.html", merchant=None)

    merchant_key = request.args.get("merchant_id", "m1")
    merchant = resolve_merchant(merchant_key) or accounts[0]

    wallet = db.ensure_wallet(merchant["user_id"], "MERCHANT")
    stats = db.get_merchant_stats(merchant["user_id"])
    pending_cashouts = db.list_cashout_requests(status="PENDING", merchant_user_id=merchant["user_id"])
    cashout_history = db.list_cashout_requests(merchant_user_id=merchant["user_id"])[:8]

    return render_template(
        "merchant.html",
        merchant=merchant,
        balance=wallet["balance"],
        stats=stats,
        merchants=accounts,
        pending_cashouts=pending_cashouts,
        cashout_history=cashout_history,
        active_nav=request.args.get("view", "dashboard"),
    )


@app.route("/api/cashout", methods=["POST"])
def api_cashout_request():
    """
    Merchant requests a cash out (logged only, no real transfer).
    Body: { merchant_id, amount } -> creates a PENDING cashout_request.
    """
    data = request.get_json(silent=True) or {}
    merchant = resolve_merchant(data.get("merchant_id"))
    if not merchant or merchant.get("user_id") is None:
        return jsonify({"success": False, "message": "Unknown merchant."}), 400

    try:
        amount = float(data.get("amount"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Amount must be a number."}), 400

    if amount <= 0:
        return jsonify({"success": False, "message": "Cash out amount must be greater than zero."}), 400

    wallet = db.get_wallet(merchant["user_id"])
    if wallet and amount > wallet["balance"]:
        return jsonify({"success": False, "message": f"Cash out exceeds available balance (₹{wallet['balance']:.2f})."}), 400

    request_id = db.create_cashout_request(merchant["user_id"], amount)
    return jsonify(
        {
            "success": True,
            "request_id": request_id,
            "message": f"Cash out request for ₹{amount:.2f} submitted (pending admin approval).",
        }
    )


@app.route("/api/merchant/stats/<merchant_id>")
def api_merchant_stats(merchant_id):
    """Revenue stats + 7-day daily chart series for a demo merchant."""
    merchant = resolve_merchant(merchant_id)
    if not merchant or merchant.get("user_id") is None:
        return jsonify({"success": False, "message": "Unknown merchant."}), 404

    stats = db.get_merchant_stats(merchant["user_id"])
    return jsonify(
        {
            "success": True,
            "merchant_id": merchant["merchant_id"],
            "name": merchant["name"],
            "balance": db.ensure_wallet(merchant["user_id"], "MERCHANT")["balance"],
            "today": stats["today_revenue"],
            "this_week": stats["week_revenue"],
            "total": stats["total_txn_count"],
            "daily_chart_data": stats["daily_chart_data"],
        }
    )


# ---------------------------------------------------------
# Payment System (Demo / Prototype) — Wallets, History, Receipt
# ---------------------------------------------------------

@app.route("/api/wallet/<user_id>")
def api_wallet(user_id):
    """
    Returns { balance, role, recent_transactions } for a wallet owner.
    Accepts a numeric users.id or a demo merchant id ("m1".."m4").
    """
    resolved_id = user_id
    merchant = resolve_merchant(user_id)
    if merchant and merchant.get("user_id") is not None:
        resolved_id = merchant["user_id"]
    else:
        try:
            resolved_id = int(user_id)
        except (TypeError, ValueError):
            return jsonify({"success": False, "message": "Unknown wallet."}), 404

    wallet = db.get_wallet(resolved_id)
    if not wallet:
        return jsonify({"success": False, "message": "Wallet not found."}), 404

    recent = db.list_transactions(
        user_id=resolved_id,
        role="consumer" if wallet["role"] == "CONSUMER" else "merchant",
        limit=10,
    )["transactions"]

    owner = db.get_user_by_id(resolved_id)
    return jsonify(
        {
            "success": True,
            "user_id": resolved_id,
            "name": owner["name"] if owner else None,
            "usn": owner["usn"] if owner else None,
            "role": wallet["role"],
            "balance": wallet["balance"],
            "recent_transactions": recent,
        }
    )


@app.route("/api/transactions")
def api_transactions():
    """
    Paginated transaction list.
    Query params: user_id, role (consumer|merchant), limit, offset, status, date_from, date_to
    """
    role = request.args.get("role", type=str)
    user_id = request.args.get("user_id", type=str)

    resolved_uid = None
    if user_id not in (None, "", "all"):
        if role == "merchant":
            merchant = resolve_merchant(user_id)
            resolved_uid = merchant["user_id"] if merchant else None
        else:
            try:
                resolved_uid = int(user_id)
            except ValueError:
                resolved_uid = None
        if resolved_uid is None:
            return jsonify({"success": False, "message": "Unknown user_id."}), 404

    result = db.list_transactions(
        user_id=resolved_uid,
        role=role,
        limit=request.args.get("limit", default=20, type=int),
        offset=request.args.get("offset", default=0, type=int),
        status=request.args.get("status", type=str) or None,
        date_from=request.args.get("date_from", type=str) or None,
        date_to=request.args.get("date_to", type=str) or None,
    )
    result["success"] = True
    return jsonify(result)


@app.route("/transactions")
def transactions_page():
    """Shared transaction history (consumer or merchant view)."""
    role = request.args.get("role", "consumer")
    if role not in ("consumer", "merchant"):
        role = "consumer"

    consumers = _list_consumers()
    merchants = list(MERCHANT_ACCOUNTS.values())

    selected_user_id = request.args.get("user_id", type=str)
    current = None
    if role == "consumer":
        try:
            uid_int = int(selected_user_id) if selected_user_id else None
        except ValueError:
            uid_int = None
        current = next((w for w in consumers if w["user_id"] == uid_int), None)
        if current is None and consumers:
            current = consumers[-1]
        resolved_uid = current["user_id"] if current else None
    else:
        merchant = resolve_merchant(selected_user_id) or (merchants[0] if merchants else None)
        current = merchant
        resolved_uid = merchant["user_id"] if merchant else None

    filters = {
        "status": request.args.get("status", "") or "",
        "date_from": request.args.get("date_from", "") or "",
        "date_to": request.args.get("date_to", "") or "",
    }

    rows = []
    total = 0
    if resolved_uid is not None:
        listing = db.list_transactions(
            user_id=resolved_uid,
            role=role,
            limit=200,
            offset=0,
            status=filters["status"] or None,
            date_from=filters["date_from"] or None,
            date_to=filters["date_to"] or None,
        )
        rows = listing["transactions"]
        total = listing["total"]

    return render_template(
        "transactions.html",
        role=role,
        current=current,
        consumers=consumers,
        merchants=merchants,
        txns=rows,
        total=total,
        filters=filters,
    )


@app.route("/api/transactions/export")
def api_transactions_export():
    """CSV export of the currently filtered transaction list (Demo / Prototype)."""
    role = request.args.get("role", "consumer")
    user_id = request.args.get("user_id", type=str)

    resolved_uid = None
    if user_id not in (None, "", "all"):
        if role == "merchant":
            merchant = resolve_merchant(user_id)
            resolved_uid = merchant["user_id"] if merchant else None
        else:
            try:
                resolved_uid = int(user_id)
            except ValueError:
                resolved_uid = None
        if resolved_uid is None:
            return jsonify({"success": False, "message": "Unknown user_id."}), 404

    listing = db.list_transactions(
        user_id=resolved_uid,
        role=role,
        limit=200,
        offset=0,
        status=request.args.get("status", type=str) or None,
        date_from=request.args.get("date_from", type=str) or None,
        date_to=request.args.get("date_to", type=str) or None,
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Transaction ID", "DateTime (UTC)", "Type", "Party", "Amount (INR)", "Status", "Biometric Score", "Note"]
    )
    for t in listing["transactions"]:
        party = t.get("merchant_name") or t.get("sender_name") or "-"
        writer.writerow(
            [
                t["id"],
                t["created_at"],
                t["type"],
                party,
                f"{t['amount']:.2f}",
                t["status"],
                f"{t['biometric_score']:.4f}" if t["biometric_score"] is not None else "",
                t.get("note") or "",
            ]
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"palmpay_demo_transactions_{role}_{stamp}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


@app.route("/receipt/<txn_id>")
def receipt_page(txn_id):
    """Printable demo receipt for one transaction."""
    txn = db.get_transaction(txn_id)
    return render_template(
        "receipt.html",
        txn=txn,
        threshold=matcher.threshold,
    )


# ---------------------------------------------------------
# Biometric REST API Endpoints
# ---------------------------------------------------------

@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """
    Real-time frame probe for live camera HUD visualizer.
    Receives single base64 frame -> returns hand detection, landmarks, and ROI polygon.
    """
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"success": False, "message": "No image data received."}), 400

    frame_bgr = pipeline.decode_base64_to_image(image_b64)
    if frame_bgr is None:
        return jsonify({"success": False, "message": "Failed to decode frame."}), 400

    # Run biometric pipeline on incoming frame
    result = pipeline.process_frame(frame_bgr, include_visuals=True)

    # Return serializable summary
    return jsonify(
        {
            "success": result["success"],
            "hand_detected": result["hand_detected"],
            "landmarks_detected": result["landmarks_detected"],
            "roi_extracted": result["roi_extracted"],
            "roi_polygon": result["roi_polygon"],
            "latency_ms": result["latency_ms"],
            "preview_roi_b64": result["preview_roi_bgr_b64"],
            "preview_enhanced_b64": result["preview_enhanced_b64"],
            "preview_annotated_b64": result["preview_annotated_b64"],
            "message": result["message"],
        }
    )


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """
    Enrolls a new user with 3-5 palm sample frames.
    Validates identity, extracts multi-scale LBP embeddings for all samples, and stores in SQLite.
    """
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    usn = str(data.get("usn", "")).strip().upper()
    samples_b64 = data.get("samples", [])

    if not name or not usn:
        return jsonify({"success": False, "message": "Name and USN / Student ID are required."}), 400

    if not isinstance(samples_b64, list) or len(samples_b64) < config.MIN_ENROLLMENT_SAMPLES:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"At least {config.MIN_ENROLLMENT_SAMPLES} palm samples are required for reliable enrollment.",
                }
            ),
            400,
        )

    # Check for duplicate USN
    existing_user = db.get_user_by_usn(usn)
    if existing_user:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"A user with USN '{usn}' is already enrolled ({existing_user['name']}).",
                }
            ),
            409,
        )

    extracted_embeddings = []
    latencies = []

    for idx, s_b64 in enumerate(samples_b64):
        frame_bgr = pipeline.decode_base64_to_image(s_b64)
        if frame_bgr is None:
            return (
                jsonify({"success": False, "message": f"Sample #{idx + 1} could not be decoded."}),
                400,
            )

        res = pipeline.process_frame(frame_bgr, include_visuals=False)
        if not res["success"] or res["embedding"] is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"Sample #{idx + 1} failed biometric extraction: {res['message']}. Please position palm steadily.",
                    }
                ),
                400,
            )

        extracted_embeddings.append(res["embedding"])
        latencies.append(res["latency_ms"])

    # Create user in database
    user_id = db.add_user(name, usn)
    if not user_id:
        return jsonify({"success": False, "message": "Failed to create user record in database."}), 500

    # Store templates
    db_success = db.add_templates(user_id, extracted_embeddings, model_name=config.MODEL_NAME)
    if not db_success:
        db.delete_user(user_id)
        return jsonify({"success": False, "message": "Failed to save biometric templates."}), 500

    # Payment system (demo): every enrolled consumer gets a wallet automatically.
    db.ensure_wallet(user_id, "CONSUMER")

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    return jsonify(
        {
            "success": True,
            "message": f"Successfully enrolled {name} ({usn}) with {len(extracted_embeddings)} palm samples.",
            "user_id": user_id,
            "name": name,
            "usn": usn,
            "sample_count": len(extracted_embeddings),
            "model_name": config.MODEL_NAME,
            "feature_dim": config.FEATURE_VECTOR_DIM,
            "avg_extraction_latency_ms": avg_latency,
        }
    )


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Performs 1:N Biometric Authentication.

    Accuracy Sprint — multi-frame fusion:
      Body may be { image: b64 } (legacy single frame) or
      { images: [b64, b64, ...] } (POS burst, up to FUSION_WINDOW frames).
    Decision = hardened mean-template match on the MEAN query embedding AND at
    least FUSION_MIN_PASS_FRAMES individual frames clearing the threshold.
    The response contract of the legacy single-frame flow is unchanged; fusion
    fields are additive.
    """
    start_total = time.perf_counter()
    data = request.get_json(silent=True) or {}

    images_b64 = data.get("images") if isinstance(data.get("images"), list) else []
    if not images_b64 and data.get("image"):
        images_b64 = [data.get("image")]

    if not images_b64:
        return jsonify({"success": False, "message": "No query image provided."}), 400

    images_b64 = images_b64[: config.FUSION_WINDOW]

    # Step 1: Run Biometric Extraction Pipeline per frame.
    # Frame 0 carries UI previews (legacy behavior); extra frames skip visuals.
    pipeline_results = []
    decode_failures = 0
    for idx, image_b64 in enumerate(images_b64):
        if not isinstance(image_b64, str) or not image_b64:
            decode_failures += 1
            continue
        frame_bgr = pipeline.decode_base64_to_image(image_b64)
        if frame_bgr is None:
            decode_failures += 1
            continue
        pipeline_results.append(
            pipeline.process_frame(frame_bgr, include_visuals=(idx == 0))
        )

    ok_frames = [
        r for r in pipeline_results if r["success"] and r["embedding"] is not None
    ]

    total_lat = (time.perf_counter() - start_total) * 1000.0

    def _failure_response(first_res=None):
        """Legacy-shaped failure payload for the whole burst."""
        res = first_res or {}
        return jsonify(
            {
                "success": False,
                "recognized": False,
                "hand_detected": bool(res.get("hand_detected")),
                "roi_extracted": bool(res.get("roi_extracted")),
                "matched_user": None,
                "similarity_score": 0.0,
                "threshold_used": matcher.threshold,
                "feature_extraction_latency_ms": sum(r["latency_ms"] for r in pipeline_results),
                "matching_latency_ms": 0.0,
                "total_latency_ms": round(total_lat, 2),
                "preview_roi_b64": res.get("preview_roi_bgr_b64", ""),
                "preview_enhanced_b64": res.get("preview_enhanced_b64", ""),
                "preview_annotated_b64": res.get("preview_annotated_b64", ""),
                "frames_received": len(images_b64),
                "frames_quality_passed": 0,
                "frames_above_threshold": 0,
                "message": (
                    res.get("message")
                    or ("Failed to decode query frame(s)." if decode_failures else "No usable palm frames.")
                ),
            }
        )

    if not pipeline_results:
        return _failure_response()

    first_res = pipeline_results[0]
    if not ok_frames:
        # Every frame failed (no hand / quality gate). Report the most useful one:
        quality_fails = [r for r in pipeline_results if r.get("failure_stage") == "quality"]
        return _failure_response(quality_fails[0] if quality_fails else first_res)

    # Step 2: Fuse embeddings -> normalized MEAN query vector.
    stacked = np.stack([np.asarray(f["embedding"], dtype=np.float32) for f in ok_frames])
    mean_query = stacked.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean_query))
    mean_query = mean_query / norm if norm > 1e-7 else stacked[0]

    # Step 3: Hardened matching on the fused query + per-frame agreement count.
    t_match_start = time.perf_counter()
    enrolled_users = db.get_all_users_with_templates()
    match_res = matcher.match_1_to_n(mean_query, enrolled_users)
    frames_above = sum(
        1 for f in ok_frames
        if matcher.frame_pass_score(f["embedding"], enrolled_users) >= matcher.threshold
    )
    matching_lat = (time.perf_counter() - t_match_start) * 1000.0
    total_lat = (time.perf_counter() - start_total) * 1000.0

    # Step 4: Fusion gate — identity stands only if enough individual frames
    # independently clear the threshold. Single-frame requests require 1/1
    # (identical to legacy behavior).
    min_frames_required = max(1, min(config.FUSION_MIN_PASS_FRAMES, len(ok_frames)))
    fusion_ok = frames_above >= min_frames_required

    if match_res["recognized"] and not fusion_ok:
        match_res["recognized"] = False
        match_res["matched_user"] = None
        match_res["reason"] = "unstable_capture"
        match_res["message"] = (
            f"Capture unstable — only {frames_above}/{len(ok_frames)} frames matched clearly. "
            "Hold the palm steady and scan again."
        )

    extraction_lat = round(sum(r["latency_ms"] for r in pipeline_results), 2)

    # Payment system (Demo): a successful server-side match opens a short-lived
    # biometric session bound to THIS browser. A failed scan invalidates any
    # previous one. The /api/recognize response contract is unchanged.
    prev_bio_token = session.get("bio_token")
    if match_res["recognized"]:
        matched = match_res["matched_user"]
        session["bio_token"] = biometric_sessions.create(
            user_id=matched["id"],
            name=matched["name"],
            usn=matched.get("usn", ""),
            score=match_res["similarity_score"],
        )
    elif prev_bio_token:
        biometric_sessions.invalidate(prev_bio_token)
        session.pop("bio_token", None)

    return jsonify(
        {
            "success": True,
            "recognized": match_res["recognized"],
            "hand_detected": True,
            "roi_extracted": True,
            "matched_user": match_res["matched_user"],
            "best_candidate": match_res.get("best_candidate"),
            "similarity_score": match_res["similarity_score"],
            "threshold_used": match_res["threshold_used"],
            "all_scores": match_res["all_scores"][:5],  # top 5
            "feature_extraction_latency_ms": extraction_lat,
            "matching_latency_ms": round(matching_lat, 2),
            "total_latency_ms": round(total_lat, 2),
            "preview_roi_b64": first_res["preview_roi_bgr_b64"] if ok_frames and ok_frames[0]["preview_roi_bgr_b64"] else "",
            "preview_enhanced_b64": (ok_frames[0]["preview_enhanced_b64"] if ok_frames else "") or first_res.get("preview_enhanced_b64", ""),
            "preview_annotated_b64": (ok_frames[0]["preview_annotated_b64"] if ok_frames else "") or first_res.get("preview_annotated_b64", ""),
            "frames_received": len(images_b64),
            "frames_quality_passed": len(ok_frames),
            "frames_above_threshold": frames_above,
            "fusion_min_frames_required": min_frames_required,
            "match_reason": match_res.get("reason"),
            "message": match_res["message"],
        }
    )


@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Lists all enrolled users."""
    users = db.get_all_users_with_templates()
    safe_users = []
    for u in users:
        safe_users.append(
            {
                "id": u["id"],
                "name": u["name"],
                "usn": u["usn"],
                "sample_count": u["sample_count"],
                "created_at": u["created_at"],
            }
        )
    return jsonify({"success": True, "users": safe_users, "total": len(safe_users)})


@app.route("/api/users/<int:user_id>/delete", methods=["POST", "DELETE"])
def api_delete_user(user_id: int):
    """Deletes an enrolled user and their templates."""
    success = db.delete_user(user_id)
    if success:
        return jsonify({"success": True, "message": f"User #{user_id} successfully deleted."})
    return jsonify({"success": False, "message": "Failed to delete user."}), 400


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Gets or sets biometric configuration and threshold."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_threshold = data.get("threshold")
        if new_threshold is not None:
            try:
                t_val = float(new_threshold)
                if 0.0 <= t_val <= 1.0:
                    matcher.set_threshold(t_val)
                    return jsonify(
                        {
                            "success": True,
                            "message": f"Threshold updated to {t_val:.4f}",
                            "threshold": matcher.threshold,
                        }
                    )
                else:
                    return jsonify({"success": False, "message": "Threshold must be between 0.0 and 1.0."}), 400
            except ValueError:
                return jsonify({"success": False, "message": "Invalid threshold number."}), 400

    return jsonify(
        {
            "threshold": matcher.threshold,
            "model_name": config.MODEL_NAME,
            "feature_dim": config.FEATURE_VECTOR_DIM,
            "roi_size": list(config.ROI_SIZE),
            "clahe_clip_limit": config.CLAHE_CLIP_LIMIT,
            "default_threshold": config.DEFAULT_RECOGNITION_THRESHOLD,
        }
    )


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Executes empirical biometric evaluation across enrolled templates.
    Computes genuine vs impostor scores, GAR, FAR, FRR, and suggested threshold.
    """
    users_data = db.get_all_users_with_templates()
    eval_result = evaluation_engine.evaluate_enrolled_dataset(users_data)
    return jsonify(eval_result)


@app.route("/api/export_csv", methods=["GET"])
def api_export_csv():
    """Generates and downloads the evaluation CSV report."""
    users_data = db.get_all_users_with_templates()
    eval_result = evaluation_engine.evaluate_enrolled_dataset(users_data)
    csv_content = evaluation_engine.generate_csv_report(eval_result)

    filename = f"palmpay_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


# ---------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PalmPay — Contactless Palm Biometric Authentication")
    print("Target Environment: Intel i3 / Windows CPU / RGB Camera")
    print(f"Local Server: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print("=" * 60)
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=config.DEBUG_MODE,
        use_reloader=False,
    )
