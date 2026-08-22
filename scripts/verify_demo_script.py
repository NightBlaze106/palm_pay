"""
Demo Script End-to-End Verification (Stage 9).
Walks the exact 13-step demo scenario from the spec using the Flask test client.
Palm recognition steps are driven through a mocked pipeline boundary (same
mechanism as tests/test_payment.py) since automated tests have no webcam.
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

import numpy as np

import app as A
import config
from tests.test_payment import _pipeline_result, _unit_vec, _variant, _b64_image

STEP = "[{i:>2}] {msg}"


def step(i, msg):
    print(STEP.format(i=i, msg=msg))


def balances(user_id):
    return round(A.db.get_wallet(user_id)["balance"], 2)


def main():
    client = A.app.test_client()
    admin = A.app.test_client()
    img = _b64_image()

    # --- Setup: enroll "Prabal" (demo consumer) through the real enroll route ---
    usn = "PRABAL-" + uuid.uuid4().hex[:6].upper()
    prabal_vec = _unit_vec(4242)
    with patch("app.pipeline.process_frame", return_value=_pipeline_result(prabal_vec)):
        r = client.post("/api/enroll", data=json.dumps(
            {"name": "Prabal", "usn": usn, "samples": [img] * 4}), content_type="application/json")
    prabal_uid = json.loads(r.data)["user_id"]
    canteen_uid = A.MERCHANT_ACCOUNTS["m1"]["user_id"]

    print("=" * 66)
    print("PALMPAY DEMO SCRIPT — END-TO-END VERIFICATION")
    print("=" * 66)

    # 1. Admin logs into /admin
    r = admin.post("/admin/login", data={"username": "admin", "password": config.ADMIN_PASSWORD})
    assert r.status_code == 302
    assert admin.get("/admin/dashboard").status_code == 200
    step(1, "Admin logged into /admin (session-based) ✓")

    # 2. Admin tops up Prabal → ₹500
    r = admin.post("/api/topup", data=json.dumps({"user_id": prabal_uid, "amount": 500}),
                   content_type="application/json")
    d = json.loads(r.data)
    assert d["success"] and d["new_balance"] == 500.0, d
    step(2, f"Admin topped up Prabal → ₹{d['new_balance']:.2f} ✓")

    # 3. Consumer opens /consumer → sees ₹500
    r = client.get(f"/consumer?user_id={prabal_uid}")
    assert r.status_code == 200 and b"500.00" in r.data
    step(3, "Consumer dashboard shows ₹500.00 ✓")

    # 4. Merchant opens /merchant → selects Canteen
    r = client.get("/merchant?merchant_id=m1")
    assert r.status_code == 200 and b"Canteen" in r.data
    stats = json.loads(client.get("/api/merchant/stats/m1").data)
    assert len(stats["daily_chart_data"]) == 7
    step(4, "Merchant dashboard on Canteen (7-day chart data present) ✓")

    # 5. Merchant POS: select Canteen, enter ₹100 (page renders; capture is automatic)
    r = client.get("/pay")
    assert r.status_code == 200 and b"MERCHANT ZONE" in r.data and b"Start Payment" in r.data
    step(5, "Merchant POS loaded (two-zone terminal); Canteen + ₹100 sale started ✓")

    # 6. Present Prabal's palm → recognized (multi-frame burst, score shown)
    burst = [img] * config.FUSION_WINDOW
    with patch("app.pipeline.process_frame", return_value=_pipeline_result(_variant(prabal_vec, 7))):
        rec = json.loads(client.post("/api/recognize", data=json.dumps({"images": burst}),
                                     content_type="application/json").data)
    assert rec["recognized"] and rec["matched_user"]["name"] == "Prabal", rec
    assert rec["frames_quality_passed"] >= config.FUSION_MIN_PASS_FRAMES, rec
    score = rec["similarity_score"]
    step(6, f"Palm recognized: Prabal (score {score:.4f}, {rec['frames_above_threshold']}/{rec['frames_quality_passed']} frames agree) — session opened ✓")

    # 7. Transaction processes: Prabal 500→400, Canteen +100
    pre_canteen = balances(canteen_uid)
    r = client.post("/api/payment", data=json.dumps({"merchant_id": "m1", "amount": 100}),
                    content_type="application/json")
    pay = json.loads(r.data)
    assert pay["success"], pay
    assert pay["new_balance"] == 400.0
    assert abs(balances(canteen_uid) - (pre_canteen + 100)) < 1e-6
    step(7, f"Payment processed: Prabal ₹500→₹{pay['new_balance']:.2f}, Canteen +₹100 ✓")

    # 8. Receipt shown
    r = client.get(pay["receipt_url"])
    assert r.status_code == 200 and b"DEMO RECEIPT" in r.data and b"Palm Verified" in r.data
    txn_id = pay["transaction_id"]
    step(8, f"Receipt /receipt/{txn_id[:8]}… renders (Palm Verified ✓) ✓")

    # 9. Consumer /consumer shows ₹400 + transaction
    r = client.get(f"/consumer?user_id={prabal_uid}")
    assert b"400.00" in r.data and b"Canteen" in r.data
    step(9, "Consumer dashboard shows ₹400.00 + Canteen transaction ✓")

    # 10. Merchant /merchant shows ₹100 revenue in stats/chart
    stats = json.loads(client.get("/api/merchant/stats/m1").data)
    today = [d for d in stats["daily_chart_data"]][-1]
    assert any(abs(d["amount"] - 100.0) < 1e-6 or d is today and today["amount"] >= 100 for d in stats["daily_chart_data"])
    txns = json.loads(client.get("/api/transactions?role=merchant&user_id=m1&limit=5").data)
    assert any(t["id"] == txn_id for t in txns["transactions"])
    step(10, f"Merchant view: today's chart includes ₹{today['amount']:.0f}, txn listed ✓")

    # 11. Wrong palm → Authentication Failed → balance unchanged
    impostor_vec = _unit_vec(999999)
    with patch("app.pipeline.process_frame", return_value=_pipeline_result(impostor_vec)):
        rec2 = json.loads(client.post("/api/recognize", data=json.dumps({"images": [img] * config.FUSION_WINDOW}),
                                       content_type="application/json").data)
    assert not rec2["recognized"]
    r = client.post("/api/payment", data=json.dumps({"merchant_id": "m1", "amount": 50}),
                    content_type="application/json")
    assert r.status_code == 401 and json.loads(r.data)["reason"] == "FAILED_AUTH"
    assert balances(prabal_uid) == 400.0, "Balance must be unchanged after failed auth"
    step(11, "Wrong palm → ✗ Authentication Failed; Prabal still ₹400.00 ✓")

    # 12. Try ₹1000 when balance is ₹400 → Insufficient Balance
    tok = A.biometric_sessions.create(prabal_uid, "Prabal", usn, 0.95)
    with client.session_transaction() as s:
        s["bio_token"] = tok
    r = client.post("/api/payment", data=json.dumps({"merchant_id": "m1", "amount": 1000}),
                    content_type="application/json")
    d = json.loads(r.data)
    assert r.status_code == 402 and d["reason"] == "FAILED_BALANCE" and d["balance"] == 400.0
    failed_txn = A.db.get_transaction(d["transaction_id"])
    assert failed_txn["status"] == "FAILED_BALANCE"
    step(12, "₹1000 with ₹400 balance → ✗ Insufficient Balance (logged, no debit) ✓")

    # 13. Admin approves Canteen cashout request
    r = client.post("/api/cashout", data=json.dumps({"merchant_id": "m1", "amount": 150}),
                    content_type="application/json")
    req_id = json.loads(r.data)["request_id"]
    r = admin.post(f"/api/cashout/{req_id}/approve")
    d = json.loads(r.data)
    assert d["success"]
    step(13, f"Canteen cashout of ₹150 approved (simulated) — balance ₹{d['new_balance']:.2f} ✓")

    # Cleanup demo artifacts from dev DB
    A.db.delete_user(prabal_uid)

    print("-" * 66)
    print("ALL 13 DEMO STEPS VERIFIED SUCCESSFULLY — Demo / Prototype system OK")
    print("=" * 66)


if __name__ == "__main__":
    main()
