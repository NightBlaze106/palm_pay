"""
PalmPay Biometric Reset Utility (Demo / Prototype).

Wipes all BIOMETRIC users + their templates/wallets/transactions while
preserving the demo MERCHANT accounts and their wallets (needed by the
payment flow). Run after any feature/model upgrade that changes the
embedding dimension — old templates are incompatible and are ignored by
the loader anyway.

Usage:
    python scripts/reset_biometrics.py [--yes]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database.db import Database


def main():
    db = Database(config.DATABASE_PATH)

    biometric_users = db.get_all_users_with_templates()  # excludes MERCH-* accounts
    print("=" * 60)
    print("PalmPay Biometric Reset")
    print(f"  Database: {config.DATABASE_PATH}")
    print(f"  Active model: {config.MODEL_NAME} ({config.FEATURE_VECTOR_DIM}-D)")
    print(f"  Biometric users to delete: {len(biometric_users)}")
    for u in biometric_users:
        print(f"    - #{u['id']} {u['name']} ({u['usn']}) [{u['sample_count']} templates]")
    print("  Merchant accounts (m1-m4) are PRESERVED.")
    print("=" * 60)

    if "--yes" not in sys.argv:
        answer = input("Proceed? [y/N]: ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    deleted = 0
    for u in biometric_users:
        if db.delete_user(u["id"]):  # cascades templates, wallet, txns, requests
            deleted += 1

    stats = db.get_stats()
    print(f"Deleted {deleted} user(s). Remaining totals: "
          f"{stats['total_users']} users, {stats['total_templates']} templates.")
    print("Re-enroll fresh via /enroll (automated scan).")


if __name__ == "__main__":
    main()
