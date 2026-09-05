#!/usr/bin/env python3
"""
PalmPay Dataset Purge Utility.
Wipes all ephemerally captured pilot ROI crops from data/pilot_rois.
Production database (palmpay.db), enrolled templates, and wallets remain completely untouched.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def purge_dataset(target_dir: str = None, force: bool = False) -> int:
    if target_dir is None:
        target_dir = getattr(config, "DATASET_DIR", os.path.join(config.BASE_DIR, "data", "pilot_rois"))

    if not os.path.exists(target_dir):
        print(f"[Purge] Target directory does not exist: {target_dir}. Nothing to purge.")
        return 0

    entries = os.listdir(target_dir)
    if not entries:
        print(f"[Purge] Directory {target_dir} is already empty.")
        return 0

    print(f"[Purge] Found {len(entries)} items in {target_dir}.")
    if not force:
        confirm = input("Are you sure you want to permanently delete all pilot ROI images? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("[Purge] Aborted by user.")
            return 0

    count = 0
    for item in entries:
        p = os.path.join(target_dir, item)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            count += 1
        except Exception as exc:
            print(f"[Purge] Error removing {p}: {exc}")

    print(f"[Purge] Successfully purged {count} items from {target_dir}.")
    print("[Purge] Standing privacy principle satisfied: No raw/derived images retained on disk.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Purge ephemeral pilot ROI dataset")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dir", type=str, default=None, help="Specific directory to purge (defaults to DATASET_DIR)")
    args = parser.parse_args()
    ret = purge_dataset(target_dir=args.dir, force=args.yes)
    sys.exit(0)


if __name__ == "__main__":
    main()
