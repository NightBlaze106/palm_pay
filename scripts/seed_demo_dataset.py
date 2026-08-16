"""
Demo Dataset Seeder for PalmPay.
Populates SQLite with sample users and realistic palmprint feature templates
for instant testing of the Biometric Evaluation, 1:N Recognition, and User Manager modules.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database.db import Database


def seed_demo_users():
    db = Database(config.DATABASE_PATH)

    demo_users = [
        {"name": "Alice Johnson", "usn": "1MS21CS001"},
        {"name": "Bob Smith", "usn": "1MS21CS045"},
        {"name": "Carol Davis", "usn": "1MS21CS089"},
        {"name": "David Wilson", "usn": "1MS21CS112"},
    ]

    print(f"Seeding demo users into database at: {config.DATABASE_PATH}")
    np.random.seed(1337)

    for user_info in demo_users:
        existing = db.get_user_by_usn(user_info["usn"])
        if existing:
            print(f"  User {user_info['name']} ({user_info['usn']}) already exists. Skipping.")
            continue

        uid = db.add_user(user_info["name"], user_info["usn"])
        if not uid:
            print(f"  Failed to insert user {user_info['name']}.")
            continue

        # Generate base palm identity embedding
        base_vector = np.random.randn(config.FEATURE_VECTOR_DIM).astype(np.float32)
        base_vector /= np.linalg.norm(base_vector)

        # Generate 4 distinct sample captures per user with intra-subject variation
        samples = []
        for s_idx in range(4):
            noise = np.random.randn(config.FEATURE_VECTOR_DIM).astype(np.float32)
            noise /= np.linalg.norm(noise)
            sample_vec = 0.95 * base_vector + 0.05 * noise
            sample_vec /= np.linalg.norm(sample_vec)
            samples.append(sample_vec)

        db.add_templates(uid, samples, model_name=config.MODEL_NAME)
        print(f"  Enrolled {user_info['name']} (ID: #{uid}) with 4 biometric templates.")

    stats = db.get_stats()
    print("\nDatabase Seeding Complete:")
    print(f"  Total Enrolled Users: {stats['total_users']}")
    print(f"  Total Templates: {stats['total_templates']}")


if __name__ == "__main__":
    seed_demo_users()
