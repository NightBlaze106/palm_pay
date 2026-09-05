import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.db import Database
import config
from core.evaluation import EvaluationEngine
from core.palm_model import get_palm_model
from core.matcher import PalmMatcher

db = Database(config.DATABASE_PATH)
users = db.get_all_users_with_templates()

if not users:
    print("No enrolled users found in the database for the active model:", config.MODEL_NAME)
    sys.exit(0)

print(f"Loaded {len(users)} users from database for model {config.MODEL_NAME}.")

matcher = PalmMatcher(threshold=config.DEFAULT_RECOGNITION_THRESHOLD)
engine = EvaluationEngine(matcher=matcher)

results = engine.evaluate_enrolled_dataset(users)

print("\n=== CURRENT EVALUATION RESULTS ===")
print("Genuine Stats (Same User):")
print(results.get("genuine_stats", {}))

print("\nImpostor Stats (Different Users):")
print(results.get("impostor_stats", {}))

if "suggested_threshold" in results:
    print(f"\nSuggested Texture Threshold: {results['suggested_threshold']}")
if "suggested_geo_max_dist" in results:
    print(f"Suggested Geo Max Dist: {results['suggested_geo_max_dist']}")

print("\nThreshold Sweep:")
for sweep in results.get("threshold_sweep", []):
    print(f"  T={sweep['threshold']:.2f} => GAR: {sweep['gar']:.4f}, FAR: {sweep['far']:.4f}")

if results.get("impostor_stats", {}).get("count", 0) > 0:
    far_0_95 = next((s['far'] for s in results['threshold_sweep'] if s['threshold'] == 0.95), "N/A")
    gar_0_95 = next((s['gar'] for s in results['threshold_sweep'] if s['threshold'] == 0.95), "N/A")
    print(f"\nAt strict threshold 0.95 -> FAR: {far_0_95}, GAR: {gar_0_95}")

