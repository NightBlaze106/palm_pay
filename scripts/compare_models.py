import sys
import os
import sqlite3
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from core.evaluation import EvaluationEngine
from core.matcher import PalmMatcher

def get_users_for_model(model_name: str, feature_dim: int):
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users ORDER BY id")
    users = [dict(row) for row in cursor.fetchall()]
    
    result = []
    for u in users:
        cursor.execute(
            "SELECT feature_blob, geo_blob FROM templates t "
            "LEFT JOIN geometry_templates g ON t.user_id = g.user_id AND t.sample_index = g.sample_index "
            "WHERE t.user_id = ? AND t.model_name = ? AND t.feature_dim = ?",
            (u['id'], model_name, feature_dim)
        )
        rows = cursor.fetchall()
        if not rows: continue
        
        user_copy = dict(u)
        user_copy['templates'] = []
        user_copy['geometry_templates'] = []
        
        for row in rows:
            vec = np.frombuffer(row['feature_blob'], dtype=np.float32)
            user_copy['templates'].append(vec)
            if row['geo_blob']:
                geo = np.frombuffer(row['geo_blob'], dtype=np.float32)
                user_copy['geometry_templates'].append(geo)
        
        result.append(user_copy)
        
    conn.close()
    return result

models_to_test = [
    {"name": "CompCode-v1-fm", "dim": 1536},
    {"name": "HogLbpFusion-v1-fm", "dim": 3652}
]

for model in models_to_test:
    print(f"\n======================================")
    print(f"EVALUATING MODEL: {model['name']}")
    print(f"======================================")
    
    users_with_templates = get_users_for_model(model['name'], model['dim'])
            
    if len(users_with_templates) < 2:
        print(f"Not enough users found for model {model['name']} to evaluate (found {len(users_with_templates)}).")
        continue
        
    print(f"Found {len(users_with_templates)} users with {sum(len(u['templates']) for u in users_with_templates)} total templates.")
    
    original_model_name = config.MODEL_NAME
    config.MODEL_NAME = model['name']
    
    matcher = PalmMatcher(threshold=0.95)
    engine = EvaluationEngine(matcher=matcher)
    
    results = engine.evaluate_enrolled_dataset(users_with_templates)
    
    print("\nGenuine Stats (Same User):")
    print(results.get("genuine_stats", {}))

    print("\nImpostor Stats (Different Users):")
    print(results.get("impostor_stats", {}))
    
    if "suggested_threshold" in results:
        print(f"\nSuggested Texture Threshold: {results['suggested_threshold']}")
        
    print("\nKey Thresholds:")
    for t_val in [0.90, 0.95, 0.98, 0.99, 0.995]:
        sweep_data = next((s for s in results.get("threshold_sweep", []) if abs(s['threshold'] - t_val) < 0.001), None)
        if sweep_data:
            print(f"  T={t_val:.3f} => GAR: {sweep_data['gar']:.4f}, FAR: {sweep_data['far']:.4f}")
            
    config.MODEL_NAME = original_model_name

