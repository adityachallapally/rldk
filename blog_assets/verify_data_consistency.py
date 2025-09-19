import pandas as pd
import json

def verify_data_consistency():
    """Verify all data consistency requirements are met."""
    print("🔍 Verifying RLDK Blog Post Data Consistency...")
    
    try:
        alerts_df = pd.read_json('artifacts/alerts.jsonl', lines=True)
        run_df = pd.read_json('artifacts/run.jsonl', lines=True)
        
        with open('comprehensive_ppo_forensics_demo/comprehensive_analysis.json', 'r') as f:
            analysis_data = json.load(f)
        
        print("✅ All data files loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data files: {e}")
        return False
    
    expected_kl_values = [0.455, 0.568, 0.688, 0.805, 0.937]
    actual_kl_values = alerts_df['kl_value'].tolist()
    
    if actual_kl_values == expected_kl_values:
        print(f"✅ KL progression correct: {actual_kl_values}")
    else:
        print(f"❌ KL progression mismatch. Expected: {expected_kl_values}, Got: {actual_kl_values}")
        return False
    
    expected_health_scores = {
        'overall_health_score': 0.603,
        'training_stability_score': 0.855,
        'convergence_quality_score': 0.959
    }
    
    for score_name, expected_value in expected_health_scores.items():
        actual_value = analysis_data[score_name]
        if actual_value == expected_value:
            print(f"✅ {score_name}: {actual_value}")
        else:
            print(f"❌ {score_name} mismatch. Expected: {expected_value}, Got: {actual_value}")
            return False
    
    termination_alerts = alerts_df[alerts_df['step'] == 44]
    if not termination_alerts.empty and 'stop' in termination_alerts.iloc[0].get('action', ''):
        print("✅ Step 44 termination documented")
    else:
        print("❌ Step 44 termination not found")
        return False
    
    if 'name' in run_df.columns:
        unique_names = run_df['name'].unique()
        expected_names = ['kl', 'reward', 'grad_norm']
        if all(name in unique_names for name in expected_names):
            print(f"✅ Run data uses 'name' field with values: {list(unique_names)}")
        else:
            print(f"❌ Missing expected metric names. Expected: {expected_names}, Got: {list(unique_names)}")
            return False
    else:
        print("❌ Run data missing 'name' field")
        return False
    
    try:
        run_timestamps = set(run_df['time'].apply(lambda x: int(float(x)) if not isinstance(x, int) else x).unique())
        alert_timestamps = set(alerts_df['timestamp'].unique())
    except (ValueError, TypeError) as e:
        print(f"❌ Error processing timestamps: {e}")
        return False
    
    if run_timestamps.intersection(alert_timestamps):
        print("✅ Timestamps consistent between run and alert data")
    else:
        print("❌ No timestamp overlap between run and alert data")
        return False
    
    print("\n🎉 All data consistency checks passed!")
    return True

if __name__ == "__main__":
    verify_data_consistency()
