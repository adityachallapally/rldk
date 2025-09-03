#!/usr/bin/env python3
"""
Wait for reward model training to complete and then continue with PPO training.
"""

import os
import time
import subprocess
import sys

def wait_for_file(filepath, timeout_hours=8):
    """Wait for a file to be created with timeout."""
    timeout_seconds = timeout_hours * 3600
    start_time = time.time()
    
    print(f"Waiting for {filepath} to be created...")
    print(f"Timeout: {timeout_hours} hours")
    
    while time.time() - start_time < timeout_seconds:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            print(f"✓ {filepath} found!")
            return True
        
        elapsed = time.time() - start_time
        print(f"Waiting... ({elapsed/60:.1f} minutes elapsed)")
        time.sleep(300)  # Check every 5 minutes
    
    print(f"✗ Timeout waiting for {filepath}")
    return False

def run_script(script_name):
    """Run a Python script."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[-500:])  # Last 500 chars
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False
    
    return True

def main():
    """Main execution flow."""
    print("RLHF Pipeline Continuation Script")
    print("=" * 50)
    
    # Wait for reward model training to complete
    rm_model_file = "./rldk_demos/rm_a/pytorch_model.bin"
    if not wait_for_file(rm_model_file, timeout_hours=8):
        print("Reward model training did not complete in time. Exiting.")
        return
    
    # Run PPO baseline training
    if not run_script("03_run_ppo_baseline_full.py"):
        print("PPO baseline training failed. Exiting.")
        return
    
    # Run PPO variants
    if not run_script("04_run_variants_with_issues_full.py"):
        print("PPO variants training failed. Exiting.")
        return
    
    # Run RLDK checks
    print(f"\n{'='*60}")
    print("Running RLDK validation checks")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(["./05_rldk_checks.sh"], 
                              capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            print("✓ RLDK checks completed successfully")
        else:
            print(f"✗ RLDK checks failed with return code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[-500:])
    except subprocess.TimeoutExpired:
        print("✗ RLDK checks timed out after 30 minutes")
    except Exception as e:
        print(f"✗ Error running RLDK checks: {e}")
    
    # Generate final report
    if not run_script("06_make_report.py"):
        print("Report generation failed.")
        return
    
    print("\n" + "="*60)
    print("RLHF PIPELINE COMPLETED!")
    print("="*60)
    print("All training runs and validation checks completed.")
    print("Check ./rldk_demos/report.md for the final report.")

if __name__ == "__main__":
    main()