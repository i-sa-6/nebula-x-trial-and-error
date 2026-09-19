"""
Official Submission Validator (simulating judge_leaderboard.py).
Verifies that predictions.zip and rail_predictions.csv strictly comply
with all competition rules and format specifications.
"""
import os
import sys
import zipfile
import pandas as pd

from src.config import SUBMISSION_DIR, CLASSES

def validate_submission(zip_path=None):
    if zip_path is None:
        zip_path = os.path.join(SUBMISSION_DIR, "predictions.zip")
        
    print("=" * 65)
    print(" 🔍 NEBULA-X SUBMISSION INTEGRITY VALIDATOR")
    print("=" * 65)
    
    # 1. Check zip exists
    if not os.path.exists(zip_path):
        print(f"❌ ERROR: Zip archive not found at {zip_path}")
        return False
    print(f"✅ Found predictions archive: {os.path.basename(zip_path)} ({os.path.getsize(zip_path)} bytes)")
    
    # 2. Check zip contents
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        print(f"[*] Files inside zip: {namelist}")
        
        # Must contain rail_predictions.csv directly at root
        if "rail_predictions.csv" not in namelist:
            print("❌ ERROR: 'rail_predictions.csv' NOT found at the root level of predictions.zip!")
            return False
            
        # Check for invalid subfolders
        for n in namelist:
            if "/" in n or "\\" in n:
                print(f"⚠️ WARNING: Subfolder detected in zip: {n}. Deliverables spec requires files directly at root.")
                
        # Read the file
        with z.open("rail_predictions.csv") as f:
            df = pd.read_csv(f)
            
    print("✅ rail_predictions.csv successfully extracted from root of zip.")
    
    # 3. Check columns
    required_cols = ["file_id", "prediction"]
    if list(df.columns) != required_cols:
        print(f"❌ ERROR: Invalid columns {list(df.columns)}. Expected exactly {required_cols}")
        return False
    print(f"✅ Column headers match required schema: {list(df.columns)}")
    
    # 4. Check row count
    expected_rows = 68
    if len(df) != expected_rows:
        print(f"⚠️ WARNING: Found {len(df)} rows. Expected {expected_rows} test rows (Test1.csv to Test68.csv).")
    else:
        print(f"✅ Row count matches exactly: {len(df)} test files.")
        
    # 5. Check permitted classes
    invalid_labels = df[~df['prediction'].isin(CLASSES)]
    if len(invalid_labels) > 0:
        print(f"❌ ERROR: Found invalid predictions: {invalid_labels['prediction'].unique()}. Permitted: {CLASSES}")
        return False
    print(f"✅ All predictions belong to valid classes: {CLASSES}")
    
    # 6. Check nulls
    if df.isnull().values.any():
        print("❌ ERROR: Missing or NaN values detected in predictions!")
        return False
    print("✅ Zero NaN or null values detected.")
    
    # 7. Distribution Breakdown
    print("\n" + "-" * 40)
    print("📊 SUBMITTED TEST PREDICTIONS DISTRIBUTION:")
    print("-" * 40)
    for cname, count in df['prediction'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  {cname:<12}: {count:>2} files ({pct:>5.1f}%)")
    print("-" * 40)
    
    print("\n[🎉] SUBMISSION VALIDATION PASSED! Ready for judge_leaderboard.py scoring.\n")
    return True

if __name__ == "__main__":
    validate_submission()
