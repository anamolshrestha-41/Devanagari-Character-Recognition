#!/usr/bin/env python3
"""
System verification script for Devanagari Character Classification
"""

import os
import sys
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[FAIL] {description}: {filepath} (NOT FOUND)")
        return False

def check_directory_structure():
    """Verify the project directory structure"""
    print("=== Directory Structure Check ===")
    
    base_dir = Path(__file__).parent
    
    # Essential files
    files_to_check = [
        (base_dir / "api" / "app.py", "FastAPI application"),
        (base_dir / "api" / "requirements.txt", "API requirements"),
        (base_dir / "utils" / "data_loader.py", "Data loader utility"),
        (base_dir / "models" / "devanagari_model.h5", "Trained model"),
        (base_dir / "models" / "class_names.json", "Class names mapping"),
        (base_dir / "notebooks" / "devanagari_classification.ipynb", "Training notebook"),
        (base_dir / "test_api.py", "API test script")
    ]
    
    all_good = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_good = False
    
    return all_good

def check_model_and_classes():
    """Check if model and class names are compatible"""
    print("\n=== Model Compatibility Check ===")
    
    try:
        # Check class names
        with open("models/class_names.json", "r", encoding="utf-8") as f:
            class_names = json.load(f)
        
        print(f"[OK] Class names loaded: {len(class_names)} classes")
        print(f"  Sample classes: {class_names[:5]}...")
        
        return True
    except Exception as e:
        print(f"[FAIL] Error loading class names: {e}")
        return False

def check_dataset():
    """Check if dataset is properly structured"""
    print("\n=== Dataset Structure Check ===")
    
    dataset_path = Path("data/archive/nhcd/nhcd")
    
    if not dataset_path.exists():
        print(f"[FAIL] Dataset not found at: {dataset_path}")
        return False
    
    categories = ["consonants", "numerals", "vowels"]
    all_good = True
    
    for category in categories:
        category_path = dataset_path / category
        if category_path.exists():
            subfolders = [d for d in category_path.iterdir() if d.is_dir()]
            print(f"[OK] {category}: {len(subfolders)} subfolders")
        else:
            print(f"[FAIL] {category}: not found")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("Devanagari Character Classification System Verification")
    print("=" * 60)
    
    checks = [
        check_directory_structure(),
        check_model_and_classes(),
        check_dataset()
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("[SUCCESS] All checks passed! System appears to be working correctly.")
        print("\nNext steps:")
        print("1. Start the API: uvicorn api.app:app --reload")
        print("2. Test the API: python test_api.py")
    else:
        print("[ERROR] Some checks failed. Please review the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())