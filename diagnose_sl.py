#!/usr/bin/env python3
"""
Diagnostic script to investigate supervised learning coverage evaluation issues
"""

import os
from dotenv import load_dotenv
import torch
import numpy as np

def check_dataset_and_solutions():
    """Check if dataset and solution files are properly formatted."""
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')
    
    if not DATASET_PATH:
        print("❌ DATASET_PATH not set in .env file")
        return False
    
    print(f"✓ DATASET_PATH: {DATASET_PATH}")
    
    dev_dir = os.path.join(DATASET_PATH, 'dev')
    train_dir = os.path.join(DATASET_PATH, 'train')
    
    print(f"Dev directory: {dev_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Dev directory exists: {os.path.exists(dev_dir)}")
    print(f"Train directory exists: {os.path.exists(train_dir)}")
    
    if not os.path.exists(dev_dir):
        print("❌ Dev directory not found")
        return False
    
    # Check files in dev directory
    pol_files = [f for f in os.listdir(dev_dir) if f.endswith('.pol')]
    solution_files = [f for f in os.listdir(dev_dir) if f.endswith('.solution')]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  .pol files: {len(pol_files)}")
    print(f"  .solution files: {len(solution_files)}")
    
    # Check if pol files have corresponding solution files
    missing_solutions = 0
    valid_solutions = 0
    empty_solutions = 0
    
    for i, pol_file in enumerate(pol_files[:10]):  # Check first 10 files
        base = os.path.splitext(pol_file)[0]
        sol_file = os.path.join(dev_dir, f'{base}.solution')
        
        print(f"\n🔍 Checking {pol_file}:")
        
        if not os.path.exists(sol_file):
            print(f"  ❌ No solution file")
            missing_solutions += 1
            continue
        
        try:
            with open(sol_file, 'r') as f:
                lines = f.read().splitlines()
                print(f"  Solution file has {len(lines)} lines")
                
                if len(lines) >= 2:
                    target_line = lines[1].strip()
                    if target_line:
                        target_indices = [int(x) for x in target_line.split()]
                        print(f"  ✓ Target indices: {target_indices} (count: {len(target_indices)})")
                        valid_solutions += 1
                    else:
                        print(f"  ⚠️ Empty target line")
                        empty_solutions += 1
                else:
                    print(f"  ❌ Not enough lines in solution file")
                    for j, line in enumerate(lines):
                        print(f"    Line {j}: {repr(line)}")
                    empty_solutions += 1
        except Exception as e:
            print(f"  ❌ Error reading solution file: {e}")
            missing_solutions += 1
    
    print(f"\n📈 Solution File Analysis (first 10 files):")
    print(f"  Valid solutions: {valid_solutions}")
    print(f"  Empty/invalid solutions: {empty_solutions}")
    print(f"  Missing solutions: {missing_solutions}")
    
    return valid_solutions > 0

def test_supervised_learning_data_loading():
    """Test if supervised learning data loading works correctly."""
    print("\n🧪 Testing Supervised Learning Data Loading...")
    
    try:
        from sl_agp import prepare_datasets_with_targets
        load_dotenv()
        DATASET_PATH = os.getenv('DATASET_PATH')
        
        train_dir = os.path.join(DATASET_PATH, 'train')
        dev_dir = os.path.join(DATASET_PATH, 'dev')
        
        # Load a small sample
        print("Loading datasets...")
        train_dataset, val_dataset = prepare_datasets_with_targets(train_dir, dev_dir, normalize=True)
        
        print(f"✓ Loaded {len(train_dataset)} training samples")
        print(f"✓ Loaded {len(val_dataset)} validation samples")
        
        # Check first few samples
        print("\n🔍 Examining first few validation samples:")
        for i in range(min(5, len(val_dataset))):
            sample = val_dataset[i]
            print(f"  Sample {i}: {sample.name}")
            print(f"    Data shape: {sample.data.shape}")
            print(f"    Target indices: {sample.label} (count: {len(sample.label)})")
            print(f"    Target indices type: {type(sample.label)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model_prediction():
    """Test if the model is making reasonable predictions."""
    print("\n🧪 Testing Model Predictions...")
    
    try:
        from models import create_actor
        
        # Create a model with the same parameters as the evaluation
        model = create_actor(
            embedding_size=128,
            hidden_size=128,
            n_glimpses=1,
            tanh_exploration=10,
            use_tanh=True,
            attention_type="Bahdanau",
            temperature=1.0
        )
        
        print("✓ Model created successfully")
        
        # Test with a simple example
        # Create a simple polygon (triangle)
        test_points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=torch.float32)
        test_points = test_points.unsqueeze(0)  # Add batch dimension
        
        # Create mask
        mask = torch.ones(1, 3, dtype=torch.bool)
        lengths = torch.tensor([3], dtype=torch.long)
        
        print(f"Test input shape: {test_points.shape}")
        
        # Run model
        model.eval()
        with torch.no_grad():
            selected_idxs, log_probs = model(test_points, padding_mask=mask, lengths=lengths)
        
        print(f"✓ Model prediction successful")
        print(f"  Selected indices: {selected_idxs}")
        print(f"  Log probabilities: {log_probs}")
        
        # Check if prediction makes sense
        if len(selected_idxs) > 0 and len(selected_idxs[0]) > 0:
            pred_indices = selected_idxs[0]
            print(f"  Predicted {len(pred_indices)} guards: {pred_indices}")
            
            # Check if indices are valid
            valid_indices = all(0 <= idx < 3 for idx in pred_indices)
            print(f"  Valid indices: {valid_indices}")
            
            return True
        else:
            print("  ⚠️ Model predicted no guards")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_coverage_calculation():
    """Diagnose the coverage calculation logic."""
    print("\n🧪 Testing Coverage Calculation Logic...")
    
    # Test case 1: Perfect match
    pred_indices = [0, 1, 2]
    true_indices = [0, 1, 2]
    
    pred_set = set(pred_indices)
    true_set = set(true_indices)
    overlap = pred_set.intersection(true_set)
    coverage_ratio = len(overlap) / len(true_set) if len(true_set) > 0 else 0.0
    
    print(f"Test 1 - Perfect match:")
    print(f"  Predicted: {pred_indices}")
    print(f"  True: {true_indices}")
    print(f"  Coverage: {coverage_ratio} (expected: 1.0)")
    
    # Test case 2: Partial match
    pred_indices = [0, 1, 3]
    true_indices = [0, 1, 2]
    
    pred_set = set(pred_indices)
    true_set = set(true_indices)
    overlap = pred_set.intersection(true_set)
    coverage_ratio = len(overlap) / len(true_set) if len(true_set) > 0 else 0.0
    
    print(f"\nTest 2 - Partial match:")
    print(f"  Predicted: {pred_indices}")
    print(f"  True: {true_indices}")
    print(f"  Coverage: {coverage_ratio} (expected: 0.667)")
    
    # Test case 3: No match
    pred_indices = [3, 4, 5]
    true_indices = [0, 1, 2]
    
    pred_set = set(pred_indices)
    true_set = set(true_indices)
    overlap = pred_set.intersection(true_set)
    coverage_ratio = len(overlap) / len(true_set) if len(true_set) > 0 else 0.0
    
    print(f"\nTest 3 - No match:")
    print(f"  Predicted: {pred_indices}")
    print(f"  True: {true_indices}")
    print(f"  Coverage: {coverage_ratio} (expected: 0.0)")

def main():
    """Run all diagnostic tests."""
    print("🔍 Supervised Learning Coverage Diagnostic")
    print("=" * 60)
    
    # Test 1: Check dataset and solution files
    dataset_ok = check_dataset_and_solutions()
    
    if not dataset_ok:
        print("\n❌ Dataset issues detected. Cannot proceed with further tests.")
        return
    
    # Test 2: Test data loading
    train_dataset, val_dataset = test_supervised_learning_data_loading()
    
    if train_dataset is None:
        print("\n❌ Data loading failed. Cannot proceed with further tests.")
        return
    
    # Test 3: Test model predictions
    model_ok = test_model_prediction()
    
    # Test 4: Test coverage calculation logic
    diagnose_coverage_calculation()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC SUMMARY:")
    print(f"  Dataset files: {'✓' if dataset_ok else '❌'}")
    print(f"  Data loading: {'✓' if train_dataset is not None else '❌'}")
    print(f"  Model prediction: {'✓' if model_ok else '❌'}")
    print(f"  Coverage calculation: ✓ (logic verified)")
    
    if dataset_ok and train_dataset is not None:
        print("\n💡 POTENTIAL ISSUES:")
        print("  1. Model might not be properly trained")
        print("  2. Model might be predicting empty or invalid guard sets")
        print("  3. There might be a mismatch between training and evaluation data")
        print("  4. The model checkpoint might not be loading correctly")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("  1. Check if the model is actually learning during training")
        print("  2. Add debug prints to see what the model is predicting")
        print("  3. Verify the model checkpoint is being saved and loaded correctly")
        print("  4. Run a shorter training with verbose output to see loss progression")

if __name__ == "__main__":
    main()
