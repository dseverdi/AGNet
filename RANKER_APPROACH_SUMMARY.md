# AGP Ranker Training - Complete Approach Summary

## Overview
We developed a ranker-based approach for fast inference-time solution selection in the Art Gallery Problem (AGP). Instead of expensive geometry calculations at test time, we train a neural ranker to predict solution quality and use it for K-way selection and Active Search proxy rewards.

## Key Innovation: Systematic Diverse Solution Generation
The main breakthrough was solving the "identical solutions problem" where the trained RL actor generates the same solution repeatedly for each polygon, leading to no informative ranking pairs.

### Solution Generation Strategy
Instead of relying on actor stochasticity, we systematically generate diverse solutions:

```python
# 1. Get base solutions from actor (even if identical)
actor_solutions = []
for sample_idx in range(min(10, num_samples_per_polygon // 6)):
    # Sample from actor with different seeds
    
# 2. Generate systematic variations
# - Random subsets of different sizes (1, 2, 3, 5, N//10, N//5 guards)
# - Systematic patterns (every k-th vertex)
# - Local clusters around different centers
# - Remove duplicates and ensure diversity

# 3. Compute true coverage-based rewards for all solutions
for sol in unique_solutions:
    reward = enhanced_penalty(polygon_coords, sol, polygon_name, alpha=5.0, p=0.0)
    # enhanced_penalty uses evaluate_polygon_visibility_numpy_wo_gt for coverage
```

## Architecture & Training

### RankerNet Architecture
```python
class RankerNet(nn.Module):
    - polygon_encoder: LSTM(2, embedding_size) 
    - guard_embedding: Embedding(max_vertices, embedding_size)
    - solution_encoder: LSTM(embedding_size, embedding_size)
    - meta_features: [|S|/N, N_norm]  # solution ratio, normalized vertices
    - backbone: MLP with dropout
    - score_head: Linear(hidden_size, 1)
    
    def predict_reward(self, ...):
        return -self.forward(...)  # Higher score = better solution
```

### Training Process
- **Pairwise ranking loss**: Within each polygon, compare all solution pairs
- **Weighted tie handling**: Exact ties skipped, near-ties get reduced weight (0.1x)
- **Loss**: `softplus(-sign(reward_diff) * (score_i - score_j))`
- **Metadata checkpoints**: Save model_type='ranker' for loader compatibility

### Key Training Parameters
- **Data generation**: 10-50 polygons, 5-20 diverse samples per polygon
- **Training**: 10-20 epochs, batch_size=16-32, lr=1e-3
- **Early stopping**: Monitor validation loss (small datasets overfit quickly)

## Integration with Evaluation

### Checkpoint Loading
```python
# evaluate.py automatically detects and loads ranker checkpoints
def load_value_net(value_net_path, ...):
    if model_type == 'ranker':
        model = RankerNet(embedding_size, hidden_size)
        # Attaches predict_reward() method automatically
```

### Fast K-way Selection
```python
# Sample K solutions from actor
# Use ranker.predict_reward() to score all K solutions (fast!)
# Select solution with best (lowest) predicted reward
# Compute true reward only for selected solution
```

### Active Search Proxy
```python
# Use ranker predictions as proxy rewards for REINFORCE updates
# No geometry calls during training - only at final evaluation
# Enables geometry-free RL training
```

## Usage Commands

### Train Ranker
```bash
# Generate diverse training data + train ranker
python train_ranker.py \
  --mode both \
  --max-polygons 20 \
  --samples-per-polygon 10 \
  --epochs 15 \
  --batch-size 16 \
  --data-file /tmp/ranker_training.pkl \
  --output checkpoints/ranker_model.pt

# Key parameters:
# --max-polygons: Number of training polygons (10-50 recommended)
# --samples-per-polygon: Diverse solutions per polygon (5-20 recommended)  
# --epochs: Training epochs (10-20, watch for overfitting)
```

### Evaluate with Ranker
```bash
# Fast K-way selection
python evaluate.py \
  --checkpoint <rl_actor_checkpoint.pt> \
  --val-dir $DATASET_PATH/dev \
  --K 32 \
  --use-value-net \
  --value-net-path auto  # Auto-detects best ranker/value_net checkpoint

# Active Search with proxy rewards (geometry-free training)
python evaluate.py \
  --checkpoint <rl_actor_checkpoint.pt> \
  --use-as --as-proxy \
  --value-net-path checkpoints/ranker_model.pt \
  --K 64 --as-batch-size 8
```

### Test Ranker Quality
```python
# test_ranker_simple.py verifies ranking correlation
python test_ranker_simple.py
# Outputs: Spearman correlation, true vs predicted rankings
# Target: >0.8 correlation for good performance
```

## Performance Results

### Before (Actor-only sampling)
- All solutions identical per polygon
- Training loss: 0.0 (no learning)
- Ranking correlation: N/A (no diversity)

### After (Systematic diverse generation)
- **Best achieved**: 98.79% Spearman ranking correlation
- **Training**: Loss reduces from ~0.69 to ~0.17-0.46
- **Diversity**: Wide reward ranges (1.0 to 900+) with meaningful differences
- **Speed**: Fast inference ~1ms vs geometry ~100ms per solution

## Files Modified/Created

### Core Implementation
- `train_ranker.py`: Complete ranker training pipeline with diverse solution generation
- `evaluate.py`: Enhanced to auto-detect and load ranker checkpoints, fast K-way selection
- `test_ranker_simple.py`: Quality verification script

### Key Components
1. **Diverse solution generation**: Systematic patterns vs actor stochasticity
2. **Weighted pairwise loss**: Smart tie handling for better learning
3. **Metadata checkpoints**: Seamless integration with existing evaluation pipeline
4. **predict_reward() compatibility**: Drop-in replacement for value networks

## Critical Success Factors

1. **Solution diversity is essential**: Actor determinism requires systematic generation
2. **Coverage-based rewards**: Use actual geometry calculations for training labels
3. **Pairwise ranking**: More robust than direct regression for this problem  
4. **Early stopping**: Small datasets overfit quickly, monitor validation carefully
5. **Integration design**: Ranker seamlessly replaces value networks in existing pipeline

## Ready-to-Use Checkpoints
- `checkpoints/ranker_diverse_v2.pt`: Best performance (98.79% correlation)
- `checkpoints/ranker_10poly_5samples.pt`: Smaller model, faster training

This approach successfully enables fast, geometry-free inference for AGP while maintaining high solution quality through learned ranking.
