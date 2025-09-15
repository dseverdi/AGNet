# Pareto Training for AGP

This script implements Pareto-optimal training for the Art Gallery Problem (AGP), exploring the trade-off between polygon coverage and solution size.

## Concept

The training uses a linear reward function that balances two objectives:
```
Reward = -α(1-coverage) - (1-α) * rel_length
```

Where:
- `α` controls the trade-off (1.0 = pure coverage focus, 0.0 = pure size focus)
- `coverage` is the fraction of polygon area covered by guards
- `rel_length = n_guards / total_vertices` is the relative solution size

## Training Process

1. **Progressive Alpha Schedule**: Start with α=1.0 and gradually decrease to α=0.0
2. **Shared Weights**: Same model continues training across all alpha values
3. **Fixed Epochs**: Train for a specified number of epochs at each alpha value
4. **Pareto Frontier**: Model learns to balance both objectives

## Usage

### Basic Usage (Using Defaults)
```bash
# Uses default dataset paths and standard parameters
python train_pareto.py

# Quick test with fewer iterations
python train_pareto.py --epochs-per-alpha 5 --alpha-steps 6 --max-instances 10
```

### Custom Dataset Paths
```bash
python train_pareto.py \
    --agp-train-dir /path/to/train/pol/files \
    --agp-val-dir /path/to/val/pol/and/solution/files \
    --epochs-per-alpha 10 \
    --alpha-steps 11
```

### Example with Custom Parameters
```bash
python train_pareto.py \
    --agp-train-dir /home/dseverdi/Radno/MLAG/dataset/AGPIL/train \
    --agp-val-dir /home/dseverdi/Radno/MLAG/dataset/AGPIL/dev \
    --epochs-per-alpha 5 \
    --alpha-steps 6 \
    --embedding-size 128 \
    --hidden-size 128 \
    --batch-size 1 \
    --lr 1e-3 \
    --max-instances 50 \
    --normalize
```

## Key Arguments

- `--epochs-per-alpha`: Number of epochs to train at each alpha value (default: 10)
- `--alpha-steps`: Number of alpha values from start to end (default: 11)
- `--alpha-start`: Starting alpha value (default: 1.0, coverage focus)
- `--alpha-end`: Ending alpha value (default: 0.0, size focus)
- `--lr-decay`: Learning rate decay per alpha phase (default: 1.0, no decay)
- `--max-instances`: Limit number of training instances for quick testing
- `--eval-during-training`: Evaluate at each alpha step (default: enabled)
- `--final-evaluation`: Comprehensive Pareto frontier evaluation (default: enabled)

## Output

The script produces:
1. **Checkpoints**: Saved at key alpha values (α=1.0, 0.5, 0.0) and final model
2. **Results**: Detailed evaluation results in `results/pareto_training_results.json`
3. **Pareto Frontier**: Evaluation across multiple alpha values showing trade-offs

## Model Usage

After training, you can use the final model with different alpha values at inference time:
```python
# Load the trained model
checkpoint = torch.load('checkpoints/pareto_model_final_epochs110.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Use with different objectives:
# α=1.0 for maximum coverage
# α=0.5 for balanced coverage-size trade-off  
# α=0.0 for minimum solution size
```

## Evaluation

The script automatically evaluates the model across the Pareto frontier, showing:
- Coverage statistics for each alpha value
- Solution size ratios compared to optimal solutions
- Trade-off curves between objectives

Check `results/pareto_training_results.json` for detailed metrics and Pareto frontier analysis.
