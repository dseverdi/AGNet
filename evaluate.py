import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import TSPDataset
from utils import visualize_tour, USE_CUDA

def evaluate_model(model, dataset_size, num_nodes, batch_size=128, name="TSP", visualize=True, max_visualize_size=100):
    """
    Evaluate a model on a test dataset and visualize results
    
    Args:
        model: The model to evaluate
        dataset_size: Size of the test dataset
        num_nodes: Number of cities in each TSP instance
        batch_size: Batch size for evaluation
        name: Name prefix for output files
        visualize: Whether to generate visualizations
        max_visualize_size: Maximum dataset size for which to generate visualizations
    """
    print(f"Evaluating {name} model on {dataset_size} samples with {num_nodes} cities...")
    
    # Create a test dataset
    test_dataset = TSPDataset(num_nodes, dataset_size, random_seed=123)  # Different seed for test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Statistics to track
    tour_lengths = []
    
    # Set model to evaluation mode
    model.eval()
    
    # Determine if we should visualize based on dataset size
    should_visualize = visualize and dataset_size <= max_visualize_size
    if visualize and not should_visualize:
        print(f"Skipping visualization because dataset size ({dataset_size}) exceeds max_visualize_size ({max_visualize_size})")
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch_id, batch in enumerate(test_loader):
            inputs = batch
            if USE_CUDA:
                inputs = inputs.cuda()
            
            # Get predictions
            R, _, actions, action_idxs = model(inputs)
            tour_lengths.extend(R.cpu().numpy())
            
            # Visualize the first tour in the first batch, but only if requested and dataset isn't too large
            if batch_id == 0 and should_visualize:
                visualize_tour(inputs[0], action_idxs, name=f"{name}-{num_nodes}")
    
    # Calculate statistics
    mean_length = np.mean(tour_lengths)
    std_length = np.std(tour_lengths)
    min_length = np.min(tour_lengths)
    max_length = np.max(tour_lengths)
    
    # Output results
    print(f"\nResults for {name} with {num_nodes} cities:")
    print(f"Mean tour length: {mean_length:.4f}")
    print(f"Std tour length: {std_length:.4f}")
    print(f"Min tour length: {min_length:.4f}")
    print(f"Max tour length: {max_length:.4f}")
    
    return mean_length, std_length, min_length, max_length
