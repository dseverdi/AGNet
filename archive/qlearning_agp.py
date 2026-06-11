"""
Q-Learning approach for Art Gallery Problem (AGP) adapted from:
"Reinforcement Learning for Optimize Coverage in Art Gallery Problem Using Q‑Learning Based in Grid World"

Key adaptations:
- State: Number of guards placed (0 to num_vertices)
- Actions: Simple toggle actions (0 to num_vertices-1) to toggle guard at specific vertex
- Reward: Total coverage after action (not incremental)
- Visibility region caching for efficient union/difference operations
"""

import numpy as np
import random
import time
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# CGAL imports
import skgeom

# Import utility functions
from utils import evaluate_polygon_visibility_numpy_wo_gt, compute_visibility, merge_polygon_sets, createPolygon


class QLearningAGP:
    def __init__(self, polygon_coords, instance_name=None, num_holes=2,
                 learning_rate=0.1, epsilon=0.3, gamma=0.9, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-learning agent for Art Gallery Problem with visibility region caching.
        
        Args:
            polygon_coords: List of (x, y) coordinates for polygon vertices
            instance_name: Name identifier for the polygon instance
            num_holes: Number of holes in the polygon (used for naming)
            learning_rate: Q-learning learning rate (alpha)
            epsilon: Initial exploration rate
            gamma: Discount factor for future rewards
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.polygon_coords = polygon_coords
        self.instance_name = instance_name or "unknown"
        self.num_holes = num_holes
        self.n_vertices = len(polygon_coords)
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: key = (num_guards), value = dict of {action: q_value}
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Track best solution found
        self.best_solution = []
        self.best_coverage = 0.0
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'episode_rewards': [],
            'best_coverage_history': [],
            'best_solution_history': [],
            'training_time': 0.0
        }
        
        # Set up visibility computation infrastructure
        self._setup_visibility_computation()
        
        # Visibility region cache: guard_index -> PolygonSet
        self.guard_visibility_cache = {}
        
        # Coverage cache for complete solutions: tuple(sorted_guards) -> coverage
        self.coverage_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Precompute visibility regions for all guards
        self._precompute_guard_visibility_regions()
    
    def _setup_visibility_computation(self):
        """Set up CGAL structures for visibility computation."""
        eps = 1e-8
        
        # Create polygon
        self.poly = createPolygon(self.polygon_coords)
        if self.poly is None or self.poly is False:
            print(f"Warning: Invalid polygon in {self.instance_name}")
            self.arrangement_ok = False
            return
        
        # Build arrangement
        self.arr = skgeom.arrangement.Arrangement()
        self.arrangement_ok = True
        
        try:
            for edge in self.poly.edges:
                self.arr.insert(edge)
            self.vs = skgeom.TriangularExpansionVisibility(self.arr)
            self.edges = list(self.poly.edges)
            self.eps = eps
            self.poly_area = abs(float(self.poly.area()))
        except RuntimeError as e:
            print(f"Warning: CGAL arrangement failed for {self.instance_name}: {e}")
            self.arrangement_ok = False
    
    def _precompute_guard_visibility_regions(self):
        """Precompute visibility regions for all possible guard positions."""
        if not self.arrangement_ok:
            return
        
        print(f"Precomputing visibility regions for {self.n_vertices} guard positions...")
        start_time = time.time()
        
        # Compute visibility regions for all vertices in parallel
        with ThreadPoolExecutor() as executor:
            futures = {
                guard_idx: executor.submit(
                    compute_visibility, 
                    self.vs, self.arr, self.poly, self.eps, self.edges, guard_idx
                )
                for guard_idx in range(self.n_vertices)
            }
            
            for guard_idx, future in futures.items():
                vis_poly, err_idx, err_q = future.result()
                if vis_poly:
                    # Store as PolygonSet for easy union operations
                    self.guard_visibility_cache[guard_idx] = skgeom.PolygonSet([vis_poly])
                else:
                    print(f"Warning: Failed to compute visibility for guard {guard_idx}")
                    # Empty visibility region
                    self.guard_visibility_cache[guard_idx] = skgeom.PolygonSet()
        
        computation_time = time.time() - start_time
        print(f"Precomputed visibility regions in {computation_time:.2f}s")
    
    def get_coverage_fast(self, solution):
        """
        Get coverage for a solution using cached visibility regions.
        
        Args:
            solution: List of guard positions (vertex indices)
            
        Returns:
            coverage: Coverage percentage (0.0 to 1.0)
        """
        if not self.arrangement_ok:
            return 0.0
        
        # Create cache key (sorted solution for consistency)
        solution_key = tuple(sorted(solution))
        
        if solution_key in self.coverage_cache:
            self.cache_hits += 1
            return self.coverage_cache[solution_key]
        
        self.cache_misses += 1
        
        # Compute coverage using cached visibility regions
        if len(solution) == 0:
            coverage = 0.0
        else:
            try:
                # Union all visibility regions for guards in solution
                union_region = skgeom.PolygonSet()
                for guard_idx in solution:
                    if guard_idx in self.guard_visibility_cache:
                        union_region = union_region.union(self.guard_visibility_cache[guard_idx])
                
                # Calculate total visible area
                total_area = 0.0
                for vis in union_region.polygons:
                    outer = abs(float(vis.outer_boundary().area()))
                    holes = sum(abs(float(h.area())) for h in vis.holes)
                    total_area += outer - holes
                
                coverage = total_area / self.poly_area if self.poly_area > 0 else 0.0
                
            except Exception as e:
                print(f"Fast coverage computation failed: {e}")
                coverage = 0.0
        
        # Cache the result
        self.coverage_cache[solution_key] = coverage
        return coverage
    
    def get_coverage_incremental(self, current_solution, action, action_type='toggle'):
        """
        Get coverage incrementally by adding/removing a single guard.
        
        Args:
            current_solution: Current list of guard positions
            action: Guard index to add/remove
            action_type: 'add', 'remove', or 'toggle'
            
        Returns:
            new_coverage: Coverage after applying the action
            new_solution: Updated solution after the action
        """
        current_solution = list(current_solution)
        
        if action_type == 'toggle':
            if action in current_solution:
                action_type = 'remove'
            else:
                action_type = 'add'
        
        # Apply action
        if action_type == 'add' and action not in current_solution:
            new_solution = current_solution + [action]
        elif action_type == 'remove' and action in current_solution:
            new_solution = [g for g in current_solution if g != action]
        else:
            # No change
            new_solution = current_solution
        
        # Get coverage for new solution
        new_coverage = self.get_coverage_fast(new_solution)
        return new_coverage, new_solution
    
    def train_episode(self, max_steps=20, target_coverage=0.95):
        """
        Train a single episode using Q-learning.
        
        Args:
            max_steps: Maximum steps per episode
            target_coverage: Early stopping if this coverage is achieved
            
        Returns:
            episode_reward: Total reward for this episode
            steps_taken: Number of steps taken
        """
        # Initialize state (empty solution)
        current_solution = []
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Current state
            state = len(current_solution)
            
            # Choose action (epsilon-greedy)
            if random.random() < self.epsilon:
                # Explore: random action
                action = random.randint(0, self.n_vertices - 1)
            else:
                # Exploit: best Q-value action
                q_values = self.q_table[state]
                if q_values:
                    action = max(q_values.keys(), key=lambda a: q_values[a])
                else:
                    action = random.randint(0, self.n_vertices - 1)
            
            # Take action and observe reward
            new_coverage, new_solution = self.get_coverage_incremental(
                current_solution, action, 'toggle'
            )
            
            reward = new_coverage  # Reward is coverage percentage
            episode_reward += reward
            
            # Update best solution if better
            if new_coverage > self.best_coverage:
                self.best_coverage = new_coverage
                self.best_solution = new_solution.copy()
            
            # Next state
            next_state = len(new_solution)
            
            # Q-learning update
            current_q = self.q_table[state][action]
            
            # Find max Q-value for next state
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            
            # Update Q-value
            target = reward + self.gamma * max_next_q
            self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
            
            # Move to next state
            current_solution = new_solution
            
            # Early stopping if target coverage achieved
            if new_coverage >= target_coverage:
                break
        
        return episode_reward, step + 1
    
    def train(self, max_episodes=100, max_steps_per_episode=20, verbose=False, 
              target_coverage=0.90, patience=30):
        """
        Train Q-learning agent for specified number of episodes.
        
        Args:
            max_episodes: Maximum number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print training progress
            target_coverage: Stop early if this coverage is achieved consistently
            patience: Stop if no improvement for this many episodes
            
        Returns:
            training_stats: Dictionary with training statistics
        """
        start_time = time.time()
        best_coverage_streak = 0
        episodes_without_improvement = 0
        last_best_coverage = 0.0
        
        for episode in range(max_episodes):
            episode_reward, steps = self.train_episode(max_steps_per_episode, target_coverage)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Track statistics
            self.training_stats['episodes'] += 1
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['best_coverage_history'].append(self.best_coverage)
            self.training_stats['best_solution_history'].append(self.best_solution.copy())
            
            # Check for improvement
            if self.best_coverage > last_best_coverage:
                last_best_coverage = self.best_coverage
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1
            
            # Early stopping conditions
            if self.best_coverage >= target_coverage:
                best_coverage_streak += 1
                if best_coverage_streak >= 10:  # Consistent good performance
                    if verbose:
                        print(f"Early stop: consistent coverage >= {target_coverage:.2f}")
                    break
            else:
                best_coverage_streak = 0
                
            if episodes_without_improvement >= patience:
                if verbose:
                    print(f"Early stop: no improvement for {patience} episodes")
                break
            
            # Print progress
            if verbose and (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{max_episodes}: "
                      f"Best Coverage: {self.best_coverage:.4f}, "
                      f"Best Solution Size: {len(self.best_solution)}, "
                      f"Cache hits/misses: {self.cache_hits}/{self.cache_misses}, "
                      f"Epsilon: {self.epsilon:.4f}")
                      
        training_time = time.time() - start_time
        self.training_stats['training_time'] = training_time
        
        if verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Episodes run: {episode + 1}/{max_episodes}")
            print(f"Best coverage: {self.best_coverage:.4f}")
            print(f"Best solution: {self.best_solution} (size: {len(self.best_solution)})")
            print(f"Cache performance: {self.cache_hits} hits, {self.cache_misses} misses")
            print(f"Cache hit rate: {self.cache_hits/(self.cache_hits + self.cache_misses)*100:.1f}%")
            
        return self.training_stats
    
    def get_best_solution(self):
        """Get the best solution found during training."""
        return self.best_solution, self.best_coverage
    
    def get_q_table_stats(self):
        """Get statistics about the Q-table."""
        total_entries = sum(len(actions) for actions in self.q_table.values())
        states_explored = len(self.q_table)
        return {
            'total_q_entries': total_entries,
            'states_explored': states_explored,
            'max_state': max(self.q_table.keys()) if self.q_table else 0
        }


def evaluate_qlearning_single_instance(polygon_coords, instance_name, **qlearning_params):
    """
    Evaluate Q-learning approach on a single polygon instance.
    
    Args:
        polygon_coords: Array of polygon vertices
        instance_name: Name of the polygon instance
        **qlearning_params: Q-learning hyperparameters
        
    Returns:
        result: Dictionary with evaluation results
    """
    # Default hyperparameters optimized for performance
    default_params = {
        'learning_rate': 0.1,
        'epsilon': 0.3,
        'gamma': 0.9,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'max_episodes': 50,  # Reduced for faster evaluation
        'max_steps_per_episode': 15,  # Reduced for faster evaluation
        'target_coverage': 0.90,  # Stop early at 90% coverage
        'patience': 20,  # Reduced patience
        'verbose': False
    }
    default_params.update(qlearning_params)
    
    # Create and train Q-learning agent
    agent = QLearningAGP(
        polygon_coords=polygon_coords,
        instance_name=instance_name,
        learning_rate=default_params['learning_rate'],
        epsilon=default_params['epsilon'],
        gamma=default_params['gamma'],
        epsilon_decay=default_params['epsilon_decay'],
        epsilon_min=default_params['epsilon_min']
    )
    
    start_time = time.time()
    training_stats = agent.train(
        max_episodes=default_params['max_episodes'],
        max_steps_per_episode=default_params['max_steps_per_episode'],
        target_coverage=default_params['target_coverage'],
        patience=default_params['patience'],
        verbose=default_params['verbose']
    )
    training_time = time.time() - start_time
    
    # Get best solution
    best_solution, best_coverage = agent.get_best_solution()
    q_stats = agent.get_q_table_stats()
    
    return {
        'best_solution': best_solution,
        'best_coverage': best_coverage,
        'best_reward': best_coverage,  # Same as coverage for this approach
        'best_size': len(best_solution),
        'size_ratio': len(best_solution) / max(1, len(polygon_coords)),
        'training_time': training_time,
        'training_stats': training_stats,
        'q_table_stats': q_stats,
        'cache_stats': {
            'hits': agent.cache_hits,
            'misses': agent.cache_misses,
            'hit_rate': agent.cache_hits / (agent.cache_hits + agent.cache_misses) if (agent.cache_hits + agent.cache_misses) > 0 else 0.0
        },
        'hyperparameters': {k: v for k, v in default_params.items() if k != 'verbose'}
    }


if __name__ == "__main__":
    # Simple test
    import os
    import sys
    sys.path.append('.')
    from dataset import agp_read_samples
    
    # Load a test polygon
    dataset_path = os.getenv("DATASET_PATH", "/home/dseverdi/Radno/MLAG/dataset/AGPIL")
    test_files = [os.path.join(dataset_path, "dev", "rand-124-93.pol")]
    
    if os.path.exists(test_files[0]):
        samples = agp_read_samples(test_files, normalize=True)
        sample = samples[0]
        polygon_coords = sample.data
        instance_name = sample.name
        
        print(f"Testing optimized Q-learning on {instance_name}")
        print(f"Polygon vertices: {len(polygon_coords)}")
        
        result = evaluate_qlearning_single_instance(
            polygon_coords, instance_name,
            max_episodes=100, verbose=False
        )
        
        print(f"\nResults:")
        print(f"Best coverage: {result['best_coverage']:.4f}")
        print(f"Best solution size: {result['best_size']}")
        print(f"Training time: {result['training_time']:.2f}s")
        print(f"Cache hit rate: {result['cache_stats']['hit_rate']*100:.1f}%")
    else:
        print(f"Test file not found: {test_files[0]}")
