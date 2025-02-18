import unittest
import numpy as np
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo import evaluate_polygon_visibility as evaluate_visibility_serial
from demo_parallel import evaluate_polygon_visibility as evaluate_visibility_parallel
from data_types import VisSample

class TestVisibility(unittest.TestCase):
    def setUp(self):
        # test instance directory
        dataset_dir = 'dataset/development/'
        instance_name = 'large/rand-800-7.pol'
        instance = os.path.join(dataset_dir, instance_name)
        self.sample = VisSample.read_samples(path=instance, sol_sample=1)[0]

    def test_single_point(self):
        solution = np.array([0])
        result_serial = evaluate_visibility_serial(self.sample, solution)
        result_parallel = evaluate_visibility_parallel(self.sample, solution)
        self.assertAlmostEqual(result_serial[-1], result_parallel[-1], places=5)  # Compare coverage
        self.assertEqual(len(result_serial[2]), len(result_parallel[2]))  # Compare number of predicted guards

    def test_half_points(self):
        solution = np.arange(len(self.sample.points[0]) // 2)
        result_serial = evaluate_visibility_serial(self.sample, solution)
        result_parallel = evaluate_visibility_parallel(self.sample, solution)
        self.assertAlmostEqual(result_serial[-1], result_parallel[-1], places=5)  # Compare coverage
        self.assertEqual(len(result_serial[2]), len(result_parallel[2]))  # Compare number of predicted guards

    def test_all_points(self):
        solution = np.arange(len(self.sample.points[0]))
        result_serial = evaluate_visibility_serial(self.sample, solution)
        result_parallel = evaluate_visibility_parallel(self.sample, solution)
        self.assertAlmostEqual(result_serial[-1], result_parallel[-1], places=5)  # Compare coverage
        self.assertEqual(len(result_serial[2]), len(result_parallel[2]))  # Compare number of predicted guards

if __name__ == '__main__':
    unittest.main(verbosity=2)
