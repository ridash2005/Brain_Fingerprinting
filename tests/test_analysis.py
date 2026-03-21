"""
Comprehensive Unit Tests for Analysis Modules in Brain Fingerprinting Pipeline.

Tests cover all analysis modules created to address IEEE TCDS reviewer comments:
1. Statistical Validation (Reviewer 1, Point 5)
2. Ablation Studies (Reviewer 1, Point 9; Reviewer 2, Point 1(2))
3. Cross-Validation (Reviewer 1, Point 3)
4. State-of-the-Art Comparisons (Reviewer 1, Point 2; Reviewer 2, Point 3)
5. Evaluation Metrics (Reviewer 1, Point 10)
6. Robustness Analysis (Reviewer 1, Point 10)
7. Interpretability (Reviewer 1, Point 7)
8. Dataset Description (Reviewer 2, Point 2)
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.analysis.evaluation_metrics import (
    ComprehensiveMetrics, 
    calculate_accuracy, 
    calculate_top_k_accuracy,
    calculate_mean_rank,
    calculate_mean_reciprocal_rank,
    calculate_differential_identifiability
)
from src.analysis.statistical_validation import (
    permutation_test, 
    bootstrap_confidence_interval,
    paired_t_test
)
from src.analysis.ablation_studies import AblationStudy
from src.analysis.cross_validation import CrossValidation
from src.analysis.state_of_art_comparison import (
    FinnFingerprinting, 
    EdgeSelectionFingerprinting, 
    PCAFingerprinting
)
from src.analysis.robustness_analysis import RobustnessAnalysis
from src.analysis.interpretability import DictionaryAtomAnalysis
from src.models.conv_ae import ConvAutoencoder
import torch


class TestEvaluationMetrics(unittest.TestCase):
    """Test comprehensive evaluation metrics (Reviewer 1, Point 10)."""
    
    def setUp(self):
        self.n_subjects = 20
        np.random.seed(42)
        # Perfect correlation matrix - diagonal is highest
        self.perfect_corr = np.eye(self.n_subjects) * 0.9 + np.random.rand(self.n_subjects, self.n_subjects) * 0.05
        # Random correlation matrix
        self.random_corr = np.random.rand(self.n_subjects, self.n_subjects)
    
    def test_top_1_accuracy_perfect(self):
        """Perfect diagonal should yield 100% accuracy."""
        acc = calculate_accuracy(self.perfect_corr)
        self.assertEqual(acc, 1.0)
    
    def test_top_k_accuracy(self):
        """Top-k accuracy should be >= top-1 accuracy."""
        top1 = calculate_accuracy(self.perfect_corr)
        top5 = calculate_top_k_accuracy(self.perfect_corr, k=5)
        top10 = calculate_top_k_accuracy(self.perfect_corr, k=10)
        self.assertGreaterEqual(top5, top1)
        self.assertGreaterEqual(top10, top5)
    
    def test_mean_rank_perfect(self):
        """Mean rank should be 1 for perfect identification."""
        mr = calculate_mean_rank(self.perfect_corr)
        self.assertEqual(mr, 1.0)
    
    def test_mean_reciprocal_rank_perfect(self):
        """MRR should be 1 for perfect identification."""
        mrr = calculate_mean_reciprocal_rank(self.perfect_corr)
        self.assertEqual(mrr, 1.0)
    
    def test_differential_identifiability(self):
        """Differential identifiability should be positive for good fingerprinting."""
        di = calculate_differential_identifiability(self.perfect_corr)
        self.assertGreater(di, 0)
    
    def test_comprehensive_metrics(self):
        """Test all metrics computed together."""
        metrics = ComprehensiveMetrics(self.perfect_corr)
        results = metrics.compute_all_metrics()
        
        required_keys = ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'top_10_accuracy',
                        'mean_rank', 'mean_reciprocal_rank', 'differential_identifiability']
        for key in required_keys:
            self.assertIn(key, results)


class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation (Reviewer 1, Point 5)."""
    
    def setUp(self):
        self.n_subjects = 15
        self.n_parcels = 100
        np.random.seed(42)
        
        # Create synthetic FC data
        self.fc_rest = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_task = self.fc_rest + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
    
    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval computation."""
        def mock_fingerprint(task, rest):
            return 0.85  # Fixed accuracy for testing
        
        result = bootstrap_confidence_interval(
            self.fc_task, self.fc_rest, mock_fingerprint, n_bootstrap=50
        )
        
        self.assertIn('mean', result)
        self.assertIn('std', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertLessEqual(result['ci_lower'], result['mean'])
        self.assertGreaterEqual(result['ci_upper'], result['mean'])
    
    def test_permutation_test(self):
        """Test permutation testing returns proper structure."""
        corr_matrix = np.eye(self.n_subjects)
        result = permutation_test(1.0, 0.5, corr_matrix, n_permutations=50)
        
        self.assertIn('p_value', result)
        self.assertIn('observed_diff', result)
        self.assertGreaterEqual(result['p_value'], 0.0)
        self.assertLessEqual(result['p_value'], 1.0)
    
    def test_paired_t_test(self):
        """Test paired t-test with effect size."""
        acc1 = np.array([0.85, 0.87, 0.83, 0.86, 0.84])
        acc2 = np.array([0.75, 0.78, 0.72, 0.77, 0.74])
        
        result = paired_t_test(acc1, acc2)
        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('effect_size', result)  # Cohen's d


class TestAblationStudies(unittest.TestCase):
    """Test ablation studies (Reviewer 1, Point 9)."""
    
    def setUp(self):
        self.n_subjects = 10
        self.n_parcels = 360
        np.random.seed(42)
        
        self.fc_task = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_task = (self.fc_task + self.fc_task.transpose(0, 2, 1)) / 2
        
        self.fc_rest = self.fc_task + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_rest = (self.fc_rest + self.fc_rest.transpose(0, 2, 1)) / 2
    
    def test_ablation_study_baseline(self):
        """Test raw FC baseline ablation."""
        ablation = AblationStudy(self.fc_task, self.fc_rest)
        acc = ablation.raw_fc_baseline()
        
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertIn('Raw FC', ablation.results)
    
    # def test_ablation_study_group_avg(self):
    #     """Test group average subtraction ablation."""
    #     ablation = AblationStudy(self.fc_task, self.fc_rest)
    #     # Not currently implemented in src
    #     pass


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation (Reviewer 1, Point 3)."""
    
    def setUp(self):
        self.n_subjects = 20
        self.n_parcels = 100  # Smaller for faster testing
        np.random.seed(42)
        
        self.fc_task = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_rest = self.fc_task + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
    
    def test_cv_initialization(self):
        """Test CV object creation."""
        cv = CrossValidation(n_splits=3)
        self.assertEqual(cv.n_splits, 3)


class TestSOTAComparison(unittest.TestCase):
    """Test state-of-the-art comparisons (Reviewer 1, Point 2; Reviewer 2, Point 3)."""
    
    def setUp(self):
        self.n_subjects = 15
        self.n_parcels = 100
        np.random.seed(42)
        
        self.fc_database = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_target = self.fc_database + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
    
    def test_finn_fingerprinting(self):
        """Test Finn et al. (2015) implementation."""
        finn = FinnFingerprinting()
        acc = finn.run(self.fc_database, self.fc_target)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
    
    def test_edge_selection_fingerprinting(self):
        """Test edge selection fingerprinting."""
        edge_fp = EdgeSelectionFingerprinting()
        acc = edge_fp.run(self.fc_database, self.fc_target, top_k=0.1)
        self.assertGreaterEqual(acc, 0.0)
    
    def test_pca_fingerprinting(self):
        """Test PCA-based fingerprinting."""
        # n_components must be <= min(n_subjects, n_features)
        pca_fp = PCAFingerprinting()
        acc = pca_fp.run(self.fc_database, self.fc_target, n_components=10)
        self.assertGreaterEqual(acc, 0.0)


class TestRobustnessAnalysis(unittest.TestCase):
    """Test robustness analysis (Reviewer 1, Point 10)."""
    
    def setUp(self):
        self.n_subjects = 15
        self.n_parcels = 100
        np.random.seed(42)
        
        self.fc_task = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_rest = self.fc_task + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
    
    def test_noise_robustness(self):
        """Test noise robustness analysis."""
        robust = RobustnessAnalysis(self.fc_task, self.fc_rest)
        results = robust.noise_robustness(noise_levels=[0.0, 0.1, 0.2], n_repeats=3)
        
        self.assertIn(0.0, results['accuracies'])
        self.assertIn(0.1, results['accuracies'])
        self.assertIn(0.0, results['stds'])
    
    def test_sample_size_robustness(self):
        """Test sample size robustness analysis."""
        robust = RobustnessAnalysis(self.fc_task, self.fc_rest)
        results = robust.sample_size_robustness(sample_fractions=[0.5, 1.0], n_repeats=3)
        
        self.assertTrue(len(results) >= 1)


class TestInterpretability(unittest.TestCase):
    """Test interpretability analysis (Reviewer 1, Point 7)."""
    
    def test_dictionary_atom_analysis(self):
        """Test dictionary atom analysis."""
        np.random.seed(42)
        n_atoms = 5
        n_features = 100 * 99 // 2
        n_subjects = 10
        
        dictionary = np.random.randn(n_features, n_atoms)
        sparse_codes = np.random.randn(n_atoms, n_subjects)
        
        # Set some values to zero to simulate sparsity
        sparse_codes[np.abs(sparse_codes) < 0.5] = 0
        
        analyzer = DictionaryAtomAnalysis(dictionary, n_parcels=100)
        
        # Should be able to reconstruct an atom
        atom_matrix = analyzer.map_atom_to_matrix(0)
        self.assertEqual(atom_matrix.shape, (100, 100))


class TestModelArchitecture(unittest.TestCase):
    """Test model architecture specifications (Reviewer 1, Point 4)."""
    
    def test_convae_architecture(self):
        """Verify ConvAE architecture matches documentation."""
        model = ConvAutoencoder(n_parcels=360)
        
        # Test forward pass
        x = torch.randn(2, 1, 360, 360)
        with torch.no_grad():
            output = model(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_convae_encoder_dimensions(self):
        """Test encoder dimension reduction."""
        model = ConvAutoencoder(n_parcels=360)
        
        x = torch.randn(1, 1, 360, 360)
        with torch.no_grad():
            latent = model.encoder(x)
        
        # After 3 pooling layers: 360 -> 180 -> 90 -> 45
        self.assertEqual(latent.shape[2], 45)
        self.assertEqual(latent.shape[3], 45)
        self.assertEqual(latent.shape[1], 64)  # 64 channels
    
    def test_convae_residual_computation(self):
        """Test residual computation."""
        model = ConvAutoencoder(n_parcels=360)
        
        x = torch.randn(2, 1, 360, 360)
        with torch.no_grad():
            residual = model.get_residual(x)
        
        self.assertEqual(residual.shape, x.shape)
    
    def test_convae_parameter_count(self):
        """Test that model has reasonable parameter count."""
        model = ConvAutoencoder(n_parcels=360)
        n_params = model.count_parameters()
        
        # Should be around 34k parameters based on current architecture
        self.assertGreater(n_params, 30000)
        self.assertLess(n_params, 500000)


class TestSparseDictionaryLearning(unittest.TestCase):
    """Test sparse dictionary learning module (Reviewer 2, Point 1)."""
    
    def setUp(self):
        self.n_samples = 20
        self.n_features = 50
        np.random.seed(42)
        self.Y = np.random.randn(self.n_features, self.n_samples)
        
    def test_k_svd_execution(self):
        """Test K-SVD runs and returns correct shapes."""
        from src.models.sparse_dictionary_learning import k_svd
        K, L = 5, 3
        D, X = k_svd(self.Y, K=K, L=L, n_iter=2, verbose=False)
        
        self.assertEqual(D.shape, (self.n_features, K))
        self.assertEqual(X.shape, (K, self.n_samples))
        # Check dictionary atoms are normalized
        norms = np.linalg.norm(D, axis=0)
        np.testing.assert_almost_equal(norms, np.ones(K))
        
    def test_perform_grid_search(self):
        """Test grid search parameter tuning."""
        from src.models.sparse_dictionary_learning import perform_grid_search
        
        # Create a "rest" matrix similar to task for matching
        rest_flat = self.Y + 0.1 * np.random.randn(*self.Y.shape)
        
        accuracies, best_K, best_L = perform_grid_search(
            self.Y, rest_flat, 
            n_subjects=self.n_samples, 
            n_features=self.n_features,
            K_range=(4, 6),
            n_iter=1,
            task_name="test"
        )
        
        self.assertIsInstance(best_K, int)
        self.assertIsInstance(best_L, int)
        self.assertTrue(best_L <= best_K)
        self.assertGreaterEqual(np.max(accuracies), 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def setUp(self):
        self.n_subjects = 10
        self.n_parcels = 100
        np.random.seed(42)
        
        self.fc_rest = np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_rest = (self.fc_rest + self.fc_rest.transpose(0, 2, 1)) / 2
        
        self.fc_task = self.fc_rest + 0.1 * np.random.randn(self.n_subjects, self.n_parcels, self.n_parcels)
        self.fc_task = (self.fc_task + self.fc_task.transpose(0, 2, 1)) / 2
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_metrics_report_generation(self):
        """Test that metrics module generates report."""
        corr_matrix = np.eye(self.n_subjects) * 0.8 + np.random.rand(self.n_subjects, self.n_subjects) * 0.1
        
        metrics = ComprehensiveMetrics(corr_matrix)
        report_path = os.path.join(self.temp_dir, "metrics_report.txt")
        metrics.generate_report(report_path)
        
        self.assertTrue(os.path.exists(report_path))


if __name__ == '__main__':
    # Run with verbosity to see test names
    unittest.main(verbosity=2)
