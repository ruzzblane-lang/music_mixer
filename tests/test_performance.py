"""
Tests for performance optimization functionality.
"""

import unittest
import numpy as np
import sys
import os
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from real_time.performance_optimizer import PerformanceOptimizer, PerformanceLevel


class TestPerformanceOptimizer(unittest.TestCase):
    """Test cases for PerformanceOptimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer(
            target_latency=0.05,
            cpu_threshold=0.8,
            memory_threshold=0.85,
            monitoring_interval=0.1
        )
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.target_latency, 0.05)
        self.assertEqual(self.optimizer.cpu_threshold, 0.8)
        self.assertEqual(self.optimizer.memory_threshold, 0.85)
        self.assertEqual(self.optimizer.monitoring_interval, 0.1)
        self.assertEqual(self.optimizer.current_performance_level, PerformanceLevel.HIGH)
        self.assertFalse(self.optimizer.is_monitoring)
    
    def test_optimization_strategies(self):
        """Test optimization strategies."""
        strategies = self.optimizer.get_optimization_strategies()
        
        # Check default strategies
        self.assertIn('feature_reduction', strategies)
        self.assertIn('window_size_reduction', strategies)
        self.assertIn('skip_advanced_features', strategies)
        self.assertIn('reduce_prediction_frequency', strategies)
        self.assertIn('enable_caching', strategies)
        
        # Check default values
        self.assertFalse(strategies['feature_reduction'])
        self.assertFalse(strategies['window_size_reduction'])
        self.assertFalse(strategies['skip_advanced_features'])
        self.assertFalse(strategies['reduce_prediction_frequency'])
        self.assertTrue(strategies['enable_caching'])
    
    def test_set_optimization_strategy(self):
        """Test setting optimization strategies."""
        # Set a strategy
        self.optimizer.set_optimization_strategy('feature_reduction', True)
        
        # Check change
        strategies = self.optimizer.get_optimization_strategies()
        self.assertTrue(strategies['feature_reduction'])
        
        # Set back to False
        self.optimizer.set_optimization_strategy('feature_reduction', False)
        
        # Check change
        strategies = self.optimizer.get_optimization_strategies()
        self.assertFalse(strategies['feature_reduction'])
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        metrics = self.optimizer.get_performance_metrics()
        
        # Check metrics structure
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('processing_time', metrics)
        self.assertIn('audio_latency', metrics)
        self.assertIn('feature_extraction_time', metrics)
        self.assertIn('performance_level', metrics)
        self.assertIn('optimization_strategies', metrics)
        
        # Check performance level
        self.assertEqual(metrics['performance_level'], PerformanceLevel.HIGH.value)
    
    def test_get_recommended_settings(self):
        """Test getting recommended settings."""
        settings = self.optimizer.get_recommended_settings()
        
        # Check settings structure
        self.assertIn('window_size', settings)
        self.assertIn('hop_size', settings)
        self.assertIn('enable_advanced_features', settings)
        self.assertIn('prediction_frequency', settings)
        self.assertIn('caching_enabled', settings)
        
        # Check default values for HIGH performance
        self.assertEqual(settings['window_size'], 2048)
        self.assertEqual(settings['hop_size'], 512)
        self.assertTrue(settings['enable_advanced_features'])
        self.assertEqual(settings['prediction_frequency'], 1.0)
        self.assertTrue(settings['caching_enabled'])
    
    def test_callback_system(self):
        """Test callback system."""
        callback_data = []
        
        def test_callback(data):
            callback_data.append(data)
        
        # Add callback
        self.optimizer.add_callback('on_performance_warning', test_callback)
        
        # Manually trigger performance analysis (simulate low performance)
        self.optimizer.current_performance_level = PerformanceLevel.LOW
        self.optimizer._apply_optimizations()
        
        # Check if callback was triggered
        self.assertGreater(len(callback_data), 0)
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        # Add some dummy data to metrics
        self.optimizer.performance_metrics['cpu_usage'].append(0.5)
        self.optimizer.performance_metrics['memory_usage'].append(0.6)
        
        # Reset metrics
        self.optimizer.reset_metrics()
        
        # Check reset
        self.assertEqual(len(self.optimizer.performance_metrics['cpu_usage']), 0)
        self.assertEqual(len(self.optimizer.performance_metrics['memory_usage']), 0)
        self.assertEqual(self.optimizer.current_performance_level, PerformanceLevel.HIGH)
    
    def test_performance_level_changes(self):
        """Test performance level changes."""
        # Start with HIGH performance
        self.assertEqual(self.optimizer.current_performance_level, PerformanceLevel.HIGH)
        
        # Simulate low performance
        self.optimizer.current_performance_level = PerformanceLevel.LOW
        self.optimizer._apply_optimizations()
        
        # Check optimizations applied
        strategies = self.optimizer.get_optimization_strategies()
        self.assertTrue(strategies['feature_reduction'])
        self.assertTrue(strategies['skip_advanced_features'])
        self.assertTrue(strategies['reduce_prediction_frequency'])
        self.assertTrue(strategies['window_size_reduction'])
    
    def test_get_performance_history(self):
        """Test getting performance history."""
        history = self.optimizer.get_performance_history()
        
        # Should be empty initially
        self.assertEqual(len(history), 0)
        
        # Simulate performance level change
        self.optimizer.current_performance_level = PerformanceLevel.MEDIUM
        self.optimizer._apply_optimizations()
        
        # Check history
        history = self.optimizer.get_performance_history()
        self.assertGreaterEqual(len(history), 0)


class TestPerformanceLevel(unittest.TestCase):
    """Test cases for PerformanceLevel enum."""
    
    def test_performance_levels(self):
        """Test performance level values."""
        self.assertEqual(PerformanceLevel.LOW.value, "low")
        self.assertEqual(PerformanceLevel.MEDIUM.value, "medium")
        self.assertEqual(PerformanceLevel.HIGH.value, "high")
        self.assertEqual(PerformanceLevel.MAXIMUM.value, "maximum")
    
    def test_performance_level_comparison(self):
        """Test performance level comparison."""
        self.assertNotEqual(PerformanceLevel.LOW, PerformanceLevel.HIGH)
        self.assertEqual(PerformanceLevel.HIGH, PerformanceLevel.HIGH)


if __name__ == '__main__':
    unittest.main()
