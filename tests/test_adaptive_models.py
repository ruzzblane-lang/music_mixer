"""
Tests for adaptive models functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from real_time.adaptive_models import AdaptiveModels, UserPreferenceModel, MixingStyleModel


class TestAdaptiveModels(unittest.TestCase):
    """Test cases for AdaptiveModels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = AdaptiveModels(
            learning_rate=0.01,
            memory_size=100,
            adaptation_threshold=0.1
        )
    
    def test_initialization(self):
        """Test models initialization."""
        self.assertEqual(self.models.learning_rate, 0.01)
        self.assertEqual(self.models.memory_size, 100)
        self.assertEqual(self.models.adaptation_threshold, 0.1)
        self.assertTrue(self.models.is_learning)
    
    def test_update_models(self):
        """Test model updates."""
        # Create test data
        audio_features = {
            'rms_energy': 0.5,
            'spectral_centroid': 2000.0,
            'tempo': 120.0
        }
        
        user_feedback = {
            'accepted': True,
            'score': 0.8
        }
        
        performance_data = {
            'processing_time': 0.05,
            'cpu_usage': 0.6
        }
        
        # Update models
        result = self.models.update_models(audio_features, user_feedback, performance_data)
        print(f"DEBUG: update_models result: {result}")  # Debug output
        self.assertTrue(result)
    
    def test_get_adaptive_recommendations(self):
        """Test getting adaptive recommendations."""
        # Create test data
        current_features = {
            'tempo': 120.0,
            'key': 'C',
            'rms_energy': 0.5
        }
        
        available_tracks = [
            {
                'id': 1,
                'title': 'Track 1',
                'features': {'tempo': 125.0, 'key': 'C', 'rms_energy': 0.6}
            },
            {
                'id': 2,
                'title': 'Track 2',
                'features': {'tempo': 80.0, 'key': 'F', 'rms_energy': 0.3}
            }
        ]
        
        # Get recommendations
        recommendations = self.models.get_adaptive_recommendations(current_features, available_tracks)
        print(f"DEBUG: recommendations: {recommendations}")  # Debug output
        
        # Check results
        self.assertEqual(len(recommendations), 2)
        self.assertIsInstance(recommendations[0], tuple)
        self.assertIsInstance(recommendations[0][1], float)
    
    def test_predict_next_features(self):
        """Test predicting next features."""
        # Create test data
        current_features = {
            'rms_energy': 0.5,
            'spectral_centroid': 2000.0,
            'tempo': 120.0
        }
        
        # Predict next features
        predicted = self.models.predict_next_features(current_features)
        
        # Check result
        self.assertIsInstance(predicted, dict)
    
    def test_get_model_status(self):
        """Test getting model status."""
        status = self.models.get_model_status()
        
        # Check status structure
        self.assertIn('user_preference_model', status)
        self.assertIn('mixing_style_model', status)
        self.assertIn('performance_model', status)
        self.assertIn('prediction_model', status)
        self.assertIn('learning_enabled', status)
    
    def test_save_load_models(self):
        """Test saving and loading models."""
        # Create test file path
        test_file = '/tmp/test_models.json'
        
        # Save models
        result = self.models.save_models(test_file)
        self.assertTrue(result)
        
        # Load models
        result = self.models.load_models(test_file)
        self.assertTrue(result)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


class TestUserPreferenceModel(unittest.TestCase):
    """Test cases for UserPreferenceModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = UserPreferenceModel(memory_size=100)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.memory_size, 100)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(len(self.model.feedback_history), 0)
    
    def test_update(self):
        """Test model update."""
        # Create test data
        features = {'rms_energy': 0.5, 'tempo': 120.0}
        feedback = {'accepted': True, 'score': 0.8}
        
        # Update model
        self.model.update(features, feedback)
        
        # Check update
        self.assertEqual(len(self.model.feedback_history), 1)
    
    def test_get_preference_adjustment(self):
        """Test getting preference adjustment."""
        # Create test track
        track = {'id': 1, 'title': 'Test Track'}
        
        # Get adjustment
        adjustment = self.model.get_preference_adjustment(track)
        
        # Check result
        self.assertIsInstance(adjustment, float)
        self.assertGreaterEqual(adjustment, 0.0)
    
    def test_get_status(self):
        """Test getting model status."""
        status = self.model.get_status()
        
        # Check status structure
        self.assertIn('memory_usage', status)
        self.assertIn('learning_rate', status)
    
    def test_serialize_deserialize(self):
        """Test model serialization and deserialization."""
        # Serialize model
        data = self.model.serialize()
        
        # Check serialized data
        self.assertIn('preference_weights', data)
        self.assertIn('learning_rate', data)
        
        # Deserialize model
        self.model.deserialize(data)
        
        # Check deserialization
        self.assertEqual(self.model.learning_rate, data['learning_rate'])


class TestMixingStyleModel(unittest.TestCase):
    """Test cases for MixingStyleModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MixingStyleModel(memory_size=100)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.memory_size, 100)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(len(self.model.mixing_history), 0)
    
    def test_update(self):
        """Test model update."""
        # Create test data
        features = {'rms_energy': 0.5, 'tempo': 120.0}
        feedback = {'accepted': True, 'score': 0.8}
        
        # Update model
        self.model.update(features, feedback)
        
        # Check update
        self.assertEqual(len(self.model.mixing_history), 1)
    
    def test_get_style_adjustment(self):
        """Test getting style adjustment."""
        # Create test track
        track = {'id': 1, 'title': 'Test Track'}
        
        # Get adjustment
        adjustment = self.model.get_style_adjustment(track)
        
        # Check result
        self.assertIsInstance(adjustment, float)
        self.assertGreaterEqual(adjustment, 0.0)
    
    def test_get_status(self):
        """Test getting model status."""
        status = self.model.get_status()
        
        # Check status structure
        self.assertIn('memory_usage', status)
        self.assertIn('learning_rate', status)
    
    def test_serialize_deserialize(self):
        """Test model serialization and deserialization."""
        # Serialize model
        data = self.model.serialize()
        
        # Check serialized data
        self.assertIn('style_weights', data)
        self.assertIn('learning_rate', data)
        
        # Deserialize model
        self.model.deserialize(data)
        
        # Check deserialization
        self.assertEqual(self.model.learning_rate, data['learning_rate'])


if __name__ == '__main__':
    unittest.main()
