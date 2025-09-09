"""
Tests for streaming audio analysis functionality.
"""

import unittest
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from real_time.streaming_analyzer import StreamingAudioAnalyzer
from real_time.sliding_window import SlidingWindowManager


class TestStreamingAnalyzer(unittest.TestCase):
    """Test cases for StreamingAudioAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StreamingAudioAnalyzer(
            sample_rate=22050,
            window_size=1024,
            hop_size=512
        )
    
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.sample_rate, 22050)
        self.assertEqual(self.analyzer.window_size, 1024)
        self.assertEqual(self.analyzer.hop_size, 512)
        self.assertFalse(self.analyzer.is_streaming)
    
    def test_add_audio_chunk(self):
        """Test adding audio chunks."""
        # Create test audio data
        audio_chunk = np.random.randn(512)
        
        # Add audio chunk
        result = self.analyzer.add_audio_chunk(audio_chunk)
        self.assertTrue(result)
        
        # Check buffer
        self.assertEqual(len(self.analyzer.audio_buffer), 512)
    
    def test_add_audio_chunk_stereo(self):
        """Test adding stereo audio chunks."""
        # Create stereo audio data
        stereo_chunk = np.random.randn(2, 512)
        
        # Add audio chunk
        result = self.analyzer.add_audio_chunk(stereo_chunk)
        self.assertTrue(result)
        
        # Should be converted to mono
        self.assertEqual(len(self.analyzer.audio_buffer), 512)
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        # Create test audio data
        audio_chunk = np.random.randn(1024)
        
        # Add enough data for window
        self.analyzer.add_audio_chunk(audio_chunk)
        self.analyzer.add_audio_chunk(audio_chunk)
        
        # Get current window
        window = self.analyzer._get_current_window()
        self.assertIsNotNone(window)
        self.assertEqual(len(window), 1024)
        
        # Extract features
        features = self.analyzer._extract_features(window)
        
        # Check basic features
        self.assertIn('rms_energy', features)
        self.assertIn('spectral_centroid', features)
        self.assertIn('zero_crossing_rate', features)
        
        # Check feature values
        self.assertGreaterEqual(features['rms_energy'], 0)
        self.assertGreaterEqual(features['spectral_centroid'], 0)
        self.assertGreaterEqual(features['zero_crossing_rate'], 0)
    
    def test_callback_system(self):
        """Test callback system."""
        callback_data = []
        
        def test_callback(data):
            callback_data.append(data)
        
        # Add callback
        self.analyzer.add_callback('on_feature_extracted', test_callback)
        
        # Process audio to trigger callback
        audio_chunk = np.random.randn(1024)
        self.analyzer.add_audio_chunk(audio_chunk)
        self.analyzer.add_audio_chunk(audio_chunk)
        
        # Check if callback was triggered
        self.assertGreater(len(callback_data), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Process some audio
        audio_chunk = np.random.randn(1024)
        self.analyzer.add_audio_chunk(audio_chunk)
        self.analyzer.add_audio_chunk(audio_chunk)
        
        # Get metrics
        metrics = self.analyzer.get_performance_metrics()
        
        # Check metrics structure
        self.assertIn('processing_time', metrics)
        self.assertIn('feature_extraction_time', metrics)
        self.assertIn('buffer_usage', metrics)
        
        # Check buffer usage
        self.assertGreater(metrics['buffer_usage']['current'], 0)
    
    def test_reset(self):
        """Test analyzer reset."""
        # Add some data
        audio_chunk = np.random.randn(1024)
        self.analyzer.add_audio_chunk(audio_chunk)
        
        # Reset
        self.analyzer.reset()
        
        # Check reset state
        self.assertEqual(len(self.analyzer.audio_buffer), 0)
        self.assertEqual(self.analyzer.current_position, 0)


class TestSlidingWindowManager(unittest.TestCase):
    """Test cases for SlidingWindowManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.window_manager = SlidingWindowManager(
            window_size=1024,
            hop_size=512,
            sample_rate=22050
        )
    
    def test_initialization(self):
        """Test window manager initialization."""
        self.assertEqual(self.window_manager.window_size, 1024)
        self.assertEqual(self.window_manager.hop_size, 512)
        self.assertEqual(self.window_manager.sample_rate, 22050)
        self.assertEqual(self.window_manager.window_function, 'hann')
    
    def test_add_audio_data(self):
        """Test adding audio data."""
        # Create test audio data
        audio_data = np.random.randn(512)
        
        # Add audio data
        result = self.window_manager.add_audio_data(audio_data)
        self.assertTrue(result)
        
        # Check total samples
        self.assertEqual(self.window_manager.total_samples, 512)
    
    def test_get_current_window(self):
        """Test getting current window."""
        # Add enough data for window
        audio_data = np.random.randn(1024)
        self.window_manager.add_audio_data(audio_data)
        
        # Get current window
        window = self.window_manager.get_current_window()
        self.assertIsNotNone(window)
        self.assertEqual(len(window), 1024)
        
        # Check window function applied
        self.assertNotEqual(np.sum(window), 0)
    
    def test_get_overlap_region(self):
        """Test getting overlap region."""
        # Add enough data for overlap
        audio_data = np.random.randn(1024)
        self.window_manager.add_audio_data(audio_data)
        
        # Get overlap region
        overlap = self.window_manager.get_overlap_region()
        self.assertIsNotNone(overlap)
        self.assertEqual(len(overlap), 512)  # window_size - hop_size
    
    def test_get_multiple_windows(self):
        """Test getting multiple windows."""
        # Add enough data for multiple windows
        audio_data = np.random.randn(2048)
        self.window_manager.add_audio_data(audio_data)
        
        # Get multiple windows
        windows = self.window_manager.get_multiple_windows(3)
        
        # Should get 3 windows
        self.assertEqual(len(windows), 3)
        
        # Each window should be correct size
        for window in windows:
            self.assertEqual(len(window), 1024)
    
    def test_get_window_with_context(self):
        """Test getting window with context."""
        # Add enough data for window with context
        audio_data = np.random.randn(2048)
        self.window_manager.add_audio_data(audio_data)
        
        # Get window with context
        result = self.window_manager.get_window_with_context(256)
        self.assertIsNotNone(result)
        
        prev_context, current_window, next_context = result
        self.assertEqual(len(prev_context), 256)
        self.assertEqual(len(current_window), 1024)
        self.assertEqual(len(next_context), 256)
    
    def test_buffer_status(self):
        """Test buffer status information."""
        # Add some data
        audio_data = np.random.randn(512)
        self.window_manager.add_audio_data(audio_data)
        
        # Get buffer status
        status = self.window_manager.get_buffer_status()
        
        # Check status fields
        self.assertIn('buffer_size', status)
        self.assertIn('current_position', status)
        self.assertIn('total_samples', status)
        self.assertIn('can_get_window', status)
        self.assertIn('can_get_overlap', status)
        
        # Check values
        self.assertEqual(status['total_samples'], 512)
        self.assertFalse(status['can_get_window'])  # Not enough data yet
    
    def test_set_window_function(self):
        """Test changing window function."""
        # Change window function
        self.window_manager.set_window_function('hamming')
        
        # Check change
        self.assertEqual(self.window_manager.window_function, 'hamming')
    
    def test_get_window_info(self):
        """Test getting window information."""
        info = self.window_manager.get_window_info()
        
        # Check info fields
        self.assertIn('window_size', info)
        self.assertIn('hop_size', info)
        self.assertIn('window_function', info)
        self.assertIn('sample_rate', info)
        self.assertIn('window_duration_ms', info)
        self.assertIn('hop_duration_ms', info)
        self.assertIn('overlap_percent', info)
        
        # Check values
        self.assertEqual(info['window_size'], 1024)
        self.assertEqual(info['hop_size'], 512)
        self.assertEqual(info['sample_rate'], 22050)
    
    def test_reset(self):
        """Test window manager reset."""
        # Add some data
        audio_data = np.random.randn(512)
        self.window_manager.add_audio_data(audio_data)
        
        # Reset
        self.window_manager.reset()
        
        # Check reset state
        self.assertEqual(self.window_manager.total_samples, 0)
        self.assertEqual(self.window_manager.buffer_position, 0)


if __name__ == '__main__':
    unittest.main()
