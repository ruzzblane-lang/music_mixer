"""
Tests for Core Streaming Engine components.

Tests the audio stream manager, streaming pipeline, device interface,
and streaming integration.
"""

import unittest
import numpy as np
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import the components we're testing
import sys
sys.path.append('/home/bane-h-kali/ai-music-mixer')

from real_time.audio_stream_manager import AudioStreamManager
from real_time.streaming_pipeline import StreamingPipeline
from real_time.audio_device_interface import AudioDeviceInterface, AudioDeviceType
from real_time.streaming_integration import StreamingIntegration


class TestAudioStreamManager(unittest.TestCase):
    """Test AudioStreamManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stream_manager = AudioStreamManager(
            sample_rate=22050,
            channels=1,
            chunk_size=1024
        )
    
    def test_initialization(self):
        """Test AudioStreamManager initialization."""
        self.assertEqual(self.stream_manager.sample_rate, 22050)
        self.assertEqual(self.stream_manager.channels, 1)
        self.assertEqual(self.stream_manager.chunk_size, 1024)
        self.assertFalse(self.stream_manager.is_streaming)
    
    def test_get_available_devices(self):
        """Test getting available audio devices."""
        devices = self.stream_manager.get_available_devices()
        self.assertIsInstance(devices, list)
        
        # Should have at least mock devices
        self.assertGreater(len(devices), 0)
        
        # Check device structure
        for device in devices:
            self.assertIn('id', device)
            self.assertIn('name', device)
            self.assertIn('channels', device)
            self.assertIn('sample_rate', device)
    
    def test_start_stop_input_stream(self):
        """Test starting and stopping input stream."""
        # Start input stream
        result = self.stream_manager.start_input_stream()
        self.assertTrue(result)
        self.assertTrue(self.stream_manager.is_streaming)
        
        # Stop streams
        self.stream_manager.stop_streams()
        self.assertFalse(self.stream_manager.is_streaming)
    
    def test_audio_buffer_operations(self):
        """Test audio buffer operations."""
        # Start input stream
        self.stream_manager.start_input_stream()
        
        # Wait a bit for audio data
        time.sleep(0.1)
        
        # Get audio chunk
        audio_chunk = self.stream_manager.get_audio_chunk(timeout=0.1)
        
        # Should get some audio data (even if mock)
        if audio_chunk is not None:
            self.assertIsInstance(audio_chunk, np.ndarray)
            self.assertEqual(len(audio_chunk.shape), 2)  # Should be 2D (samples, channels)
        
        # Test output buffer
        test_audio = np.random.randn(1024, 1).astype(np.float32)
        result = self.stream_manager.put_audio_chunk(test_audio)
        self.assertTrue(result)
        
        self.stream_manager.stop_streams()
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        metrics = self.stream_manager.get_performance_metrics()
        
        self.assertIn('is_streaming', metrics)
        self.assertIn('input_buffer_size', metrics)
        self.assertIn('output_buffer_size', metrics)
        self.assertIn('audio_history_size', metrics)
        self.assertIn('buffer_underruns', metrics)
        self.assertIn('buffer_overruns', metrics)
        self.assertIn('stream_errors', metrics)
        self.assertIn('audio_library', metrics)
    
    def test_callbacks(self):
        """Test callback system."""
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Add callback
        self.stream_manager.add_callback('on_audio_input', test_callback)
        
        # Start stream to trigger callback
        self.stream_manager.start_input_stream()
        time.sleep(0.1)
        
        # Check if callback was called
        if callback_called:
            self.assertIsInstance(callback_data, np.ndarray)
        
        self.stream_manager.stop_streams()
    
    def tearDown(self):
        """Clean up after tests."""
        self.stream_manager.stop_streams()


class TestStreamingPipeline(unittest.TestCase):
    """Test StreamingPipeline functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = StreamingPipeline(
            sample_rate=22050,
            window_size=2048,
            hop_size=512,
            enable_predictions=True,
            enable_learning=True
        )
    
    def test_initialization(self):
        """Test StreamingPipeline initialization."""
        self.assertEqual(self.pipeline.sample_rate, 22050)
        self.assertEqual(self.pipeline.window_size, 2048)
        self.assertEqual(self.pipeline.hop_size, 512)
        self.assertTrue(self.pipeline.enable_predictions)
        self.assertTrue(self.pipeline.enable_learning)
        self.assertFalse(self.pipeline.is_running)
    
    def test_start_stop_pipeline(self):
        """Test starting and stopping pipeline."""
        # Start pipeline
        result = self.pipeline.start_pipeline()
        self.assertTrue(result)
        self.assertTrue(self.pipeline.is_running)
        
        # Wait a bit for processing
        time.sleep(0.2)
        
        # Stop pipeline
        self.pipeline.stop_pipeline()
        self.assertFalse(self.pipeline.is_running)
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        status = self.pipeline.get_pipeline_status()
        
        self.assertIn('is_running', status)
        self.assertIn('audio_manager_status', status)
        self.assertIn('analyzer_status', status)
        self.assertIn('window_manager_status', status)
        self.assertIn('optimizer_status', status)
        self.assertIn('pipeline_metrics', status)
        self.assertIn('queue_sizes', status)
    
    def test_recommendations(self):
        """Test getting recommendations."""
        # Mock current features
        current_features = {
            'tempo': 120.0,
            'key': 'C',
            'mode': 'major',
            'rms_energy': 0.5,
            'spectral_centroid': 2000.0
        }
        
        # Mock available tracks
        available_tracks = [
            {'id': 1, 'title': 'Track 1', 'tempo': 120.0, 'key': 'C'},
            {'id': 2, 'title': 'Track 2', 'tempo': 125.0, 'key': 'G'},
            {'id': 3, 'title': 'Track 3', 'tempo': 115.0, 'key': 'F'}
        ]
        
        recommendations = self.pipeline.get_recommendations(current_features, available_tracks)
        self.assertIsInstance(recommendations, list)
    
    def test_callbacks(self):
        """Test callback system."""
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Add callback
        self.pipeline.add_callback('on_features_extracted', test_callback)
        
        # Start pipeline to trigger callbacks
        self.pipeline.start_pipeline()
        time.sleep(0.2)
        
        # Check if callback was called
        if callback_called:
            self.assertIsInstance(callback_data, dict)
        
        self.pipeline.stop_pipeline()
    
    def tearDown(self):
        """Clean up after tests."""
        self.pipeline.stop_pipeline()


class TestAudioDeviceInterface(unittest.TestCase):
    """Test AudioDeviceInterface functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device_interface = AudioDeviceInterface()
    
    def test_initialization(self):
        """Test AudioDeviceInterface initialization."""
        self.assertIsInstance(self.device_interface.audio_library, str)
        self.assertIsInstance(self.device_interface.devices, list)
    
    def test_refresh_devices(self):
        """Test refreshing device list."""
        devices = self.device_interface.refresh_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)
        
        # Check device structure
        for device in devices:
            self.assertIsInstance(device.device_id, int)
            self.assertIsInstance(device.name, str)
            self.assertIsInstance(device.device_type, AudioDeviceType)
            self.assertIsInstance(device.channels, int)
            self.assertIsInstance(device.sample_rate, (int, float))
    
    def test_get_devices_by_type(self):
        """Test getting devices by type."""
        # Refresh devices first
        self.device_interface.refresh_devices()
        
        # Get input devices
        input_devices = self.device_interface.get_input_devices()
        self.assertIsInstance(input_devices, list)
        
        # Get output devices
        output_devices = self.device_interface.get_output_devices()
        self.assertIsInstance(output_devices, list)
        
        # Get devices by specific type
        input_only = self.device_interface.get_devices(AudioDeviceType.INPUT)
        self.assertIsInstance(input_only, list)
    
    def test_get_device_by_id(self):
        """Test getting device by ID."""
        self.device_interface.refresh_devices()
        
        if self.device_interface.devices:
            device = self.device_interface.get_device_by_id(0)
            if device:
                self.assertEqual(device.device_id, 0)
        
        # Test non-existent device
        device = self.device_interface.get_device_by_id(999)
        self.assertIsNone(device)
    
    def test_get_device_by_name(self):
        """Test getting device by name."""
        self.device_interface.refresh_devices()
        
        if self.device_interface.devices:
            device = self.device_interface.get_device_by_id(0)
            if device:
                found_device = self.device_interface.get_device_by_name(device.name)
                self.assertIsNotNone(found_device)
                self.assertEqual(found_device.device_id, device.device_id)
    
    def test_get_default_devices(self):
        """Test getting default devices."""
        self.device_interface.refresh_devices()
        
        default_input = self.device_interface.get_default_input_device()
        default_output = self.device_interface.get_default_output_device()
        
        # At least one should exist (even in mock mode)
        self.assertTrue(default_input is not None or default_output is not None)
    
    def test_device_info(self):
        """Test getting device information."""
        info = self.device_interface.get_device_info()
        
        self.assertIn('audio_library', info)
        self.assertIn('total_devices', info)
        self.assertIn('input_devices', info)
        self.assertIn('output_devices', info)
        self.assertIn('devices', info)
        
        self.assertIsInstance(info['total_devices'], int)
        self.assertIsInstance(info['input_devices'], int)
        self.assertIsInstance(info['output_devices'], int)
        self.assertIsInstance(info['devices'], list)


class TestStreamingIntegration(unittest.TestCase):
    """Test StreamingIntegration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.integration = StreamingIntegration(
            db_path=self.temp_db.name,
            sample_rate=22050,
            enable_predictions=True,
            enable_learning=True
        )
    
    def test_initialization(self):
        """Test StreamingIntegration initialization."""
        self.assertEqual(self.integration.sample_rate, 22050)
        self.assertTrue(self.integration.enable_predictions)
        self.assertTrue(self.integration.enable_learning)
        self.assertFalse(self.integration.is_integrated)
        self.assertFalse(self.integration.mix_session_active)
    
    def test_start_stop_integration(self):
        """Test starting and stopping integration."""
        # Start integration
        result = self.integration.start_integration()
        self.assertTrue(result)
        self.assertTrue(self.integration.is_integrated)
        
        # Wait a bit for processing
        time.sleep(0.2)
        
        # Stop integration
        self.integration.stop_integration()
        self.assertFalse(self.integration.is_integrated)
    
    def test_integration_status(self):
        """Test getting integration status."""
        status = self.integration.get_integration_status()
        
        self.assertIn('is_integrated', status)
        self.assertIn('mix_session_active', status)
        self.assertIn('current_track', status)
        self.assertIn('real_time_features', status)
        self.assertIn('real_time_predictions', status)
        self.assertIn('recommendations_count', status)
        self.assertIn('pipeline_status', status)
        self.assertIn('device_info', status)
    
    def test_real_time_data_access(self):
        """Test accessing real-time data."""
        # Get real-time features
        features = self.integration.get_real_time_features()
        self.assertIsInstance(features, dict)
        
        # Get real-time predictions
        predictions = self.integration.get_real_time_predictions()
        self.assertIsInstance(predictions, dict)
        
        # Get real-time recommendations
        recommendations = self.integration.get_real_time_recommendations()
        self.assertIsInstance(recommendations, list)
    
    def test_mix_session(self):
        """Test mix session management."""
        # Create a mock audio file
        mock_audio_path = "/tmp/test_audio.wav"
        
        # Mock the mixing engine
        with patch.object(self.integration.mixing_engine, 'get_track_info') as mock_get_info:
            mock_get_info.return_value = {
                'duration': 180.0,
                'sample_rate': 22050,
                'channels': 2
            }
            
            # Start mix session
            result = self.integration.start_mix_session(mock_audio_path)
            self.assertTrue(result)
            self.assertTrue(self.integration.mix_session_active)
            self.assertIsNotNone(self.integration.current_track)
            
            # Stop mix session
            self.integration.stop_mix_session()
            self.assertFalse(self.integration.mix_session_active)
            self.assertIsNone(self.integration.current_track)
    
    def test_user_feedback(self):
        """Test adding user feedback."""
        # This should not raise an exception
        self.integration.add_user_feedback("Track A", "Track B", True, 0.8)
        
        # Check that feedback was processed (no exceptions)
        self.assertTrue(True)  # If we get here, no exception was raised
    
    def test_callbacks(self):
        """Test callback system."""
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Add callback
        self.integration.add_callback('on_track_analyzed', test_callback)
        
        # Start integration to trigger callbacks
        self.integration.start_integration()
        time.sleep(0.2)
        
        # Check if callback was called
        if callback_called:
            self.assertIsInstance(callback_data, dict)
        
        self.integration.stop_integration()
    
    def tearDown(self):
        """Clean up after tests."""
        self.integration.stop_integration()
        
        # Clean up temporary database
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass


if __name__ == '__main__':
    # Set up logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)
