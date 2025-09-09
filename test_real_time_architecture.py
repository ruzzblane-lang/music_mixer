#!/usr/bin/env python3
"""
Test script for real-time audio analysis architecture

This script tests the basic functionality of the real-time audio analysis
architecture without requiring actual audio files.
"""

import sys
import os
import numpy as np
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all real-time modules can be imported."""
    print("🧪 Testing Real-Time Architecture Imports...")
    
    try:
        from real_time.streaming_analyzer import StreamingAudioAnalyzer
        print("✅ StreamingAudioAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ StreamingAudioAnalyzer import failed: {e}")
        return False
    
    try:
        from real_time.sliding_window import SlidingWindowManager
        print("✅ SlidingWindowManager imported successfully")
    except ImportError as e:
        print(f"❌ SlidingWindowManager import failed: {e}")
        return False
    
    try:
        from real_time.adaptive_models import AdaptiveModels
        print("✅ AdaptiveModels imported successfully")
    except ImportError as e:
        print(f"❌ AdaptiveModels import failed: {e}")
        return False
    
    try:
        from real_time.prediction_engine import RealTimePredictor
        print("✅ RealTimePredictor imported successfully")
    except ImportError as e:
        print(f"❌ RealTimePredictor import failed: {e}")
        return False
    
    try:
        from real_time.performance_optimizer import PerformanceOptimizer
        print("✅ PerformanceOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ PerformanceOptimizer import failed: {e}")
        print("💡 Install psutil: pip install psutil")
        return False
    
    return True

def test_streaming_analyzer():
    """Test StreamingAudioAnalyzer functionality."""
    print("\n🎵 Testing StreamingAudioAnalyzer...")
    
    try:
        from real_time.streaming_analyzer import StreamingAudioAnalyzer
        
        # Create analyzer
        analyzer = StreamingAudioAnalyzer(
            sample_rate=22050,
            window_size=1024,
            hop_size=512
        )
        
        # Test initialization
        assert analyzer.sample_rate == 22050
        assert analyzer.window_size == 1024
        assert analyzer.hop_size == 512
        print("✅ Initialization successful")
        
        # Test audio chunk addition
        audio_chunk = np.random.randn(512)
        result = analyzer.add_audio_chunk(audio_chunk)
        assert result == True
        print("✅ Audio chunk addition successful")
        
        # Test feature extraction
        analyzer.add_audio_chunk(audio_chunk)  # Add more data
        window = analyzer._get_current_window()
        if window is not None:
            features = analyzer._extract_features(window)
            assert 'rms_energy' in features
            assert 'spectral_centroid' in features
            print("✅ Feature extraction successful")
        
        # Test performance metrics
        metrics = analyzer.get_performance_metrics()
        assert 'processing_time' in metrics
        assert 'buffer_usage' in metrics
        print("✅ Performance metrics successful")
        
        return True
        
    except Exception as e:
        print(f"❌ StreamingAudioAnalyzer test failed: {e}")
        return False

def test_sliding_window():
    """Test SlidingWindowManager functionality."""
    print("\n🪟 Testing SlidingWindowManager...")
    
    try:
        from real_time.sliding_window import SlidingWindowManager
        
        # Create window manager
        window_manager = SlidingWindowManager(
            window_size=1024,
            hop_size=512,
            window_function='hann'
        )
        
        # Test initialization
        assert window_manager.window_size == 1024
        assert window_manager.hop_size == 512
        assert window_manager.window_function == 'hann'
        print("✅ Initialization successful")
        
        # Test audio data addition
        audio_data = np.random.randn(1024)
        result = window_manager.add_audio_data(audio_data)
        assert result == True
        print("✅ Audio data addition successful")
        
        # Test window retrieval
        window = window_manager.get_current_window()
        assert window is not None
        assert len(window) == 1024
        print("✅ Window retrieval successful")
        
        # Test buffer status
        status = window_manager.get_buffer_status()
        assert 'total_samples' in status
        assert 'can_get_window' in status
        print("✅ Buffer status successful")
        
        return True
        
    except Exception as e:
        print(f"❌ SlidingWindowManager test failed: {e}")
        return False

def test_adaptive_models():
    """Test AdaptiveModels functionality."""
    print("\n🧠 Testing AdaptiveModels...")
    
    try:
        from real_time.adaptive_models import AdaptiveModels
        
        # Create models
        models = AdaptiveModels(
            learning_rate=0.01,
            memory_size=100
        )
        
        # Test initialization
        assert models.learning_rate == 0.01
        assert models.memory_size == 100
        assert models.is_learning == True
        print("✅ Initialization successful")
        
        # Test model update
        audio_features = {'rms_energy': 0.5, 'tempo': 120.0}
        user_feedback = {'accepted': True, 'score': 0.8}
        
        result = models.update_models(audio_features, user_feedback)
        assert result == True
        print("✅ Model update successful")
        
        # Test recommendations
        available_tracks = [
            {'id': 1, 'features': {'tempo': 125.0, 'key': 'C', 'rms_energy': 0.6}},
            {'id': 2, 'features': {'tempo': 80.0, 'key': 'F', 'rms_energy': 0.3}}
        ]
        
        recommendations = models.get_adaptive_recommendations(audio_features, available_tracks)
        assert len(recommendations) == 2
        print("✅ Recommendations successful")
        
        # Test model status
        status = models.get_model_status()
        assert 'user_preference_model' in status
        assert 'learning_enabled' in status
        print("✅ Model status successful")
        
        return True
        
    except Exception as e:
        print(f"❌ AdaptiveModels test failed: {e}")
        return False

def test_prediction_engine():
    """Test RealTimePredictor functionality."""
    print("\n🔮 Testing RealTimePredictor...")
    
    try:
        from real_time.prediction_engine import RealTimePredictor
        
        # Create predictor
        predictor = RealTimePredictor(
            prediction_window=2.0,
            confidence_threshold=0.7
        )
        
        # Test initialization
        assert predictor.prediction_window == 2.0
        assert predictor.confidence_threshold == 0.7
        print("✅ Initialization successful")
        
        # Test beat prediction
        current_audio = np.random.randn(2048)
        tempo_history = [120.0, 121.0, 120.5]
        
        predicted_time, confidence = predictor.predict_next_beat(
            current_audio, tempo_history
        )
        assert isinstance(predicted_time, float)
        assert isinstance(confidence, float)
        print("✅ Beat prediction successful")
        
        # Test energy transition prediction
        energy_history = [0.3, 0.4, 0.5, 0.6]
        transition_type, predicted_time, confidence = predictor.predict_energy_transition(
            energy_history, {}
        )
        assert isinstance(transition_type, str)
        assert isinstance(predicted_time, float)
        assert isinstance(confidence, float)
        print("✅ Energy transition prediction successful")
        
        # Test performance metrics
        metrics = predictor.get_performance_metrics()
        assert 'avg_time' in metrics
        assert 'total_predictions' in metrics
        print("✅ Performance metrics successful")
        
        return True
        
    except Exception as e:
        print(f"❌ RealTimePredictor test failed: {e}")
        return False

def test_performance_optimizer():
    """Test PerformanceOptimizer functionality."""
    print("\n⚡ Testing PerformanceOptimizer...")
    
    try:
        from real_time.performance_optimizer import PerformanceOptimizer, PerformanceLevel
        
        # Create optimizer
        optimizer = PerformanceOptimizer(
            target_latency=0.05,
            cpu_threshold=0.8
        )
        
        # Test initialization
        assert optimizer.target_latency == 0.05
        assert optimizer.cpu_threshold == 0.8
        assert optimizer.current_performance_level == PerformanceLevel.HIGH
        print("✅ Initialization successful")
        
        # Test optimization strategies
        strategies = optimizer.get_optimization_strategies()
        assert 'feature_reduction' in strategies
        assert 'skip_advanced_features' in strategies
        print("✅ Optimization strategies successful")
        
        # Test recommended settings
        settings = optimizer.get_recommended_settings()
        assert 'window_size' in settings
        assert 'enable_advanced_features' in settings
        print("✅ Recommended settings successful")
        
        # Test performance metrics
        metrics = optimizer.get_performance_metrics()
        assert 'cpu_usage' in metrics
        assert 'performance_level' in metrics
        print("✅ Performance metrics successful")
        
        return True
        
    except Exception as e:
        print(f"❌ PerformanceOptimizer test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\n🔗 Testing Component Integration...")
    
    try:
        from real_time.streaming_analyzer import StreamingAudioAnalyzer
        from real_time.adaptive_models import AdaptiveModels
        from real_time.performance_optimizer import PerformanceOptimizer
        
        # Create components
        analyzer = StreamingAudioAnalyzer()
        models = AdaptiveModels()
        optimizer = PerformanceOptimizer()
        
        # Test integration
        audio_chunk = np.random.randn(1024)
        analyzer.add_audio_chunk(audio_chunk)
        analyzer.add_audio_chunk(audio_chunk)
        
        # Get features
        window = analyzer._get_current_window()
        if window is not None:
            features = analyzer._extract_features(window)
            
            # Update models
            models.update_models(features, {'accepted': True, 'score': 0.8})
            
            # Get performance metrics
            perf_metrics = optimizer.get_performance_metrics()
            
            print("✅ Component integration successful")
            return True
        else:
            print("⚠️ Integration test skipped (insufficient data)")
            return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎵 AI Music Mixer - Real-Time Architecture Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("StreamingAudioAnalyzer", test_streaming_analyzer),
        ("SlidingWindowManager", test_sliding_window),
        ("AdaptiveModels", test_adaptive_models),
        ("RealTimePredictor", test_prediction_engine),
        ("PerformanceOptimizer", test_performance_optimizer),
        ("Integration Test", test_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Real-time architecture is working correctly.")
        print("\n🚀 Ready for Phase 7A.2: Core Streaming Engine implementation!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
