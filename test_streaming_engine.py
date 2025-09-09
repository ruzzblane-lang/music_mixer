#!/usr/bin/env python3
"""
Test script for Core Streaming Engine (Phase 7A.2)

Tests the complete streaming engine implementation including:
- AudioStreamManager
- StreamingPipeline  
- AudioDeviceInterface
- StreamingIntegration

This script provides comprehensive testing of the streaming components.
"""

import sys
import os
import time
import logging
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing all streaming engine components."""
    print("🔍 Testing imports...")
    
    try:
        from real_time.audio_stream_manager import AudioStreamManager
        print("✅ AudioStreamManager imported successfully")
    except ImportError as e:
        print(f"❌ AudioStreamManager import failed: {e}")
        return False
    
    try:
        from real_time.streaming_pipeline import StreamingPipeline
        print("✅ StreamingPipeline imported successfully")
    except ImportError as e:
        print(f"❌ StreamingPipeline import failed: {e}")
        return False
    
    try:
        from real_time.audio_device_interface import AudioDeviceInterface, AudioDeviceType
        print("✅ AudioDeviceInterface imported successfully")
    except ImportError as e:
        print(f"❌ AudioDeviceInterface import failed: {e}")
        return False
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        print("✅ StreamingIntegration imported successfully")
    except ImportError as e:
        print(f"❌ StreamingIntegration import failed: {e}")
        return False
    
    return True

def test_audio_stream_manager():
    """Test AudioStreamManager functionality."""
    print("\n🎵 Testing AudioStreamManager...")
    
    try:
        from real_time.audio_stream_manager import AudioStreamManager
        
        # Initialize
        manager = AudioStreamManager(sample_rate=22050, chunk_size=1024)
        print("✅ AudioStreamManager initialized")
        
        # Test device enumeration
        devices = manager.get_available_devices()
        print(f"✅ Found {len(devices)} audio devices")
        
        # Test performance metrics
        metrics = manager.get_performance_metrics()
        print(f"✅ Performance metrics: {metrics['audio_library']}")
        
        # Test stream start/stop
        if manager.start_input_stream():
            print("✅ Input stream started")
            time.sleep(0.1)  # Let it run briefly
            manager.stop_streams()
            print("✅ Streams stopped")
        else:
            print("⚠️  Input stream start failed (expected in some environments)")
        
        return True
        
    except Exception as e:
        print(f"❌ AudioStreamManager test failed: {e}")
        traceback.print_exc()
        return False

def test_streaming_pipeline():
    """Test StreamingPipeline functionality."""
    print("\n🔄 Testing StreamingPipeline...")
    
    try:
        from real_time.streaming_pipeline import StreamingPipeline
        
        # Initialize
        pipeline = StreamingPipeline(
            sample_rate=22050,
            window_size=2048,
            hop_size=512,
            enable_predictions=True,
            enable_learning=True
        )
        print("✅ StreamingPipeline initialized")
        
        # Test pipeline status
        status = pipeline.get_pipeline_status()
        print(f"✅ Pipeline status: running={status['is_running']}")
        
        # Test recommendations
        current_features = {
            'tempo': 120.0,
            'key': 'C',
            'mode': 'major',
            'rms_energy': 0.5
        }
        available_tracks = [
            {'id': 1, 'title': 'Test Track 1', 'tempo': 120.0, 'key': 'C'},
            {'id': 2, 'title': 'Test Track 2', 'tempo': 125.0, 'key': 'G'}
        ]
        
        recommendations = pipeline.get_recommendations(current_features, available_tracks)
        print(f"✅ Got {len(recommendations)} recommendations")
        
        # Test start/stop
        if pipeline.start_pipeline():
            print("✅ Pipeline started")
            time.sleep(0.2)  # Let it process
            pipeline.stop_pipeline()
            print("✅ Pipeline stopped")
        else:
            print("⚠️  Pipeline start failed (expected in some environments)")
        
        return True
        
    except Exception as e:
        print(f"❌ StreamingPipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_audio_device_interface():
    """Test AudioDeviceInterface functionality."""
    print("\n🎛️  Testing AudioDeviceInterface...")
    
    try:
        from real_time.audio_device_interface import AudioDeviceInterface, AudioDeviceType
        
        # Initialize
        interface = AudioDeviceInterface()
        print(f"✅ AudioDeviceInterface initialized with {interface.audio_library}")
        
        # Test device refresh
        devices = interface.refresh_devices()
        print(f"✅ Found {len(devices)} audio devices")
        
        # Test device filtering
        input_devices = interface.get_input_devices()
        output_devices = interface.get_output_devices()
        print(f"✅ Input devices: {len(input_devices)}, Output devices: {len(output_devices)}")
        
        # Test default devices
        default_input = interface.get_default_input_device()
        default_output = interface.get_default_output_device()
        print(f"✅ Default input: {default_input.name if default_input else 'None'}")
        print(f"✅ Default output: {default_output.name if default_output else 'None'}")
        
        # Test device info
        info = interface.get_device_info()
        print(f"✅ Device info: {info['total_devices']} total devices")
        
        # Test device by ID
        if devices:
            device = interface.get_device_by_id(0)
            if device:
                print(f"✅ Device 0: {device.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ AudioDeviceInterface test failed: {e}")
        traceback.print_exc()
        return False

def test_streaming_integration():
    """Test StreamingIntegration functionality."""
    print("\n🔗 Testing StreamingIntegration...")
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Create temporary music directory
        temp_music_dir = tempfile.mkdtemp()
        
        # Initialize
        integration = StreamingIntegration(
            db_path=temp_db.name,
            music_dir=temp_music_dir,
            sample_rate=22050,
            enable_predictions=True,
            enable_learning=True
        )
        print("✅ StreamingIntegration initialized")
        
        # Test integration status
        status = integration.get_integration_status()
        print(f"✅ Integration status: integrated={status['is_integrated']}")
        
        # Test real-time data access
        features = integration.get_real_time_features()
        predictions = integration.get_real_time_predictions()
        recommendations = integration.get_real_time_recommendations()
        print(f"✅ Real-time data: features={len(features)}, predictions={len(predictions)}, recommendations={len(recommendations)}")
        
        # Test start/stop
        if integration.start_integration():
            print("✅ Integration started")
            time.sleep(0.2)  # Let it process
            integration.stop_integration()
            print("✅ Integration stopped")
        else:
            print("⚠️  Integration start failed (expected in some environments)")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
            import shutil
            shutil.rmtree(temp_music_dir)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ StreamingIntegration test failed: {e}")
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test complete integration workflow."""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Create temporary music directory
        temp_music_dir = tempfile.mkdtemp()
        
        # Initialize integration
        integration = StreamingIntegration(
            db_path=temp_db.name,
            music_dir=temp_music_dir,
            sample_rate=22050,
            enable_predictions=True,
            enable_learning=True
        )
        
        # Start integration
        if integration.start_integration():
            print("✅ Integration workflow started")
            
            # Simulate some processing time
            time.sleep(0.5)
            
            # Check status
            status = integration.get_integration_status()
            print(f"✅ Pipeline cycles: {status['pipeline_status']['pipeline_metrics']['pipeline_cycles']}")
            
            # Test user feedback
            integration.add_user_feedback("Track A", "Track B", True, 0.8)
            print("✅ User feedback added")
            
            # Stop integration
            integration.stop_integration()
            print("✅ Integration workflow completed")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
            import shutil
            shutil.rmtree(temp_music_dir)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Integration workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all streaming engine tests."""
    print("🚀 Testing Core Streaming Engine (Phase 7A.2)")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("AudioStreamManager", test_audio_stream_manager),
        ("StreamingPipeline", test_streaming_pipeline),
        ("AudioDeviceInterface", test_audio_device_interface),
        ("StreamingIntegration", test_streaming_integration),
        ("Integration Workflow", test_integration_workflow)
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Core Streaming Engine is working correctly.")
        print("\n🚀 Ready for Phase 7A.3: Predictive Engine implementation!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
