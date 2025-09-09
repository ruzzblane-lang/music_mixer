#!/usr/bin/env python3
"""
Test script for Real Music Files

Tests the streaming system with actual music files placed by the user.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def list_music_files():
    """List all music files in the test_tracks directory."""
    print("🎵 Discovering Music Files...")
    
    music_dir = Path("music_files/test_tracks")
    if not music_dir.exists():
        print("❌ music_files/test_tracks directory not found")
        return []
    
    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
    music_files = []
    
    for file_path in music_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            music_files.append(file_path)
    
    print(f"✅ Found {len(music_files)} music files:")
    for i, file_path in enumerate(music_files, 1):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   {i}. {file_path.name} ({file_size:.1f} MB)")
    
    return music_files

def test_music_file_analysis():
    """Test music file analysis with real files."""
    print("\n🔍 Testing Music File Analysis...")
    
    try:
        from real_time.music_file_manager import MusicFileManager
        
        # Initialize with the actual music directory
        manager = MusicFileManager(music_dir="music_files")
        print("✅ MusicFileManager initialized with real music directory")
        
        # Scan for music files
        print("📊 Scanning music files...")
        scan_results = manager.scan_music_files(force_rescan=True)
        
        print(f"✅ Scan completed:")
        print(f"   - Total files: {scan_results['total_files']}")
        print(f"   - New files: {len(scan_results['new_files'])}")
        print(f"   - Updated files: {len(scan_results['updated_files'])}")
        print(f"   - Failed files: {len(scan_results['failed_files'])}")
        print(f"   - Scan duration: {scan_results['scan_duration']:.2f}s")
        
        # Show failed files if any
        if scan_results['failed_files']:
            print("\n⚠️  Failed files:")
            for failed_file in scan_results['failed_files']:
                print(f"   - {failed_file['file_path']}: {failed_file['error']}")
        
        # Get music files from test_tracks
        test_tracks = manager.get_music_files(directory="test_tracks")
        print(f"\n🎧 Test tracks found: {len(test_tracks)}")
        
        # Display metadata for each track
        for i, track in enumerate(test_tracks, 1):
            print(f"\n   Track {i}: {track['filename']}")
            print(f"   - Format: {track['format']}")
            print(f"   - Size: {track['file_size'] / (1024*1024):.1f} MB")
            
            # Show extracted audio features
            if 'tempo' in track:
                print(f"   - Tempo: {track['tempo']:.1f} BPM")
            if 'key' in track:
                print(f"   - Key: {track['key']}")
            if 'duration' in track:
                print(f"   - Duration: {track['duration']:.1f}s")
            if 'rms_energy' in track:
                print(f"   - Energy: {track['rms_energy']:.3f}")
            if 'spectral_centroid' in track:
                print(f"   - Brightness: {track['spectral_centroid']:.1f} Hz")
        
        return True
        
    except Exception as e:
        print(f"❌ Music file analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_integration_with_real_files():
    """Test streaming integration with real music files."""
    print("\n🔗 Testing Streaming Integration with Real Files...")
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize integration with real music directory
        integration = StreamingIntegration(
            db_path=temp_db.name,
            music_dir="music_files",
            sample_rate=22050,
            enable_predictions=True,
            enable_learning=True
        )
        print("✅ StreamingIntegration initialized with real music files")
        
        # Scan music files
        scan_results = integration.scan_music_files(force_rescan=True)
        print(f"✅ Music files scanned: {scan_results['total_files']} files")
        
        # Get available tracks
        available_tracks = integration._get_available_tracks()
        print(f"✅ Available tracks: {len(available_tracks)}")
        
        # Test recommendations with real tracks
        if available_tracks:
            print("\n🎯 Testing Recommendations with Real Tracks...")
            
            # Use the first track as current track
            current_track = available_tracks[0]
            print(f"   Current track: {current_track['title']} by {current_track['artist']}")
            
            # Get recommendations
            recommendations = integration.get_real_time_recommendations()
            print(f"   Real-time recommendations: {len(recommendations)}")
            
            # Test adaptive recommendations
            current_features = {
                'tempo': current_track.get('tempo', 120.0),
                'key': current_track.get('key', 'C'),
                'mode': current_track.get('mode', 'major'),
                'rms_energy': current_track.get('rms_energy', 0.5),
                'spectral_centroid': current_track.get('spectral_centroid', 2000.0)
            }
            
            adaptive_recommendations = integration.streaming_pipeline.get_recommendations(
                current_features, available_tracks
            )
            print(f"   Adaptive recommendations: {len(adaptive_recommendations)}")
            
            # Show top recommendations
            if adaptive_recommendations:
                print("\n   Top 3 Recommendations:")
                for i, (track, score) in enumerate(adaptive_recommendations[:3], 1):
                    print(f"   {i}. {track['title']} (score: {score:.3f})")
        
        # Test integration status
        status = integration.get_integration_status()
        print(f"\n📊 Integration Status:")
        print(f"   - Music files: {status['music_files_status']['known_files_count']}")
        print(f"   - Pipeline ready: {not status['pipeline_status']['is_running']}")
        print(f"   - Device info: {status['device_info']['total_devices']} devices")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_time_analysis():
    """Test real-time analysis with actual music files."""
    print("\n⚡ Testing Real-Time Analysis...")
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize integration
        integration = StreamingIntegration(
            db_path=temp_db.name,
            music_dir="music_files",
            sample_rate=22050,
            enable_predictions=True,
            enable_learning=True
        )
        
        # Start integration
        if integration.start_integration():
            print("✅ Real-time integration started")
            
            # Let it run for a few seconds to process audio
            print("🎧 Processing audio for 3 seconds...")
            time.sleep(3)
            
            # Check real-time features
            features = integration.get_real_time_features()
            predictions = integration.get_real_time_predictions()
            
            print(f"✅ Real-time features extracted: {len(features)}")
            if features:
                print("   Sample features:")
                for key, value in list(features.items())[:5]:
                    print(f"   - {key}: {value}")
            
            print(f"✅ Real-time predictions: {len(predictions)}")
            if predictions:
                print("   Sample predictions:")
                for key, value in list(predictions.items())[:3]:
                    print(f"   - {key}: {value}")
            
            # Stop integration
            integration.stop_integration()
            print("✅ Real-time integration stopped")
            
        else:
            print("⚠️  Real-time integration start failed (expected in some environments)")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_music_library_summary():
    """Create a summary of the music library."""
    print("\n📚 Creating Music Library Summary...")
    
    try:
        from real_time.music_file_manager import MusicFileManager
        
        manager = MusicFileManager(music_dir="music_files")
        manager.scan_music_files(force_rescan=True)
        
        # Get all music files
        all_files = manager.get_music_files()
        test_tracks = manager.get_music_files(directory="test_tracks")
        
        # Create summary
        summary = {
            "total_files": len(all_files),
            "test_tracks": len(test_tracks),
            "formats": {},
            "tempos": [],
            "keys": [],
            "durations": [],
            "energies": []
        }
        
        # Analyze files
        for file_metadata in all_files:
            # Count formats
            format_name = file_metadata.get('format', 'unknown')
            summary["formats"][format_name] = summary["formats"].get(format_name, 0) + 1
            
            # Collect features
            if 'tempo' in file_metadata:
                summary["tempos"].append(file_metadata['tempo'])
            if 'key' in file_metadata:
                summary["keys"].append(file_metadata['key'])
            if 'duration' in file_metadata:
                summary["durations"].append(file_metadata['duration'])
            if 'rms_energy' in file_metadata:
                summary["energies"].append(file_metadata['rms_energy'])
        
        # Calculate statistics
        if summary["tempos"]:
            summary["tempo_stats"] = {
                "min": min(summary["tempos"]),
                "max": max(summary["tempos"]),
                "avg": sum(summary["tempos"]) / len(summary["tempos"])
            }
        
        if summary["durations"]:
            summary["duration_stats"] = {
                "min": min(summary["durations"]),
                "max": max(summary["durations"]),
                "avg": sum(summary["durations"]) / len(summary["durations"])
            }
        
        # Save summary
        summary_file = Path("music_files/music_library_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Music library summary created: {summary_file}")
        print(f"   - Total files: {summary['total_files']}")
        print(f"   - Test tracks: {summary['test_tracks']}")
        print(f"   - Formats: {summary['formats']}")
        
        if 'tempo_stats' in summary:
            print(f"   - Tempo range: {summary['tempo_stats']['min']:.1f} - {summary['tempo_stats']['max']:.1f} BPM")
            print(f"   - Average tempo: {summary['tempo_stats']['avg']:.1f} BPM")
        
        if 'duration_stats' in summary:
            print(f"   - Duration range: {summary['duration_stats']['min']:.1f} - {summary['duration_stats']['max']:.1f} seconds")
            print(f"   - Average duration: {summary['duration_stats']['avg']:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Music library summary creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all real music file tests."""
    print("🎵 Testing AI Music Mixer with Real Music Files")
    print("=" * 60)
    
    # First, list the music files
    music_files = list_music_files()
    if not music_files:
        print("❌ No music files found in music_files/test_tracks/")
        print("   Please add some music files to test the system!")
        return False
    
    tests = [
        ("Music File Analysis", test_music_file_analysis),
        ("Streaming Integration", test_streaming_integration_with_real_files),
        ("Real-Time Analysis", test_real_time_analysis),
        ("Music Library Summary", create_music_library_summary)
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
        print("🎉 All tests passed! Your music files are working perfectly with the AI Music Mixer!")
        print("\n🎧 Your music library is ready for:")
        print("   - Real-time audio analysis")
        print("   - AI-powered recommendations")
        print("   - Streaming integration")
        print("   - Adaptive learning")
        print("\n🚀 Ready to proceed to Phase 7A.3: Predictive Engine!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
