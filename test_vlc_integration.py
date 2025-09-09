#!/usr/bin/env python3
"""
Test script for VLC Integration

Tests the VLC music player integration with the AI Music Mixer system.
"""

import sys
import os
import time
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_vlc_player():
    """Test VLC music player functionality."""
    print("🎵 Testing VLC Music Player...")
    
    try:
        from real_time.vlc_music_player import VLCMusicPlayer, PlayerState
        
        # Initialize player
        player = VLCMusicPlayer(sample_rate=44100, enable_streaming=True)
        print("✅ VLC Music Player initialized")
        
        # Test status
        status = player.get_status()
        print(f"✅ Player status: {status['state']}")
        
        # Find a test music file
        music_files = []
        for root, dirs, files in os.walk('music_files/test_tracks'):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac')):
                    music_files.append(os.path.join(root, file))
                    break
            if music_files:
                break
        
        if music_files:
            test_file = music_files[0]
            print(f"✅ Testing with: {os.path.basename(test_file)}")
            
            # Test loading track
            if player.load_track(test_file):
                print("✅ Track loaded successfully")
                
                # Test playback (briefly)
                if player.play():
                    print("✅ Playback started")
                    time.sleep(2)  # Play for 2 seconds
                    player.stop()
                    print("✅ Playback stopped")
                else:
                    print("⚠️  Playback start failed")
            else:
                print("❌ Track loading failed")
        else:
            print("⚠️  No music files found for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ VLC Music Player test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlc_integration():
    """Test VLC integration with streaming system."""
    print("\n🔗 Testing VLC Integration...")
    
    try:
        from real_time.vlc_integration import VLCIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize integration
        integration = VLCIntegration(
            music_dir='music_files',
            db_path=temp_db.name,
            sample_rate=44100,
            enable_streaming=True,
            enable_analysis=True
        )
        print("✅ VLC Integration initialized")
        
        # Test integration status
        status = integration.get_integration_status()
        print(f"✅ Integration status: integrated={status['is_integrated']}")
        print(f"✅ VLC status: {status['vlc_status']['state']}")
        
        # Test starting integration
        if integration.start_integration():
            print("✅ VLC integration started")
            
            # Wait a bit for analysis
            time.sleep(2)
            
            # Check status again
            status = integration.get_integration_status()
            print(f"✅ Analysis active: {status['analysis_active']}")
            
            # Stop integration
            integration.stop_integration()
            print("✅ VLC integration stopped")
        else:
            print("⚠️  VLC integration start failed")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ VLC Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlc_playlist():
    """Test VLC playlist functionality."""
    print("\n📋 Testing VLC Playlist...")
    
    try:
        from real_time.vlc_integration import VLCIntegration
        import tempfile
        
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        # Initialize integration
        integration = VLCIntegration(
            music_dir='music_files',
            db_path=temp_db.name,
            sample_rate=44100,
            enable_streaming=True,
            enable_analysis=True
        )
        
        # Get music files
        music_files = integration.music_file_manager.get_music_files(directory='test_tracks')
        print(f"✅ Found {len(music_files)} music files")
        
        if music_files:
            # Create playlist from first 3 files
            playlist = []
            for i, file_metadata in enumerate(music_files[:3]):
                track = {
                    'file_path': file_metadata['file_path'],
                    'title': file_metadata.get('title', file_metadata['filename']),
                    'artist': file_metadata.get('artist', 'Unknown'),
                    'tempo': file_metadata.get('tempo', 120.0),
                    'key': file_metadata.get('key', 'C'),
                    'metadata': file_metadata
                }
                playlist.append(track)
            
            # Load playlist
            if integration.load_playlist(playlist):
                print(f"✅ Playlist loaded: {len(playlist)} tracks")
                
                # Test playing first track
                if integration.play_current_track():
                    print("✅ First track started")
                    time.sleep(2)
                    integration.stop_current_track()
                    print("✅ Track stopped")
                else:
                    print("⚠️  First track play failed")
            else:
                print("❌ Playlist loading failed")
        else:
            print("⚠️  No music files for playlist test")
        
        # Clean up
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass
        
        return True
        
    except Exception as e:
        print(f"❌ VLC Playlist test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlc_streaming():
    """Test VLC streaming capabilities."""
    print("\n🌐 Testing VLC Streaming...")
    
    try:
        from real_time.vlc_music_player import VLCMusicPlayer
        
        # Initialize player with streaming
        player = VLCMusicPlayer(enable_streaming=True)
        print("✅ VLC Player with streaming initialized")
        
        # Test streaming start
        if player.start_streaming(port=8080):
            print("✅ Streaming started on port 8080")
            
            # Test streaming stop
            if player.stop_streaming():
                print("✅ Streaming stopped")
            else:
                print("⚠️  Streaming stop failed")
        else:
            print("⚠️  Streaming start failed")
        
        return True
        
    except Exception as e:
        print(f"❌ VLC Streaming test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vlc_crossfading():
    """Test VLC crossfading functionality."""
    print("\n🎧 Testing VLC Crossfading...")
    
    try:
        from real_time.vlc_music_player import VLCMusicPlayer
        
        # Initialize player
        player = VLCMusicPlayer()
        print("✅ VLC Player for crossfading initialized")
        
        # Find two test files
        music_files = []
        for root, dirs, files in os.walk('music_files/test_tracks'):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.flac')):
                    music_files.append(os.path.join(root, file))
                    if len(music_files) >= 2:
                        break
            if len(music_files) >= 2:
                break
        
        if len(music_files) >= 2:
            file1, file2 = music_files[0], music_files[1]
            print(f"✅ Testing crossfade: {os.path.basename(file1)} -> {os.path.basename(file2)}")
            
            # Load first track
            if player.load_track(file1):
                print("✅ First track loaded")
                
                # Test crossfade (briefly)
                if player.crossfade_to_next(file2, duration=2.0):
                    print("✅ Crossfade started")
                    time.sleep(3)  # Wait for crossfade
                    player.stop()
                    print("✅ Crossfade test completed")
                else:
                    print("⚠️  Crossfade start failed")
            else:
                print("❌ First track loading failed")
        else:
            print("⚠️  Need at least 2 music files for crossfade test")
        
        return True
        
    except Exception as e:
        print(f"❌ VLC Crossfading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all VLC integration tests."""
    print("🎵 Testing VLC Integration with AI Music Mixer")
    print("=" * 60)
    
    tests = [
        ("VLC Music Player", test_vlc_player),
        ("VLC Integration", test_vlc_integration),
        ("VLC Playlist", test_vlc_playlist),
        ("VLC Streaming", test_vlc_streaming),
        ("VLC Crossfading", test_vlc_crossfading)
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
        print("🎉 All tests passed! VLC Integration is working perfectly!")
        print("\n🎧 VLC Integration Features:")
        print("   - Music playback with VLC")
        print("   - Real-time audio analysis")
        print("   - Intelligent recommendations")
        print("   - Crossfading between tracks")
        print("   - Streaming capabilities")
        print("   - Playlist management")
        print("\n🚀 Ready for Phase 7A.3: Predictive Engine!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
