#!/usr/bin/env python3
"""
Test script for Music File Management

Tests the music file manager and integration with the streaming system.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_music_file_manager():
    """Test MusicFileManager functionality."""
    print("🎵 Testing MusicFileManager...")
    
    try:
        from real_time.music_file_manager import MusicFileManager
        
        # Initialize with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MusicFileManager(music_dir=temp_dir)
            print("✅ MusicFileManager initialized")
            
            # Test directory structure
            for dir_name, dir_path in manager.directories.items():
                if dir_path.exists():
                    print(f"✅ Directory created: {dir_name}")
                else:
                    print(f"❌ Directory missing: {dir_name}")
            
            # Test performance metrics
            metrics = manager.get_performance_metrics()
            print(f"✅ Performance metrics: {metrics['known_files_count']} files")
            
            # Test file discovery
            files = manager._discover_music_files()
            print(f"✅ File discovery: {len(files)} files found")
            
            # Test scan
            results = manager.scan_music_files()
            print(f"✅ File scan: {results['total_files']} files processed")
            
            return True
            
    except Exception as e:
        print(f"❌ MusicFileManager test failed: {e}")
        return False

def test_music_file_validation():
    """Test music file validation."""
    print("\n🔍 Testing Music File Validation...")
    
    try:
        from real_time.music_file_manager import MusicFileManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MusicFileManager(music_dir=temp_dir)
            
            # Test validation of non-existent file
            result = manager.validate_file("/nonexistent/file.wav")
            if not result['valid'] and "File does not exist" in result['errors']:
                print("✅ Non-existent file validation works")
            else:
                print("❌ Non-existent file validation failed")
                return False
            
            # Test validation of unsupported format
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("This is not audio")
            
            result = manager.validate_file(str(test_file))
            if not result['valid'] and "Unsupported format" in result['errors']:
                print("✅ Unsupported format validation works")
            else:
                print("❌ Unsupported format validation failed")
                return False
            
            return True
            
    except Exception as e:
        print(f"❌ Music file validation test failed: {e}")
        return False

def test_music_file_metadata():
    """Test music file metadata handling."""
    print("\n📋 Testing Music File Metadata...")
    
    try:
        from real_time.music_file_manager import MusicFileManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MusicFileManager(music_dir=temp_dir)
            
            # Create a test file
            test_file = Path(temp_dir) / "test_tracks" / "test.wav"
            test_file.parent.mkdir(exist_ok=True)
            test_file.write_bytes(b"fake audio data")
            
            # Test metadata extraction
            metadata = manager._extract_file_metadata(test_file)
            print(f"✅ Metadata extracted: {metadata['filename']}")
            
            # Test custom metadata
            custom_metadata = {
                "title": "Test Track",
                "artist": "Test Artist",
                "genre": "Test Genre",
                "tempo": 128.0
            }
            
            success = manager.add_custom_metadata(str(test_file), custom_metadata)
            if success:
                print("✅ Custom metadata added")
            else:
                print("❌ Custom metadata addition failed")
                return False
            
            # Test getting files with filters
            files = manager.get_music_files(directory="test_tracks")
            print(f"✅ Filtered files: {len(files)} files in test_tracks")
            
            return True
            
    except Exception as e:
        print(f"❌ Music file metadata test failed: {e}")
        return False

def test_streaming_integration_with_music_files():
    """Test streaming integration with music files."""
    print("\n🔗 Testing Streaming Integration with Music Files...")
    
    try:
        from real_time.streaming_integration import StreamingIntegration
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary database
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.close()
            
            # Initialize integration with music files
            integration = StreamingIntegration(
                db_path=temp_db.name,
                music_dir=temp_dir,
                sample_rate=22050,
                enable_predictions=True,
                enable_learning=True
            )
            print("✅ StreamingIntegration with music files initialized")
            
            # Test music file scanning
            scan_results = integration.scan_music_files()
            print(f"✅ Music file scan: {scan_results['total_files']} files")
            
            # Test getting music files
            music_files = integration.get_music_files()
            print(f"✅ Music files retrieved: {len(music_files)} files")
            
            # Test integration status
            status = integration.get_integration_status()
            if 'music_files_status' in status:
                print("✅ Music files status included in integration status")
            else:
                print("❌ Music files status missing from integration status")
                return False
            
            # Clean up
            try:
                os.unlink(temp_db.name)
            except OSError:
                pass
            
            return True
            
    except Exception as e:
        print(f"❌ Streaming integration with music files test failed: {e}")
        return False

def test_music_file_structure():
    """Test music file directory structure."""
    print("\n📁 Testing Music File Directory Structure...")
    
    try:
        # Check if music_files directory exists
        music_dir = Path("music_files")
        if music_dir.exists():
            print("✅ music_files directory exists")
        else:
            print("❌ music_files directory missing")
            return False
        
        # Check subdirectories
        subdirs = ['test_tracks', 'sample_mixes', 'reference_tracks', 'user_uploads']
        for subdir in subdirs:
            subdir_path = music_dir / subdir
            if subdir_path.exists():
                print(f"✅ {subdir} directory exists")
            else:
                print(f"❌ {subdir} directory missing")
                return False
        
        # Check .gitkeep files
        for subdir in subdirs:
            gitkeep_file = music_dir / subdir / '.gitkeep'
            if gitkeep_file.exists():
                print(f"✅ {subdir}/.gitkeep exists")
            else:
                print(f"❌ {subdir}/.gitkeep missing")
                return False
        
        # Check README
        readme_file = music_dir / 'README.md'
        if readme_file.exists():
            print("✅ README.md exists")
        else:
            print("❌ README.md missing")
            return False
        
        # Check .gitignore
        gitignore_file = music_dir / '.gitignore'
        if gitignore_file.exists():
            print("✅ .gitignore exists")
        else:
            print("❌ .gitignore missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Music file structure test failed: {e}")
        return False

def create_sample_music_file():
    """Create a sample music file for testing."""
    print("\n🎼 Creating Sample Music File...")
    
    try:
        # Create a simple WAV file header (44 bytes)
        # This is a minimal WAV file that will pass basic validation
        wav_header = bytearray([
            # RIFF header
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x24, 0x00, 0x00, 0x00,  # File size - 8
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            
            # fmt chunk
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Chunk size
            0x01, 0x00,              # Audio format (PCM)
            0x01, 0x00,              # Number of channels
            0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
            0x88, 0x58, 0x01, 0x00,  # Byte rate
            0x02, 0x00,              # Block align
            0x10, 0x00,              # Bits per sample
            
            # data chunk
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00   # Data size
        ])
        
        # Create sample file
        sample_file = Path("music_files/test_tracks/sample.wav")
        sample_file.parent.mkdir(exist_ok=True)
        
        with open(sample_file, 'wb') as f:
            f.write(wav_header)
        
        # Create metadata file
        metadata = {
            "title": "Sample Track",
            "artist": "Test Artist",
            "album": "Test Album",
            "genre": "Electronic",
            "year": 2024,
            "duration": 30.0,
            "tempo": 128.0,
            "key": "C",
            "mode": "major",
            "energy": 0.7,
            "danceability": 0.8
        }
        
        metadata_file = sample_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Sample music file created: {sample_file}")
        print(f"✅ Sample metadata file created: {metadata_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample music file creation failed: {e}")
        return False

def main():
    """Run all music file tests."""
    print("🎵 Testing Music File Management System")
    print("=" * 60)
    
    tests = [
        ("Music File Structure", test_music_file_structure),
        ("MusicFileManager", test_music_file_manager),
        ("Music File Validation", test_music_file_validation),
        ("Music File Metadata", test_music_file_metadata),
        ("Streaming Integration", test_streaming_integration_with_music_files),
        ("Sample File Creation", create_sample_music_file)
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
        print("🎉 All tests passed! Music File Management is working correctly.")
        print("\n📁 Music files directory structure:")
        print("   music_files/")
        print("   ├── test_tracks/     # Test audio files")
        print("   ├── sample_mixes/    # Sample mixed tracks")
        print("   ├── reference_tracks/ # Reference tracks")
        print("   └── user_uploads/    # User-uploaded files")
        print("\n🎧 Ready to add music files and test the streaming system!")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
