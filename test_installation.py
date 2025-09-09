#!/usr/bin/env python3
"""
Test script to verify AI Music Mixer installation
"""

import sys
import importlib

def test_imports():
    """Test if all required modules can be imported."""
    required_modules = [
        'librosa',
        'numpy',
        'scipy',
        'sklearn',
        'pydub',
        'click',
        'colorama',
        'tqdm'
    ]
    
    print("Testing imports...")
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required modules imported successfully!")
        return True

def test_project_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    try:
        from features.extractor import AudioFeatureExtractor
        print("✅ AudioFeatureExtractor")
    except ImportError as e:
        print(f"❌ AudioFeatureExtractor: {e}")
        return False
    
    try:
        from features.database import MusicDatabase
        print("✅ MusicDatabase")
    except ImportError as e:
        print(f"❌ MusicDatabase: {e}")
        return False
    
    try:
        from recommend.engine import RecommendationEngine
        print("✅ RecommendationEngine")
    except ImportError as e:
        print(f"❌ RecommendationEngine: {e}")
        return False
    
    try:
        from mix.engine import MixingEngine
        print("✅ MixingEngine")
    except ImportError as e:
        print(f"❌ MixingEngine: {e}")
        return False
    
    print("\n✅ All project modules imported successfully!")
    return True

def main():
    """Run all tests."""
    print("🎵 AI Music Mixer - Installation Test")
    print("=" * 40)
    
    # Test external dependencies
    deps_ok = test_imports()
    
    # Test project modules
    modules_ok = test_project_modules()
    
    if deps_ok and modules_ok:
        print("\n🎉 Installation test passed! You're ready to use AI Music Mixer.")
        print("\nNext steps:")
        print("1. Run: python main.py scanmusic /path/to/your/music")
        print("2. Run: python main.py mix --interactive")
    else:
        print("\n❌ Installation test failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
