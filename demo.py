#!/usr/bin/env python3
"""
AI Music Mixer Demo Script

This script demonstrates the key features of the AI Music Mixer
without requiring a full music library.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.database import MusicDatabase
from recommend.engine import RecommendationEngine
from mix.engine import MixingEngine
from features.extractor import AudioFeatureExtractor

def demo_feature_extraction():
    """Demonstrate audio feature extraction."""
    print("🎵 AI Music Mixer - Feature Extraction Demo")
    print("=" * 50)
    
    extractor = AudioFeatureExtractor()
    
    # Create sample features (simulating extracted features)
    sample_features = {
        'tempo': 128.5,
        'key': 'C',
        'mode': 'major',
        'rms_energy': 0.3,
        'brightness': 2500.0,
        'spectral_rolloff': 4000.0,
        'spectral_bandwidth': 1500.0,
        'zero_crossing_rate': 0.05,
        'onset_strength': 0.8,
        'tempo_stability': 0.2,
        'spectral_contrast': 0.6,
        'mfcc_means': [0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.1, 0.2]
    }
    
    print("Sample audio features:")
    for key, value in sample_features.items():
        if key == 'mfcc_means':
            print(f"  {key}: {len(value)} coefficients")
        else:
            print(f"  {key}: {value}")
    
    # Convert to feature vector
    feature_vector = extractor.get_feature_vector(sample_features)
    print(f"\nFeature vector length: {len(feature_vector)}")
    print(f"Feature vector (first 10): {feature_vector[:10]}")
    
    return sample_features

def demo_database():
    """Demonstrate database operations."""
    print("\n🗄️ Database Demo")
    print("=" * 30)
    
    # Create database
    db = MusicDatabase("demo_music.db")
    
    # Add sample tracks
    sample_tracks = [
        {
            'file_path': '/music/track1.mp3',
            'file_hash': 'abc123',
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen',
            'album': 'A Night at the Opera',
            'duration': 355.0,
            'tempo': 72.0,
            'key': 'Bb',
            'mode': 'major',
            'rms_energy': 0.4,
            'brightness': 2000.0,
            'spectral_rolloff': 3500.0,
            'spectral_bandwidth': 1200.0,
            'zero_crossing_rate': 0.03,
            'onset_strength': 0.7,
            'tempo_stability': 0.1,
            'spectral_contrast': 0.5,
            'mfcc_means': ','.join(map(str, [0.1] * 13)),
            'last_analyzed': 1234567890
        },
        {
            'file_path': '/music/track2.mp3',
            'file_hash': 'def456',
            'title': 'Hotel California',
            'artist': 'Eagles',
            'album': 'Hotel California',
            'duration': 391.0,
            'tempo': 75.0,
            'key': 'Bm',
            'mode': 'minor',
            'rms_energy': 0.35,
            'brightness': 2200.0,
            'spectral_rolloff': 3800.0,
            'spectral_bandwidth': 1300.0,
            'zero_crossing_rate': 0.04,
            'onset_strength': 0.6,
            'tempo_stability': 0.15,
            'spectral_contrast': 0.55,
            'mfcc_means': ','.join(map(str, [0.2] * 13)),
            'last_analyzed': 1234567890
        }
    ]
    
    # Add tracks to database
    for track in sample_tracks:
        track_id = db.add_track(track)
        print(f"Added track: {track['title']} (ID: {track_id})")
    
    # Get library stats
    stats = db.get_library_stats()
    print(f"\nLibrary Statistics:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Unique artists: {stats['unique_artists']}")
    print(f"  Tempo range: {stats['tempo_min']:.1f} - {stats['tempo_max']:.1f} BPM")
    
    # Search tracks
    search_results = db.search_tracks("Queen", limit=5)
    print(f"\nSearch results for 'Queen':")
    for track in search_results:
        print(f"  - {track['title']} - {track['artist']}")
    
    return db

def demo_recommendations(db):
    """Demonstrate recommendation engine."""
    print("\n🎯 Recommendation Engine Demo")
    print("=" * 35)
    
    engine = RecommendationEngine("demo_music.db")
    
    # Get recommendations
    recommendations = engine.get_recommendations("Bohemian Rhapsody", count=3)
    
    if recommendations:
        print("Recommendations for 'Bohemian Rhapsody':")
        for i, (track, score) in enumerate(recommendations, 1):
            print(f"  {i}. {track['title']} - {track['artist']}")
            print(f"     Tempo: {track['tempo']:.1f} BPM, Key: {track['key']} {track['mode']}")
            print(f"     Compatibility Score: {score:.2f}")
    else:
        print("No recommendations found (need more tracks for ML model)")
    
    # Demonstrate key compatibility
    print(f"\nKey Compatibility Examples:")
    compatibility = engine._get_key_compatibility('C', 'G')
    print(f"  C -> G: {compatibility:.2f}")
    
    compatibility = engine._get_key_compatibility('C', 'Am')
    print(f"  C -> Am: {compatibility:.2f}")
    
    compatibility = engine._get_key_compatibility('C', 'F#')
    print(f"  C -> F#: {compatibility:.2f}")

def demo_mixing_engine():
    """Demonstrate mixing engine capabilities."""
    print("\n🎧 Mixing Engine Demo")
    print("=" * 25)
    
    engine = MixingEngine()
    
    # Show supported formats
    print("Supported audio formats:")
    formats = ['.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma']
    for fmt in formats:
        print(f"  - {fmt}")
    
    # Demonstrate crossfade calculation
    print(f"\nCrossfade Configuration:")
    print(f"  Default crossfade duration: 8.0 seconds")
    print(f"  Beat alignment: Enabled")
    print(f"  EQ ducking: Enabled")
    
    # Show transition effects
    print(f"\nAvailable Transition Effects:")
    effects = ['reverb', 'echo', 'filter_sweep']
    for effect in effects:
        print(f"  - {effect}")

def main():
    """Run the complete demo."""
    print("🎵 AI-Assisted Music Mixer - Complete Demo")
    print("=" * 50)
    print("This demo shows the key features without requiring audio files.")
    print()
    
    try:
        # Demo feature extraction
        features = demo_feature_extraction()
        
        # Demo database operations
        db = demo_database()
        
        # Demo recommendations
        demo_recommendations(db)
        
        # Demo mixing engine
        demo_mixing_engine()
        
        print("\n🎉 Demo Complete!")
        print("=" * 20)
        print("The AI Music Mixer is ready to use with your music library!")
        print("\nNext steps:")
        print("1. Run: python main.py scanmusic /path/to/your/music")
        print("2. Run: python main.py mix --interactive")
        
        # Clean up demo database
        if os.path.exists("demo_music.db"):
            os.remove("demo_music.db")
            print("\nDemo database cleaned up.")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
