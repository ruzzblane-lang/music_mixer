"""
Music Library Scanner

Scans and analyzes entire music libraries, extracting features
and storing them in a database for later use.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import hashlib
import time

from .extractor import AudioFeatureExtractor
from .database import MusicDatabase

logger = logging.getLogger(__name__)


class MusicScanner:
    """Scans and analyzes music libraries."""
    
    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    def __init__(self, db_path: str = "data/music_library.db", verbose: bool = False):
        """
        Initialize the music scanner.
        
        Args:
            db_path: Path to the SQLite database
            verbose: Enable verbose logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self.extractor = AudioFeatureExtractor()
        self.db = MusicDatabase(db_path)
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
    
    def scan_directory(self, directory_path: str, force_rescan: bool = False) -> Dict:
        """
        Scan a directory for music files and analyze them.
        
        Args:
            directory_path: Path to the music directory
            force_rescan: Force re-analysis of already processed tracks
            
        Returns:
            Dictionary with scan results
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all audio files
        audio_files = self._find_audio_files(directory)
        
        if not audio_files:
            logger.warning(f"No audio files found in {directory_path}")
            return {'total_tracks': 0, 'new_tracks': 0, 'updated_tracks': 0, 'errors': 0}
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process files
        results = {
            'total_tracks': len(audio_files),
            'new_tracks': 0,
            'updated_tracks': 0,
            'errors': 0
        }
        
        # Create progress bar
        with tqdm(total=len(audio_files), desc="Analyzing tracks") as pbar:
            for audio_file in audio_files:
                try:
                    result = self._process_track(audio_file, force_rescan)
                    if result == 'new':
                        results['new_tracks'] += 1
                    elif result == 'updated':
                        results['updated_tracks'] += 1
                    
                    pbar.set_postfix({
                        'New': results['new_tracks'],
                        'Updated': results['updated_tracks'],
                        'Errors': results['errors']
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing {audio_file}: {e}")
                    results['errors'] += 1
                
                pbar.update(1)
        
        return results
    
    def _find_audio_files(self, directory: Path) -> List[Path]:
        """Find all audio files in a directory tree."""
        audio_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    audio_files.append(file_path)
        
        return sorted(audio_files)
    
    def _process_track(self, audio_file: Path, force_rescan: bool = False) -> str:
        """
        Process a single audio track.
        
        Args:
            audio_file: Path to the audio file
            force_rescan: Force re-analysis even if already processed
            
        Returns:
            'new', 'updated', or 'skipped'
        """
        # Calculate file hash for change detection
        file_hash = self._calculate_file_hash(audio_file)
        
        # Check if track already exists in database
        existing_track = self.db.get_track_by_path(str(audio_file))
        
        if existing_track and not force_rescan:
            # Check if file has changed
            if existing_track['file_hash'] == file_hash:
                return 'skipped'
        
        # Extract features
        try:
            features = self.extractor.extract_features(str(audio_file))
        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_file}: {e}")
            raise
        
        # Extract metadata
        metadata = self._extract_metadata(audio_file)
        
        # Prepare track data
        track_data = {
            'file_path': str(audio_file),
            'file_hash': file_hash,
            'title': metadata['title'],
            'artist': metadata['artist'],
            'album': metadata['album'],
            'duration': features['duration'],
            'tempo': features['tempo'],
            'key': features['key'],
            'mode': features['mode'],
            'rms_energy': features['rms_energy'],
            'brightness': features['brightness'],
            'spectral_rolloff': features['spectral_rolloff'],
            'spectral_bandwidth': features['spectral_bandwidth'],
            'zero_crossing_rate': features['zero_crossing_rate'],
            'onset_strength': features['onset_strength'],
            'tempo_stability': features['tempo_stability'],
            'spectral_contrast': features['spectral_contrast'],
            'mfcc_means': ','.join(map(str, features['mfcc_means'])),
            'last_analyzed': int(time.time())
        }
        
        # Save to database
        if existing_track:
            self.db.update_track(track_data)
            return 'updated'
        else:
            self.db.add_track(track_data)
            return 'new'
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""
    
    def _extract_metadata(self, audio_file: Path) -> Dict:
        """
        Extract basic metadata from audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Dictionary with metadata
        """
        # For now, extract from filename
        # In a full implementation, you'd use mutagen or similar library
        filename = audio_file.stem
        
        # Try to parse "Artist - Title" format
        if ' - ' in filename:
            parts = filename.split(' - ', 1)
            artist = parts[0].strip()
            title = parts[1].strip()
        else:
            artist = "Unknown Artist"
            title = filename
        
        return {
            'title': title,
            'artist': artist,
            'album': audio_file.parent.name  # Use directory name as album
        }
    
    def get_library_stats(self) -> Dict:
        """Get statistics about the analyzed music library."""
        return self.db.get_library_stats()
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tracks by title or artist."""
        return self.db.search_tracks(query, limit)
