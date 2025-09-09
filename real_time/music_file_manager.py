"""
Music File Manager

Manages music files in the dedicated music_files directory.
Handles file discovery, validation, and integration with the streaming system.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import mimetypes

# Audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MusicFileManager:
    """
    Manages music files in the dedicated music_files directory.
    
    Handles file discovery, validation, metadata extraction,
    and integration with the streaming system.
    """
    
    def __init__(self, music_dir: str = "music_files"):
        """
        Initialize the music file manager.
        
        Args:
            music_dir: Path to the music files directory
        """
        self.music_dir = Path(music_dir)
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
        
        # Directory structure
        self.directories = {
            'test_tracks': self.music_dir / 'test_tracks',
            'sample_mixes': self.music_dir / 'sample_mixes',
            'reference_tracks': self.music_dir / 'reference_tracks',
            'user_uploads': self.music_dir / 'user_uploads'
        }
        
        # File tracking
        self.known_files = {}  # file_path -> metadata
        self.file_hashes = {}  # file_path -> hash
        
        # Performance tracking
        self.performance_metrics = {
            'files_scanned': 0,
            'files_processed': 0,
            'files_failed': 0,
            'last_scan_time': None,
            'scan_duration': 0
        }
        
        # Setup directories
        self._setup_directories()
        
        logger.info(f"MusicFileManager initialized with directory: {self.music_dir}")
    
    def _setup_directories(self):
        """Create directory structure if it doesn't exist."""
        try:
            # Create main directory
            self.music_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            for dir_name, dir_path in self.directories.items():
                dir_path.mkdir(exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            
            # Create .gitkeep files if they don't exist
            for dir_path in self.directories.values():
                gitkeep_file = dir_path / '.gitkeep'
                if not gitkeep_file.exists():
                    gitkeep_file.write_text(f"# This file ensures the {dir_path.name} directory is tracked by git\n")
            
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
    
    def scan_music_files(self, force_rescan: bool = False) -> Dict[str, Any]:
        """
        Scan for music files in all directories.
        
        Args:
            force_rescan: Force rescan even if files haven't changed
            
        Returns:
            Scan results dictionary
        """
        start_time = datetime.now()
        logger.info("Starting music file scan...")
        
        results = {
            'new_files': [],
            'updated_files': [],
            'removed_files': [],
            'failed_files': [],
            'total_files': 0,
            'scan_duration': 0
        }
        
        try:
            # Get all music files
            all_files = self._discover_music_files()
            results['total_files'] = len(all_files)
            
            # Process each file
            for file_path in all_files:
                try:
                    file_result = self._process_music_file(file_path, force_rescan)
                    
                    if file_result['status'] == 'new':
                        results['new_files'].append(file_result)
                    elif file_result['status'] == 'updated':
                        results['updated_files'].append(file_result)
                    elif file_result['status'] == 'failed':
                        results['failed_files'].append(file_result)
                    
                    self.performance_metrics['files_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    results['failed_files'].append({
                        'file_path': str(file_path),
                        'error': str(e),
                        'status': 'failed'
                    })
                    self.performance_metrics['files_failed'] += 1
            
            # Check for removed files
            current_files = {str(f) for f in all_files}
            known_files = set(self.known_files.keys())
            removed_files = known_files - current_files
            
            for removed_file in removed_files:
                results['removed_files'].append({
                    'file_path': removed_file,
                    'status': 'removed'
                })
                # Clean up from tracking
                if removed_file in self.known_files:
                    del self.known_files[removed_file]
                if removed_file in self.file_hashes:
                    del self.file_hashes[removed_file]
            
            # Update performance metrics
            end_time = datetime.now()
            scan_duration = (end_time - start_time).total_seconds()
            results['scan_duration'] = scan_duration
            
            self.performance_metrics['files_scanned'] = len(all_files)
            self.performance_metrics['last_scan_time'] = end_time
            self.performance_metrics['scan_duration'] = scan_duration
            
            logger.info(f"Music file scan completed: {len(all_files)} files, {scan_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during music file scan: {e}")
            results['error'] = str(e)
        
        return results
    
    def _discover_music_files(self) -> List[Path]:
        """Discover all music files in the directory structure."""
        music_files = []
        
        for dir_name, dir_path in self.directories.items():
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                        music_files.append(file_path)
        
        return music_files
    
    def _process_music_file(self, file_path: Path, force_rescan: bool = False) -> Dict[str, Any]:
        """
        Process a single music file.
        
        Args:
            file_path: Path to the music file
            force_rescan: Force rescan even if file hasn't changed
            
        Returns:
            Processing result dictionary
        """
        file_path_str = str(file_path)
        
        # Check if file has changed
        current_hash = self._calculate_file_hash(file_path)
        known_hash = self.file_hashes.get(file_path_str)
        
        if not force_rescan and current_hash == known_hash:
            # File hasn't changed, return existing metadata
            return {
                'file_path': file_path_str,
                'status': 'unchanged',
                'metadata': self.known_files.get(file_path_str, {})
            }
        
        # Process the file
        metadata = self._extract_file_metadata(file_path)
        
        # Update tracking
        self.file_hashes[file_path_str] = current_hash
        self.known_files[file_path_str] = metadata
        
        # Determine status
        if known_hash is None:
            status = 'new'
        else:
            status = 'updated'
        
        return {
            'file_path': file_path_str,
            'status': status,
            'metadata': metadata,
            'hash': current_hash
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of a file for change detection."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a music file.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'file_path': str(file_path),
            'filename': file_path.name,
            'directory': file_path.parent.name,
            'file_size': file_path.stat().st_size,
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'format': file_path.suffix.lower(),
            'mime_type': mimetypes.guess_type(str(file_path))[0]
        }
        
        # Try to extract audio metadata
        try:
            if LIBROSA_AVAILABLE:
                audio_metadata = self._extract_librosa_metadata(file_path)
                metadata.update(audio_metadata)
            elif PYDUB_AVAILABLE:
                audio_metadata = self._extract_pydub_metadata(file_path)
                metadata.update(audio_metadata)
            else:
                logger.warning("No audio processing library available for metadata extraction")
                
        except Exception as e:
            logger.error(f"Error extracting audio metadata from {file_path}: {e}")
            metadata['audio_extraction_error'] = str(e)
        
        # Try to load custom metadata file
        try:
            metadata_file = file_path.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    custom_metadata = json.load(f)
                    metadata.update(custom_metadata)
        except Exception as e:
            logger.debug(f"No custom metadata file for {file_path}: {e}")
        
        return metadata
    
    def _extract_librosa_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using librosa."""
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=None, duration=30)  # Load first 30 seconds
            
            # Extract basic features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            rms_energy = librosa.feature.rms(y=y).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
            
            # Estimate key (simplified)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = chroma.mean(axis=1)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            estimated_key = key_names[key_profile.argmax()]
            
            return {
                'duration': len(y) / sr,
                'sample_rate': sr,
                'tempo': float(tempo),
                'key': estimated_key,
                'spectral_centroid': float(spectral_centroid),
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zero_crossing_rate),
                'extraction_method': 'librosa'
            }
            
        except Exception as e:
            logger.error(f"Error extracting librosa metadata: {e}")
            return {'extraction_error': str(e)}
    
    def _extract_pydub_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using pydub."""
        try:
            audio = AudioSegment.from_file(str(file_path))
            
            return {
                'duration': len(audio) / 1000.0,  # Convert to seconds
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'bit_depth': audio.sample_width * 8,
                'extraction_method': 'pydub'
            }
            
        except Exception as e:
            logger.error(f"Error extracting pydub metadata: {e}")
            return {'extraction_error': str(e)}
    
    def get_music_files(self, directory: Optional[str] = None, format_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of music files with metadata.
        
        Args:
            directory: Filter by directory name
            format_filter: Filter by file format
            
        Returns:
            List of music file metadata dictionaries
        """
        files = []
        
        for file_path, metadata in self.known_files.items():
            # Apply filters
            if directory and metadata.get('directory') != directory:
                continue
            
            if format_filter and metadata.get('format') != format_filter.lower():
                continue
            
            files.append(metadata)
        
        return files
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.known_files.get(file_path)
    
    def add_custom_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Add custom metadata to a file.
        
        Args:
            file_path: Path to the music file
            metadata: Custom metadata to add
            
        Returns:
            True if successful
        """
        try:
            metadata_file = Path(file_path).with_suffix('.json')
            
            # Load existing metadata if it exists
            existing_metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            
            # Update with new metadata
            existing_metadata.update(metadata)
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            # Update in-memory tracking
            if file_path in self.known_files:
                self.known_files[file_path].update(metadata)
            
            logger.info(f"Added custom metadata to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding custom metadata to {file_path}: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'files_scanned': self.performance_metrics['files_scanned'],
            'files_processed': self.performance_metrics['files_processed'],
            'files_failed': self.performance_metrics['files_failed'],
            'last_scan_time': self.performance_metrics['last_scan_time'].isoformat() if self.performance_metrics['last_scan_time'] else None,
            'scan_duration': self.performance_metrics['scan_duration'],
            'known_files_count': len(self.known_files),
            'supported_formats': list(self.supported_formats),
            'directories': {name: str(path) for name, path in self.directories.items()}
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a music file.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            Validation result dictionary
        """
        result = {
            'file_path': file_path,
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                result['errors'].append("File does not exist")
                return result
            
            # Check file format
            if file_path_obj.suffix.lower() not in self.supported_formats:
                result['errors'].append(f"Unsupported format: {file_path_obj.suffix}")
                return result
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            if file_size == 0:
                result['errors'].append("File is empty")
                return result
            
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                result['warnings'].append("File is very large (>100MB)")
            
            # Try to load the file
            if LIBROSA_AVAILABLE:
                try:
                    y, sr = librosa.load(str(file_path_obj), sr=None, duration=1)  # Load 1 second
                    if len(y) == 0:
                        result['errors'].append("File contains no audio data")
                        return result
                except Exception as e:
                    result['errors'].append(f"Error loading audio: {e}")
                    return result
            
            result['valid'] = True
            
        except Exception as e:
            result['errors'].append(f"Validation error: {e}")
        
        return result
