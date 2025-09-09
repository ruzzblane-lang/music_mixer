"""
Music Database Module

Handles SQLite database operations for storing and retrieving
music track information and features.
"""

import sqlite3
import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class MusicDatabase:
    """Handles database operations for music library."""
    
    def __init__(self, db_path: str = "data/music_library.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tracks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT,
                    title TEXT,
                    artist TEXT,
                    album TEXT,
                    duration REAL,
                    tempo REAL,
                    key TEXT,
                    mode TEXT,
                    rms_energy REAL,
                    brightness REAL,
                    spectral_rolloff REAL,
                    spectral_bandwidth REAL,
                    zero_crossing_rate REAL,
                    onset_strength REAL,
                    tempo_stability REAL,
                    spectral_contrast REAL,
                    mfcc_means TEXT,
                    last_analyzed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create user feedback table for learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_track_id INTEGER,
                    recommended_track_id INTEGER,
                    accepted BOOLEAN,
                    feedback_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (current_track_id) REFERENCES tracks (id),
                    FOREIGN KEY (recommended_track_id) REFERENCES tracks (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_title ON tracks(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_tempo ON tracks(tempo)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_key ON tracks(key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path)')
            
            conn.commit()
    
    def add_track(self, track_data: Dict) -> int:
        """
        Add a new track to the database.
        
        Args:
            track_data: Dictionary containing track information
            
        Returns:
            ID of the inserted track
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tracks (
                    file_path, file_hash, title, artist, album, duration,
                    tempo, key, mode, rms_energy, brightness, spectral_rolloff,
                    spectral_bandwidth, zero_crossing_rate, onset_strength,
                    tempo_stability, spectral_contrast, mfcc_means, last_analyzed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                track_data['file_path'],
                track_data['file_hash'],
                track_data['title'],
                track_data['artist'],
                track_data['album'],
                track_data['duration'],
                track_data['tempo'],
                track_data['key'],
                track_data['mode'],
                track_data['rms_energy'],
                track_data['brightness'],
                track_data['spectral_rolloff'],
                track_data['spectral_bandwidth'],
                track_data['zero_crossing_rate'],
                track_data['onset_strength'],
                track_data['tempo_stability'],
                track_data['spectral_contrast'],
                track_data['mfcc_means'],
                track_data['last_analyzed']
            ))
            
            track_id = cursor.lastrowid
            conn.commit()
            return track_id
    
    def update_track(self, track_data: Dict) -> bool:
        """
        Update an existing track in the database.
        
        Args:
            track_data: Dictionary containing updated track information
            
        Returns:
            True if update was successful
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE tracks SET
                    file_hash = ?, title = ?, artist = ?, album = ?, duration = ?,
                    tempo = ?, key = ?, mode = ?, rms_energy = ?, brightness = ?,
                    spectral_rolloff = ?, spectral_bandwidth = ?, zero_crossing_rate = ?,
                    onset_strength = ?, tempo_stability = ?, spectral_contrast = ?,
                    mfcc_means = ?, last_analyzed = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
            ''', (
                track_data['file_hash'],
                track_data['title'],
                track_data['artist'],
                track_data['album'],
                track_data['duration'],
                track_data['tempo'],
                track_data['key'],
                track_data['mode'],
                track_data['rms_energy'],
                track_data['brightness'],
                track_data['spectral_rolloff'],
                track_data['spectral_bandwidth'],
                track_data['zero_crossing_rate'],
                track_data['onset_strength'],
                track_data['tempo_stability'],
                track_data['spectral_contrast'],
                track_data['mfcc_means'],
                track_data['last_analyzed'],
                track_data['file_path']
            ))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def _parse_track_data(self, row_dict: Dict) -> Dict:
        """
        Parse track data from database, converting JSON strings back to objects.
        
        Args:
            row_dict: Dictionary from database row
            
        Returns:
            Parsed track data
        """
        try:
            # Parse MFCC means from JSON
            if 'mfcc_means' in row_dict and row_dict['mfcc_means']:
                if isinstance(row_dict['mfcc_means'], str):
                    row_dict['mfcc_means'] = json.loads(row_dict['mfcc_means'])
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing MFCC data: {e}")
            row_dict['mfcc_means'] = []
        
        return row_dict
    
    def get_track_by_path(self, file_path: str) -> Optional[Dict]:
        """
        Get track information by file path.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with track information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tracks WHERE file_path = ?', (file_path,))
            row = cursor.fetchone()
            
            if row:
                return self._parse_track_data(dict(row))
            return None
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        """
        Get track information by ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            Dictionary with track information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tracks WHERE id = ?', (track_id,))
            row = cursor.fetchone()
            
            if row:
                return self._parse_track_data(dict(row))
            return None
    
    def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for tracks by title or artist.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching tracks
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            search_pattern = f"%{query}%"
            cursor.execute('''
                SELECT * FROM tracks 
                WHERE title LIKE ? OR artist LIKE ? 
                ORDER BY title
                LIMIT ?
            ''', (search_pattern, search_pattern, limit))
            
            rows = cursor.fetchall()
            return [self._parse_track_data(dict(row)) for row in rows]
    
    def get_tracks_by_tempo_range(self, min_tempo: float, max_tempo: float) -> List[Dict]:
        """
        Get tracks within a tempo range.
        
        Args:
            min_tempo: Minimum tempo
            max_tempo: Maximum tempo
            
        Returns:
            List of tracks in tempo range
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM tracks 
                WHERE tempo BETWEEN ? AND ?
                ORDER BY tempo
            ''', (min_tempo, max_tempo))
            
            rows = cursor.fetchall()
            return [self._parse_track_data(dict(row)) for row in rows]
    
    def get_tracks_by_key(self, key: str) -> List[Dict]:
        """
        Get tracks with a specific key.
        
        Args:
            key: Musical key (e.g., 'C', 'D#', 'F')
            
        Returns:
            List of tracks with the specified key
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tracks WHERE key = ?', (key,))
            rows = cursor.fetchall()
            return [self._parse_track_data(dict(row)) for row in rows]
    
    def get_all_tracks(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all tracks from the database.
        
        Args:
            limit: Maximum number of tracks to return
            
        Returns:
            List of all tracks
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if limit:
                cursor.execute('SELECT * FROM tracks ORDER BY title LIMIT ?', (limit,))
            else:
                cursor.execute('SELECT * FROM tracks ORDER BY title')
            
            rows = cursor.fetchall()
            return [self._parse_track_data(dict(row)) for row in rows]
    
    def get_library_stats(self) -> Dict:
        """
        Get statistics about the music library.
        
        Returns:
            Dictionary with library statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total tracks
            cursor.execute('SELECT COUNT(*) FROM tracks')
            total_tracks = cursor.fetchone()[0]
            
            # Unique artists
            cursor.execute('SELECT COUNT(DISTINCT artist) FROM tracks')
            unique_artists = cursor.fetchone()[0]
            
            # Unique albums
            cursor.execute('SELECT COUNT(DISTINCT album) FROM tracks')
            unique_albums = cursor.fetchone()[0]
            
            # Tempo range
            cursor.execute('SELECT MIN(tempo), MAX(tempo), AVG(tempo) FROM tracks')
            tempo_stats = cursor.fetchone()
            
            # Key distribution
            cursor.execute('SELECT key, COUNT(*) FROM tracks GROUP BY key ORDER BY COUNT(*) DESC')
            key_distribution = dict(cursor.fetchall())
            
            return {
                'total_tracks': total_tracks,
                'unique_artists': unique_artists,
                'unique_albums': unique_albums,
                'tempo_min': tempo_stats[0] if tempo_stats[0] else 0,
                'tempo_max': tempo_stats[1] if tempo_stats[1] else 0,
                'tempo_avg': tempo_stats[2] if tempo_stats[2] else 0,
                'key_distribution': key_distribution
            }
    
    def add_user_feedback(self, current_track_id: int, recommended_track_id: int, 
                         accepted: bool, feedback_score: Optional[float] = None):
        """
        Add user feedback for learning.
        
        Args:
            current_track_id: ID of the current track
            recommended_track_id: ID of the recommended track
            accepted: Whether the recommendation was accepted
            feedback_score: Optional feedback score (0-1)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_feedback (current_track_id, recommended_track_id, accepted, feedback_score)
                VALUES (?, ?, ?, ?)
            ''', (current_track_id, recommended_track_id, accepted, feedback_score))
            
            conn.commit()
    
    def get_user_feedback(self, limit: int = 100) -> List[Dict]:
        """
        Get user feedback data for learning.
        
        Args:
            limit: Maximum number of feedback records to return
            
        Returns:
            List of feedback records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM user_feedback 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            return [self._parse_track_data(dict(row)) for row in rows]
