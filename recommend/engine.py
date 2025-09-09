"""
Recommendation Engine

Provides intelligent track recommendations based on audio features,
compatibility rules, and machine learning models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging

from features.database import MusicDatabase
from features.extractor import AudioFeatureExtractor

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Generates track recommendations using multiple approaches."""
    
    def __init__(self, db_path: str = "data/music_library.db"):
        """
        Initialize the recommendation engine.
        
        Args:
            db_path: Path to the music database
        """
        self.db = MusicDatabase(db_path)
        self.extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.ml_model = None
        self.feature_matrix = None
        self.track_ids = None
        self._build_ml_model()
    
    def _build_ml_model(self):
        """Build the machine learning model for recommendations."""
        try:
            # Get all tracks from database
            tracks = self.db.get_all_tracks()
            
            if len(tracks) < 2:
                logger.warning("Not enough tracks for ML model. Using rule-based recommendations only.")
                return
            
            # Extract feature vectors
            feature_vectors = []
            track_ids = []
            
            for track in tracks:
                try:
                    # Convert track data to feature vector
                    features = self._track_to_features(track)
                    feature_vector = self.extractor.get_feature_vector(features)
                    
                    feature_vectors.append(feature_vector)
                    track_ids.append(track['id'])
                    
                except Exception as e:
                    logger.warning(f"Could not process track {track['id']}: {e}")
                    continue
            
            if len(feature_vectors) < 2:
                logger.warning("Not enough valid feature vectors for ML model.")
                return
            
            # Normalize features
            self.feature_matrix = self.scaler.fit_transform(feature_vectors)
            self.track_ids = np.array(track_ids)
            
            # Build k-NN model
            n_neighbors = min(10, len(feature_vectors) - 1)
            self.ml_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='cosine',
                algorithm='auto'
            )
            self.ml_model.fit(self.feature_matrix)
            
            logger.info(f"ML model built with {len(feature_vectors)} tracks")
            
        except Exception as e:
            logger.error(f"Error building ML model: {e}")
            self.ml_model = None
    
    def _track_to_features(self, track: Dict) -> Dict:
        """Convert database track to features dictionary."""
        # Parse MFCC means from string
        mfcc_means = []
        if track['mfcc_means']:
            try:
                mfcc_means = [float(x) for x in track['mfcc_means'].split(',')]
            except:
                mfcc_means = [0.0] * 13
        
        return {
            'tempo': track['tempo'],
            'key': track['key'],
            'mode': track['mode'],
            'rms_energy': track['rms_energy'],
            'brightness': track['brightness'],
            'spectral_rolloff': track['spectral_rolloff'],
            'spectral_bandwidth': track['spectral_bandwidth'],
            'zero_crossing_rate': track['zero_crossing_rate'],
            'onset_strength': track['onset_strength'],
            'tempo_stability': track['tempo_stability'],
            'spectral_contrast': track['spectral_contrast'],
            'mfcc_means': mfcc_means
        }
    
    def get_recommendations(self, track_name: str, count: int = 5, 
                          use_ml: bool = True) -> List[Tuple[Dict, float]]:
        """
        Get recommendations for a given track.
        
        Args:
            track_name: Name of the track to get recommendations for
            count: Number of recommendations to return
            use_ml: Whether to use ML model or rule-based recommendations
            
        Returns:
            List of (track, score) tuples
        """
        # Find the track
        tracks = self.db.search_tracks(track_name, limit=1)
        if not tracks:
            logger.warning(f"Track not found: {track_name}")
            return []
        
        current_track = tracks[0]
        
        if use_ml and self.ml_model is not None:
            return self._get_ml_recommendations(current_track, count)
        else:
            return self._get_rule_based_recommendations(current_track, count)
    
    def _get_ml_recommendations(self, current_track: Dict, count: int) -> List[Tuple[Dict, float]]:
        """Get recommendations using machine learning model."""
        try:
            # Convert current track to feature vector
            features = self._track_to_features(current_track)
            feature_vector = self.extractor.get_feature_vector(features)
            
            # Normalize the feature vector
            normalized_vector = self.scaler.transform([feature_vector])
            
            # Find nearest neighbors
            distances, indices = self.ml_model.kneighbors(normalized_vector)
            
            # Get recommended tracks
            recommendations = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if i == 0:  # Skip the first one (it's the current track itself)
                    continue
                
                track_id = self.track_ids[idx]
                track = self.db.get_track_by_id(track_id)
                
                if track and track['id'] != current_track['id']:
                    # Convert distance to similarity score (0-1)
                    score = 1.0 - distance
                    recommendations.append((track, score))
            
            return recommendations[:count]
            
        except Exception as e:
            logger.error(f"Error in ML recommendations: {e}")
            return self._get_rule_based_recommendations(current_track, count)
    
    def _get_rule_based_recommendations(self, current_track: Dict, count: int) -> List[Tuple[Dict, float]]:
        """Get recommendations using rule-based approach."""
        recommendations = []
        
        # Get all other tracks
        all_tracks = self.db.get_all_tracks()
        
        for track in all_tracks:
            if track['id'] == current_track['id']:
                continue
            
            # Calculate compatibility score
            score = self._calculate_compatibility_score(current_track, track)
            
            if score > 0.3:  # Only include tracks with reasonable compatibility
                recommendations.append((track, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:count]
    
    def _calculate_compatibility_score(self, track1: Dict, track2: Dict) -> float:
        """
        Calculate compatibility score between two tracks.
        
        Args:
            track1: First track
            track2: Second track
            
        Returns:
            Compatibility score (0-1)
        """
        score = 0.0
        
        # Tempo compatibility (40% weight)
        tempo_diff = abs(track1['tempo'] - track2['tempo'])
        tempo_score = max(0, 1.0 - (tempo_diff / 50.0))  # 50 BPM tolerance
        score += tempo_score * 0.4
        
        # Key compatibility (30% weight)
        key_score = self._get_key_compatibility(track1['key'], track2['key'])
        score += key_score * 0.3
        
        # Energy compatibility (20% weight)
        energy_diff = abs(track1['rms_energy'] - track2['rms_energy'])
        energy_score = max(0, 1.0 - (energy_diff / 0.5))  # 0.5 energy tolerance
        score += energy_score * 0.2
        
        # Brightness compatibility (10% weight)
        brightness_diff = abs(track1['brightness'] - track2['brightness'])
        brightness_score = max(0, 1.0 - (brightness_diff / 2000.0))  # 2000 Hz tolerance
        score += brightness_score * 0.1
        
        return min(1.0, score)
    
    def _get_key_compatibility(self, key1: str, key2: str) -> float:
        """
        Calculate key compatibility between two tracks.
        
        Args:
            key1: First key
            key2: Second key
            
        Returns:
            Key compatibility score (0-1)
        """
        # Circle of fifths for key relationships
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        try:
            idx1 = circle_of_fifths.index(key1)
            idx2 = circle_of_fifths.index(key2)
            
            # Calculate distance in circle of fifths
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            
            # Perfect match
            if distance == 0:
                return 1.0
            # Relative major/minor or perfect fifth
            elif distance == 1 or distance == 11:
                return 0.8
            # Subdominant/dominant
            elif distance == 2 or distance == 10:
                return 0.6
            # Other relationships
            elif distance <= 3:
                return 0.4
            else:
                return 0.2
                
        except ValueError:
            # If key not found in circle of fifths, return neutral score
            return 0.5
    
    def get_tempo_based_recommendations(self, tempo: float, tolerance: float = 10.0, 
                                      count: int = 10) -> List[Dict]:
        """
        Get recommendations based on tempo matching.
        
        Args:
            tempo: Target tempo
            tolerance: Tempo tolerance in BPM
            count: Number of recommendations
            
        Returns:
            List of recommended tracks
        """
        min_tempo = tempo - tolerance
        max_tempo = tempo + tolerance
        
        tracks = self.db.get_tracks_by_tempo_range(min_tempo, max_tempo)
        
        # Sort by tempo proximity
        tracks.sort(key=lambda x: abs(x['tempo'] - tempo))
        
        return tracks[:count]
    
    def get_key_based_recommendations(self, key: str, count: int = 10) -> List[Dict]:
        """
        Get recommendations based on key matching.
        
        Args:
            key: Target key
            count: Number of recommendations
            
        Returns:
            List of recommended tracks
        """
        tracks = self.db.get_tracks_by_key(key)
        
        # Sort by energy level for variety
        tracks.sort(key=lambda x: x['rms_energy'], reverse=True)
        
        return tracks[:count]
    
    def retrain_model(self):
        """Retrain the ML model with updated data."""
        logger.info("Retraining ML model...")
        self._build_ml_model()
    
    def add_feedback(self, current_track_name: str, recommended_track_name: str, 
                    accepted: bool, feedback_score: Optional[float] = None):
        """
        Add user feedback for learning.
        
        Args:
            current_track_name: Name of the current track
            recommended_track_name: Name of the recommended track
            accepted: Whether the recommendation was accepted
            feedback_score: Optional feedback score (0-1)
        """
        try:
            # Find tracks
            current_tracks = self.db.search_tracks(current_track_name, limit=1)
            recommended_tracks = self.db.search_tracks(recommended_track_name, limit=1)
            
            if current_tracks and recommended_tracks:
                current_track = current_tracks[0]
                recommended_track = recommended_tracks[0]
                
                self.db.add_user_feedback(
                    current_track['id'],
                    recommended_track['id'],
                    accepted,
                    feedback_score
                )
                
                logger.info(f"Added feedback: {accepted} for {recommended_track_name}")
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
