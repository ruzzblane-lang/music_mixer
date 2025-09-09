"""
Streaming Integration

Integrates the streaming pipeline with the existing AI Music Mixer system.
Provides seamless connection between real-time analysis and the main application.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .streaming_pipeline import StreamingPipeline
from .audio_device_interface import AudioDeviceInterface, AudioDeviceType
from .music_file_manager import MusicFileManager

# Optional imports for existing components
try:
    from features.database import MusicDatabase
    from recommend.engine import RecommendationEngine
    from mix.engine import MixingEngine
    EXISTING_COMPONENTS_AVAILABLE = True
except ImportError:
    # Mock classes for testing
    class MusicDatabase:
        def __init__(self, db_path): pass
        def get_all_tracks(self, limit=50): return []
    
    class RecommendationEngine:
        def __init__(self, db_path): pass
        def get_recommendations(self, track, count=5): return []
        def add_feedback(self, current, recommended, accepted, score): pass
    
    class MixingEngine:
        def __init__(self): pass
        def get_track_info(self, path): return {'duration': 180.0}
    
    EXISTING_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamingIntegration:
    """
    Integrates streaming pipeline with the main AI Music Mixer system.
    
    Provides seamless connection between real-time analysis
    and the existing recommendation and mixing systems.
    """
    
    def __init__(self, 
                 db_path: str = "data/music_library.db",
                 music_dir: str = "music_files",
                 sample_rate: int = 22050,
                 enable_predictions: bool = True,
                 enable_learning: bool = True):
        """
        Initialize streaming integration.
        
        Args:
            db_path: Path to the music database
            music_dir: Path to the music files directory
            sample_rate: Audio sample rate
            enable_predictions: Enable predictive analysis
            enable_learning: Enable adaptive learning
        """
        self.db_path = db_path
        self.music_dir = music_dir
        self.sample_rate = sample_rate
        self.enable_predictions = enable_predictions
        self.enable_learning = enable_learning
        
        # Initialize components
        self.streaming_pipeline = StreamingPipeline(
            sample_rate=sample_rate,
            enable_predictions=enable_predictions,
            enable_learning=enable_learning
        )
        
        self.device_interface = AudioDeviceInterface()
        self.music_file_manager = MusicFileManager(music_dir)
        self.database = MusicDatabase(db_path)
        self.recommendation_engine = RecommendationEngine(db_path)
        self.mixing_engine = MixingEngine()
        
        # Log component availability
        if EXISTING_COMPONENTS_AVAILABLE:
            logger.info("Existing components (database, recommend, mix) available")
        else:
            logger.info("Using mock components for testing")
        
        # Integration state
        self.is_integrated = False
        self.current_track = None
        self.current_features = {}
        self.recommendation_history = []
        self.mix_session_active = False
        
        # Real-time data
        self.real_time_features = {}
        self.real_time_predictions = {}
        self.real_time_recommendations = []
        
        # Callbacks
        self.callbacks = {
            'on_track_analyzed': [],
            'on_recommendations_updated': [],
            'on_mix_ready': [],
            'on_integration_error': []
        }
        
        # Setup integration
        self._setup_integration()
        
        logger.info("StreamingIntegration initialized")
    
    def _setup_integration(self):
        """Setup integration between components."""
        # Connect streaming pipeline callbacks
        self.streaming_pipeline.add_callback('on_features_extracted', self._on_features_extracted)
        self.streaming_pipeline.add_callback('on_prediction_made', self._on_prediction_made)
        self.streaming_pipeline.add_callback('on_model_updated', self._on_model_updated)
        self.streaming_pipeline.add_callback('on_pipeline_error', self._on_pipeline_error)
    
    def start_integration(self) -> bool:
        """
        Start the streaming integration.
        
        Returns:
            True if integration started successfully
        """
        try:
            if self.is_integrated:
                logger.warning("Integration already active")
                return False
            
            # Start streaming pipeline
            if not self.streaming_pipeline.start_pipeline():
                logger.error("Failed to start streaming pipeline")
                return False
            
            self.is_integrated = True
            logger.info("Streaming integration started")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting integration: {e}")
            self._trigger_callbacks('on_integration_error', {'error': str(e)})
            return False
    
    def stop_integration(self):
        """Stop the streaming integration."""
        try:
            if not self.is_integrated:
                return
            
            # Stop streaming pipeline
            self.streaming_pipeline.stop_pipeline()
            
            self.is_integrated = False
            logger.info("Streaming integration stopped")
            
        except Exception as e:
            logger.error(f"Error stopping integration: {e}")
    
    def _on_features_extracted(self, features: Dict):
        """Handle features extracted from streaming pipeline."""
        try:
            # Update real-time features
            self.real_time_features.update(features)
            
            # Update current features for recommendations
            self.current_features.update(features)
            
            # Trigger track analysis if we have enough features
            if self._has_sufficient_features():
                self._analyze_current_track()
            
            # Update recommendations if we have a current track
            if self.current_track:
                self._update_recommendations()
            
        except Exception as e:
            logger.error(f"Error handling features: {e}")
    
    def _on_prediction_made(self, predictions: Dict):
        """Handle predictions made by streaming pipeline."""
        try:
            # Update real-time predictions
            self.real_time_predictions.update(predictions)
            
            # Use predictions for enhanced recommendations
            if self.current_track:
                self._update_recommendations_with_predictions()
            
        except Exception as e:
            logger.error(f"Error handling predictions: {e}")
    
    def _on_model_updated(self, model_data: Dict):
        """Handle model updates from streaming pipeline."""
        try:
            # Models have been updated with new data
            logger.debug("Models updated with streaming data")
            
            # Refresh recommendations with updated models
            if self.current_track:
                self._update_recommendations()
            
        except Exception as e:
            logger.error(f"Error handling model updates: {e}")
    
    def _on_pipeline_error(self, error_data: Dict):
        """Handle pipeline errors."""
        logger.error(f"Pipeline error: {error_data}")
        self._trigger_callbacks('on_integration_error', error_data)
    
    def _has_sufficient_features(self) -> bool:
        """Check if we have sufficient features for track analysis."""
        required_features = ['rms_energy', 'spectral_centroid', 'zero_crossing_rate']
        return all(feature in self.real_time_features for feature in required_features)
    
    def _analyze_current_track(self):
        """Analyze the current track based on real-time features."""
        try:
            # Create a mock track entry for real-time analysis
            track_data = {
                'title': 'Real-time Track',
                'artist': 'Live Audio',
                'tempo': self.real_time_features.get('tempo', 120.0),
                'key': self.real_time_features.get('key', 'C'),
                'mode': self.real_time_features.get('mode', 'major'),
                'rms_energy': self.real_time_features.get('rms_energy', 0.5),
                'brightness': self.real_time_features.get('spectral_centroid', 2000.0),
                'features': self.real_time_features
            }
            
            self.current_track = track_data
            
            # Trigger callbacks
            self._trigger_callbacks('on_track_analyzed', track_data)
            
            logger.debug("Current track analyzed with real-time features")
            
        except Exception as e:
            logger.error(f"Error analyzing current track: {e}")
    
    def _update_recommendations(self):
        """Update recommendations based on current features."""
        try:
            if not self.current_track:
                return
            
            # Get recommendations from the engine
            recommendations = self.recommendation_engine.get_recommendations(
                self.current_track['title'], count=5
            )
            
            # Also get recommendations from streaming pipeline
            if self.streaming_pipeline.adaptive_models:
                available_tracks = self._get_available_tracks()
                streaming_recommendations = self.streaming_pipeline.get_recommendations(
                    self.current_features, available_tracks
                )
                
                # Combine recommendations
                recommendations = self._combine_recommendations(recommendations, streaming_recommendations)
            
            self.real_time_recommendations = recommendations
            self.recommendation_history.append({
                'timestamp': time.time(),
                'track': self.current_track,
                'recommendations': recommendations
            })
            
            # Trigger callbacks
            self._trigger_callbacks('on_recommendations_updated', recommendations)
            
        except Exception as e:
            logger.error(f"Error updating recommendations: {e}")
    
    def _update_recommendations_with_predictions(self):
        """Update recommendations using predictions."""
        try:
            if not self.real_time_predictions:
                return
            
            # Use predictions to enhance recommendations
            enhanced_features = self.current_features.copy()
            
            # Add predicted features
            if 'beat_prediction' in self.real_time_predictions:
                beat_pred = self.real_time_predictions['beat_prediction']
                enhanced_features['predicted_beat_time'] = beat_pred['predicted_time']
                enhanced_features['beat_confidence'] = beat_pred['confidence']
            
            if 'energy_prediction' in self.real_time_predictions:
                energy_pred = self.real_time_predictions['energy_prediction']
                enhanced_features['predicted_energy_transition'] = energy_pred['transition_type']
                enhanced_features['energy_confidence'] = energy_pred['confidence']
            
            # Update recommendations with enhanced features
            available_tracks = self._get_available_tracks()
            enhanced_recommendations = self.streaming_pipeline.get_recommendations(
                enhanced_features, available_tracks
            )
            
            if enhanced_recommendations:
                self.real_time_recommendations = enhanced_recommendations
                self._trigger_callbacks('on_recommendations_updated', enhanced_recommendations)
            
        except Exception as e:
            logger.error(f"Error updating recommendations with predictions: {e}")
    
    def _get_available_tracks(self) -> List[Dict]:
        """Get available tracks from music files and database."""
        try:
            # Get tracks from music file manager
            music_files = self.music_file_manager.get_music_files()
            
            # Convert to track format
            tracks = []
            for file_metadata in music_files:
                track = {
                    'id': hash(file_metadata['file_path']),  # Simple ID generation
                    'title': file_metadata.get('title', file_metadata['filename']),
                    'artist': file_metadata.get('artist', 'Unknown'),
                    'file_path': file_metadata['file_path'],
                    'tempo': file_metadata.get('tempo', 120.0),
                    'key': file_metadata.get('key', 'C'),
                    'mode': file_metadata.get('mode', 'major'),
                    'duration': file_metadata.get('duration', 0),
                    'rms_energy': file_metadata.get('rms_energy', 0.5),
                    'spectral_centroid': file_metadata.get('spectral_centroid', 2000.0),
                    'metadata': file_metadata
                }
                tracks.append(track)
            
            # Also get tracks from database
            try:
                db_tracks = self.database.get_all_tracks(limit=50)
                tracks.extend(db_tracks)
            except Exception as e:
                logger.warning(f"Could not get tracks from database: {e}")
            
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting available tracks: {e}")
            return []
    
    def _combine_recommendations(self, 
                               engine_recommendations: List[tuple], 
                               streaming_recommendations: List[tuple]) -> List[tuple]:
        """Combine recommendations from different sources."""
        try:
            # Simple combination - prioritize streaming recommendations
            combined = []
            
            # Add streaming recommendations first
            for track, score in streaming_recommendations:
                combined.append((track, score * 1.1))  # Boost streaming recommendations
            
            # Add engine recommendations
            for track, score in engine_recommendations:
                # Check if track already exists
                if not any(t['id'] == track['id'] for t, s in combined):
                    combined.append((track, score))
            
            # Sort by score
            combined.sort(key=lambda x: x[1], reverse=True)
            
            return combined[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            return engine_recommendations
    
    def start_mix_session(self, track_path: str) -> bool:
        """
        Start a mix session with a specific track.
        
        Args:
            track_path: Path to the track file
            
        Returns:
            True if mix session started successfully
        """
        try:
            if self.mix_session_active:
                logger.warning("Mix session already active")
                return False
            
            # Load track information
            track_info = self.mixing_engine.get_track_info(track_path)
            if not track_info:
                logger.error(f"Could not load track info for {track_path}")
                return False
            
            # Set current track
            self.current_track = {
                'file_path': track_path,
                'title': Path(track_path).stem,
                'artist': 'Unknown',
                'duration': track_info['duration'],
                'info': track_info
            }
            
            self.mix_session_active = True
            
            # Trigger callbacks
            self._trigger_callbacks('on_mix_ready', {
                'track': self.current_track,
                'session_active': True
            })
            
            logger.info(f"Mix session started with {track_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting mix session: {e}")
            return False
    
    def stop_mix_session(self):
        """Stop the current mix session."""
        try:
            self.mix_session_active = False
            self.current_track = None
            
            logger.info("Mix session stopped")
            
        except Exception as e:
            logger.error(f"Error stopping mix session: {e}")
    
    def get_integration_status(self) -> Dict:
        """Get current integration status."""
        return {
            'is_integrated': self.is_integrated,
            'mix_session_active': self.mix_session_active,
            'current_track': self.current_track,
            'real_time_features': self.real_time_features,
            'real_time_predictions': self.real_time_predictions,
            'recommendations_count': len(self.real_time_recommendations),
            'recommendation_history_count': len(self.recommendation_history),
            'pipeline_status': self.streaming_pipeline.get_pipeline_status(),
            'device_info': self.device_interface.get_device_info(),
            'music_files_status': self.music_file_manager.get_performance_metrics()
        }
    
    def get_real_time_recommendations(self) -> List[tuple]:
        """Get current real-time recommendations."""
        return self.real_time_recommendations.copy()
    
    def get_real_time_features(self) -> Dict:
        """Get current real-time features."""
        return self.real_time_features.copy()
    
    def get_real_time_predictions(self) -> Dict:
        """Get current real-time predictions."""
        return self.real_time_predictions.copy()
    
    def add_user_feedback(self, current_track: str, recommended_track: str, accepted: bool, score: Optional[float] = None):
        """Add user feedback for learning."""
        try:
            # Add feedback to streaming pipeline
            self.streaming_pipeline.add_user_feedback(current_track, recommended_track, accepted, score)
            
            # Add feedback to recommendation engine
            self.recommendation_engine.add_feedback(current_track, recommended_track, accepted, score)
            
            logger.info(f"User feedback added: {current_track} -> {recommended_track}, accepted: {accepted}")
            
        except Exception as e:
            logger.error(f"Error adding user feedback: {e}")
    
    def scan_music_files(self, force_rescan: bool = False) -> Dict[str, Any]:
        """
        Scan for music files in the music directory.
        
        Args:
            force_rescan: Force rescan even if files haven't changed
            
        Returns:
            Scan results dictionary
        """
        try:
            results = self.music_file_manager.scan_music_files(force_rescan)
            logger.info(f"Music file scan completed: {results['total_files']} files found")
            return results
        except Exception as e:
            logger.error(f"Error scanning music files: {e}")
            return {'error': str(e)}
    
    def get_music_files(self, directory: Optional[str] = None, format_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of music files with metadata.
        
        Args:
            directory: Filter by directory name
            format_filter: Filter by file format
            
        Returns:
            List of music file metadata dictionaries
        """
        return self.music_file_manager.get_music_files(directory, format_filter)
    
    def add_custom_metadata(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Add custom metadata to a music file.
        
        Args:
            file_path: Path to the music file
            metadata: Custom metadata to add
            
        Returns:
            True if successful
        """
        return self.music_file_manager.add_custom_metadata(file_path, metadata)
    
    def validate_music_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a music file.
        
        Args:
            file_path: Path to the music file
            
        Returns:
            Validation result dictionary
        """
        return self.music_file_manager.validate_file(file_path)
    
    def add_callback(self, event_name: str, callback: Callable):
        """Add a callback for an event."""
        if event_name in self.callbacks:
            self.callbacks[event_name].append(callback)
        else:
            logger.warning(f"Unknown event name: {event_name}")
    
    def remove_callback(self, event_name: str, callback: Callable):
        """Remove a callback for an event."""
        if event_name in self.callbacks and callback in self.callbacks[event_name]:
            self.callbacks[event_name].remove(callback)
    
    def _trigger_callbacks(self, event_name: str, data: Any):
        """Trigger callbacks for an event."""
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback {event_name}: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_integration()
