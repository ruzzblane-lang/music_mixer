"""
VLC Integration

Integrates VLC music player with the streaming system for real-time
audio analysis, recommendations, and mixing capabilities.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .vlc_music_player import VLCMusicPlayer, PlayerState
from .streaming_integration import StreamingIntegration
from .music_file_manager import MusicFileManager

logger = logging.getLogger(__name__)


class VLCIntegration:
    """
    Integrates VLC music player with the AI Music Mixer streaming system.
    
    Provides seamless music playback with real-time analysis, recommendations,
    and intelligent mixing capabilities.
    """
    
    def __init__(self, 
                 music_dir: str = "music_files",
                 db_path: str = "data/music_library.db",
                 sample_rate: int = 44100,
                 enable_streaming: bool = True,
                 enable_analysis: bool = True):
        """
        Initialize VLC integration.
        
        Args:
            music_dir: Path to music files directory
            db_path: Path to music database
            sample_rate: Audio sample rate
            enable_streaming: Enable streaming capabilities
            enable_analysis: Enable real-time analysis
        """
        self.music_dir = music_dir
        self.db_path = db_path
        self.sample_rate = sample_rate
        self.enable_streaming = enable_streaming
        self.enable_analysis = enable_analysis
        
        # Initialize components
        self.vlc_player = VLCMusicPlayer(
            sample_rate=sample_rate,
            enable_streaming=enable_streaming
        )
        
        self.streaming_integration = StreamingIntegration(
            db_path=db_path,
            music_dir=music_dir,
            sample_rate=sample_rate,
            enable_predictions=True,
            enable_learning=True
        )
        
        self.music_file_manager = MusicFileManager(music_dir)
        
        # Integration state
        self.is_integrated = False
        self.analysis_active = False
        self.current_session = None
        self.playlist = []
        self.current_playlist_index = 0
        
        # Real-time data
        self.live_analysis_data = {}
        self.live_recommendations = []
        self.mixing_suggestions = []
        
        # Callbacks
        self.callbacks = {
            'on_track_started': [],
            'on_track_ended': [],
            'on_recommendations_updated': [],
            'on_mixing_suggestion': [],
            'on_analysis_data': [],
            'on_integration_error': []
        }
        
        # Setup integration
        self._setup_integration()
        
        logger.info("VLCIntegration initialized")
    
    def _setup_integration(self):
        """Setup integration between VLC player and streaming system."""
        # Connect VLC player callbacks
        self.vlc_player.add_callback('on_track_started', self._on_track_started)
        self.vlc_player.add_callback('on_track_ended', self._on_track_ended)
        self.vlc_player.add_callback('on_position_changed', self._on_position_changed)
        self.vlc_player.add_callback('on_crossfade_completed', self._on_crossfade_completed)
        
        # Connect streaming integration callbacks
        self.streaming_integration.add_callback('on_features_extracted', self._on_features_extracted)
        self.streaming_integration.add_callback('on_recommendations_updated', self._on_recommendations_updated)
        self.streaming_integration.add_callback('on_pipeline_error', self._on_pipeline_error)
    
    def start_integration(self) -> bool:
        """
        Start the VLC integration.
        
        Returns:
            True if integration started successfully
        """
        try:
            if self.is_integrated:
                logger.warning("Integration already active")
                return False
            
            # Start streaming integration
            if not self.streaming_integration.start_integration():
                logger.error("Failed to start streaming integration")
                return False
            
            # Start analysis if enabled
            if self.enable_analysis:
                self.analysis_active = True
                self._start_analysis_thread()
            
            self.is_integrated = True
            logger.info("VLC integration started")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting VLC integration: {e}")
            self._trigger_callbacks('on_integration_error', {'error': str(e)})
            return False
    
    def stop_integration(self):
        """Stop the VLC integration."""
        try:
            if not self.is_integrated:
                return
            
            # Stop analysis
            self.analysis_active = False
            
            # Stop streaming integration
            self.streaming_integration.stop_integration()
            
            # Stop VLC player
            self.vlc_player.stop()
            
            self.is_integrated = False
            logger.info("VLC integration stopped")
            
        except Exception as e:
            logger.error(f"Error stopping VLC integration: {e}")
    
    def _start_analysis_thread(self):
        """Start real-time analysis thread."""
        def analysis_worker():
            while self.analysis_active and self.is_integrated:
                try:
                    # Get current track info
                    if self.vlc_player.current_track:
                        track_info = self.vlc_player.current_track
                        
                        # Get real-time features
                        features = self.streaming_integration.get_real_time_features()
                        predictions = self.streaming_integration.get_real_time_predictions()
                        
                        # Update live analysis data
                        self.live_analysis_data = {
                            'track': track_info,
                            'features': features,
                            'predictions': predictions,
                            'position': self.vlc_player.position,
                            'duration': self.vlc_player.duration,
                            'timestamp': time.time()
                        }
                        
                        # Trigger analysis callbacks
                        self._trigger_callbacks('on_analysis_data', self.live_analysis_data)
                        
                        # Generate mixing suggestions
                        self._generate_mixing_suggestions()
                    
                    time.sleep(0.5)  # Update every 500ms
                    
                except Exception as e:
                    logger.error(f"Error in analysis thread: {e}")
                    time.sleep(1.0)
        
        analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
        analysis_thread.start()
    
    def _generate_mixing_suggestions(self):
        """Generate mixing suggestions based on current analysis."""
        try:
            if not self.live_analysis_data or not self.live_analysis_data.get('features'):
                return
            
            features = self.live_analysis_data['features']
            current_track = self.live_analysis_data['track']
            
            # Get recommendations
            available_tracks = self.streaming_integration._get_available_tracks()
            recommendations = self.streaming_integration.get_real_time_recommendations()
            
            if recommendations:
                # Generate mixing suggestions
                suggestions = []
                for track, score in recommendations[:3]:  # Top 3
                    suggestion = {
                        'track': track,
                        'score': score,
                        'mix_type': self._determine_mix_type(features, track),
                        'transition_time': self._calculate_transition_time(features, track),
                        'confidence': score
                    }
                    suggestions.append(suggestion)
                
                self.mixing_suggestions = suggestions
                self._trigger_callbacks('on_mixing_suggestion', suggestions)
            
        except Exception as e:
            logger.error(f"Error generating mixing suggestions: {e}")
    
    def _determine_mix_type(self, current_features: Dict, next_track: Dict) -> str:
        """Determine the best mix type based on features."""
        try:
            current_tempo = current_features.get('tempo', 120.0)
            next_tempo = next_track.get('tempo', 120.0)
            current_energy = current_features.get('rms_energy', 0.5)
            next_energy = next_track.get('rms_energy', 0.5)
            
            tempo_diff = abs(current_tempo - next_tempo)
            energy_diff = abs(current_energy - next_energy)
            
            if tempo_diff < 5 and energy_diff < 0.1:
                return "seamless"
            elif tempo_diff < 10 and energy_diff < 0.2:
                return "crossfade"
            elif tempo_diff < 20:
                return "tempo_match"
            else:
                return "energy_build"
                
        except Exception as e:
            logger.error(f"Error determining mix type: {e}")
            return "crossfade"
    
    def _calculate_transition_time(self, current_features: Dict, next_track: Dict) -> float:
        """Calculate optimal transition time."""
        try:
            current_tempo = current_features.get('tempo', 120.0)
            next_tempo = next_track.get('tempo', 120.0)
            
            # Base transition time on tempo difference
            tempo_diff = abs(current_tempo - next_tempo)
            
            if tempo_diff < 5:
                return 2.0  # Quick transition
            elif tempo_diff < 15:
                return 4.0  # Medium transition
            else:
                return 8.0  # Long transition
                
        except Exception as e:
            logger.error(f"Error calculating transition time: {e}")
            return 4.0
    
    def load_playlist(self, tracks: List[Dict]) -> bool:
        """
        Load a playlist of tracks.
        
        Args:
            tracks: List of track dictionaries
            
        Returns:
            True if playlist loaded successfully
        """
        try:
            self.playlist = tracks
            self.current_playlist_index = 0
            
            if tracks:
                # Load first track
                first_track = tracks[0]
                file_path = first_track.get('file_path', first_track.get('file_path'))
                
                if file_path and os.path.exists(file_path):
                    self.vlc_player.load_track(file_path, first_track)
                    logger.info(f"Playlist loaded: {len(tracks)} tracks")
                    return True
            
            logger.warning("No valid tracks in playlist")
            return False
            
        except Exception as e:
            logger.error(f"Error loading playlist: {e}")
            return False
    
    def play_current_track(self) -> bool:
        """Play the current track."""
        return self.vlc_player.play()
    
    def pause_current_track(self) -> bool:
        """Pause the current track."""
        return self.vlc_player.pause()
    
    def stop_current_track(self) -> bool:
        """Stop the current track."""
        return self.vlc_player.stop()
    
    def next_track(self) -> bool:
        """Play next track in playlist."""
        try:
            if self.current_playlist_index < len(self.playlist) - 1:
                self.current_playlist_index += 1
                next_track = self.playlist[self.current_playlist_index]
                file_path = next_track.get('file_path', next_track.get('file_path'))
                
                if file_path and os.path.exists(file_path):
                    # Use crossfade if available
                    if self.vlc_player.crossfade_active:
                        self.vlc_player.crossfade_to_next(file_path)
                    else:
                        self.vlc_player.load_track(file_path, next_track)
                        self.vlc_player.play()
                    
                    return True
            
            logger.warning("No next track available")
            return False
            
        except Exception as e:
            logger.error(f"Error playing next track: {e}")
            return False
    
    def previous_track(self) -> bool:
        """Play previous track in playlist."""
        try:
            if self.current_playlist_index > 0:
                self.current_playlist_index -= 1
                prev_track = self.playlist[self.current_playlist_index]
                file_path = prev_track.get('file_path', prev_track.get('file_path'))
                
                if file_path and os.path.exists(file_path):
                    self.vlc_player.load_track(file_path, prev_track)
                    self.vlc_player.play()
                    return True
            
            logger.warning("No previous track available")
            return False
            
        except Exception as e:
            logger.error(f"Error playing previous track: {e}")
            return False
    
    def apply_mixing_suggestion(self, suggestion_index: int) -> bool:
        """
        Apply a mixing suggestion.
        
        Args:
            suggestion_index: Index of suggestion to apply
            
        Returns:
            True if suggestion applied successfully
        """
        try:
            if suggestion_index >= len(self.mixing_suggestions):
                logger.warning("Invalid suggestion index")
                return False
            
            suggestion = self.mixing_suggestions[suggestion_index]
            track = suggestion['track']
            mix_type = suggestion['mix_type']
            transition_time = suggestion['transition_time']
            
            file_path = track.get('file_path', track.get('file_path'))
            if not file_path or not os.path.exists(file_path):
                logger.error("Track file not found")
                return False
            
            # Apply the mix
            if mix_type in ['seamless', 'crossfade']:
                self.vlc_player.crossfade_to_next(file_path, transition_time)
            else:
                # Load and play immediately
                self.vlc_player.load_track(file_path, track)
                self.vlc_player.play()
            
            logger.info(f"Applied mixing suggestion: {mix_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying mixing suggestion: {e}")
            return False
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'is_integrated': self.is_integrated,
            'analysis_active': self.analysis_active,
            'vlc_status': self.vlc_player.get_status(),
            'streaming_status': self.streaming_integration.get_integration_status(),
            'playlist_info': {
                'total_tracks': len(self.playlist),
                'current_index': self.current_playlist_index,
                'current_track': self.playlist[self.current_playlist_index] if self.playlist else None
            },
            'live_analysis': self.live_analysis_data,
            'mixing_suggestions': self.mixing_suggestions
        }
    
    def _on_track_started(self, data: Dict):
        """Handle track started event."""
        self._trigger_callbacks('on_track_started', data)
    
    def _on_track_ended(self, data: Dict):
        """Handle track ended event."""
        self._trigger_callbacks('on_track_ended', data)
    
    def _on_position_changed(self, data: Dict):
        """Handle position changed event."""
        # Update live analysis data
        if self.live_analysis_data:
            self.live_analysis_data['position'] = data.get('position', 0.0)
    
    def _on_crossfade_completed(self, data: Dict):
        """Handle crossfade completed event."""
        logger.info("Crossfade completed")
    
    def _on_features_extracted(self, data: Dict):
        """Handle features extracted event."""
        # Update live analysis data
        if self.live_analysis_data:
            self.live_analysis_data['features'] = data
    
    def _on_recommendations_updated(self, data: Dict):
        """Handle recommendations updated event."""
        self.live_recommendations = data
        self._trigger_callbacks('on_recommendations_updated', data)
    
    def _on_pipeline_error(self, data: Dict):
        """Handle pipeline error event."""
        self._trigger_callbacks('on_integration_error', data)
    
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
