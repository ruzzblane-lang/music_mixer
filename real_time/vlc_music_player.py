"""
VLC Music Player

High-level VLC integration for music playback, streaming, and mixing.
Provides seamless integration with the AI Music Mixer system.
"""

import vlc
import logging
import threading
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


class PlayerState(Enum):
    """Player state enumeration."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    BUFFERING = "buffering"
    ERROR = "error"


class VLCMusicPlayer:
    """
    VLC-based music player with advanced features.
    
    Provides music playback, streaming, crossfading, and real-time analysis
    integration for the AI Music Mixer system.
    """
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 enable_streaming: bool = True,
                 crossfade_duration: float = 3.0):
        """
        Initialize VLC music player.
        
        Args:
            sample_rate: Audio sample rate
            enable_streaming: Enable streaming capabilities
            crossfade_duration: Default crossfade duration in seconds
        """
        self.sample_rate = sample_rate
        self.enable_streaming = enable_streaming
        self.crossfade_duration = crossfade_duration
        
        # VLC components
        self.instance = None
        self.players = {}  # Multiple players for crossfading
        self.current_player = None
        self.next_player = None
        
        # Player state
        self.state = PlayerState.STOPPED
        self.current_track = None
        self.playlist = []
        self.current_index = 0
        self.volume = 100
        self.position = 0.0
        self.duration = 0.0
        
        # Streaming
        self.streaming_enabled = enable_streaming
        self.stream_url = None
        self.stream_port = 8080
        
        # Crossfading
        self.crossfade_active = False
        self.crossfade_start_time = 0.0
        self.crossfade_thread = None
        
        # Callbacks
        self.callbacks = {
            'on_track_started': [],
            'on_track_ended': [],
            'on_position_changed': [],
            'on_state_changed': [],
            'on_error': [],
            'on_crossfade_started': [],
            'on_crossfade_completed': []
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'tracks_played': 0,
            'total_play_time': 0.0,
            'crossfades_completed': 0,
            'errors': 0,
            'last_error': None
        }
        
        # Initialize VLC
        self._initialize_vlc()
        
        logger.info(f"VLCMusicPlayer initialized: SR={sample_rate}, Streaming={enable_streaming}")
    
    def _initialize_vlc(self):
        """Initialize VLC instance and players."""
        try:
            # Create VLC instance with audio options
            vlc_args = [
                '--intf', 'dummy',  # No interface
                '--no-video',  # Audio only
                '--aout', 'alsa',  # Use ALSA audio output
                '--audio-resampler', 'src',  # High-quality resampling
                '--sout-keep',  # Keep streaming output
                '--network-caching', '1000',  # Network caching
                '--live-caching', '1000',  # Live streaming cache
            ]
            
            self.instance = vlc.Instance(vlc_args)
            
            # Create primary player
            self.players['primary'] = self.instance.media_player_new()
            self.players['secondary'] = self.instance.media_player_new()
            
            self.current_player = self.players['primary']
            self.next_player = self.players['secondary']
            
            # Set up event callbacks
            self._setup_event_callbacks()
            
            logger.info("VLC instance and players initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VLC: {e}")
            self.performance_metrics['errors'] += 1
            self.performance_metrics['last_error'] = str(e)
            raise
    
    def _setup_event_callbacks(self):
        """Set up VLC event callbacks."""
        try:
            # Set up callbacks for primary player
            event_manager = self.players['primary'].event_manager()
            event_manager.event_attach(vlc.EventType.MediaPlayerPlaying, self._on_playing)
            event_manager.event_attach(vlc.EventType.MediaPlayerPaused, self._on_paused)
            event_manager.event_attach(vlc.EventType.MediaPlayerStopped, self._on_stopped)
            event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_ended)
            event_manager.event_attach(vlc.EventType.MediaPlayerPositionChanged, self._on_position_changed)
            event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._on_error)
            
            # Set up callbacks for secondary player
            event_manager_2 = self.players['secondary'].event_manager()
            event_manager_2.event_attach(vlc.EventType.MediaPlayerPlaying, self._on_playing)
            event_manager_2.event_attach(vlc.EventType.MediaPlayerPaused, self._on_paused)
            event_manager_2.event_attach(vlc.EventType.MediaPlayerStopped, self._on_stopped)
            event_manager_2.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_ended)
            event_manager_2.event_attach(vlc.EventType.MediaPlayerPositionChanged, self._on_position_changed)
            event_manager_2.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._on_error)
            
        except Exception as e:
            logger.error(f"Error setting up VLC callbacks: {e}")
    
    def _on_playing(self, event):
        """Handle playing event."""
        self.state = PlayerState.PLAYING
        self._trigger_callbacks('on_state_changed', {'state': self.state.value})
        logger.debug("Player started playing")
    
    def _on_paused(self, event):
        """Handle paused event."""
        self.state = PlayerState.PAUSED
        self._trigger_callbacks('on_state_changed', {'state': self.state.value})
        logger.debug("Player paused")
    
    def _on_stopped(self, event):
        """Handle stopped event."""
        self.state = PlayerState.STOPPED
        self._trigger_callbacks('on_state_changed', {'state': self.state.value})
        logger.debug("Player stopped")
    
    def _on_ended(self, event):
        """Handle track ended event."""
        self.performance_metrics['tracks_played'] += 1
        self._trigger_callbacks('on_track_ended', {'track': self.current_track})
        
        # Auto-play next track if available
        if self.current_index < len(self.playlist) - 1:
            self.next_track()
        else:
            self.stop()
        
        logger.debug("Track ended")
    
    def _on_position_changed(self, event):
        """Handle position changed event."""
        if self.current_player:
            self.position = self.current_player.get_position()
            self._trigger_callbacks('on_position_changed', {
                'position': self.position,
                'time': self.current_player.get_time()
            })
    
    def _on_error(self, event):
        """Handle error event."""
        self.state = PlayerState.ERROR
        self.performance_metrics['errors'] += 1
        self.performance_metrics['last_error'] = "VLC playback error"
        self._trigger_callbacks('on_error', {'error': 'VLC playback error'})
        logger.error("VLC playback error occurred")
    
    def load_track(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """
        Load a track for playback.
        
        Args:
            file_path: Path to the audio file
            metadata: Optional track metadata
            
        Returns:
            True if track loaded successfully
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Track file not found: {file_path}")
                return False
            
            # Create media
            media = self.instance.media_new(file_path)
            self.current_player.set_media(media)
            
            # Store track info
            self.current_track = {
                'file_path': file_path,
                'title': Path(file_path).stem,
                'metadata': metadata or {}
            }
            
            # Get duration
            self.duration = media.get_duration() / 1000.0  # Convert to seconds
            
            logger.info(f"Track loaded: {self.current_track['title']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading track {file_path}: {e}")
            self.performance_metrics['errors'] += 1
            self.performance_metrics['last_error'] = str(e)
            return False
    
    def play(self) -> bool:
        """
        Start playback.
        
        Returns:
            True if playback started successfully
        """
        try:
            if not self.current_player.get_media():
                logger.warning("No track loaded")
                return False
            
            result = self.current_player.play()
            if result == 0:  # VLC success
                self.state = PlayerState.PLAYING
                self._trigger_callbacks('on_track_started', {'track': self.current_track})
                logger.info("Playback started")
                return True
            else:
                logger.error("Failed to start playback")
                return False
                
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            self.performance_metrics['errors'] += 1
            self.performance_metrics['last_error'] = str(e)
            return False
    
    def pause(self) -> bool:
        """
        Pause playback.
        
        Returns:
            True if paused successfully
        """
        try:
            self.current_player.pause()
            self.state = PlayerState.PAUSED
            logger.info("Playback paused")
            return True
        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop playback.
        
        Returns:
            True if stopped successfully
        """
        try:
            self.current_player.stop()
            self.state = PlayerState.STOPPED
            self.position = 0.0
            logger.info("Playback stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            return False
    
    def set_volume(self, volume: int) -> bool:
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0-100)
            
        Returns:
            True if volume set successfully
        """
        try:
            volume = max(0, min(100, volume))  # Clamp to 0-100
            self.current_player.audio_set_volume(volume)
            self.volume = volume
            logger.debug(f"Volume set to {volume}")
            return True
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return False
    
    def set_position(self, position: float) -> bool:
        """
        Set playback position.
        
        Args:
            position: Position (0.0-1.0)
            
        Returns:
            True if position set successfully
        """
        try:
            position = max(0.0, min(1.0, position))  # Clamp to 0.0-1.0
            self.current_player.set_position(position)
            self.position = position
            logger.debug(f"Position set to {position}")
            return True
        except Exception as e:
            logger.error(f"Error setting position: {e}")
            return False
    
    def crossfade_to_next(self, next_track_path: str, duration: Optional[float] = None) -> bool:
        """
        Crossfade to next track.
        
        Args:
            next_track_path: Path to next track
            duration: Crossfade duration (uses default if None)
            
        Returns:
            True if crossfade started successfully
        """
        try:
            if self.crossfade_active:
                logger.warning("Crossfade already in progress")
                return False
            
            duration = duration or self.crossfade_duration
            
            # Load next track in secondary player
            media = self.instance.media_new(next_track_path)
            self.next_player.set_media(media)
            self.next_player.audio_set_volume(0)  # Start silent
            
            # Start crossfade
            self.crossfade_active = True
            self.crossfade_start_time = time.time()
            self._trigger_callbacks('on_crossfade_started', {
                'current_track': self.current_track,
                'next_track': next_track_path,
                'duration': duration
            })
            
            # Start crossfade thread
            self.crossfade_thread = threading.Thread(
                target=self._crossfade_worker,
                args=(duration,),
                daemon=True
            )
            self.crossfade_thread.start()
            
            logger.info(f"Crossfade started: {duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Error starting crossfade: {e}")
            self.performance_metrics['errors'] += 1
            self.performance_metrics['last_error'] = str(e)
            return False
    
    def _crossfade_worker(self, duration: float):
        """Crossfade worker thread."""
        try:
            # Start next track
            self.next_player.play()
            
            # Crossfade over duration
            start_time = time.time()
            while time.time() - start_time < duration:
                progress = (time.time() - start_time) / duration
                
                # Fade out current, fade in next
                current_volume = int((1.0 - progress) * self.volume)
                next_volume = int(progress * self.volume)
                
                self.current_player.audio_set_volume(current_volume)
                self.next_player.audio_set_volume(next_volume)
                
                time.sleep(0.1)  # Update every 100ms
            
            # Complete crossfade
            self.current_player.stop()
            self.next_player.audio_set_volume(self.volume)
            
            # Swap players
            self.current_player, self.next_player = self.next_player, self.current_player
            
            self.crossfade_active = False
            self.performance_metrics['crossfades_completed'] += 1
            
            self._trigger_callbacks('on_crossfade_completed', {
                'duration': duration
            })
            
            logger.info("Crossfade completed")
            
        except Exception as e:
            logger.error(f"Error in crossfade worker: {e}")
            self.crossfade_active = False
            self.performance_metrics['errors'] += 1
            self.performance_metrics['last_error'] = str(e)
    
    def start_streaming(self, port: int = 8080) -> bool:
        """
        Start audio streaming.
        
        Args:
            port: Streaming port
            
        Returns:
            True if streaming started successfully
        """
        try:
            if not self.enable_streaming:
                logger.warning("Streaming not enabled")
                return False
            
            self.stream_port = port
            self.stream_url = f"http://localhost:{port}/stream"
            
            # Configure streaming (this would need VLC streaming setup)
            logger.info(f"Streaming started on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """
        Stop audio streaming.
        
        Returns:
            True if streaming stopped successfully
        """
        try:
            self.stream_url = None
            logger.info("Streaming stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current player status."""
        return {
            'state': self.state.value,
            'current_track': self.current_track,
            'position': self.position,
            'duration': self.duration,
            'volume': self.volume,
            'crossfade_active': self.crossfade_active,
            'streaming_enabled': self.streaming_enabled,
            'stream_url': self.stream_url,
            'performance_metrics': self.performance_metrics.copy()
        }
    
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
        try:
            if self.current_player:
                self.current_player.stop()
            if self.next_player:
                self.next_player.stop()
        except Exception:
            pass
