"""
Streaming Audio Analyzer

Core streaming analysis engine for real-time audio processing.
Handles audio streams, feature extraction, and adaptive analysis.
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque
import time
import threading

logger = logging.getLogger(__name__)


class StreamingAudioAnalyzer:
    """
    Main real-time audio analysis engine.
    
    Processes audio streams in real-time, extracts features,
    and provides adaptive analysis capabilities.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 window_size: int = 2048,
                 hop_size: int = 512,
                 buffer_size: int = 1000):
        """
        Initialize the streaming analyzer.
        
        Args:
            sample_rate: Audio sample rate
            window_size: Size of analysis window
            hop_size: Hop size for windowing
            buffer_size: Size of audio buffer
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.buffer_size = buffer_size
        
        # Audio buffer for streaming
        self.audio_buffer = deque(maxlen=buffer_size)
        self.current_position = 0
        self.is_streaming = False
        
        # Feature extraction
        self.feature_extractors = {}
        self.feature_cache = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'processing_time': [],
            'feature_extraction_time': [],
            'buffer_usage': [],
            'cpu_usage': []
        }
        
        # Callbacks for real-time events
        self.callbacks = {
            'on_feature_extracted': [],
            'on_beat_detected': [],
            'on_energy_change': [],
            'on_performance_warning': []
        }
        
        logger.info(f"StreamingAudioAnalyzer initialized: SR={sample_rate}, WS={window_size}, HS={hop_size}")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Add new audio data to the streaming buffer.
        
        Args:
            audio_chunk: New audio data chunk
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Ensure audio chunk is the right shape
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)  # Convert to mono
            
            # Add to buffer
            self.audio_buffer.extend(audio_chunk)
            self.current_position += len(audio_chunk)
            
            # Process if we have enough data
            if len(self.audio_buffer) >= self.window_size:
                self._process_current_window()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
            return False
    
    def _process_current_window(self):
        """Process the current audio window."""
        try:
            start_time = time.time()
            
            # Get current window
            current_window = self._get_current_window()
            if current_window is None:
                return
            
            # Extract features
            features = self._extract_features(current_window)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics['processing_time'].append(processing_time)
            
            # Trigger callbacks
            self._trigger_callbacks('on_feature_extracted', features)
            
            # Check for performance issues
            if processing_time > 0.1:  # 100ms threshold
                self._trigger_callbacks('on_performance_warning', {
                    'processing_time': processing_time,
                    'threshold': 0.1
                })
            
        except Exception as e:
            logger.error(f"Error processing current window: {e}")
    
    def _get_current_window(self) -> Optional[np.ndarray]:
        """Get the current analysis window."""
        try:
            if len(self.audio_buffer) < self.window_size:
                return None
            
            # Get the most recent window
            window_data = np.array(list(self.audio_buffer)[-self.window_size:])
            
            # Apply windowing function (Hann window)
            window = np.hanning(self.window_size)
            windowed_data = window_data * window
            
            return windowed_data
            
        except Exception as e:
            logger.error(f"Error getting current window: {e}")
            return None
    
    def _extract_features(self, audio_window: np.ndarray) -> Dict:
        """Extract features from audio window."""
        try:
            start_time = time.time()
            
            features = {}
            
            # Basic features (always extract)
            features['rms_energy'] = self._extract_rms_energy(audio_window)
            features['spectral_centroid'] = self._extract_spectral_centroid(audio_window)
            features['zero_crossing_rate'] = self._extract_zcr(audio_window)
            
            # Advanced features (if performance allows)
            if self._can_extract_advanced_features():
                features['mfcc'] = self._extract_mfcc(audio_window)
                features['chroma'] = self._extract_chroma(audio_window)
                features['spectral_rolloff'] = self._extract_spectral_rolloff(audio_window)
            
            # Update feature extraction time
            extraction_time = time.time() - start_time
            self.performance_metrics['feature_extraction_time'].append(extraction_time)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _extract_rms_energy(self, audio: np.ndarray) -> float:
        """Extract RMS energy."""
        return float(np.sqrt(np.mean(audio**2)))
    
    def _extract_spectral_centroid(self, audio: np.ndarray) -> float:
        """Extract spectral centroid."""
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        if np.sum(magnitude) == 0:
            return 0.0
        
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))
    
    def _extract_zcr(self, audio: np.ndarray) -> float:
        """Extract zero crossing rate."""
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        return float(zero_crossings / len(audio))
    
    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features."""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            return np.mean(mfcc, axis=1)
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return np.zeros(13)
    
    def _extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features."""
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            return np.mean(chroma, axis=1)
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {e}")
            return np.zeros(12)
    
    def _extract_spectral_rolloff(self, audio: np.ndarray) -> float:
        """Extract spectral rolloff."""
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            return float(np.mean(rolloff))
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {e}")
            return 0.0
    
    def _can_extract_advanced_features(self) -> bool:
        """Check if we can extract advanced features based on performance."""
        if not self.performance_metrics['processing_time']:
            return True
        
        recent_times = self.performance_metrics['processing_time'][-10:]
        avg_time = np.mean(recent_times)
        
        return avg_time < 0.05  # 50ms threshold
    
    def _trigger_callbacks(self, event_name: str, data: any):
        """Trigger callbacks for an event."""
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback {event_name}: {e}")
    
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
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        metrics = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                metrics[key] = {
                    'current': values[-1] if values else 0,
                    'average': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
            else:
                metrics[key] = {'current': 0, 'average': 0, 'max': 0, 'min': 0}
        
        # Buffer usage
        metrics['buffer_usage'] = {
            'current': len(self.audio_buffer),
            'max': self.buffer_size,
            'percentage': len(self.audio_buffer) / self.buffer_size * 100
        }
        
        return metrics
    
    def start_streaming(self):
        """Start the streaming analysis."""
        self.is_streaming = True
        logger.info("Streaming analysis started")
    
    def stop_streaming(self):
        """Stop the streaming analysis."""
        self.is_streaming = False
        logger.info("Streaming analysis stopped")
    
    def reset(self):
        """Reset the analyzer state."""
        self.audio_buffer.clear()
        self.current_position = 0
        self.feature_cache.clear()
        
        # Clear performance metrics
        for key in self.performance_metrics:
            self.performance_metrics[key].clear()
        
        logger.info("Streaming analyzer reset")
