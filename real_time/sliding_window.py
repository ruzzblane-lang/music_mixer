"""
Sliding Window Manager

Manages audio data windows for real-time analysis.
Handles windowing, overlap, and buffer management.
"""

import numpy as np
from typing import Optional, Tuple, List
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SlidingWindowManager:
    """
    Manages sliding windows for audio analysis.
    
    Provides efficient windowing with overlap management,
    circular buffering, and window function application.
    """
    
    def __init__(self, 
                 window_size: int = 2048,
                 hop_size: int = 512,
                 sample_rate: int = 22050,
                 window_function: str = 'hann'):
        """
        Initialize the sliding window manager.
        
        Args:
            window_size: Size of analysis window in samples
            hop_size: Hop size between windows in samples
            sample_rate: Audio sample rate
            window_function: Window function to apply ('hann', 'hamming', 'blackman')
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window_function = window_function
        
        # Circular buffer for audio data
        self.buffer_size = window_size * 4  # 4x window size for overlap
        self.audio_buffer = np.zeros(self.buffer_size)
        self.buffer_position = 0
        self.total_samples = 0
        
        # Window function
        self.window = self._create_window_function()
        
        # Overlap management
        self.overlap_samples = window_size - hop_size
        self.overlap_buffer = np.zeros(self.overlap_samples)
        
        logger.info(f"SlidingWindowManager initialized: WS={window_size}, HS={hop_size}, WF={window_function}")
    
    def _create_window_function(self) -> np.ndarray:
        """Create the window function."""
        if self.window_function == 'hann':
            return np.hanning(self.window_size)
        elif self.window_function == 'hamming':
            return np.hamming(self.window_size)
        elif self.window_function == 'blackman':
            return np.blackman(self.window_size)
        else:
            logger.warning(f"Unknown window function: {self.window_function}, using Hann")
            return np.hanning(self.window_size)
    
    def add_audio_data(self, audio_data: np.ndarray) -> bool:
        """
        Add new audio data to the buffer.
        
        Args:
            audio_data: New audio samples to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Ensure audio data is 1D
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Add to circular buffer
            for sample in audio_data:
                self.audio_buffer[self.buffer_position] = sample
                self.buffer_position = (self.buffer_position + 1) % self.buffer_size
                self.total_samples += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding audio data: {e}")
            return False
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """
        Get the current analysis window.
        
        Returns:
            Current windowed audio data or None if not enough data
        """
        try:
            if self.total_samples < self.window_size:
                return None
            
            # Get the most recent window_size samples
            window_data = np.zeros(self.window_size)
            
            for i in range(self.window_size):
                buffer_idx = (self.buffer_position - self.window_size + i) % self.buffer_size
                window_data[i] = self.audio_buffer[buffer_idx]
            
            # Apply window function
            windowed_data = window_data * self.window
            
            return windowed_data
            
        except Exception as e:
            logger.error(f"Error getting current window: {e}")
            return None
    
    def get_overlap_region(self) -> Optional[np.ndarray]:
        """
        Get the overlap region for smooth transitions.
        
        Returns:
            Overlap region data or None if not enough data
        """
        try:
            if self.total_samples < self.overlap_samples:
                return None
            
            # Get the overlap region
            overlap_data = np.zeros(self.overlap_samples)
            
            for i in range(self.overlap_samples):
                buffer_idx = (self.buffer_position - self.overlap_samples + i) % self.buffer_size
                overlap_data[i] = self.audio_buffer[buffer_idx]
            
            return overlap_data
            
        except Exception as e:
            logger.error(f"Error getting overlap region: {e}")
            return None
    
    def get_multiple_windows(self, num_windows: int) -> List[np.ndarray]:
        """
        Get multiple overlapping windows.
        
        Args:
            num_windows: Number of windows to retrieve
            
        Returns:
            List of windowed audio data
        """
        windows = []
        
        try:
            for i in range(num_windows):
                # Calculate offset for this window
                offset = i * self.hop_size
                
                if self.total_samples < self.window_size + offset:
                    break
                
                # Get window data
                window_data = np.zeros(self.window_size)
                
                for j in range(self.window_size):
                    buffer_idx = (self.buffer_position - self.window_size - offset + j) % self.buffer_size
                    window_data[j] = self.audio_buffer[buffer_idx]
                
                # Apply window function
                windowed_data = window_data * self.window
                windows.append(windowed_data)
            
        except Exception as e:
            logger.error(f"Error getting multiple windows: {e}")
        
        return windows
    
    def get_window_with_context(self, context_samples: int = 1024) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get current window with context (previous and next samples).
        
        Args:
            context_samples: Number of context samples on each side
            
        Returns:
            Tuple of (previous_context, current_window, next_context) or None
        """
        try:
            if self.total_samples < self.window_size + context_samples * 2:
                return None
            
            # Get previous context
            prev_context = np.zeros(context_samples)
            for i in range(context_samples):
                buffer_idx = (self.buffer_position - self.window_size - context_samples + i) % self.buffer_size
                prev_context[i] = self.audio_buffer[buffer_idx]
            
            # Get current window
            current_window = self.get_current_window()
            if current_window is None:
                return None
            
            # Get next context (this is limited by available data)
            next_context = np.zeros(context_samples)
            for i in range(context_samples):
                buffer_idx = (self.buffer_position - context_samples + i) % self.buffer_size
                next_context[i] = self.audio_buffer[buffer_idx]
            
            return prev_context, current_window, next_context
            
        except Exception as e:
            logger.error(f"Error getting window with context: {e}")
            return None
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status information."""
        return {
            'buffer_size': self.buffer_size,
            'current_position': self.buffer_position,
            'total_samples': self.total_samples,
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'overlap_samples': self.overlap_samples,
            'buffer_usage_percent': (self.total_samples % self.buffer_size) / self.buffer_size * 100,
            'can_get_window': self.total_samples >= self.window_size,
            'can_get_overlap': self.total_samples >= self.overlap_samples
        }
    
    def reset(self):
        """Reset the sliding window manager."""
        self.audio_buffer.fill(0)
        self.buffer_position = 0
        self.total_samples = 0
        self.overlap_buffer.fill(0)
        logger.info("SlidingWindowManager reset")
    
    def set_window_function(self, window_function: str):
        """
        Change the window function.
        
        Args:
            window_function: New window function ('hann', 'hamming', 'blackman')
        """
        self.window_function = window_function
        self.window = self._create_window_function()
        logger.info(f"Window function changed to: {window_function}")
    
    def get_window_info(self) -> dict:
        """Get information about the current window configuration."""
        return {
            'window_size': self.window_size,
            'hop_size': self.hop_size,
            'window_function': self.window_function,
            'sample_rate': self.sample_rate,
            'window_duration_ms': (self.window_size / self.sample_rate) * 1000,
            'hop_duration_ms': (self.hop_size / self.sample_rate) * 1000,
            'overlap_percent': (self.overlap_samples / self.window_size) * 100
        }
