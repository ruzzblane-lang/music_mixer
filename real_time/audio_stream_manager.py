"""
Audio Stream Manager

Handles real-time audio input/output for streaming analysis.
Manages audio devices, streams, and synchronization.
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import queue

# Optional audio I/O libraries
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages real-time audio streaming for analysis.
    
    Handles audio input/output, device management,
    and stream synchronization.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 device: Optional[int] = None):
        """
        Initialize the audio stream manager.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_size: Size of audio chunks
            device: Audio device ID (None for default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        
        # Audio I/O state
        self.is_streaming = False
        self.input_stream = None
        self.output_stream = None
        
        # Audio buffers
        self.input_buffer = queue.Queue(maxsize=100)
        self.output_buffer = queue.Queue(maxsize=100)
        self.audio_history = deque(maxlen=1000)
        
        # Callbacks
        self.callbacks = {
            'on_audio_input': [],
            'on_audio_output': [],
            'on_stream_error': [],
            'on_device_changed': []
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'input_latency': deque(maxlen=100),
            'output_latency': deque(maxlen=100),
            'buffer_underruns': 0,
            'buffer_overruns': 0,
            'stream_errors': 0
        }
        
        # Threading
        self.stream_thread = None
        self.stop_event = threading.Event()
        
        # Check available audio libraries
        self.audio_library = self._detect_audio_library()
        
        logger.info(f"AudioStreamManager initialized: SR={sample_rate}, CH={channels}, CS={chunk_size}")
        logger.info(f"Audio library: {self.audio_library}")
    
    def _detect_audio_library(self) -> str:
        """Detect which audio library is available."""
        if SOUNDDEVICE_AVAILABLE:
            return "sounddevice"
        elif PYAUDIO_AVAILABLE:
            return "pyaudio"
        else:
            logger.warning("No audio I/O library available - using mock mode")
            return "mock"
    
    def get_available_devices(self) -> List[Dict]:
        """Get list of available audio devices."""
        devices = []
        
        if self.audio_library == "sounddevice":
            try:
                sd_devices = sd.query_devices()
                for i, device in enumerate(sd_devices):
                    devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            except Exception as e:
                logger.error(f"Error querying sounddevice: {e}")
        
        elif self.audio_library == "pyaudio":
            try:
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    device_info = p.get_device_info_by_index(i)
                    devices.append({
                        'id': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
                p.terminate()
            except Exception as e:
                logger.error(f"Error querying pyaudio: {e}")
        
        else:
            # Mock devices for testing
            devices = [
                {'id': 0, 'name': 'Mock Input Device', 'channels': 1, 'sample_rate': 22050},
                {'id': 1, 'name': 'Mock Output Device', 'channels': 2, 'sample_rate': 44100}
            ]
        
        return devices
    
    def start_input_stream(self, callback: Optional[Callable] = None) -> bool:
        """
        Start audio input stream.
        
        Args:
            callback: Optional callback for audio data
            
        Returns:
            True if stream started successfully
        """
        try:
            if self.is_streaming:
                logger.warning("Stream already running")
                return False
            
            if callback:
                self.add_callback('on_audio_input', callback)
            
            if self.audio_library == "sounddevice":
                return self._start_sounddevice_input()
            elif self.audio_library == "pyaudio":
                return self._start_pyaudio_input()
            else:
                return self._start_mock_input()
                
        except Exception as e:
            logger.error(f"Error starting input stream: {e}")
            self._trigger_callbacks('on_stream_error', {'error': str(e)})
            return False
    
    def start_output_stream(self, callback: Optional[Callable] = None) -> bool:
        """
        Start audio output stream.
        
        Args:
            callback: Optional callback for output events
            
        Returns:
            True if stream started successfully
        """
        try:
            if callback:
                self.add_callback('on_audio_output', callback)
            
            if self.audio_library == "sounddevice":
                return self._start_sounddevice_output()
            elif self.audio_library == "pyaudio":
                return self._start_pyaudio_output()
            else:
                return self._start_mock_output()
                
        except Exception as e:
            logger.error(f"Error starting output stream: {e}")
            self._trigger_callbacks('on_stream_error', {'error': str(e)})
            return False
    
    def _start_sounddevice_input(self) -> bool:
        """Start sounddevice input stream."""
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Sounddevice status: {status}")
                
                # Convert to numpy array
                audio_data = indata.copy()
                
                # Add to buffer
                try:
                    self.input_buffer.put_nowait(audio_data)
                    self.audio_history.append(audio_data)
                except queue.Full:
                    self.performance_metrics['buffer_overruns'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('on_audio_input', audio_data)
            
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device,
                callback=audio_callback
            )
            
            self.input_stream.start()
            self.is_streaming = True
            
            logger.info("Sounddevice input stream started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting sounddevice input: {e}")
            return False
    
    def _start_pyaudio_input(self) -> bool:
        """Start pyaudio input stream."""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            def audio_callback(in_data, frame_count, time_info, status):
                # Convert bytes to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.float32)
                audio_data = audio_data.reshape(-1, self.channels)
                
                # Add to buffer
                try:
                    self.input_buffer.put_nowait(audio_data)
                    self.audio_history.append(audio_data)
                except queue.Full:
                    self.performance_metrics['buffer_overruns'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('on_audio_input', audio_data)
                
                return (in_data, pyaudio.paContinue)
            
            self.input_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device,
                frames_per_buffer=self.chunk_size,
                stream_callback=audio_callback
            )
            
            self.input_stream.start_stream()
            self.is_streaming = True
            
            logger.info("PyAudio input stream started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting pyaudio input: {e}")
            return False
    
    def _start_mock_input(self) -> bool:
        """Start mock input stream for testing."""
        def mock_audio_generator():
            while not self.stop_event.is_set():
                # Generate mock audio data
                audio_data = np.random.randn(self.chunk_size, self.channels).astype(np.float32)
                
                # Add to buffer
                try:
                    self.input_buffer.put_nowait(audio_data)
                    self.audio_history.append(audio_data)
                except queue.Full:
                    self.performance_metrics['buffer_overruns'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('on_audio_input', audio_data)
                
                time.sleep(self.chunk_size / self.sample_rate)
        
        self.stream_thread = threading.Thread(target=mock_audio_generator, daemon=True)
        self.stream_thread.start()
        self.is_streaming = True
        
        logger.info("Mock input stream started")
        return True
    
    def _start_sounddevice_output(self) -> bool:
        """Start sounddevice output stream."""
        try:
            def output_callback(outdata, frames, time, status):
                if status:
                    logger.warning(f"Sounddevice output status: {status}")
                
                try:
                    # Get audio data from output buffer
                    audio_data = self.output_buffer.get_nowait()
                    outdata[:] = audio_data
                except queue.Empty:
                    # Generate silence if no data
                    outdata.fill(0)
                    self.performance_metrics['buffer_underruns'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('on_audio_output', outdata)
            
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                blocksize=self.chunk_size,
                device=self.device,
                callback=output_callback
            )
            
            self.output_stream.start()
            
            logger.info("Sounddevice output stream started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting sounddevice output: {e}")
            return False
    
    def _start_pyaudio_output(self) -> bool:
        """Start pyaudio output stream."""
        try:
            if not hasattr(self, 'pyaudio_instance'):
                self.pyaudio_instance = pyaudio.PyAudio()
            
            def output_callback(in_data, frame_count, time_info, status):
                try:
                    # Get audio data from output buffer
                    audio_data = self.output_buffer.get_nowait()
                    # Convert to bytes
                    out_data = audio_data.tobytes()
                except queue.Empty:
                    # Generate silence if no data
                    out_data = np.zeros(frame_count * self.channels, dtype=np.float32).tobytes()
                    self.performance_metrics['buffer_underruns'] += 1
                
                # Trigger callbacks
                self._trigger_callbacks('on_audio_output', audio_data)
                
                return (out_data, pyaudio.paContinue)
            
            self.output_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device,
                frames_per_buffer=self.chunk_size,
                stream_callback=output_callback
            )
            
            self.output_stream.start_stream()
            
            logger.info("PyAudio output stream started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting pyaudio output: {e}")
            return False
    
    def _start_mock_output(self) -> bool:
        """Start mock output stream for testing."""
        logger.info("Mock output stream started")
        return True
    
    def stop_streams(self):
        """Stop all audio streams."""
        try:
            self.is_streaming = False
            self.stop_event.set()
            
            if self.input_stream:
                if self.audio_library == "sounddevice":
                    self.input_stream.stop()
                    self.input_stream.close()
                elif self.audio_library == "pyaudio":
                    self.input_stream.stop_stream()
                    self.input_stream.close()
                
                self.input_stream = None
            
            if self.output_stream:
                if self.audio_library == "sounddevice":
                    self.output_stream.stop()
                    self.output_stream.close()
                elif self.audio_library == "pyaudio":
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                
                self.output_stream = None
            
            if hasattr(self, 'pyaudio_instance'):
                self.pyaudio_instance.terminate()
                delattr(self, 'pyaudio_instance')
            
            if self.stream_thread and self.stream_thread.is_alive():
                self.stream_thread.join(timeout=1.0)
            
            logger.info("Audio streams stopped")
            
        except Exception as e:
            logger.error(f"Error stopping streams: {e}")
    
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get audio chunk from input buffer.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk or None if no data available
        """
        try:
            return self.input_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Put audio chunk to output buffer.
        
        Args:
            audio_data: Audio data to output
            
        Returns:
            True if successfully added to buffer
        """
        try:
            self.output_buffer.put_nowait(audio_data)
            return True
        except queue.Full:
            self.performance_metrics['buffer_overruns'] += 1
            return False
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        return {
            'is_streaming': self.is_streaming,
            'input_buffer_size': self.input_buffer.qsize(),
            'output_buffer_size': self.output_buffer.qsize(),
            'audio_history_size': len(self.audio_history),
            'buffer_underruns': self.performance_metrics['buffer_underruns'],
            'buffer_overruns': self.performance_metrics['buffer_overruns'],
            'stream_errors': self.performance_metrics['stream_errors'],
            'audio_library': self.audio_library
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
        self.stop_streams()
