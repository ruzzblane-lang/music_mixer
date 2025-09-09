"""
Audio Device Interface

Provides a unified interface for audio device management
across different audio libraries and platforms.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

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


class AudioDeviceType(Enum):
    """Audio device types."""
    INPUT = "input"
    OUTPUT = "output"
    DUPLEX = "duplex"


class AudioDevice:
    """Represents an audio device."""
    
    def __init__(self, 
                 device_id: int,
                 name: str,
                 device_type: AudioDeviceType,
                 channels: int,
                 sample_rate: float,
                 is_default: bool = False):
        """
        Initialize audio device.
        
        Args:
            device_id: Device ID
            name: Device name
            device_type: Type of device (input/output/duplex)
            channels: Number of channels
            sample_rate: Sample rate
            is_default: Whether this is the default device
        """
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_default = is_default
    
    def to_dict(self) -> Dict:
        """Convert device to dictionary."""
        return {
            'id': self.device_id,
            'name': self.name,
            'type': self.device_type.value,
            'channels': self.channels,
            'sample_rate': self.sample_rate,
            'is_default': self.is_default
        }
    
    def __str__(self):
        return f"AudioDevice(id={self.device_id}, name='{self.name}', type={self.device_type.value})"


class AudioDeviceInterface:
    """
    Unified interface for audio device management.
    
    Provides a consistent API across different audio libraries
    and platforms for device enumeration and management.
    """
    
    def __init__(self):
        """Initialize the audio device interface."""
        self.audio_library = self._detect_audio_library()
        self.devices = []
        self.default_input_device = None
        self.default_output_device = None
        
        logger.info(f"AudioDeviceInterface initialized with {self.audio_library}")
    
    def _detect_audio_library(self) -> str:
        """Detect which audio library is available."""
        if SOUNDDEVICE_AVAILABLE:
            return "sounddevice"
        elif PYAUDIO_AVAILABLE:
            return "pyaudio"
        else:
            logger.warning("No audio I/O library available - using mock mode")
            return "mock"
    
    def refresh_devices(self) -> List[AudioDevice]:
        """
        Refresh the list of available audio devices.
        
        Returns:
            List of available audio devices
        """
        self.devices = []
        
        if self.audio_library == "sounddevice":
            self._refresh_sounddevice_devices()
        elif self.audio_library == "pyaudio":
            self._refresh_pyaudio_devices()
        else:
            self._refresh_mock_devices()
        
        logger.info(f"Found {len(self.devices)} audio devices")
        return self.devices
    
    def _refresh_sounddevice_devices(self):
        """Refresh devices using sounddevice."""
        try:
            sd_devices = sd.query_devices()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            
            for i, device in enumerate(sd_devices):
                # Determine device type
                if device['max_input_channels'] > 0 and device['max_output_channels'] > 0:
                    device_type = AudioDeviceType.DUPLEX
                elif device['max_input_channels'] > 0:
                    device_type = AudioDeviceType.INPUT
                elif device['max_output_channels'] > 0:
                    device_type = AudioDeviceType.OUTPUT
                else:
                    continue  # Skip devices with no channels
                
                # Create device object
                audio_device = AudioDevice(
                    device_id=i,
                    name=device['name'],
                    device_type=device_type,
                    channels=max(device['max_input_channels'], device['max_output_channels']),
                    sample_rate=device['default_samplerate'],
                    is_default=(i == default_input or i == default_output)
                )
                
                self.devices.append(audio_device)
                
                # Set default devices
                if i == default_input:
                    self.default_input_device = audio_device
                if i == default_output:
                    self.default_output_device = audio_device
            
        except Exception as e:
            logger.error(f"Error refreshing sounddevice devices: {e}")
    
    def _refresh_pyaudio_devices(self):
        """Refresh devices using pyaudio."""
        try:
            p = pyaudio.PyAudio()
            default_input = p.get_default_input_device_info()['index']
            default_output = p.get_default_output_device_info()['index']
            
            for i in range(p.get_device_count()):
                try:
                    device_info = p.get_device_info_by_index(i)
                    
                    # Determine device type
                    if device_info['maxInputChannels'] > 0 and device_info['maxOutputChannels'] > 0:
                        device_type = AudioDeviceType.DUPLEX
                    elif device_info['maxInputChannels'] > 0:
                        device_type = AudioDeviceType.INPUT
                    elif device_info['maxOutputChannels'] > 0:
                        device_type = AudioDeviceType.OUTPUT
                    else:
                        continue  # Skip devices with no channels
                    
                    # Create device object
                    audio_device = AudioDevice(
                        device_id=i,
                        name=device_info['name'],
                        device_type=device_type,
                        channels=max(device_info['maxInputChannels'], device_info['maxOutputChannels']),
                        sample_rate=device_info['defaultSampleRate'],
                        is_default=(i == default_input or i == default_output)
                    )
                    
                    self.devices.append(audio_device)
                    
                    # Set default devices
                    if i == default_input:
                        self.default_input_device = audio_device
                    if i == default_output:
                        self.default_output_device = audio_device
                
                except Exception as e:
                    logger.warning(f"Error getting device {i} info: {e}")
                    continue
            
            p.terminate()
            
        except Exception as e:
            logger.error(f"Error refreshing pyaudio devices: {e}")
    
    def _refresh_mock_devices(self):
        """Refresh devices using mock data."""
        # Create mock devices for testing
        mock_devices = [
            AudioDevice(0, "Mock Input Device", AudioDeviceType.INPUT, 1, 22050, True),
            AudioDevice(1, "Mock Output Device", AudioDeviceType.OUTPUT, 2, 44100, True),
            AudioDevice(2, "Mock Duplex Device", AudioDeviceType.DUPLEX, 2, 48000, False)
        ]
        
        self.devices = mock_devices
        self.default_input_device = mock_devices[0]
        self.default_output_device = mock_devices[1]
    
    def get_devices(self, device_type: Optional[AudioDeviceType] = None) -> List[AudioDevice]:
        """
        Get list of audio devices.
        
        Args:
            device_type: Filter by device type (None for all)
            
        Returns:
            List of audio devices
        """
        if not self.devices:
            self.refresh_devices()
        
        if device_type is None:
            return self.devices.copy()
        else:
            return [device for device in self.devices if device.device_type == device_type]
    
    def get_input_devices(self) -> List[AudioDevice]:
        """Get list of input devices."""
        return self.get_devices(AudioDeviceType.INPUT) + self.get_devices(AudioDeviceType.DUPLEX)
    
    def get_output_devices(self) -> List[AudioDevice]:
        """Get list of output devices."""
        return self.get_devices(AudioDeviceType.OUTPUT) + self.get_devices(AudioDeviceType.DUPLEX)
    
    def get_device_by_id(self, device_id: int) -> Optional[AudioDevice]:
        """
        Get device by ID.
        
        Args:
            device_id: Device ID
            
        Returns:
            Audio device or None if not found
        """
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None
    
    def get_device_by_name(self, name: str) -> Optional[AudioDevice]:
        """
        Get device by name.
        
        Args:
            name: Device name
            
        Returns:
            Audio device or None if not found
        """
        for device in self.devices:
            if name.lower() in device.name.lower():
                return device
        return None
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get default input device."""
        if not self.devices:
            self.refresh_devices()
        return self.default_input_device
    
    def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get default output device."""
        if not self.devices:
            self.refresh_devices()
        return self.default_output_device
    
    def test_device(self, device: AudioDevice, duration: float = 1.0) -> Dict:
        """
        Test an audio device.
        
        Args:
            device: Device to test
            duration: Test duration in seconds
            
        Returns:
            Test results dictionary
        """
        results = {
            'device_id': device.device_id,
            'device_name': device.name,
            'test_passed': False,
            'error': None,
            'latency': None,
            'sample_rate': None
        }
        
        try:
            if self.audio_library == "sounddevice":
                results.update(self._test_sounddevice_device(device, duration))
            elif self.audio_library == "pyaudio":
                results.update(self._test_pyaudio_device(device, duration))
            else:
                results.update(self._test_mock_device(device, duration))
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error testing device {device.name}: {e}")
        
        return results
    
    def _test_sounddevice_device(self, device: AudioDevice, duration: float) -> Dict:
        """Test device using sounddevice."""
        results = {'test_passed': False, 'error': None}
        
        try:
            # Test input if device supports it
            if device.device_type in [AudioDeviceType.INPUT, AudioDeviceType.DUPLEX]:
                # Record a short sample
                import numpy as np
                sample = sd.rec(int(duration * device.sample_rate), 
                              samplerate=device.sample_rate, 
                              channels=min(device.channels, 2),
                              device=device.device_id)
                sd.wait()
                
                if len(sample) > 0:
                    results['test_passed'] = True
                    results['sample_rate'] = device.sample_rate
                    results['latency'] = 0.05  # Mock latency
            
            # Test output if device supports it
            elif device.device_type in [AudioDeviceType.OUTPUT, AudioDeviceType.DUPLEX]:
                # Play a short tone
                import numpy as np
                t = np.linspace(0, duration, int(duration * device.sample_rate))
                tone = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
                
                sd.play(tone, samplerate=device.sample_rate, device=device.device_id)
                sd.wait()
                
                results['test_passed'] = True
                results['sample_rate'] = device.sample_rate
                results['latency'] = 0.05  # Mock latency
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_pyaudio_device(self, device: AudioDevice, duration: float) -> Dict:
        """Test device using pyaudio."""
        results = {'test_passed': False, 'error': None}
        
        try:
            p = pyaudio.PyAudio()
            
            # Test input if device supports it
            if device.device_type in [AudioDeviceType.INPUT, AudioDeviceType.DUPLEX]:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=min(device.channels, 2),
                    rate=int(device.sample_rate),
                    input=True,
                    input_device_index=device.device_id,
                    frames_per_buffer=1024
                )
                
                # Record a short sample
                frames = []
                for _ in range(int(duration * device.sample_rate / 1024)):
                    data = stream.read(1024)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                
                if frames:
                    results['test_passed'] = True
                    results['sample_rate'] = device.sample_rate
                    results['latency'] = 0.05  # Mock latency
            
            # Test output if device supports it
            elif device.device_type in [AudioDeviceType.OUTPUT, AudioDeviceType.DUPLEX]:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=min(device.channels, 2),
                    rate=int(device.sample_rate),
                    output=True,
                    output_device_index=device.device_id,
                    frames_per_buffer=1024
                )
                
                # Play a short tone
                import numpy as np
                t = np.linspace(0, duration, int(duration * device.sample_rate))
                tone = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
                
                stream.write(tone.tobytes())
                stream.stop_stream()
                stream.close()
                
                results['test_passed'] = True
                results['sample_rate'] = device.sample_rate
                results['latency'] = 0.05  # Mock latency
            
            p.terminate()
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_mock_device(self, device: AudioDevice, duration: float) -> Dict:
        """Test device using mock implementation."""
        return {
            'test_passed': True,
            'error': None,
            'latency': 0.01,
            'sample_rate': device.sample_rate
        }
    
    def get_device_info(self) -> Dict:
        """Get information about the audio system."""
        if not self.devices:
            self.refresh_devices()
        
        return {
            'audio_library': self.audio_library,
            'total_devices': len(self.devices),
            'input_devices': len(self.get_input_devices()),
            'output_devices': len(self.get_output_devices()),
            'default_input': self.default_input_device.to_dict() if self.default_input_device else None,
            'default_output': self.default_output_device.to_dict() if self.default_output_device else None,
            'devices': [device.to_dict() for device in self.devices]
        }
