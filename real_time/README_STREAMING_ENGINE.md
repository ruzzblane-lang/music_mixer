# Core Streaming Engine (Phase 7A.2)

## Overview

The Core Streaming Engine provides real-time audio processing capabilities for the AI Music Mixer. It enables live audio analysis, feature extraction, and integration with the existing recommendation and mixing systems.

## Components

### 1. AudioStreamManager (`audio_stream_manager.py`)

**Purpose**: Manages real-time audio input/output streams.

**Key Features**:
- Cross-platform audio I/O support (sounddevice, pyaudio, mock)
- Audio buffer management with queue-based system
- Performance monitoring and metrics
- Callback system for real-time events
- Automatic device detection and management

**Usage**:
```python
from real_time.audio_stream_manager import AudioStreamManager

# Initialize
manager = AudioStreamManager(sample_rate=22050, chunk_size=1024)

# Get available devices
devices = manager.get_available_devices()

# Start input stream
manager.start_input_stream()

# Get audio data
audio_chunk = manager.get_audio_chunk(timeout=0.1)

# Stop streams
manager.stop_streams()
```

### 2. StreamingPipeline (`streaming_pipeline.py`)

**Purpose**: Coordinates the complete real-time audio analysis pipeline.

**Key Features**:
- Integrates all streaming components
- Manages feature extraction pipeline
- Handles prediction and learning queues
- Performance optimization and monitoring
- Real-time recommendation generation

**Usage**:
```python
from real_time.streaming_pipeline import StreamingPipeline

# Initialize
pipeline = StreamingPipeline(
    sample_rate=22050,
    window_size=2048,
    hop_size=512,
    enable_predictions=True,
    enable_learning=True
)

# Start pipeline
pipeline.start_pipeline()

# Get recommendations
recommendations = pipeline.get_recommendations(current_features, available_tracks)

# Stop pipeline
pipeline.stop_pipeline()
```

### 3. AudioDeviceInterface (`audio_device_interface.py`)

**Purpose**: Provides unified interface for audio device management.

**Key Features**:
- Cross-platform device enumeration
- Device type filtering (input/output/duplex)
- Device testing and validation
- Default device detection
- Comprehensive device information

**Usage**:
```python
from real_time.audio_device_interface import AudioDeviceInterface, AudioDeviceType

# Initialize
interface = AudioDeviceInterface()

# Refresh device list
devices = interface.refresh_devices()

# Get input devices
input_devices = interface.get_input_devices()

# Get default devices
default_input = interface.get_default_input_device()
default_output = interface.get_default_output_device()

# Test device
test_result = interface.test_device(device, duration=1.0)
```

### 4. StreamingIntegration (`streaming_integration.py`)

**Purpose**: Integrates streaming pipeline with the main AI Music Mixer system.

**Key Features**:
- Seamless connection to existing systems
- Real-time feature processing
- Live recommendation updates
- Mix session management
- User feedback integration

**Usage**:
```python
from real_time.streaming_integration import StreamingIntegration

# Initialize
integration = StreamingIntegration(
    db_path="data/music_library.db",
    sample_rate=22050,
    enable_predictions=True,
    enable_learning=True
)

# Start integration
integration.start_integration()

# Get real-time data
features = integration.get_real_time_features()
predictions = integration.get_real_time_predictions()
recommendations = integration.get_real_time_recommendations()

# Add user feedback
integration.add_user_feedback("Track A", "Track B", True, 0.8)

# Stop integration
integration.stop_integration()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    StreamingIntegration                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Database      │  │ Recommendation  │  │ Mix Engine  │ │
│  │   Connection    │  │    Engine       │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 StreamingPipeline                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Analyzer  │  │   Predictor │  │  Adaptive Models    │ │
│  │             │  │             │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               AudioStreamManager                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Input     │  │   Output    │  │   Buffer Manager    │ │
│  │   Stream    │  │   Stream    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              AudioDeviceInterface                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Device     │  │  Device     │  │   Device Testing    │ │
│  │  Detection  │  │  Management │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Audio Input**: AudioStreamManager captures real-time audio
2. **Feature Extraction**: StreamingPipeline processes audio through analyzer
3. **Prediction**: Real-time predictions are generated
4. **Learning**: Adaptive models are updated with new data
5. **Integration**: StreamingIntegration connects to main system
6. **Recommendations**: Real-time recommendations are generated
7. **Feedback**: User feedback is incorporated for learning

## Performance Considerations

### Latency Optimization
- Chunk-based processing for low latency
- Queue-based buffering to prevent blocking
- Performance monitoring and adaptive optimization

### Memory Management
- Sliding window buffers with size limits
- Automatic cleanup of old data
- Memory usage monitoring

### CPU Optimization
- Efficient numpy operations
- Optional JIT compilation with numba
- Background processing threads

## Testing

### Unit Tests
```bash
# Run streaming engine tests
python -m pytest tests/test_streaming_engine.py -v

# Run specific component tests
python -m pytest tests/test_streaming_engine.py::TestAudioStreamManager -v
```

### Integration Tests
```bash
# Run complete streaming engine test
python test_streaming_engine.py
```

### Manual Testing
```bash
# Test with real audio (requires audio device)
python -c "
from real_time.streaming_integration import StreamingIntegration
integration = StreamingIntegration()
integration.start_integration()
# Let it run for a few seconds
import time; time.sleep(5)
integration.stop_integration()
"
```

## Dependencies

### Required
- `numpy`: Numerical operations
- `librosa`: Audio feature extraction
- `scipy`: Signal processing
- `psutil`: System monitoring

### Optional (for real audio I/O)
- `sounddevice`: Cross-platform audio I/O
- `pyaudio`: Alternative audio I/O library

### Testing
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting

## Configuration

### Audio Settings
```python
# Default configuration
SAMPLE_RATE = 22050
CHUNK_SIZE = 1024
WINDOW_SIZE = 2048
HOP_SIZE = 512
```

### Performance Settings
```python
# Buffer sizes
INPUT_BUFFER_SIZE = 100
OUTPUT_BUFFER_SIZE = 100
FEATURE_QUEUE_SIZE = 100
PREDICTION_QUEUE_SIZE = 50
LEARNING_QUEUE_SIZE = 50
```

## Error Handling

### Graceful Degradation
- Falls back to mock mode if no audio libraries available
- Continues operation with reduced functionality
- Comprehensive error logging

### Recovery Mechanisms
- Automatic stream restart on errors
- Buffer overflow/underflow handling
- Device disconnection recovery

## Future Enhancements

### Phase 7A.3: Predictive Engine
- LSTM models for beat prediction
- Energy transition detection
- Harmonic progression models

### Phase 7A.4: Performance Optimization
- GPU acceleration support
- Intelligent caching
- Adaptive scheduling

## Troubleshooting

### Common Issues

1. **No audio devices found**
   - Install audio drivers
   - Check device permissions
   - Use mock mode for testing

2. **High CPU usage**
   - Reduce chunk size
   - Disable predictions/learning
   - Check for audio feedback loops

3. **Audio dropouts**
   - Increase buffer sizes
   - Reduce processing complexity
   - Check system performance

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for all components
```

## API Reference

See individual component documentation for detailed API reference:
- `AudioStreamManager`: Audio I/O management
- `StreamingPipeline`: Pipeline coordination
- `AudioDeviceInterface`: Device management
- `StreamingIntegration`: System integration
