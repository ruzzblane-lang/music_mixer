# Real-Time Audio Analysis Module

This module provides real-time audio analysis capabilities for the AI Music Mixer, enabling live audio processing, adaptive feature extraction, and predictive audio intelligence.

## 🎯 Overview

The real-time module transforms the AI Music Mixer from a batch-processing system into a truly intelligent, real-time audio companion that can:

- Process audio streams in real-time
- Extract features from sliding windows
- Predict future audio characteristics
- Adapt processing based on system performance
- Learn from user feedback and improve over time

## 📁 Module Structure

```
real_time/
├── __init__.py                 # Module initialization
├── streaming_analyzer.py       # Core streaming analysis engine
├── sliding_window.py          # Sliding window management
├── adaptive_models.py         # Adaptive learning models
├── prediction_engine.py       # Predictive audio intelligence
├── performance_optimizer.py   # Real-time performance optimization
└── README.md                  # This file
```

## 🚀 Key Components

### 1. StreamingAudioAnalyzer
**Core streaming analysis engine**

- **Real-time Processing**: Analyzes audio as it streams
- **Feature Extraction**: Extracts features from sliding windows
- **Adaptive Analysis**: Adjusts processing based on performance
- **Callback System**: Provides real-time event notifications

**Key Features:**
- RMS energy, spectral centroid, zero crossing rate
- MFCC and chroma features (when performance allows)
- Real-time performance monitoring
- Event-driven architecture with callbacks

### 2. SlidingWindowManager
**Manages audio data windows for analysis**

- **Circular Buffering**: Efficient memory management
- **Window Functions**: Hann, Hamming, Blackman windows
- **Overlap Management**: Smooth transitions between windows
- **Context Awareness**: Provides audio context for analysis

**Key Features:**
- Configurable window size and hop size
- Multiple window retrieval methods
- Buffer status monitoring
- Window function customization

### 3. AdaptiveModels
**Models that learn and adapt in real-time**

- **User Preference Learning**: Learns from user feedback
- **Mixing Style Adaptation**: Adapts to user's mixing style
- **Performance Optimization**: Optimizes based on system performance
- **Prediction Models**: Predicts future audio characteristics

**Key Features:**
- Real-time model updates
- Adaptive recommendation scoring
- Performance-based optimization
- Model serialization and persistence

### 4. RealTimePredictor
**Predictive audio intelligence**

- **Beat Prediction**: Predicts where the next beat will land
- **Energy Transition Prediction**: Predicts energy changes
- **Harmonic Flow Prediction**: Predicts harmonic progressions
- **Tempo Change Prediction**: Predicts tempo variations

**Key Features:**
- Multiple prediction models
- Confidence scoring
- Accuracy tracking
- Performance monitoring

### 5. PerformanceOptimizer
**Real-time performance optimization**

- **System Monitoring**: Monitors CPU, memory, and latency
- **Adaptive Optimization**: Adjusts processing based on performance
- **Performance Levels**: LOW, MEDIUM, HIGH, MAXIMUM
- **Optimization Strategies**: Feature reduction, caching, etc.

**Key Features:**
- Real-time performance monitoring
- Automatic optimization strategies
- Performance level management
- Callback system for performance events

## 🎵 Usage Examples

### Basic Real-Time Analysis
```python
from real_time.streaming_analyzer import StreamingAudioAnalyzer

# Create analyzer
analyzer = StreamingAudioAnalyzer(
    sample_rate=22050,
    window_size=2048,
    hop_size=512
)

# Add callback for feature extraction
def on_features_extracted(features):
    print(f"Features: {features}")

analyzer.add_callback('on_feature_extracted', on_features_extracted)

# Start streaming
analyzer.start_streaming()

# Add audio chunks
audio_chunk = np.random.randn(1024)
analyzer.add_audio_chunk(audio_chunk)
```

### Sliding Window Management
```python
from real_time.sliding_window import SlidingWindowManager

# Create window manager
window_manager = SlidingWindowManager(
    window_size=2048,
    hop_size=512,
    window_function='hann'
)

# Add audio data
audio_data = np.random.randn(4096)
window_manager.add_audio_data(audio_data)

# Get current window
current_window = window_manager.get_current_window()

# Get multiple windows
windows = window_manager.get_multiple_windows(3)
```

### Adaptive Models
```python
from real_time.adaptive_models import AdaptiveModels

# Create adaptive models
models = AdaptiveModels(
    learning_rate=0.01,
    memory_size=1000
)

# Update models with user feedback
audio_features = {'rms_energy': 0.5, 'tempo': 120.0}
user_feedback = {'accepted': True, 'score': 0.8}

models.update_models(audio_features, user_feedback)

# Get adaptive recommendations
recommendations = models.get_adaptive_recommendations(
    current_features, available_tracks
)
```

### Real-Time Prediction
```python
from real_time.prediction_engine import RealTimePredictor

# Create predictor
predictor = RealTimePredictor(
    prediction_window=2.0,
    confidence_threshold=0.7
)

# Predict next beat
current_audio = np.random.randn(2048)
tempo_history = [120.0, 121.0, 120.5]

predicted_time, confidence = predictor.predict_next_beat(
    current_audio, tempo_history
)

# Predict energy transition
energy_history = [0.3, 0.4, 0.5, 0.6]
transition_type, predicted_time, confidence = predictor.predict_energy_transition(
    energy_history, {}
)
```

### Performance Optimization
```python
from real_time.performance_optimizer import PerformanceOptimizer

# Create optimizer
optimizer = PerformanceOptimizer(
    target_latency=0.05,
    cpu_threshold=0.8
)

# Start monitoring
optimizer.start_monitoring()

# Get current optimization strategies
strategies = optimizer.get_optimization_strategies()

# Get recommended settings
settings = optimizer.get_recommended_settings()
```

## 🔧 Configuration

### Performance Levels
- **LOW**: Aggressive optimizations, basic features only
- **MEDIUM**: Moderate optimizations, limited advanced features
- **HIGH**: Light optimizations, most features enabled
- **MAXIMUM**: No optimizations, all features enabled

### Optimization Strategies
- **feature_reduction**: Reduce feature extraction complexity
- **window_size_reduction**: Use smaller analysis windows
- **skip_advanced_features**: Skip computationally expensive features
- **reduce_prediction_frequency**: Reduce prediction update frequency
- **enable_caching**: Enable feature caching

## 📊 Performance Targets

### Real-Time Constraints
- **Audio Latency**: < 50ms
- **Feature Extraction**: < 10ms per window
- **Prediction Time**: < 5ms per prediction
- **CPU Usage**: < 70% on average
- **Memory Usage**: < 500MB

### Quality Targets
- **Beat Prediction Accuracy**: > 85%
- **Energy Transition Detection**: > 80%
- **Feature Extraction Quality**: > 90% of batch processing
- **User Satisfaction**: > 4.0/5.0

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all real-time tests
python -m pytest tests/test_streaming.py -v
python -m pytest tests/test_adaptive_models.py -v
python -m pytest tests/test_performance.py -v

# Run specific test
python -m pytest tests/test_streaming.py::TestStreamingAnalyzer::test_feature_extraction -v
```

## 🚀 Future Enhancements

### Planned Features
- **Deep Learning Integration**: Transformer-based models
- **Advanced Audio Effects**: Real-time audio processing
- **Hardware Acceleration**: GPU-accelerated feature extraction
- **Multi-Channel Support**: Stereo and surround sound analysis
- **Cloud Integration**: Remote processing capabilities

### Research Areas
- **Neural Audio Analysis**: Deep learning for audio understanding
- **Predictive Mixing**: AI-powered mixing decisions
- **Emotional Intelligence**: Emotion-aware audio processing
- **Cultural Context**: Culturally-aware audio analysis

## 📚 Dependencies

### Core Dependencies
- `numpy`: Numerical computing
- `librosa`: Audio analysis
- `psutil`: System monitoring
- `threading`: Multi-threading support

### Optional Dependencies
- `torch`: Deep learning models
- `scikit-learn`: Machine learning
- `sounddevice`: Audio I/O
- `websockets`: Real-time communication

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Ensure all tests pass**
5. **Submit a pull request**

## 📄 License

This module is part of the AI Music Mixer project and follows the same license terms.

---

**Ready to experience real-time audio intelligence?** 🎧✨

The real-time module provides the foundation for truly intelligent, adaptive audio processing that learns and improves with every interaction.
