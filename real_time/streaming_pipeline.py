"""
Streaming Pipeline

Coordinates all streaming components for real-time audio analysis.
Manages the complete pipeline from audio input to feature output.
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import queue

from .audio_stream_manager import AudioStreamManager
from .streaming_analyzer import StreamingAudioAnalyzer
from .sliding_window import SlidingWindowManager
from .adaptive_models import AdaptiveModels
from .prediction_engine import RealTimePredictor
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """
    Main streaming pipeline coordinator.
    
    Manages the complete real-time audio analysis pipeline
    from audio input to feature extraction and predictions.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 window_size: int = 2048,
                 hop_size: int = 512,
                 enable_predictions: bool = True,
                 enable_learning: bool = True):
        """
        Initialize the streaming pipeline.
        
        Args:
            sample_rate: Audio sample rate
            window_size: Analysis window size
            hop_size: Hop size for windowing
            enable_predictions: Enable predictive analysis
            enable_learning: Enable adaptive learning
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.enable_predictions = enable_predictions
        self.enable_learning = enable_learning
        
        # Initialize components
        self.audio_manager = AudioStreamManager(
            sample_rate=sample_rate,
            chunk_size=hop_size
        )
        
        self.analyzer = StreamingAudioAnalyzer(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size
        )
        
        self.window_manager = SlidingWindowManager(
            window_size=window_size,
            hop_size=hop_size,
            sample_rate=sample_rate
        )
        
        self.adaptive_models = AdaptiveModels() if enable_learning else None
        self.predictor = RealTimePredictor() if enable_predictions else None
        self.optimizer = PerformanceOptimizer()
        
        # Pipeline state
        self.is_running = False
        self.pipeline_thread = None
        self.stop_event = threading.Event()
        
        # Data flow
        self.feature_queue = queue.Queue(maxsize=100)
        self.prediction_queue = queue.Queue(maxsize=50)
        self.learning_queue = queue.Queue(maxsize=50)
        
        # Performance tracking
        self.performance_metrics = {
            'pipeline_cycles': 0,
            'features_extracted': 0,
            'predictions_made': 0,
            'models_updated': 0,
            'processing_time': deque(maxlen=100),
            'pipeline_latency': deque(maxlen=100)
        }
        
        # Callbacks
        self.callbacks = {
            'on_features_extracted': [],
            'on_prediction_made': [],
            'on_model_updated': [],
            'on_pipeline_error': [],
            'on_performance_warning': []
        }
        
        # Setup component integration
        self._setup_component_integration()
        
        logger.info(f"StreamingPipeline initialized: SR={sample_rate}, WS={window_size}, HS={hop_size}")
    
    def _setup_component_integration(self):
        """Setup integration between components."""
        # Connect audio manager to analyzer
        self.audio_manager.add_callback('on_audio_input', self._on_audio_input)
        
        # Connect analyzer to window manager
        self.analyzer.add_callback('on_feature_extracted', self._on_feature_extracted)
        
        # Connect optimizer to analyzer
        self.optimizer.add_callback('on_optimization_applied', self._on_optimization_applied)
        
        # Connect performance monitoring
        self.optimizer.add_callback('on_performance_warning', self._on_performance_warning)
    
    def start_pipeline(self) -> bool:
        """
        Start the streaming pipeline.
        
        Returns:
            True if pipeline started successfully
        """
        try:
            if self.is_running:
                logger.warning("Pipeline already running")
                return False
            
            # Start audio input
            if not self.audio_manager.start_input_stream():
                logger.error("Failed to start audio input")
                return False
            
            # Start performance monitoring
            self.optimizer.start_monitoring()
            
            # Start pipeline processing thread
            self.stop_event.clear()
            self.pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self.pipeline_thread.start()
            
            self.is_running = True
            logger.info("Streaming pipeline started")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting pipeline: {e}")
            self._trigger_callbacks('on_pipeline_error', {'error': str(e)})
            return False
    
    def stop_pipeline(self):
        """Stop the streaming pipeline."""
        try:
            self.is_running = False
            self.stop_event.set()
            
            # Stop audio streams
            self.audio_manager.stop_streams()
            
            # Stop performance monitoring
            self.optimizer.stop_monitoring()
            
            # Wait for pipeline thread
            if self.pipeline_thread and self.pipeline_thread.is_alive():
                self.pipeline_thread.join(timeout=2.0)
            
            logger.info("Streaming pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    def _pipeline_loop(self):
        """Main pipeline processing loop."""
        logger.info("Pipeline processing loop started")
        
        while not self.stop_event.is_set():
            try:
                start_time = time.time()
                
                # Process audio chunks
                self._process_audio_chunks()
                
                # Process feature queue
                self._process_feature_queue()
                
                # Process prediction queue
                if self.enable_predictions:
                    self._process_prediction_queue()
                
                # Process learning queue
                if self.enable_learning:
                    self._process_learning_queue()
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.performance_metrics['processing_time'].append(processing_time)
                self.performance_metrics['pipeline_cycles'] += 1
                
                # Check for performance issues
                if processing_time > 0.1:  # 100ms threshold
                    self._trigger_callbacks('on_performance_warning', {
                        'processing_time': processing_time,
                        'threshold': 0.1
                    })
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")
                self._trigger_callbacks('on_pipeline_error', {'error': str(e)})
                time.sleep(0.1)  # Wait before retrying
        
        logger.info("Pipeline processing loop stopped")
    
    def _process_audio_chunks(self):
        """Process audio chunks from input buffer."""
        try:
            # Get audio chunk with timeout
            audio_chunk = self.audio_manager.get_audio_chunk(timeout=0.01)
            
            if audio_chunk is not None:
                # Add to window manager
                self.window_manager.add_audio_data(audio_chunk.flatten())
                
                # Get current window for analysis
                current_window = self.window_manager.get_current_window()
                
                if current_window is not None:
                    # Extract features
                    features = self.analyzer._extract_features(current_window)
                    
                    if features:
                        # Add to feature queue
                        try:
                            self.feature_queue.put_nowait({
                                'features': features,
                                'timestamp': time.time(),
                                'window': current_window
                            })
                            self.performance_metrics['features_extracted'] += 1
                        except queue.Full:
                            logger.warning("Feature queue full, dropping features")
                
        except Exception as e:
            logger.error(f"Error processing audio chunks: {e}")
    
    def _process_feature_queue(self):
        """Process features from the feature queue."""
        try:
            while not self.feature_queue.empty():
                feature_data = self.feature_queue.get_nowait()
                features = feature_data['features']
                
                # Trigger feature callbacks
                self._trigger_callbacks('on_features_extracted', features)
                
                # Add to prediction queue if predictions enabled
                if self.enable_predictions and self.predictor:
                    try:
                        self.prediction_queue.put_nowait(feature_data)
                    except queue.Full:
                        logger.warning("Prediction queue full, dropping features")
                
                # Add to learning queue if learning enabled
                if self.enable_learning and self.adaptive_models:
                    try:
                        self.learning_queue.put_nowait(feature_data)
                    except queue.Full:
                        logger.warning("Learning queue full, dropping features")
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing feature queue: {e}")
    
    def _process_prediction_queue(self):
        """Process predictions from the prediction queue."""
        try:
            while not self.prediction_queue.empty():
                feature_data = self.prediction_queue.get_nowait()
                features = feature_data['features']
                
                # Make predictions
                predictions = self._make_predictions(features)
                
                if predictions:
                    # Trigger prediction callbacks
                    self._trigger_callbacks('on_prediction_made', predictions)
                    self.performance_metrics['predictions_made'] += 1
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing prediction queue: {e}")
    
    def _process_learning_queue(self):
        """Process learning from the learning queue."""
        try:
            while not self.learning_queue.empty():
                feature_data = self.learning_queue.get_nowait()
                features = feature_data['features']
                
                # Update models (this will handle timing internally)
                if self.adaptive_models.update_models(features):
                    self.performance_metrics['models_updated'] += 1
                    
                    # Trigger model update callbacks
                    self._trigger_callbacks('on_model_updated', {
                        'features': features,
                        'timestamp': time.time()
                    })
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing learning queue: {e}")
    
    def _make_predictions(self, features: Dict) -> Optional[Dict]:
        """Make predictions based on features."""
        try:
            if not self.predictor:
                return None
            
            predictions = {}
            
            # Beat prediction (if tempo is available)
            if 'tempo' in features:
                # Mock tempo history for now
                tempo_history = [features.get('tempo', 120.0)]
                current_audio = np.random.randn(2048)  # Mock audio
                
                predicted_time, confidence = self.predictor.predict_next_beat(
                    current_audio, tempo_history
                )
                predictions['beat_prediction'] = {
                    'predicted_time': predicted_time,
                    'confidence': confidence
                }
            
            # Energy transition prediction
            if 'rms_energy' in features:
                # Mock energy history
                energy_history = [features.get('rms_energy', 0.5)]
                
                transition_type, predicted_time, confidence = self.predictor.predict_energy_transition(
                    energy_history, {}
                )
                predictions['energy_prediction'] = {
                    'transition_type': transition_type,
                    'predicted_time': predicted_time,
                    'confidence': confidence
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def _on_audio_input(self, audio_data: np.ndarray):
        """Handle audio input from audio manager."""
        # This is called by the audio manager callback
        pass
    
    def _on_feature_extracted(self, features: Dict):
        """Handle features extracted by analyzer."""
        # This is called by the analyzer callback
        pass
    
    def _on_optimization_applied(self, optimizations: Dict):
        """Handle optimization applied by optimizer."""
        logger.info(f"Optimizations applied: {optimizations}")
    
    def _on_performance_warning(self, warning: Dict):
        """Handle performance warning from optimizer."""
        self._trigger_callbacks('on_performance_warning', warning)
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'is_running': self.is_running,
            'audio_manager_status': self.audio_manager.get_performance_metrics(),
            'analyzer_status': self.analyzer.get_performance_metrics(),
            'window_manager_status': self.window_manager.get_buffer_status(),
            'optimizer_status': self.optimizer.get_performance_metrics(),
            'pipeline_metrics': self.performance_metrics.copy(),
            'queue_sizes': {
                'feature_queue': self.feature_queue.qsize(),
                'prediction_queue': self.prediction_queue.qsize(),
                'learning_queue': self.learning_queue.qsize()
            }
        }
    
    def get_recommendations(self, current_features: Dict, available_tracks: List[Dict]) -> List[tuple]:
        """Get recommendations using adaptive models."""
        if not self.adaptive_models:
            return []
        
        return self.adaptive_models.get_adaptive_recommendations(current_features, available_tracks)
    
    def add_user_feedback(self, current_track: str, recommended_track: str, accepted: bool, score: Optional[float] = None):
        """Add user feedback for learning."""
        if not self.adaptive_models:
            return
        
        # This would be called when user provides feedback
        # For now, we'll just log it
        logger.info(f"User feedback: {current_track} -> {recommended_track}, accepted: {accepted}, score: {score}")
    
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
        self.stop_pipeline()
