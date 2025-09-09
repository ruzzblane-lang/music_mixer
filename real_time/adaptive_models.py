"""
Adaptive Models

Models that adapt and learn in real-time based on user feedback
and audio analysis results.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import json

logger = logging.getLogger(__name__)


class AdaptiveModels:
    """
    Manages adaptive models that learn and improve in real-time.
    
    Includes user preference models, mixing style models,
    and performance optimization models.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 memory_size: int = 1000,
                 adaptation_threshold: float = 0.1):
        """
        Initialize adaptive models.
        
        Args:
            learning_rate: Learning rate for model updates
            memory_size: Size of memory buffer for learning
            adaptation_threshold: Threshold for triggering model updates
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.adaptation_threshold = adaptation_threshold
        
        # Model components
        self.user_preference_model = UserPreferenceModel(memory_size)
        self.mixing_style_model = MixingStyleModel(memory_size)
        self.performance_model = PerformanceModel(memory_size)
        self.prediction_model = PredictionModel(memory_size)
        
        # Learning state
        self.is_learning = True
        self.last_update_time = 0  # Allow first update immediately
        self.update_frequency = 1.0  # Update every 1 second
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        
        logger.info(f"AdaptiveModels initialized: LR={learning_rate}, MS={memory_size}")
    
    def update_models(self, 
                     audio_features: Dict,
                     user_feedback: Optional[Dict] = None,
                     performance_data: Optional[Dict] = None) -> bool:
        """
        Update models based on new data.
        
        Args:
            audio_features: Current audio features
            user_feedback: User feedback data
            performance_data: Performance metrics
            
        Returns:
            True if models were updated, False otherwise
        """
        try:
            current_time = time.time()
            
            # Check if it's time to update (allow first update or if enough time has passed)
            if self.last_update_time > 0 and current_time - self.last_update_time < self.update_frequency:
                return False
            
            # Update individual models
            self.user_preference_model.update(audio_features, user_feedback)
            self.mixing_style_model.update(audio_features, user_feedback)
            self.performance_model.update(performance_data)
            self.prediction_model.update(audio_features, user_feedback)
            
            # Update learning state
            self.last_update_time = current_time
            
            # Track performance
            self._track_performance()
            
            logger.debug("Models updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return False
    
    def get_adaptive_recommendations(self, 
                                   current_features: Dict,
                                   available_tracks: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Get recommendations using adaptive models.
        
        Args:
            current_features: Current audio features
            available_tracks: List of available tracks
            
        Returns:
            List of (track, score) tuples
        """
        try:
            recommendations = []
            
            for track in available_tracks:
                # Get base compatibility score
                base_score = self._calculate_base_compatibility(current_features, track)
                
                # Apply user preference adjustments
                user_adjustment = self.user_preference_model.get_preference_adjustment(track)
                
                # Apply mixing style adjustments
                style_adjustment = self.mixing_style_model.get_style_adjustment(track)
                
                # Apply performance-based adjustments
                performance_adjustment = self.performance_model.get_performance_adjustment()
                
                # Calculate final score
                final_score = base_score * user_adjustment * style_adjustment * performance_adjustment
                
                recommendations.append((track, final_score))
            
            # Sort by score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting adaptive recommendations: {e}")
            return []
    
    def predict_next_features(self, 
                            current_features: Dict,
                            history_length: int = 10) -> Dict:
        """
        Predict next audio features based on current state.
        
        Args:
            current_features: Current audio features
            history_length: Length of history to consider
            
        Returns:
            Predicted next features
        """
        try:
            return self.prediction_model.predict_next(current_features, history_length)
            
        except Exception as e:
            logger.error(f"Error predicting next features: {e}")
            return {}
    
    def get_model_status(self) -> Dict:
        """Get current status of all models."""
        return {
            'user_preference_model': self.user_preference_model.get_status(),
            'mixing_style_model': self.mixing_style_model.get_status(),
            'performance_model': self.performance_model.get_status(),
            'prediction_model': self.prediction_model.get_status(),
            'learning_enabled': self.is_learning,
            'last_update': self.last_update_time,
            'performance_history': list(self.performance_history),
            'accuracy_history': list(self.accuracy_history)
        }
    
    def _calculate_base_compatibility(self, features1: Dict, track2: Dict) -> float:
        """Calculate base compatibility between features and track."""
        try:
            # Extract track features
            track_features = track2.get('features', {})
            
            # Calculate compatibility scores
            tempo_score = self._tempo_compatibility(features1.get('tempo', 120), track_features.get('tempo', 120))
            key_score = self._key_compatibility(features1.get('key', 'C'), track_features.get('key', 'C'))
            energy_score = self._energy_compatibility(features1.get('rms_energy', 0.5), track_features.get('rms_energy', 0.5))
            
            # Weighted combination
            final_score = (tempo_score * 0.4 + key_score * 0.3 + energy_score * 0.3)
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating base compatibility: {e}")
            return 0.5
    
    def _tempo_compatibility(self, tempo1: float, tempo2: float) -> float:
        """Calculate tempo compatibility."""
        tempo_diff = abs(tempo1 - tempo2)
        return max(0.0, 1.0 - (tempo_diff / 50.0))  # 50 BPM tolerance
    
    def _key_compatibility(self, key1: str, key2: str) -> float:
        """Calculate key compatibility."""
        # Simple key compatibility (can be enhanced)
        if key1 == key2:
            return 1.0
        elif key1 in ['C', 'G', 'D', 'A', 'E', 'B', 'F#'] and key2 in ['C', 'G', 'D', 'A', 'E', 'B', 'F#']:
            return 0.8
        else:
            return 0.5
    
    def _energy_compatibility(self, energy1: float, energy2: float) -> float:
        """Calculate energy compatibility."""
        energy_diff = abs(energy1 - energy2)
        return max(0.0, 1.0 - (energy_diff / 0.5))  # 0.5 energy tolerance
    
    def _track_performance(self):
        """Track model performance."""
        try:
            # Calculate current performance metrics
            current_performance = {
                'timestamp': time.time(),
                'learning_rate': self.learning_rate,
                'memory_usage': sum(getattr(model, 'get_memory_usage', lambda: 0)() for model in [
                    self.user_preference_model,
                    self.mixing_style_model,
                    self.performance_model,
                    self.prediction_model
                ]),
                'update_frequency': self.update_frequency
            }
            
            self.performance_history.append(current_performance)
            
        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
    
    def save_models(self, filepath: str) -> bool:
        """Save models to file."""
        try:
            model_data = {
                'user_preference_model': self.user_preference_model.serialize(),
                'mixing_style_model': self.mixing_style_model.serialize(),
                'performance_model': self.performance_model.serialize(),
                'prediction_model': self.prediction_model.serialize(),
                'learning_rate': self.learning_rate,
                'adaptation_threshold': self.adaptation_threshold
            }
            
            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load models from file."""
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            self.user_preference_model.deserialize(model_data.get('user_preference_model', {}))
            self.mixing_style_model.deserialize(model_data.get('mixing_style_model', {}))
            self.performance_model.deserialize(model_data.get('performance_model', {}))
            self.prediction_model.deserialize(model_data.get('prediction_model', {}))
            
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.adaptation_threshold = model_data.get('adaptation_threshold', self.adaptation_threshold)
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


class UserPreferenceModel:
    """Model for learning user preferences."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.preference_weights = {}
        self.feedback_history = deque(maxlen=memory_size)
        self.learning_rate = 0.01
    
    def update(self, features: Dict, feedback: Optional[Dict]):
        """Update model with new data."""
        if feedback:
            self.feedback_history.append({
                'features': features,
                'feedback': feedback,
                'timestamp': time.time()
            })
            self._update_weights(features, feedback)
    
    def get_preference_adjustment(self, track: Dict) -> float:
        """Get preference adjustment for a track."""
        # Simple implementation - can be enhanced
        return 1.0
    
    def get_status(self) -> Dict:
        """Get model status."""
        return {
            'memory_usage': len(self.feedback_history),
            'learning_rate': self.learning_rate
        }
    
    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return len(self.feedback_history)
    
    def _update_weights(self, features: Dict, feedback: Dict):
        """Update preference weights."""
        # Implementation for weight updates
        pass
    
    def serialize(self) -> Dict:
        """Serialize model data."""
        return {
            'preference_weights': self.preference_weights,
            'learning_rate': self.learning_rate
        }
    
    def deserialize(self, data: Dict):
        """Deserialize model data."""
        self.preference_weights = data.get('preference_weights', {})
        self.learning_rate = data.get('learning_rate', self.learning_rate)


class MixingStyleModel:
    """Model for learning mixing style preferences."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.style_weights = {}
        self.mixing_history = deque(maxlen=memory_size)
        self.learning_rate = 0.01
    
    def update(self, features: Dict, feedback: Optional[Dict]):
        """Update model with new data."""
        if feedback:
            self.mixing_history.append({
                'features': features,
                'feedback': feedback,
                'timestamp': time.time()
            })
    
    def get_style_adjustment(self, track: Dict) -> float:
        """Get style adjustment for a track."""
        return 1.0
    
    def get_status(self) -> Dict:
        """Get model status."""
        return {
            'memory_usage': len(self.mixing_history),
            'learning_rate': self.learning_rate
        }
    
    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return len(self.mixing_history)
    
    def serialize(self) -> Dict:
        """Serialize model data."""
        return {
            'style_weights': self.style_weights,
            'learning_rate': self.learning_rate
        }
    
    def deserialize(self, data: Dict):
        """Deserialize model data."""
        self.style_weights = data.get('style_weights', {})
        self.learning_rate = data.get('learning_rate', self.learning_rate)


class PerformanceModel:
    """Model for performance optimization."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.performance_history = deque(maxlen=memory_size)
        self.optimization_rules = {}
    
    def update(self, performance_data: Optional[Dict]):
        """Update model with performance data."""
        if performance_data:
            self.performance_history.append({
                'data': performance_data,
                'timestamp': time.time()
            })
    
    def get_performance_adjustment(self) -> float:
        """Get performance-based adjustment."""
        return 1.0
    
    def get_status(self) -> Dict:
        """Get model status."""
        return {
            'memory_usage': len(self.performance_history)
        }
    
    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return len(self.performance_history)
    
    def serialize(self) -> Dict:
        """Serialize model data."""
        return {
            'optimization_rules': self.optimization_rules
        }
    
    def deserialize(self, data: Dict):
        """Deserialize model data."""
        self.optimization_rules = data.get('optimization_rules', {})


class PredictionModel:
    """Model for predicting future audio features."""
    
    def __init__(self, memory_size: int = 1000):
        self.memory_size = memory_size
        self.feature_history = deque(maxlen=memory_size)
        self.prediction_weights = {}
    
    def update(self, features: Dict, feedback: Optional[Dict]):
        """Update model with new features."""
        self.feature_history.append({
            'features': features,
            'timestamp': time.time()
        })
    
    def predict_next(self, current_features: Dict, history_length: int = 10) -> Dict:
        """Predict next features."""
        # Simple prediction - can be enhanced with ML models
        return current_features.copy()
    
    def get_status(self) -> Dict:
        """Get model status."""
        return {
            'memory_usage': len(self.feature_history)
        }
    
    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return len(self.feature_history)
    
    def serialize(self) -> Dict:
        """Serialize model data."""
        return {
            'prediction_weights': self.prediction_weights
        }
    
    def deserialize(self, data: Dict):
        """Deserialize model data."""
        self.prediction_weights = data.get('prediction_weights', {})
