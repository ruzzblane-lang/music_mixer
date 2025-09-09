"""
Prediction Engine

Predictive capabilities for real-time audio analysis.
Includes beat prediction, energy transition prediction, and harmonic flow prediction.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import librosa

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """
    Real-time prediction engine for audio characteristics.
    
    Predicts future audio features including beats, energy transitions,
    and harmonic progressions.
    """
    
    def __init__(self, 
                 prediction_window: float = 2.0,
                 confidence_threshold: float = 0.7,
                 history_length: int = 100):
        """
        Initialize the prediction engine.
        
        Args:
            prediction_window: Time window for predictions in seconds
            confidence_threshold: Minimum confidence for predictions
            history_length: Length of history to maintain
        """
        self.prediction_window = prediction_window
        self.confidence_threshold = confidence_threshold
        self.history_length = history_length
        
        # Prediction components
        self.beat_predictor = BeatPredictor(history_length)
        self.energy_predictor = EnergyPredictor(history_length)
        self.harmonic_predictor = HarmonicPredictor(history_length)
        self.tempo_predictor = TempoPredictor(history_length)
        
        # Prediction history
        self.prediction_history = deque(maxlen=history_length)
        self.accuracy_history = deque(maxlen=history_length)
        
        # Performance tracking
        self.prediction_times = deque(maxlen=100)
        self.last_prediction_time = 0
        
        logger.info(f"RealTimePredictor initialized: PW={prediction_window}, CT={confidence_threshold}")
    
    def predict_next_beat(self, 
                         current_audio: np.ndarray,
                         tempo_history: List[float],
                         sample_rate: int = 22050) -> Tuple[float, float]:
        """
        Predict where the next beat will land.
        
        Args:
            current_audio: Current audio window
            tempo_history: History of tempo values
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (predicted_time, confidence)
        """
        try:
            start_time = time.time()
            
            # Get current tempo
            current_tempo = tempo_history[-1] if tempo_history else 120.0
            
            # Calculate beat interval
            beat_interval = 60.0 / current_tempo  # seconds per beat
            
            # Predict next beat time
            predicted_time = beat_interval
            
            # Calculate confidence based on tempo stability
            tempo_stability = self._calculate_tempo_stability(tempo_history)
            confidence = min(0.95, tempo_stability)
            
            # Update predictor
            self.beat_predictor.update(current_audio, current_tempo, sample_rate)
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.debug(f"Beat prediction: {predicted_time:.3f}s, confidence: {confidence:.3f}")
            
            return predicted_time, confidence
            
        except Exception as e:
            logger.error(f"Error predicting next beat: {e}")
            return 0.5, 0.0
    
    def predict_energy_transition(self, 
                                energy_history: List[float],
                                audio_context: Dict) -> Tuple[str, float, float]:
        """
        Predict when energy will change.
        
        Args:
            energy_history: History of energy values
            audio_context: Current audio context
            
        Returns:
            Tuple of (transition_type, predicted_time, confidence)
        """
        try:
            start_time = time.time()
            
            if len(energy_history) < 3:
                return "stable", 0.0, 0.0
            
            # Analyze energy trends
            recent_energy = energy_history[-10:]
            energy_trend = self._calculate_energy_trend(recent_energy)
            
            # Predict transition type
            if energy_trend > 0.1:
                transition_type = "build"
                predicted_time = self._predict_build_time(energy_history)
            elif energy_trend < -0.1:
                transition_type = "drop"
                predicted_time = self._predict_drop_time(energy_history)
            else:
                transition_type = "stable"
                predicted_time = 0.0
            
            # Calculate confidence
            confidence = self._calculate_energy_confidence(energy_history, transition_type)
            
            # Update predictor
            self.energy_predictor.update(energy_history, audio_context)
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.debug(f"Energy transition: {transition_type}, time: {predicted_time:.3f}s, confidence: {confidence:.3f}")
            
            return transition_type, predicted_time, confidence
            
        except Exception as e:
            logger.error(f"Error predicting energy transition: {e}")
            return "stable", 0.0, 0.0
    
    def predict_harmonic_flow(self, 
                            harmonic_history: List[str],
                            current_chord: str) -> Tuple[str, float]:
        """
        Predict harmonic progression.
        
        Args:
            harmonic_history: History of harmonic changes
            current_chord: Current chord
            
        Returns:
            Tuple of (predicted_chord, confidence)
        """
        try:
            start_time = time.time()
            
            if not harmonic_history:
                return current_chord, 0.5
            
            # Analyze harmonic patterns
            pattern = self._analyze_harmonic_pattern(harmonic_history)
            
            # Predict next chord
            predicted_chord = self._predict_next_chord(current_chord, pattern)
            
            # Calculate confidence
            confidence = self._calculate_harmonic_confidence(harmonic_history, predicted_chord)
            
            # Update predictor
            self.harmonic_predictor.update(harmonic_history, current_chord)
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.debug(f"Harmonic prediction: {predicted_chord}, confidence: {confidence:.3f}")
            
            return predicted_chord, confidence
            
        except Exception as e:
            logger.error(f"Error predicting harmonic flow: {e}")
            return current_chord, 0.0
    
    def predict_tempo_change(self, 
                           tempo_history: List[float],
                           audio_features: Dict) -> Tuple[float, float, float]:
        """
        Predict tempo changes.
        
        Args:
            tempo_history: History of tempo values
            audio_features: Current audio features
            
        Returns:
            Tuple of (predicted_tempo, change_rate, confidence)
        """
        try:
            start_time = time.time()
            
            if len(tempo_history) < 2:
                return tempo_history[-1] if tempo_history else 120.0, 0.0, 0.0
            
            # Analyze tempo trends
            recent_tempo = tempo_history[-5:]
            tempo_trend = self._calculate_tempo_trend(recent_tempo)
            
            # Predict next tempo
            current_tempo = tempo_history[-1]
            predicted_tempo = current_tempo + (tempo_trend * 0.1)  # Small adjustment
            
            # Calculate change rate
            change_rate = tempo_trend
            
            # Calculate confidence
            confidence = self._calculate_tempo_confidence(tempo_history)
            
            # Update predictor
            self.tempo_predictor.update(tempo_history, audio_features)
            
            # Track prediction time
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.debug(f"Tempo prediction: {predicted_tempo:.1f} BPM, change: {change_rate:.3f}, confidence: {confidence:.3f}")
            
            return predicted_tempo, change_rate, confidence
            
        except Exception as e:
            logger.error(f"Error predicting tempo change: {e}")
            return tempo_history[-1] if tempo_history else 120.0, 0.0, 0.0
    
    def get_prediction_accuracy(self) -> Dict:
        """Get prediction accuracy metrics."""
        if not self.accuracy_history:
            return {'overall': 0.0, 'beat': 0.0, 'energy': 0.0, 'harmonic': 0.0}
        
        accuracies = np.array(self.accuracy_history)
        
        return {
            'overall': float(np.mean(accuracies)),
            'beat': float(np.mean([a.get('beat', 0) for a in accuracies])),
            'energy': float(np.mean([a.get('energy', 0) for a in accuracies])),
            'harmonic': float(np.mean([a.get('harmonic', 0) for a in accuracies])),
            'recent': float(np.mean(accuracies[-10:])) if len(accuracies) >= 10 else 0.0
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        if not self.prediction_times:
            return {'avg_time': 0.0, 'max_time': 0.0, 'min_time': 0.0}
        
        times = np.array(self.prediction_times)
        
        return {
            'avg_time': float(np.mean(times)),
            'max_time': float(np.max(times)),
            'min_time': float(np.min(times)),
            'total_predictions': len(self.prediction_times)
        }
    
    def _calculate_tempo_stability(self, tempo_history: List[float]) -> float:
        """Calculate tempo stability."""
        if len(tempo_history) < 2:
            return 0.5
        
        tempo_std = np.std(tempo_history[-10:])
        stability = max(0.0, 1.0 - (tempo_std / 10.0))  # 10 BPM std threshold
        
        return stability
    
    def _calculate_energy_trend(self, energy_history: List[float]) -> float:
        """Calculate energy trend."""
        if len(energy_history) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(energy_history))
        y = np.array(energy_history)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return 0.0
    
    def _predict_build_time(self, energy_history: List[float]) -> float:
        """Predict when energy will peak."""
        # Simple prediction based on current trend
        return 2.0  # 2 seconds default
    
    def _predict_drop_time(self, energy_history: List[float]) -> float:
        """Predict when energy will drop."""
        # Simple prediction based on current trend
        return 1.5  # 1.5 seconds default
    
    def _calculate_energy_confidence(self, energy_history: List[float], transition_type: str) -> float:
        """Calculate confidence in energy prediction."""
        if len(energy_history) < 3:
            return 0.0
        
        # Calculate trend consistency
        recent_trend = self._calculate_energy_trend(energy_history[-5:])
        
        if transition_type == "build" and recent_trend > 0:
            return min(0.9, abs(recent_trend) * 10)
        elif transition_type == "drop" and recent_trend < 0:
            return min(0.9, abs(recent_trend) * 10)
        else:
            return 0.3
    
    def _analyze_harmonic_pattern(self, harmonic_history: List[str]) -> Dict:
        """Analyze harmonic patterns."""
        if len(harmonic_history) < 2:
            return {}
        
        # Simple pattern analysis
        pattern = {
            'common_chords': {},
            'transitions': {},
            'length': len(harmonic_history)
        }
        
        # Count chord frequencies
        for chord in harmonic_history:
            pattern['common_chords'][chord] = pattern['common_chords'].get(chord, 0) + 1
        
        # Count transitions
        for i in range(len(harmonic_history) - 1):
            transition = f"{harmonic_history[i]}->{harmonic_history[i+1]}"
            pattern['transitions'][transition] = pattern['transitions'].get(transition, 0) + 1
        
        return pattern
    
    def _predict_next_chord(self, current_chord: str, pattern: Dict) -> str:
        """Predict next chord based on pattern."""
        if not pattern.get('transitions'):
            return current_chord
        
        # Find most common transition from current chord
        current_transitions = {k: v for k, v in pattern['transitions'].items() 
                             if k.startswith(current_chord + '->')}
        
        if current_transitions:
            most_common = max(current_transitions, key=current_transitions.get)
            return most_common.split('->')[1]
        else:
            return current_chord
    
    def _calculate_harmonic_confidence(self, harmonic_history: List[str], predicted_chord: str) -> float:
        """Calculate confidence in harmonic prediction."""
        if len(harmonic_history) < 2:
            return 0.0
        
        # Calculate based on pattern consistency
        pattern = self._analyze_harmonic_pattern(harmonic_history)
        
        if predicted_chord in pattern.get('common_chords', {}):
            frequency = pattern['common_chords'][predicted_chord]
            confidence = min(0.9, frequency / len(harmonic_history) * 2)
            return confidence
        else:
            return 0.2
    
    def _calculate_tempo_trend(self, tempo_history: List[float]) -> float:
        """Calculate tempo trend."""
        if len(tempo_history) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(tempo_history))
        y = np.array(tempo_history)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        else:
            return 0.0
    
    def _calculate_tempo_confidence(self, tempo_history: List[float]) -> float:
        """Calculate confidence in tempo prediction."""
        if len(tempo_history) < 3:
            return 0.0
        
        # Calculate based on tempo stability
        tempo_std = np.std(tempo_history[-5:])
        confidence = max(0.0, 1.0 - (tempo_std / 15.0))  # 15 BPM std threshold
        
        return confidence


class BeatPredictor:
    """Specialized beat prediction model."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.beat_history = deque(maxlen=history_length)
        self.tempo_history = deque(maxlen=history_length)
    
    def update(self, audio: np.ndarray, tempo: float, sample_rate: int):
        """Update beat predictor with new data."""
        self.tempo_history.append(tempo)
        # Additional beat analysis can be added here
    
    def predict_next_beat(self, current_tempo: float) -> float:
        """Predict next beat time."""
        return 60.0 / current_tempo


class EnergyPredictor:
    """Specialized energy prediction model."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.energy_history = deque(maxlen=history_length)
        self.transition_history = deque(maxlen=history_length)
    
    def update(self, energy_history: List[float], audio_context: Dict):
        """Update energy predictor with new data."""
        if energy_history:
            self.energy_history.append(energy_history[-1])
    
    def predict_energy_transition(self, current_energy: float) -> Tuple[str, float]:
        """Predict energy transition."""
        if len(self.energy_history) < 2:
            return "stable", 0.0
        
        recent_energy = list(self.energy_history)[-5:]
        trend = np.polyfit(range(len(recent_energy)), recent_energy, 1)[0]
        
        if trend > 0.01:
            return "build", min(0.9, trend * 100)
        elif trend < -0.01:
            return "drop", min(0.9, abs(trend) * 100)
        else:
            return "stable", 0.5


class HarmonicPredictor:
    """Specialized harmonic prediction model."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.chord_history = deque(maxlen=history_length)
        self.progression_patterns = {}
    
    def update(self, harmonic_history: List[str], current_chord: str):
        """Update harmonic predictor with new data."""
        if current_chord:
            self.chord_history.append(current_chord)
    
    def predict_next_chord(self, current_chord: str) -> str:
        """Predict next chord."""
        if len(self.chord_history) < 2:
            return current_chord
        
        # Simple prediction based on common progressions
        return current_chord


class TempoPredictor:
    """Specialized tempo prediction model."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.tempo_history = deque(maxlen=history_length)
        self.tempo_changes = deque(maxlen=history_length)
    
    def update(self, tempo_history: List[float], audio_features: Dict):
        """Update tempo predictor with new data."""
        if tempo_history:
            self.tempo_history.append(tempo_history[-1])
            
            if len(tempo_history) > 1:
                change = tempo_history[-1] - tempo_history[-2]
                self.tempo_changes.append(change)
    
    def predict_tempo_change(self) -> float:
        """Predict tempo change."""
        if not self.tempo_changes:
            return 0.0
        
        return np.mean(list(self.tempo_changes)[-5:])
