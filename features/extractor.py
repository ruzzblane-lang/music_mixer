"""
Audio Feature Extractor

Extracts various audio features from music tracks including:
- Tempo (BPM)
- Key/mode detection
- Energy levels
- Spectral features (brightness, percussiveness)
"""

import librosa
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts audio features from music tracks."""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Sample rate for audio loading
            hop_length: Hop length for STFT computation
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def extract_features(self, audio_path: str) -> Dict:
        """
        Extract all features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            features = {}
            
            # Extract tempo and beat tracking
            tempo, beats = self._extract_tempo(y, sr)
            features['tempo'] = tempo
            features['beats'] = beats
            
            # Extract key and mode
            key, mode = self._extract_key(y, sr)
            features['key'] = key
            features['mode'] = mode
            
            # Extract energy features
            energy_features = self._extract_energy_features(y, sr)
            features.update(energy_features)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(y, sr)
            features.update(spectral_features)
            
            # Extract rhythm features
            rhythm_features = self._extract_rhythm_features(y, sr)
            features.update(rhythm_features)
            
            # Add metadata
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            raise
    
    def _extract_tempo(self, y: np.ndarray, sr: int) -> Tuple[float, np.ndarray]:
        """Extract tempo and beat positions."""
        try:
            # Use librosa's tempo estimation
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            return float(tempo), beats
        except Exception as e:
            logger.warning(f"Tempo extraction failed: {e}")
            return 120.0, np.array([])
    
    def _extract_key(self, y: np.ndarray, sr: int) -> Tuple[str, str]:
        """Extract key and mode using chroma features."""
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            
            # Average chroma across time
            chroma_mean = np.mean(chroma, axis=1)
            
            # Simple key detection (can be improved with more sophisticated methods)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_idx = np.argmax(chroma_mean)
            key = key_names[key_idx]
            
            # Simple mode detection (major/minor)
            # This is a simplified approach - in practice, you'd use more sophisticated methods
            mode = 'major'  # Default to major
            
            return key, mode
            
        except Exception as e:
            logger.warning(f"Key extraction failed: {e}")
            return 'C', 'major'
    
    def _extract_energy_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract energy-related features."""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            rms_mean = float(np.mean(rms))
            rms_std = float(np.std(rms))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            zcr_mean = float(np.mean(zcr))
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            brightness = float(np.mean(spectral_centroids))
            
            return {
                'rms_energy': rms_mean,
                'rms_energy_std': rms_std,
                'zero_crossing_rate': zcr_mean,
                'brightness': brightness
            }
            
        except Exception as e:
            logger.warning(f"Energy feature extraction failed: {e}")
            return {
                'rms_energy': 0.0,
                'rms_energy_std': 0.0,
                'zero_crossing_rate': 0.0,
                'brightness': 0.0
            }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract spectral features."""
        try:
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
            rolloff_mean = float(np.mean(rolloff))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
            bandwidth_mean = float(np.mean(bandwidth))
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            mfcc_means = [float(np.mean(mfcc)) for mfcc in mfccs]
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            contrast_mean = float(np.mean(contrast))
            
            return {
                'spectral_rolloff': rolloff_mean,
                'spectral_bandwidth': bandwidth_mean,
                'mfcc_means': mfcc_means,
                'spectral_contrast': contrast_mean
            }
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
            return {
                'spectral_rolloff': 0.0,
                'spectral_bandwidth': 0.0,
                'mfcc_means': [0.0] * 13,
                'spectral_contrast': 0.0
            }
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract rhythm-related features."""
        try:
            # Onset strength
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
            onset_mean = float(np.mean(onset_strength))
            
            # Tempo stability (variance in tempo over time)
            tempo_stability = float(np.std(onset_strength))
            
            return {
                'onset_strength': onset_mean,
                'tempo_stability': tempo_stability
            }
            
        except Exception as e:
            logger.warning(f"Rhythm feature extraction failed: {e}")
            return {
                'onset_strength': 0.0,
                'tempo_stability': 0.0
            }
    
    def get_feature_vector(self, features: Dict) -> np.ndarray:
        """
        Convert features dictionary to a numerical vector for ML models.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Numerical feature vector
        """
        vector = []
        
        # Tempo (normalized)
        vector.append(features.get('tempo', 120.0) / 200.0)  # Normalize to 0-1 range
        
        # Key (one-hot encoded)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = features.get('key', 'C')
        key_vector = [1.0 if k == key else 0.0 for k in key_names]
        vector.extend(key_vector)
        
        # Mode (major=1, minor=0)
        mode = features.get('mode', 'major')
        vector.append(1.0 if mode == 'major' else 0.0)
        
        # Energy features
        vector.append(features.get('rms_energy', 0.0))
        vector.append(features.get('rms_energy_std', 0.0))
        vector.append(features.get('zero_crossing_rate', 0.0))
        vector.append(features.get('brightness', 0.0))
        
        # Spectral features
        vector.append(features.get('spectral_rolloff', 0.0))
        vector.append(features.get('spectral_bandwidth', 0.0))
        vector.append(features.get('spectral_contrast', 0.0))
        
        # MFCC features
        mfcc_means = features.get('mfcc_means', [0.0] * 13)
        vector.extend(mfcc_means)
        
        # Rhythm features
        vector.append(features.get('onset_strength', 0.0))
        vector.append(features.get('tempo_stability', 0.0))
        
        return np.array(vector)
