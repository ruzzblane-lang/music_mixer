"""
Audio Mixing Engine

Handles audio mixing, crossfading, and transition effects
for seamless track transitions.
"""

import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
from typing import Dict, List, Optional, Tuple
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class MixingEngine:
    """Handles audio mixing and transition effects."""
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the mixing engine.
        
        Args:
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.current_track = None
        self.next_track = None
        self.mix_position = 0.0
    
    def load_track(self, file_path: str) -> AudioSegment:
        """
        Load an audio track.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioSegment object
        """
        try:
            # Load with pydub
            audio = AudioSegment.from_file(file_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate
            audio = audio.set_frame_rate(self.sample_rate)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading track {file_path}: {e}")
            raise
    
    def create_crossfade(self, track1: AudioSegment, track2: AudioSegment, 
                        crossfade_duration: float = 8.0, 
                        mix_point1: Optional[float] = None,
                        mix_point2: Optional[float] = None) -> AudioSegment:
        """
        Create a crossfade between two tracks.
        
        Args:
            track1: First track (outgoing)
            track2: Second track (incoming)
            crossfade_duration: Duration of crossfade in seconds
            mix_point1: Mix point in track1 (seconds from start)
            mix_point2: Mix point in track2 (seconds from start)
            
        Returns:
            Mixed audio segment
        """
        try:
            # Convert duration to milliseconds
            crossfade_ms = int(crossfade_duration * 1000)
            
            # Determine mix points
            if mix_point1 is None:
                # Mix from 75% of track1
                mix_point1 = len(track1) * 0.75
            else:
                mix_point1 = mix_point1 * 1000  # Convert to ms
            
            if mix_point2 is None:
                # Mix from start of track2
                mix_point2 = 0
            else:
                mix_point2 = mix_point2 * 1000  # Convert to ms
            
            # Ensure mix points are valid
            mix_point1 = min(mix_point1, len(track1) - crossfade_ms)
            mix_point2 = min(mix_point2, len(track2) - crossfade_ms)
            
            # Extract segments for mixing
            track1_segment = track1[int(mix_point1):int(mix_point1) + crossfade_ms]
            track2_segment = track2[int(mix_point2):int(mix_point2) + crossfade_ms]
            
            # Apply crossfade
            crossfaded = track1_segment.fade_out(crossfade_ms).overlay(
                track2_segment.fade_in(crossfade_ms)
            )
            
            # Combine: track1 start + crossfade + track2 end
            result = track1[:int(mix_point1)] + crossfaded + track2[int(mix_point2) + crossfade_ms:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating crossfade: {e}")
            raise
    
    def create_beat_aligned_crossfade(self, track1: AudioSegment, track2: AudioSegment,
                                    track1_tempo: float, track2_tempo: float,
                                    crossfade_duration: float = 8.0) -> AudioSegment:
        """
        Create a beat-aligned crossfade between two tracks.
        
        Args:
            track1: First track (outgoing)
            track2: Second track (incoming)
            track1_tempo: Tempo of track1 in BPM
            track2_tempo: Tempo of track2 in BPM
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            Beat-aligned mixed audio segment
        """
        try:
            # Calculate beat intervals
            beat_interval1 = 60.0 / track1_tempo  # seconds per beat
            beat_interval2 = 60.0 / track2_tempo
            
            # Find optimal mix points (on beats)
            track1_duration = len(track1) / 1000.0  # Convert to seconds
            track2_duration = len(track2) / 1000.0
            
            # Find mix point in track1 (75% through, aligned to beat)
            target_mix_point1 = track1_duration * 0.75
            mix_point1 = self._align_to_beat(target_mix_point1, beat_interval1)
            
            # Find mix point in track2 (start, aligned to beat)
            mix_point2 = 0.0  # Start from beginning
            
            # Adjust crossfade duration to be beat-aligned
            avg_tempo = (track1_tempo + track2_tempo) / 2.0
            beat_interval = 60.0 / avg_tempo
            crossfade_beats = max(4, int(crossfade_duration / beat_interval))
            crossfade_duration = crossfade_beats * beat_interval
            
            return self.create_crossfade(track1, track2, crossfade_duration, mix_point1, mix_point2)
            
        except Exception as e:
            logger.error(f"Error creating beat-aligned crossfade: {e}")
            # Fallback to regular crossfade
            return self.create_crossfade(track1, track2, crossfade_duration)
    
    def _align_to_beat(self, time_point: float, beat_interval: float) -> float:
        """Align a time point to the nearest beat."""
        beat_number = round(time_point / beat_interval)
        return beat_number * beat_interval
    
    def apply_eq_ducking(self, track1: AudioSegment, track2: AudioSegment,
                        crossfade_duration: float = 8.0) -> AudioSegment:
        """
        Apply EQ ducking to prevent bass conflicts during transitions.
        
        Args:
            track1: First track (outgoing)
            track2: Second track (incoming)
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            EQ-processed audio segment
        """
        try:
            # This is a simplified EQ ducking implementation
            # In a full implementation, you'd use more sophisticated EQ processing
            
            crossfade_ms = int(crossfade_duration * 1000)
            
            # Apply high-pass filter to track1 during crossfade (reduce bass)
            track1_start = len(track1) - crossfade_ms
            track1_end = track1[track1_start:]
            
            # Simple bass reduction (this is a placeholder - real EQ would be more complex)
            track1_end = track1_end.low_pass_filter(200)  # Reduce frequencies below 200Hz
            
            # Combine tracks
            result = track1[:track1_start] + track1_end.overlay(track2[:crossfade_ms])
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying EQ ducking: {e}")
            # Fallback to regular crossfade
            return self.create_crossfade(track1, track2, crossfade_duration)
    
    def apply_transition_effect(self, audio: AudioSegment, effect_type: str = "reverb") -> AudioSegment:
        """
        Apply transition effects to audio.
        
        Args:
            audio: Audio segment to process
            effect_type: Type of effect to apply
            
        Returns:
            Processed audio segment
        """
        try:
            if effect_type == "reverb":
                # Simple reverb simulation (placeholder)
                # In a real implementation, you'd use proper reverb algorithms
                return audio + AudioSegment.silent(duration=500)  # Add some tail
            
            elif effect_type == "filter_sweep":
                # Simple filter sweep (placeholder)
                # This would involve frequency filtering over time
                return audio
            
            elif effect_type == "echo":
                # Simple echo effect
                echo = audio - 10  # Reduce volume
                return audio.overlay(echo, position=200)  # 200ms delay
            
            else:
                return audio
                
        except Exception as e:
            logger.error(f"Error applying transition effect {effect_type}: {e}")
            return audio
    
    def export_mix(self, audio: AudioSegment, output_path: str, format: str = "mp3") -> bool:
        """
        Export mixed audio to file.
        
        Args:
            audio: Audio segment to export
            output_path: Output file path
            format: Output format (mp3, wav, etc.)
            
        Returns:
            True if export was successful
        """
        try:
            # Normalize audio before export
            audio = normalize(audio)
            
            # Export
            audio.export(output_path, format=format)
            
            logger.info(f"Exported mix to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting mix: {e}")
            return False
    
    def play_audio(self, audio: AudioSegment) -> bool:
        """
        Play audio (requires pygame or similar).
        
        Args:
            audio: Audio segment to play
            
        Returns:
            True if playback was successful
        """
        try:
            # This is a placeholder - in a real implementation you'd use pygame or similar
            logger.info("Audio playback not implemented - use export_mix() to save file")
            return False
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def get_track_info(self, file_path: str) -> Dict:
        """
        Get information about a track.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary with track information
        """
        try:
            audio = self.load_track(file_path)
            
            return {
                'duration': len(audio) / 1000.0,  # seconds
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'bit_depth': audio.sample_width * 8,
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting track info for {file_path}: {e}")
            return {}
    
    def create_playlist_mix(self, track_paths: List[str], output_path: str,
                          crossfade_duration: float = 8.0) -> bool:
        """
        Create a continuous mix from a playlist of tracks.
        
        Args:
            track_paths: List of track file paths
            output_path: Output file path
            crossfade_duration: Duration of crossfades between tracks
            
        Returns:
            True if mix creation was successful
        """
        try:
            if len(track_paths) < 2:
                logger.error("Need at least 2 tracks for a mix")
                return False
            
            # Load first track
            current_mix = self.load_track(track_paths[0])
            
            # Mix with remaining tracks
            for i, track_path in enumerate(track_paths[1:], 1):
                logger.info(f"Mixing track {i+1}/{len(track_paths)}: {track_path}")
                
                next_track = self.load_track(track_path)
                current_mix = self.create_crossfade(current_mix, next_track, crossfade_duration)
            
            # Export final mix
            return self.export_mix(current_mix, output_path)
            
        except Exception as e:
            logger.error(f"Error creating playlist mix: {e}")
            return False
