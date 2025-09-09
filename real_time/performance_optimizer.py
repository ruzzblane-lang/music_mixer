"""
Performance Optimizer

Real-time performance optimization for audio analysis.
Monitors system performance and adapts processing strategies.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create a mock psutil for when it's not available
    class MockPsutil:
        @staticmethod
        def cpu_percent(interval=0.1):
            return 50.0  # Mock CPU usage
        
        @staticmethod
        def virtual_memory():
            class MockMemory:
                percent = 60.0  # Mock memory usage
            return MockMemory()
    
    psutil = MockPsutil()

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance levels for adaptive processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class PerformanceOptimizer:
    """
    Real-time performance optimizer for audio analysis.
    
    Monitors system performance and adapts processing strategies
    to maintain real-time constraints.
    """
    
    def __init__(self, 
                 target_latency: float = 0.05,  # 50ms
                 cpu_threshold: float = 0.8,    # 80%
                 memory_threshold: float = 0.85, # 85%
                 monitoring_interval: float = 0.1):  # 100ms
        """
        Initialize the performance optimizer.
        
        Args:
            target_latency: Target processing latency in seconds
            cpu_threshold: CPU usage threshold for optimization
            memory_threshold: Memory usage threshold for optimization
            monitoring_interval: Monitoring interval in seconds
        """
        self.target_latency = target_latency
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.monitoring_interval = monitoring_interval
        
        # Performance monitoring
        self.performance_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'processing_time': deque(maxlen=100),
            'audio_latency': deque(maxlen=100),
            'feature_extraction_time': deque(maxlen=100)
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            'feature_reduction': False,
            'window_size_reduction': False,
            'skip_advanced_features': False,
            'reduce_prediction_frequency': False,
            'enable_caching': True
        }
        
        # Performance level
        self.current_performance_level = PerformanceLevel.HIGH
        self.performance_history = deque(maxlen=50)
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitoring_lock = threading.Lock()
        
        # Callbacks for performance events
        self.callbacks = {
            'on_performance_warning': [],
            'on_performance_critical': [],
            'on_optimization_applied': []
        }
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - using mock performance data")
        
        logger.info(f"PerformanceOptimizer initialized: TL={target_latency}, CT={cpu_threshold}")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                with self.monitoring_lock:
                    self._collect_metrics()
                    self._analyze_performance()
                    self._apply_optimizations()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self):
        """Collect current performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.performance_metrics['cpu_usage'].append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            self.performance_metrics['memory_usage'].append(memory_percent)
            
            # Audio latency (placeholder - would be measured from audio system)
            audio_latency = self._estimate_audio_latency()
            self.performance_metrics['audio_latency'].append(audio_latency)
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    def _estimate_audio_latency(self) -> float:
        """Estimate audio latency based on system performance."""
        # This is a simplified estimation
        # In a real implementation, this would measure actual audio latency
        
        cpu_usage = self.performance_metrics['cpu_usage'][-1] if self.performance_metrics['cpu_usage'] else 0
        memory_usage = self.performance_metrics['memory_usage'][-1] if self.performance_metrics['memory_usage'] else 0
        
        # Estimate latency based on system load
        base_latency = 0.02  # 20ms base latency
        cpu_factor = cpu_usage / 100.0 * 0.05  # Up to 50ms additional from CPU
        memory_factor = memory_usage * 0.03  # Up to 30ms additional from memory
        
        estimated_latency = base_latency + cpu_factor + memory_factor
        return min(0.2, estimated_latency)  # Cap at 200ms
    
    def _analyze_performance(self):
        """Analyze current performance and determine optimization needs."""
        try:
            if not self.performance_metrics['cpu_usage']:
                return
            
            # Get recent metrics
            recent_cpu = np.mean(list(self.performance_metrics['cpu_usage'])[-10:])
            recent_memory = np.mean(list(self.performance_metrics['memory_usage'])[-10:])
            recent_latency = np.mean(list(self.performance_metrics['audio_latency'])[-10:])
            
            # Determine performance level
            if recent_cpu > 90 or recent_memory > 0.95 or recent_latency > 0.1:
                new_level = PerformanceLevel.LOW
            elif recent_cpu > 70 or recent_memory > 0.85 or recent_latency > 0.075:
                new_level = PerformanceLevel.MEDIUM
            elif recent_cpu > 50 or recent_memory > 0.75 or recent_latency > 0.05:
                new_level = PerformanceLevel.HIGH
            else:
                new_level = PerformanceLevel.MAXIMUM
            
            # Update performance level if changed
            if new_level != self.current_performance_level:
                old_level = self.current_performance_level
                self.current_performance_level = new_level
                self.performance_history.append({
                    'timestamp': time.time(),
                    'old_level': old_level.value,
                    'new_level': new_level.value,
                    'cpu': recent_cpu,
                    'memory': recent_memory,
                    'latency': recent_latency
                })
                
                logger.info(f"Performance level changed: {old_level.value} -> {new_level.value}")
                
                # Trigger callbacks
                if new_level in [PerformanceLevel.LOW, PerformanceLevel.MEDIUM]:
                    self._trigger_callbacks('on_performance_warning', {
                        'level': new_level.value,
                        'cpu': recent_cpu,
                        'memory': recent_memory,
                        'latency': recent_latency
                    })
                
                if new_level == PerformanceLevel.LOW:
                    self._trigger_callbacks('on_performance_critical', {
                        'cpu': recent_cpu,
                        'memory': recent_memory,
                        'latency': recent_latency
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    def _apply_optimizations(self):
        """Apply optimizations based on current performance level."""
        try:
            optimizations_applied = []
            
            if self.current_performance_level == PerformanceLevel.LOW:
                # Apply aggressive optimizations
                if not self.optimization_strategies['feature_reduction']:
                    self.optimization_strategies['feature_reduction'] = True
                    optimizations_applied.append('feature_reduction')
                
                if not self.optimization_strategies['skip_advanced_features']:
                    self.optimization_strategies['skip_advanced_features'] = True
                    optimizations_applied.append('skip_advanced_features')
                
                if not self.optimization_strategies['reduce_prediction_frequency']:
                    self.optimization_strategies['reduce_prediction_frequency'] = True
                    optimizations_applied.append('reduce_prediction_frequency')
                
                if not self.optimization_strategies['window_size_reduction']:
                    self.optimization_strategies['window_size_reduction'] = True
                    optimizations_applied.append('window_size_reduction')
            
            elif self.current_performance_level == PerformanceLevel.MEDIUM:
                # Apply moderate optimizations
                if not self.optimization_strategies['skip_advanced_features']:
                    self.optimization_strategies['skip_advanced_features'] = True
                    optimizations_applied.append('skip_advanced_features')
                
                if not self.optimization_strategies['reduce_prediction_frequency']:
                    self.optimization_strategies['reduce_prediction_frequency'] = True
                    optimizations_applied.append('reduce_prediction_frequency')
            
            elif self.current_performance_level == PerformanceLevel.HIGH:
                # Apply light optimizations
                if not self.optimization_strategies['reduce_prediction_frequency']:
                    self.optimization_strategies['reduce_prediction_frequency'] = True
                    optimizations_applied.append('reduce_prediction_frequency')
            
            else:  # MAXIMUM performance
                # Remove optimizations
                if self.optimization_strategies['feature_reduction']:
                    self.optimization_strategies['feature_reduction'] = False
                    optimizations_applied.append('remove_feature_reduction')
                
                if self.optimization_strategies['skip_advanced_features']:
                    self.optimization_strategies['skip_advanced_features'] = False
                    optimizations_applied.append('remove_skip_advanced_features')
                
                if self.optimization_strategies['reduce_prediction_frequency']:
                    self.optimization_strategies['reduce_prediction_frequency'] = False
                    optimizations_applied.append('remove_reduce_prediction_frequency')
                
                if self.optimization_strategies['window_size_reduction']:
                    self.optimization_strategies['window_size_reduction'] = False
                    optimizations_applied.append('remove_window_size_reduction')
            
            # Trigger callbacks for applied optimizations
            if optimizations_applied:
                self._trigger_callbacks('on_optimization_applied', {
                    'optimizations': optimizations_applied,
                    'performance_level': self.current_performance_level.value
                })
                
                logger.info(f"Optimizations applied: {optimizations_applied}")
            
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
    
    def get_optimization_strategies(self) -> Dict[str, bool]:
        """Get current optimization strategies."""
        return self.optimization_strategies.copy()
    
    def set_optimization_strategy(self, strategy: str, enabled: bool):
        """Set a specific optimization strategy."""
        if strategy in self.optimization_strategies:
            self.optimization_strategies[strategy] = enabled
            logger.info(f"Optimization strategy '{strategy}' set to {enabled}")
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        metrics = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                metrics[key] = {
                    'current': values[-1],
                    'average': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'trend': self._calculate_trend(values)
                }
            else:
                metrics[key] = {
                    'current': 0,
                    'average': 0,
                    'max': 0,
                    'min': 0,
                    'trend': 0
                }
        
        # Add performance level
        metrics['performance_level'] = self.current_performance_level.value
        
        # Add optimization strategies
        metrics['optimization_strategies'] = self.optimization_strategies.copy()
        
        return metrics
    
    def _calculate_trend(self, values: deque) -> float:
        """Calculate trend in values."""
        if len(values) < 2:
            return 0.0
        
        recent_values = list(values)[-10:]
        if len(recent_values) < 2:
            return 0.0
        
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return 0.0
    
    def _trigger_callbacks(self, event_name: str, data: Any):
        """Trigger callbacks for an event."""
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback {event_name}: {e}")
    
    def add_callback(self, event_name: str, callback):
        """Add a callback for an event."""
        if event_name in self.callbacks:
            self.callbacks[event_name].append(callback)
        else:
            logger.warning(f"Unknown event name: {event_name}")
    
    def remove_callback(self, event_name: str, callback):
        """Remove a callback for an event."""
        if event_name in self.callbacks and callback in self.callbacks[event_name]:
            self.callbacks[event_name].remove(callback)
    
    def get_performance_history(self) -> List[Dict]:
        """Get performance history."""
        return list(self.performance_history)
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        for key in self.performance_metrics:
            self.performance_metrics[key].clear()
        
        self.performance_history.clear()
        self.current_performance_level = PerformanceLevel.HIGH
        
        logger.info("Performance metrics reset")
    
    def get_recommended_settings(self) -> Dict:
        """Get recommended settings based on current performance."""
        recommendations = {
            'window_size': 2048,
            'hop_size': 512,
            'enable_advanced_features': True,
            'prediction_frequency': 1.0,
            'caching_enabled': True
        }
        
        if self.current_performance_level == PerformanceLevel.LOW:
            recommendations.update({
                'window_size': 1024,
                'hop_size': 256,
                'enable_advanced_features': False,
                'prediction_frequency': 0.5,
                'caching_enabled': True
            })
        elif self.current_performance_level == PerformanceLevel.MEDIUM:
            recommendations.update({
                'window_size': 1536,
                'hop_size': 384,
                'enable_advanced_features': False,
                'prediction_frequency': 0.75,
                'caching_enabled': True
            })
        elif self.current_performance_level == PerformanceLevel.HIGH:
            recommendations.update({
                'window_size': 2048,
                'hop_size': 512,
                'enable_advanced_features': True,
                'prediction_frequency': 1.0,
                'caching_enabled': True
            })
        else:  # MAXIMUM
            recommendations.update({
                'window_size': 4096,
                'hop_size': 1024,
                'enable_advanced_features': True,
                'prediction_frequency': 1.0,
                'caching_enabled': True
            })
        
        return recommendations
