"""
Performance monitoring utilities for tracking endpoint and function execution times.
"""
import time
import logging
from functools import wraps
from typing import Any, Callable, Optional
import asyncio

logger = logging.getLogger(__name__)

# Storage for performance metrics
performance_metrics = {
    "endpoints": {},  # {endpoint_name: [durations]}
    "functions": {},  # {function_name: [durations]}
}

def timing_decorator(name: Optional[str] = None, category: str = "functions"):
    """
    Decorator to measure and log execution time of functions.
    
    Args:
        name: Custom name for the metric (defaults to function name)
        category: Category of metric ("endpoints" or "functions")
        
    Usage:
        @timing_decorator()
        def my_function():
            pass
            
        @timing_decorator(name="custom_name")
        async def my_async_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _record_metric(category, metric_name, duration_ms)
                _log_performance(metric_name, duration_ms, category)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                _record_metric(category, metric_name, duration_ms)
                _log_performance(metric_name, duration_ms, category)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def endpoint_timing(name: Optional[str] = None):
    """
    Specialized decorator for API endpoints.
    
    Usage:
        @endpoint_timing()
        async def get_dataset():
            pass
    """
    return timing_decorator(name=name, category="endpoints")


def _record_metric(category: str, name: str, duration_ms: float):
    """Record a performance metric."""
    if category not in performance_metrics:
        performance_metrics[category] = {}
    
    if name not in performance_metrics[category]:
        performance_metrics[category][name] = []
    
    # Keep only last 1000 measurements to avoid memory bloat
    metrics_list = performance_metrics[category][name]
    metrics_list.append(duration_ms)
    if len(metrics_list) > 1000:
        metrics_list.pop(0)


def _log_performance(name: str, duration_ms: float, category: str):
    """Log performance metrics with appropriate level based on duration."""
    if duration_ms > 5000:  # > 5s: WARNING
        logger.warning(f"⚠️  {category[:-1].upper()} SLOW: {name} took {duration_ms:.0f}ms")
    elif duration_ms > 1000:  # > 1s: INFO
        logger.info(f"⏱️  {category[:-1].upper()}: {name} took {duration_ms:.0f}ms")
    else:  # < 1s: DEBUG
        logger.debug(f"⚡ {category[:-1].upper()}: {name} took {duration_ms:.1f}ms")


def get_performance_stats():
    """
    Get aggregated performance statistics.
    
    Returns:
        Dictionary with performance stats for all tracked metrics
    """
    import statistics
    
    stats = {
        "endpoints": {},
        "functions": {}
    }
    
    for category in ["endpoints", "functions"]:
        for name, durations in performance_metrics[category].items():
            if not durations:
                continue
            
            stats[category][name] = {
                "count": len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "p95_ms": _percentile(durations, 95),
                "p99_ms": _percentile(durations, 99),
            }
    
    return stats


def _percentile(data: list[float], percentile: int) -> float:
    """Calculate percentile of a list of numbers."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def reset_metrics():
    """Clear all performance metrics (useful for testing)."""
    performance_metrics["endpoints"] = {}
    performance_metrics["functions"] = {}
    logger.info("Performance metrics reset")


# Context manager for measuring code blocks
class TimingContext:
    """
    Context manager for measuring execution time of code blocks.
    
    Usage:
        with TimingContext("my_operation"):
            # code to measure
            pass
    """
    def __init__(self, name: str, category: str = "functions"):
        self.name = name
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        _record_metric(self.category, self.name, duration_ms)
        _log_performance(self.name, duration_ms, self.category)
