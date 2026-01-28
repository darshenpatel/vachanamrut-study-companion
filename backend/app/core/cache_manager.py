"""
Optimized data cache manager for the Vachanamrut Study Companion

This module provides intelligent caching strategies for:
- PDF passages and search results
- AI-generated responses
- Theme data and related content
- Memory management and cleanup

Performance improvements:
- LRU caching with configurable sizes
- Memory-efficient storage
- Cache hit rate monitoring
- Automatic cleanup and eviction
"""

import hashlib
import logging
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from threading import RLock
import json
import gc
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction"""
    value: Any
    timestamp: float
    access_count: int
    memory_size: int
    last_access: float
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """
    High-performance LRU cache with memory management
    
    Features:
    - Thread-safe operations
    - Memory usage tracking
    - Access pattern analytics
    - Automatic eviction based on memory limits
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, name: str = "cache"):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.name = name
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_memory = 0
        
        logger.info(f"Initialized {name} cache: max_size={max_size}, max_memory={max_memory_mb}MB")
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._calculate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(
                    self._calculate_size(k) + self._calculate_size(v) 
                    for k, v in obj.items()
                )
            else:
                # Fallback: use string representation size
                return len(str(obj).encode('utf-8'))
        except Exception:
            return 1024  # Default size if calculation fails
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            entry = self.cache.pop(key)
            entry.update_access()
            self.cache[key] = entry
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache with intelligent eviction"""
        with self.lock:
            memory_size = self._calculate_size(value)
            
            # If single item is too large, don't cache it
            if memory_size > self.max_memory_bytes // 2:
                logger.warning(f"Item too large for {self.name} cache: {memory_size} bytes")
                return
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_memory -= old_entry.memory_size
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=1,
                memory_size=memory_size,
                last_access=time.time()
            )
            
            # Evict items if necessary
            self._evict_if_needed(memory_size)
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory += memory_size
    
    def _evict_if_needed(self, new_item_size: int):
        """Evict items to make space for new item"""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Check memory limit
        while (self.current_memory + new_item_size) > self.max_memory_bytes:
            if not self.cache:
                break
            self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Remove first item (least recently used)
        key, entry = self.cache.popitem(last=False)
        self.current_memory -= entry.memory_size
        self.evictions += 1
        
        logger.debug(f"Evicted from {self.name} cache: {key} ({entry.memory_size} bytes)")
    
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
            logger.info(f"Cleared {self.name} cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'name': self.name,
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_used_mb': round(self.current_memory / (1024 * 1024), 2),
                'memory_limit_mb': round(self.max_memory_bytes / (1024 * 1024), 2),
                'hit_rate': round(hit_rate, 2),
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'total_requests': total_requests
            }


class CompressedCache:
    """
    Cache with optional compression for large text data
    """
    
    def __init__(self, base_cache: LRUCache, compress_threshold: int = 1024):
        self.base_cache = base_cache
        self.compress_threshold = compress_threshold
    
    def _should_compress(self, data: str) -> bool:
        """Check if data should be compressed"""
        return isinstance(data, str) and len(data) > self.compress_threshold
    
    def _compress_data(self, data: str) -> bytes:
        """Compress string data using gzip"""
        import gzip
        return gzip.compress(data.encode('utf-8'))
    
    def _decompress_data(self, data: bytes) -> str:
        """Decompress gzipped data"""
        import gzip
        return gzip.decompress(data).decode('utf-8')
    
    def get(self, key: str) -> Optional[Any]:
        """Get with automatic decompression"""
        value = self.base_cache.get(key)
        if value is None:
            return None
        
        # Check if data is compressed
        if isinstance(value, tuple) and len(value) == 2 and value[0] == '__compressed__':
            return self._decompress_data(value[1])
        
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Put with automatic compression for large strings"""
        if self._should_compress(value):
            compressed = self._compress_data(value)
            # Store as tuple to mark as compressed
            self.base_cache.put(key, ('__compressed__', compressed))
            logger.debug(f"Compressed cache entry {key}: {len(value)} -> {len(compressed)} bytes")
        else:
            self.base_cache.put(key, value)


class CacheManager:
    """
    Centralized cache manager for the application
    
    Manages multiple cache instances for different data types:
    - Search results cache
    - Passage content cache  
    - AI response cache
    - Theme data cache
    """
    
    def __init__(self):
        # Initialize different cache types
        self.search_cache = LRUCache(
            max_size=500, 
            max_memory_mb=50, 
            name="search_results"
        )
        
        self.passage_cache = CompressedCache(
            LRUCache(
                max_size=1000, 
                max_memory_mb=100, 
                name="passages"
            )
        )
        
        self.ai_response_cache = CompressedCache(
            LRUCache(
                max_size=200, 
                max_memory_mb=30, 
                name="ai_responses"
            )
        )
        
        self.theme_cache = LRUCache(
            max_size=100, 
            max_memory_mb=20, 
            name="themes"
        )
        
        # Memory monitoring
        self.memory_warning_threshold = 0.85  # 85% of system memory
        self.last_memory_check = 0
        self.memory_check_interval = 60  # seconds
        
        logger.info("Cache manager initialized with all cache layers")
    
    def get_search_result(self, query_hash: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        return self.search_cache.get(query_hash)
    
    def cache_search_result(self, query_hash: str, results: List[Dict]) -> None:
        """Cache search results"""
        self.search_cache.put(query_hash, results)
    
    def get_passage_content(self, passage_id: str) -> Optional[str]:
        """Get cached passage content"""
        return self.passage_cache.get(passage_id)
    
    def cache_passage_content(self, passage_id: str, content: str) -> None:
        """Cache passage content"""
        self.passage_cache.put(passage_id, content)
    
    def get_ai_response(self, prompt_hash: str) -> Optional[str]:
        """Get cached AI response"""
        return self.ai_response_cache.get(prompt_hash)
    
    def cache_ai_response(self, prompt_hash: str, response: str) -> None:
        """Cache AI response"""
        self.ai_response_cache.put(prompt_hash, response)
    
    def get_theme_data(self, theme_name: str) -> Optional[Dict]:
        """Get cached theme data"""
        return self.theme_cache.get(theme_name)
    
    def cache_theme_data(self, theme_name: str, data: Dict) -> None:
        """Cache theme data"""
        self.theme_cache.put(theme_name, data)
    
    def create_query_hash(self, query: str, filters: Optional[Dict] = None) -> str:
        """Create hash for query + filters combination"""
        hash_input = f"{query}_{json.dumps(filters, sort_keys=True) if filters else ''}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def create_prompt_hash(self, user_question: str, context: List[Dict], theme: Optional[str] = None) -> str:
        """Create hash for AI prompt components"""
        context_str = json.dumps([c.get('content', '')[:100] for c in context], sort_keys=True)
        hash_input = f"{user_question}_{context_str}_{theme or ''}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage and cache statistics"""
        current_time = time.time()
        
        # Throttle memory checks
        if current_time - self.last_memory_check < self.memory_check_interval:
            return {}
        
        self.last_memory_check = current_time
        
        # Get system memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100
        
        # Get cache statistics
        cache_stats = {
            'search_cache': self.search_cache.get_stats(),
            'passage_cache': self.passage_cache.base_cache.get_stats(),
            'ai_response_cache': self.ai_response_cache.base_cache.get_stats(),
            'theme_cache': self.theme_cache.get_stats()
        }
        
        total_cache_memory = sum(
            stats['memory_used_mb'] for stats in cache_stats.values()
        )
        
        stats = {
            'system_memory_percent': round(memory_percent * 100, 2),
            'system_memory_available_gb': round(memory.available / (1024**3), 2),
            'total_cache_memory_mb': round(total_cache_memory, 2),
            'cache_details': cache_stats,
            'memory_warning': memory_percent > self.memory_warning_threshold
        }
        
        # Log warning if memory usage is high
        if stats['memory_warning']:
            logger.warning(f"High memory usage detected: {stats['system_memory_percent']}%")
            self._emergency_cache_cleanup()
        
        return stats
    
    def _emergency_cache_cleanup(self):
        """Emergency cache cleanup when memory is running low"""
        logger.info("Performing emergency cache cleanup due to high memory usage")
        
        # Clear least important caches first
        self.ai_response_cache.base_cache.clear()
        self.theme_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Emergency cache cleanup completed")
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.search_cache.clear()
        self.passage_cache.base_cache.clear()
        self.ai_response_cache.base_cache.clear()
        self.theme_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("All caches cleared")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all caches and system"""
        memory_stats = self.check_memory_usage()
        
        return {
            'cache_manager_status': 'active',
            'total_caches': 4,
            **memory_stats,
            'performance_recommendations': self._get_performance_recommendations()
        }
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations based on cache statistics"""
        recommendations = []
        
        stats = self.check_memory_usage()
        cache_details = stats.get('cache_details', {})
        
        # Check hit rates
        for cache_name, cache_stats in cache_details.items():
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate < 50:
                recommendations.append(f"Low hit rate in {cache_name} ({hit_rate}%) - consider adjusting cache size or TTL")
        
        # Check memory usage
        total_memory = stats.get('total_cache_memory_mb', 0)
        if total_memory > 200:
            recommendations.append(f"High cache memory usage ({total_memory}MB) - consider reducing cache limits")
        
        # System memory check
        if stats.get('memory_warning', False):
            recommendations.append("High system memory usage - consider clearing caches or restarting application")
        
        return recommendations


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_search_results(func):
    """Decorator for caching search results"""
    def wrapper(*args, **kwargs):
        cache_manager = get_cache_manager()
        
        # Create cache key from arguments
        cache_key = cache_manager.create_query_hash(
            str(args) + str(kwargs)
        )
        
        # Try to get from cache
        cached_result = cache_manager.get_search_result(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for search: {cache_key}")
            return cached_result
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        cache_manager.cache_search_result(cache_key, result)
        logger.debug(f"Cached search result: {cache_key}")
        
        return result
    
    return wrapper


def cache_ai_responses(func):
    """Decorator for caching AI responses"""
    def wrapper(*args, **kwargs):
        cache_manager = get_cache_manager()
        
        # Extract key parameters for hashing
        user_question = kwargs.get('user_question', args[0] if args else '')
        context_passages = kwargs.get('context_passages', args[1] if len(args) > 1 else [])
        theme = kwargs.get('theme', args[2] if len(args) > 2 else None)
        
        cache_key = cache_manager.create_prompt_hash(user_question, context_passages, theme)
        
        # Try to get from cache
        cached_response = cache_manager.get_ai_response(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for AI response: {cache_key}")
            return cached_response
        
        # Execute function and cache result
        result = func(*args, **kwargs)
        cache_manager.cache_ai_response(cache_key, result)
        logger.debug(f"Cached AI response: {cache_key}")
        
        return result
    
    return wrapper