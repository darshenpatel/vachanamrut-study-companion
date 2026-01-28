"""
Lazy loading utilities for large data structures

This module provides lazy loading mechanisms to avoid loading large files (like the 34MB embeddings_store.pkl)
into memory at startup. Instead, data is loaded on-demand and cached intelligently.

Key features:
- Deferred loading of large files
- Memory-efficient streaming for JSON files
- Async loading with progress tracking
- Intelligent prefetching for common queries
- Memory pressure awareness
"""

import asyncio
import logging
import json
import pickle
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from threading import Lock, RLock
from dataclasses import dataclass
import weakref
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class LoadStatus:
    """Status of a lazy loading operation"""
    is_loaded: bool = False
    is_loading: bool = False
    load_time: Optional[float] = None
    file_size_mb: Optional[float] = None
    error: Optional[str] = None
    last_accessed: Optional[float] = None


class LazyLoader:
    """
    Base class for lazy loading of large data structures
    
    Features:
    - Deferred loading until first access
    - Thread-safe loading operations
    - Memory usage tracking
    - Automatic unloading under memory pressure
    """
    
    def __init__(self, file_path: str, loader_func: Callable[[], Any], name: str = "data"):
        self.file_path = Path(file_path)
        self.loader_func = loader_func
        self.name = name
        self.status = LoadStatus()
        self.data: Any = None
        self.lock = RLock()
        self._weak_refs = set()
        
        # Calculate file size
        if self.file_path.exists():
            size_bytes = self.file_path.stat().st_size
            self.status.file_size_mb = size_bytes / (1024 * 1024)
            logger.info(f"LazyLoader '{name}' initialized for {self.status.file_size_mb:.1f}MB file")
    
    def is_loaded(self) -> bool:
        """Check if data is loaded"""
        with self.lock:
            return self.status.is_loaded and self.data is not None
    
    def get_status(self) -> LoadStatus:
        """Get current loading status"""
        with self.lock:
            return self.status
    
    def load(self) -> Any:
        """Load data synchronously (blocking)"""
        with self.lock:
            if self.status.is_loaded and self.data is not None:
                self.status.last_accessed = time.time()
                return self.data
            
            if self.status.is_loading:
                # Wait for concurrent loading to complete
                while self.status.is_loading:
                    time.sleep(0.1)
                return self.data
            
            try:
                self.status.is_loading = True
                start_time = time.time()
                
                logger.info(f"Loading {self.name} from {self.file_path}")
                
                self.data = self.loader_func()
                
                self.status.load_time = time.time() - start_time
                self.status.is_loaded = True
                self.status.last_accessed = time.time()
                
                logger.info(f"Loaded {self.name} in {self.status.load_time:.2f}s")
                
                return self.data
                
            except Exception as e:
                self.status.error = str(e)
                logger.error(f"Failed to load {self.name}: {e}")
                raise
            
            finally:
                self.status.is_loading = False
    
    async def load_async(self) -> Any:
        """Load data asynchronously (non-blocking)"""
        if self.is_loaded():
            return self.data
        
        # Run loading in thread pool to avoid blocking event loop
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        try:
            return await loop.run_in_executor(executor, self.load)
        finally:
            executor.shutdown(wait=False)
    
    def unload(self, force: bool = False) -> bool:
        """Unload data from memory"""
        with self.lock:
            if not self.status.is_loaded or self.data is None:
                return True
            
            # Check if there are active references
            if not force and len(self._weak_refs) > 0:
                logger.debug(f"Cannot unload {self.name}: {len(self._weak_refs)} active references")
                return False
            
            self.data = None
            self.status.is_loaded = False
            self.status.last_accessed = None
            
            logger.info(f"Unloaded {self.name} from memory")
            return True
    
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage of loaded data"""
        if not self.is_loaded():
            return 0.0
        
        try:
            import sys
            return sys.getsizeof(self.data) / (1024 * 1024)
        except:
            # Fallback to file size estimate
            return self.status.file_size_mb or 0.0
    
    def add_reference(self, obj) -> None:
        """Add weak reference to track usage"""
        self._weak_refs.add(weakref.ref(obj))
    
    def remove_reference(self, obj) -> None:
        """Remove weak reference"""
        ref = weakref.ref(obj)
        self._weak_refs.discard(ref)


class JSONLazyLoader(LazyLoader):
    """Specialized lazy loader for JSON files with streaming support"""
    
    def __init__(self, file_path: str, name: str = "json_data", 
                 streaming: bool = False, chunk_size: int = 1000):
        self.streaming = streaming
        self.chunk_size = chunk_size
        
        def load_json():
            if self.streaming:
                return self._load_streaming()
            else:
                return self._load_full()
        
        super().__init__(file_path, load_json, name)
    
    def _load_full(self) -> Any:
        """Load entire JSON file into memory"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_streaming(self) -> Dict[str, Any]:
        """Load JSON file with streaming parser for large files"""
        import ijson  # Requires installation: pip install ijson
        
        result = {
            'metadata': {},
            'data_chunks': [],
            'total_items': 0
        }
        
        try:
            with open(self.file_path, 'rb') as f:
                # Parse metadata first
                if 'total_passages' in f.read(1000).decode('utf-8', errors='ignore'):
                    f.seek(0)
                    parser = ijson.parse(f)
                    for prefix, event, value in parser:
                        if prefix.endswith('.total_passages'):
                            result['metadata']['total_passages'] = value
                        elif prefix == 'passages.item':
                            result['data_chunks'].append(value)
                            result['total_items'] += 1
                            
                            # Process in chunks to avoid memory overflow
                            if len(result['data_chunks']) >= self.chunk_size:
                                yield result
                                result['data_chunks'] = []
            
            # Return final chunk
            if result['data_chunks']:
                yield result
                
        except ImportError:
            logger.warning("ijson not available, falling back to full JSON loading")
            return self._load_full()
    
    def get_item_count(self) -> int:
        """Get count of items without loading full data"""
        if self.is_loaded():
            data = self.data
            if isinstance(data, dict):
                if 'total_passages' in data:
                    return data['total_passages']
                elif 'passages' in data:
                    return len(data['passages'])
        
        # Try to parse count from file header
        try:
            with open(self.file_path, 'r') as f:
                header = f.read(500)  # Read first 500 chars
                if '"total_passages":' in header:
                    import re
                    match = re.search(r'"total_passages":\s*(\d+)', header)
                    if match:
                        return int(match.group(1))
        except Exception:
            pass
        
        return 0


class PickleLazyLoader(LazyLoader):
    """Specialized lazy loader for pickle files with compression support"""
    
    def __init__(self, file_path: str, name: str = "pickle_data", 
                 compressed: bool = False):
        self.compressed = compressed
        
        def load_pickle():
            if self.compressed:
                with gzip.open(self.file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(self.file_path, 'rb') as f:
                    return pickle.load(f)
        
        super().__init__(file_path, load_pickle, name)


class LazyDataManager:
    """
    Manager for multiple lazy loaders with intelligent loading strategies
    
    Features:
    - Coordinated loading of related data
    - Memory pressure management
    - Prefetching for anticipated needs
    - Usage analytics
    """
    
    def __init__(self, memory_limit_mb: int = 500):
        self.memory_limit_mb = memory_limit_mb
        self.loaders: Dict[str, LazyLoader] = {}
        self.lock = Lock()
        
        # Usage tracking
        self.access_history: List[Tuple[str, float]] = []
        self.prefetch_predictions: Dict[str, float] = {}
        
        logger.info(f"LazyDataManager initialized with {memory_limit_mb}MB memory limit")
    
    def register_loader(self, name: str, loader: LazyLoader) -> None:
        """Register a lazy loader"""
        with self.lock:
            self.loaders[name] = loader
            logger.info(f"Registered lazy loader: {name}")
    
    def get_loader(self, name: str) -> Optional[LazyLoader]:
        """Get a lazy loader by name"""
        return self.loaders.get(name)
    
    async def load_data(self, name: str, prefetch_related: bool = True) -> Any:
        """Load data with optional prefetching of related data"""
        loader = self.loaders.get(name)
        if not loader:
            raise ValueError(f"No loader registered for '{name}'")
        
        # Record access
        self._record_access(name)
        
        # Check memory pressure before loading
        await self._manage_memory_pressure()
        
        # Load the requested data
        data = await loader.load_async()
        
        # Prefetch related data if enabled
        if prefetch_related:
            await self._prefetch_related_data(name)
        
        return data
    
    def _record_access(self, name: str) -> None:
        """Record data access for usage analytics"""
        current_time = time.time()
        self.access_history.append((name, current_time))
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.access_history = [
            (n, t) for n, t in self.access_history 
            if t > cutoff_time
        ]
    
    async def _manage_memory_pressure(self) -> None:
        """Manage memory pressure by unloading least used data"""
        current_memory = self._calculate_total_memory_usage()
        
        if current_memory <= self.memory_limit_mb:
            return
        
        logger.info(f"Memory pressure detected: {current_memory:.1f}MB > {self.memory_limit_mb}MB")
        
        # Find least recently used loaders
        lru_loaders = self._get_lru_loaders()
        
        # Unload data until memory usage is acceptable
        for loader_name in lru_loaders:
            loader = self.loaders[loader_name]
            if loader.unload():
                logger.info(f"Unloaded {loader_name} due to memory pressure")
                current_memory = self._calculate_total_memory_usage()
                if current_memory <= self.memory_limit_mb:
                    break
    
    def _calculate_total_memory_usage(self) -> float:
        """Calculate total memory usage of all loaded data"""
        return sum(
            loader.get_memory_usage_mb() 
            for loader in self.loaders.values()
        )
    
    def _get_lru_loaders(self) -> List[str]:
        """Get list of loaders sorted by least recently used"""
        loader_access_times = {}
        
        for name, loader in self.loaders.items():
            if loader.is_loaded():
                last_access = loader.status.last_accessed or 0
                loader_access_times[name] = last_access
        
        # Sort by access time (oldest first)
        return sorted(
            loader_access_times.keys(), 
            key=lambda x: loader_access_times[x]
        )
    
    async def _prefetch_related_data(self, accessed_name: str) -> None:
        """Prefetch data that might be needed based on access patterns"""
        # Simple heuristics for prefetching
        prefetch_rules = {
            'pdf_store': ['theme_data'],  # After loading PDF, often need themes
            'embeddings_store': ['pdf_store'],  # Embeddings usually need PDF data
            'theme_data': ['pdf_store'],  # Themes need PDF context
        }
        
        related_names = prefetch_rules.get(accessed_name, [])
        
        for name in related_names:
            loader = self.loaders.get(name)
            if loader and not loader.is_loaded():
                logger.info(f"Prefetching related data: {name}")
                try:
                    await loader.load_async()
                except Exception as e:
                    logger.warning(f"Failed to prefetch {name}: {e}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary of all loaders"""
        with self.lock:
            summary = {
                'total_loaders': len(self.loaders),
                'loaded_count': sum(1 for l in self.loaders.values() if l.is_loaded()),
                'total_memory_mb': round(self._calculate_total_memory_usage(), 2),
                'memory_limit_mb': self.memory_limit_mb,
                'memory_usage_percent': round(
                    (self._calculate_total_memory_usage() / self.memory_limit_mb) * 100, 1
                ),
                'loaders': {}
            }
            
            for name, loader in self.loaders.items():
                status = loader.get_status()
                summary['loaders'][name] = {
                    'is_loaded': status.is_loaded,
                    'file_size_mb': status.file_size_mb,
                    'memory_usage_mb': loader.get_memory_usage_mb(),
                    'load_time': status.load_time,
                    'last_accessed': status.last_accessed,
                    'error': status.error
                }
        
        return summary
    
    def unload_all(self) -> None:
        """Unload all data"""
        with self.lock:
            for loader in self.loaders.values():
                loader.unload(force=True)
        logger.info("Unloaded all lazy loaded data")


# Global lazy data manager
_lazy_manager: Optional[LazyDataManager] = None


def get_lazy_manager() -> LazyDataManager:
    """Get global lazy data manager"""
    global _lazy_manager
    if _lazy_manager is None:
        _lazy_manager = LazyDataManager()
    return _lazy_manager


def create_pdf_store_loader(file_path: str) -> JSONLazyLoader:
    """Create lazy loader for PDF store JSON file"""
    return JSONLazyLoader(
        file_path=file_path,
        name="pdf_store",
        streaming=True,
        chunk_size=100
    )


def create_embeddings_loader(file_path: str) -> PickleLazyLoader:
    """Create lazy loader for embeddings pickle file"""
    return PickleLazyLoader(
        file_path=file_path,
        name="embeddings_store",
        compressed=False
    )


# Convenience function for backward compatibility
async def lazy_load_pdf_store(file_path: str) -> Any:
    """Lazy load PDF store with automatic management"""
    manager = get_lazy_manager()
    
    if "pdf_store" not in manager.loaders:
        loader = create_pdf_store_loader(file_path)
        manager.register_loader("pdf_store", loader)
    
    return await manager.load_data("pdf_store")


async def lazy_load_embeddings(file_path: str) -> Any:
    """Lazy load embeddings with automatic management"""
    manager = get_lazy_manager()
    
    if "embeddings_store" not in manager.loaders:
        loader = create_embeddings_loader(file_path)
        manager.register_loader("embeddings_store", loader)
    
    return await manager.load_data("embeddings_store")