"""
Data compression utilities for efficient storage and transfer

This module provides compression utilities optimized for the Vachanamrut Study Companion:
- JSON compression/decompression with intelligent algorithms
- Text content compression for passages
- Binary data optimization
- Performance monitoring and benchmarking

Key benefits:
- Reduced memory usage
- Faster data transfer
- Optimized storage efficiency
- Intelligent compression algorithm selection
"""

import gzip
import zlib
import bz2
import lzma
import json
import pickle
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of compression operation with metrics"""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    algorithm: str
    
    @property
    def space_saved_mb(self) -> float:
        """Space saved in MB"""
        return (self.original_size - self.compressed_size) / (1024 * 1024)
    
    @property
    def compression_efficiency(self) -> float:
        """Compression efficiency score (0-100)"""
        return (1 - self.compression_ratio) * 100


class CompressionAnalyzer:
    """
    Analyzes data characteristics to choose optimal compression algorithm
    """
    
    def __init__(self):
        self.algorithm_performance: Dict[str, Dict[str, float]] = {}
        self.data_type_preferences: Dict[str, str] = {
            'json': 'gzip',
            'text': 'gzip', 
            'binary': 'lzma',
            'mixed': 'gzip'
        }
    
    def analyze_data_type(self, data: Union[str, bytes, Dict, List]) -> str:
        """Analyze data type and characteristics"""
        if isinstance(data, dict) or isinstance(data, list):
            return 'json'
        elif isinstance(data, str):
            # Check if it's JSON-like text
            if data.strip().startswith(('{', '[')):
                try:
                    json.loads(data)
                    return 'json'
                except:
                    pass
            return 'text'
        elif isinstance(data, bytes):
            return 'binary'
        else:
            return 'mixed'
    
    def recommend_algorithm(self, data: Any, 
                          priority: str = 'balanced') -> str:
        """
        Recommend compression algorithm based on data and priority
        
        Priority options:
        - 'speed': Fastest compression
        - 'size': Best compression ratio
        - 'balanced': Balance between speed and size
        """
        data_type = self.analyze_data_type(data)
        
        if priority == 'speed':
            return 'gzip'  # Generally fastest
        elif priority == 'size':
            return 'lzma'  # Best compression ratio
        else:  # balanced
            return self.data_type_preferences.get(data_type, 'gzip')
    
    def benchmark_algorithms(self, sample_data: Any) -> Dict[str, CompressionResult]:
        """Benchmark all algorithms on sample data"""
        compressor = DataCompressor()
        algorithms = ['gzip', 'zlib', 'bz2', 'lzma']
        results = {}
        
        for algo in algorithms:
            try:
                result = compressor.compress(sample_data, algorithm=algo)
                results[algo] = result
                logger.debug(f"{algo}: {result.compression_ratio:.3f} ratio, {result.compression_time:.3f}s")
            except Exception as e:
                logger.warning(f"Failed to benchmark {algo}: {e}")
        
        return results


class DataCompressor:
    """
    Main compression class with support for multiple algorithms
    """
    
    def __init__(self):
        self.analyzer = CompressionAnalyzer()
        self.compression_stats: Dict[str, int] = {
            'total_operations': 0,
            'total_bytes_saved': 0,
            'total_time_spent': 0.0
        }
    
    def compress(self, data: Any, algorithm: Optional[str] = None, 
                json_separators: Tuple[str, str] = (',', ':')) -> CompressionResult:
        """
        Compress data using specified or optimal algorithm
        
        Args:
            data: Data to compress
            algorithm: Compression algorithm ('gzip', 'zlib', 'bz2', 'lzma')
            json_separators: JSON separators for compact serialization
        """
        start_time = time.time()
        
        # Convert data to bytes if needed
        if isinstance(data, (dict, list)):
            original_data = json.dumps(data, separators=json_separators, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, str):
            original_data = data.encode('utf-8')
        elif isinstance(data, bytes):
            original_data = data
        else:
            # Fallback to pickle for complex objects
            original_data = pickle.dumps(data)
        
        original_size = len(original_data)
        
        # Choose algorithm if not specified
        if algorithm is None:
            algorithm = self.analyzer.recommend_algorithm(data)
        
        # Compress using selected algorithm
        compressed_data = self._compress_bytes(original_data, algorithm)
        compressed_size = len(compressed_data)
        
        compression_time = time.time() - start_time
        compression_ratio = compressed_size / original_size
        
        # Update statistics
        self.compression_stats['total_operations'] += 1
        self.compression_stats['total_bytes_saved'] += (original_size - compressed_size)
        self.compression_stats['total_time_spent'] += compression_time
        
        result = CompressionResult(
            compressed_data=compressed_data,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            algorithm=algorithm
        )
        
        logger.debug(f"Compressed {original_size} -> {compressed_size} bytes "
                    f"({result.compression_efficiency:.1f}% efficiency) using {algorithm}")
        
        return result
    
    def decompress(self, compressed_data: bytes, algorithm: str, 
                  output_type: str = 'auto') -> Any:
        """
        Decompress data and optionally convert back to original type
        
        Args:
            compressed_data: Compressed bytes
            algorithm: Algorithm used for compression
            output_type: 'auto', 'bytes', 'str', 'json'
        """
        start_time = time.time()
        
        # Decompress bytes
        decompressed_bytes = self._decompress_bytes(compressed_data, algorithm)
        
        # Convert to requested type
        if output_type == 'bytes':
            result = decompressed_bytes
        elif output_type == 'str':
            result = decompressed_bytes.decode('utf-8')
        elif output_type == 'json':
            result = json.loads(decompressed_bytes.decode('utf-8'))
        else:  # auto
            # Try to detect original type
            try:
                text = decompressed_bytes.decode('utf-8')
                # Check if it's JSON
                if text.strip().startswith(('{', '[')):
                    result = json.loads(text)
                else:
                    result = text
            except UnicodeDecodeError:
                # Probably binary data or pickled object
                try:
                    result = pickle.loads(decompressed_bytes)
                except:
                    result = decompressed_bytes
        
        decompression_time = time.time() - start_time
        logger.debug(f"Decompressed {len(compressed_data)} bytes using {algorithm} "
                    f"in {decompression_time:.3f}s")
        
        return result
    
    def _compress_bytes(self, data: bytes, algorithm: str) -> bytes:
        """Compress bytes using specified algorithm"""
        if algorithm == 'gzip':
            return gzip.compress(data, compresslevel=6)  # Balanced compression level
        elif algorithm == 'zlib':
            return zlib.compress(data, level=6)
        elif algorithm == 'bz2':
            return bz2.compress(data, compresslevel=6)
        elif algorithm == 'lzma':
            return lzma.compress(data, preset=6)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
    
    def _decompress_bytes(self, data: bytes, algorithm: str) -> bytes:
        """Decompress bytes using specified algorithm"""
        if algorithm == 'gzip':
            return gzip.decompress(data)
        elif algorithm == 'zlib':
            return zlib.decompress(data)
        elif algorithm == 'bz2':
            return bz2.decompress(data)
        elif algorithm == 'lzma':
            return lzma.decompress(data)
        else:
            raise ValueError(f"Unsupported decompression algorithm: {algorithm}")
    
    def compress_json_file(self, input_path: str, output_path: str, 
                          algorithm: str = 'gzip') -> CompressionResult:
        """Compress a JSON file"""
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        # Load JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Compress data
        result = self.compress(data, algorithm)
        
        # Save compressed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            f.write(result.compressed_data)
        
        logger.info(f"Compressed JSON file: {input_path} -> {output_path} "
                   f"({result.space_saved_mb:.1f}MB saved)")
        
        return result
    
    def decompress_json_file(self, input_path: str, output_path: str, 
                            algorithm: str = 'gzip') -> Dict:
        """Decompress a JSON file"""
        input_file = Path(input_path)
        output_file = Path(output_path)
        
        # Load compressed data
        with open(input_file, 'rb') as f:
            compressed_data = f.read()
        
        # Decompress
        data = self.decompress(compressed_data, algorithm, 'json')
        
        # Save decompressed JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Decompressed JSON file: {input_path} -> {output_path}")
        
        return data
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        stats = self.compression_stats.copy()
        if stats['total_operations'] > 0:
            stats['average_bytes_saved'] = stats['total_bytes_saved'] / stats['total_operations']
            stats['average_time_per_operation'] = stats['total_time_spent'] / stats['total_operations']
            stats['total_mb_saved'] = stats['total_bytes_saved'] / (1024 * 1024)
        
        return stats


class AsyncDataCompressor:
    """
    Async version of data compressor for non-blocking operations
    """
    
    def __init__(self, max_workers: int = 4):
        self.compressor = DataCompressor()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def compress_async(self, data: Any, algorithm: Optional[str] = None) -> CompressionResult:
        """Compress data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.compressor.compress, 
            data, 
            algorithm
        )
    
    async def decompress_async(self, compressed_data: bytes, algorithm: str, 
                             output_type: str = 'auto') -> Any:
        """Decompress data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.compressor.decompress,
            compressed_data,
            algorithm,
            output_type
        )
    
    async def compress_multiple(self, data_items: List[Tuple[str, Any]]) -> Dict[str, CompressionResult]:
        """Compress multiple items concurrently"""
        tasks = []
        for name, data in data_items:
            task = asyncio.create_task(
                self.compress_async(data),
                name=f"compress_{name}"
            )
            tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Failed to compress {name}: {e}")
        
        return results
    
    def __del__(self):
        """Cleanup executor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class CompressionCache:
    """
    Cache for compressed data with intelligent eviction
    """
    
    def __init__(self, max_items: int = 100, max_memory_mb: int = 50):
        self.max_items = max_items
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, Tuple[bytes, str, float]] = {}  # hash -> (compressed_data, algorithm, timestamp)
        self.memory_usage = 0
        
    def _generate_key(self, data: Any) -> str:
        """Generate cache key for data"""
        if isinstance(data, (dict, list)):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_compressed(self, data: Any) -> Optional[Tuple[bytes, str]]:
        """Get compressed data from cache"""
        key = self._generate_key(data)
        
        if key in self.cache:
            compressed_data, algorithm, timestamp = self.cache[key]
            # Update timestamp
            self.cache[key] = (compressed_data, algorithm, time.time())
            return compressed_data, algorithm
        
        return None
    
    def put_compressed(self, data: Any, compressed_data: bytes, algorithm: str):
        """Put compressed data in cache"""
        key = self._generate_key(data)
        size = len(compressed_data)
        
        # Check if we need to evict items
        while (len(self.cache) >= self.max_items or 
               self.memory_usage + size > self.max_memory_bytes):
            if not self._evict_lru():
                break  # No more items to evict
        
        # Add to cache
        self.cache[key] = (compressed_data, algorithm, time.time())
        self.memory_usage += size
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Find LRU item
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][2])
        
        # Remove from cache
        compressed_data, _, _ = self.cache.pop(lru_key)
        self.memory_usage -= len(compressed_data)
        
        return True
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.memory_usage = 0


# Utility functions for common operations

def compress_pdf_store(file_path: str, output_path: Optional[str] = None, 
                      algorithm: str = 'gzip') -> CompressionResult:
    """Compress PDF store JSON file"""
    compressor = DataCompressor()
    
    if output_path is None:
        path = Path(file_path)
        output_path = str(path.with_suffix(f'.{algorithm}.compressed'))
    
    return compressor.compress_json_file(file_path, output_path, algorithm)


def decompress_pdf_store(compressed_path: str, output_path: Optional[str] = None,
                        algorithm: str = 'gzip') -> Dict:
    """Decompress PDF store file"""
    compressor = DataCompressor()
    
    if output_path is None:
        path = Path(compressed_path)
        output_path = str(path.with_suffix('.json'))
    
    return compressor.decompress_json_file(compressed_path, output_path, algorithm)


async def compress_large_dataset(data: Dict[str, Any], 
                                chunk_size: int = 1000) -> List[CompressionResult]:
    """Compress large dataset in chunks"""
    compressor = AsyncDataCompressor()
    
    # Split data into chunks
    if 'passages' in data and isinstance(data['passages'], list):
        passages = data['passages']
        chunks = []
        
        for i in range(0, len(passages), chunk_size):
            chunk_data = {
                'metadata': data.get('metadata', {}),
                'passages': passages[i:i + chunk_size],
                'chunk_info': {
                    'chunk_index': i // chunk_size,
                    'total_chunks': (len(passages) + chunk_size - 1) // chunk_size,
                    'chunk_size': len(passages[i:i + chunk_size])
                }
            }
            chunks.append((f"chunk_{i // chunk_size}", chunk_data))
        
        # Compress chunks concurrently
        return await compressor.compress_multiple(chunks)
    
    else:
        # Compress entire dataset
        result = await compressor.compress_async(data)
        return [result]


# Global instances
_compressor: Optional[DataCompressor] = None
_async_compressor: Optional[AsyncDataCompressor] = None


def get_compressor() -> DataCompressor:
    """Get global synchronous compressor instance"""
    global _compressor
    if _compressor is None:
        _compressor = DataCompressor()
    return _compressor


def get_async_compressor() -> AsyncDataCompressor:
    """Get global asynchronous compressor instance"""
    global _async_compressor
    if _async_compressor is None:
        _async_compressor = AsyncDataCompressor()
    return _async_compressor