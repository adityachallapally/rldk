"""
Caching infrastructure for tracking system performance optimization.
"""

import asyncio
import hashlib
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrackingCache:
    """TTL-based cache for tracking system with memory management."""

    def __init__(self, cache_dir: Optional[Path] = None, ttl: int = 3600, max_memory_mb: int = 512):
        self.cache_dir = cache_dir or Path.cwd() / ".rldk_cache"
        self.ttl = ttl  # Time to live in seconds
        self.max_memory_mb = max_memory_mb
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, key: str) -> str:
        """Generate a safe cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl

    def _evict_expired(self) -> None:
        """Remove expired entries from memory cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._memory_cache.items()
            if current_time - entry["timestamp"] > self.ttl
        ]
        for key in expired_keys:
            del self._memory_cache[key]
            self._cache_stats["evictions"] += 1

    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in MB."""
        try:
            total_size = 0
            for entry in self._memory_cache.values():
                total_size += len(pickle.dumps(entry["data"]))
            return total_size // (1024 * 1024)  # Convert to MB
        except Exception:
            return 0

    def _evict_lru(self) -> None:
        """Evict least recently used entries if memory limit exceeded."""
        while (self._estimate_memory_usage() > self.max_memory_mb and
               len(self._memory_cache) > 0):
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]["timestamp"]
            )
            del self._memory_cache[oldest_key]
            self._cache_stats["evictions"] += 1

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (sync version) with thread safety."""
        with self._lock:
            self._evict_expired()

            cache_key = self._get_cache_key(key)

            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                if not self._is_expired(entry["timestamp"]):
                    self._cache_stats["hits"] += 1
                    entry["timestamp"] = time.time()
                    return entry["data"]
                else:
                    del self._memory_cache[cache_key]

            cache_file = self.cache_dir / f"{cache_key}.cache"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)

                    if not self._is_expired(entry["timestamp"]):
                        self._memory_cache[cache_key] = entry
                        self._evict_lru()  # Manage memory
                        self._cache_stats["hits"] += 1
                        return entry["data"]
                    else:
                        cache_file.unlink()
                except Exception:
                    if cache_file.exists():
                        cache_file.unlink()

            self._cache_stats["misses"] += 1
            return None

    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from cache (async version)."""
        return await asyncio.get_running_loop().run_in_executor(None, self.get, key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache (sync version) with thread safety."""
        with self._lock:
            cache_key = self._get_cache_key(key)
            entry = {
                "data": value,
                "timestamp": time.time()
            }

            self._memory_cache[cache_key] = entry
            self._evict_lru()  # Manage memory

            cache_file = self.cache_dir / f"{cache_key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception:
                pass  # Disk cache is optional

    async def set_async(self, key: str, value: Any) -> None:
        """Set value in cache (async version)."""
        await asyncio.get_running_loop().run_in_executor(None, self.set, key, value)

    def clear(self) -> None:
        """Clear all cache entries with thread safety."""
        with self._lock:
            self._memory_cache.clear()

            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with thread safety."""
        with self._lock:
            return {
                **self._cache_stats,
                "memory_entries": len(self._memory_cache),
                "estimated_memory_mb": self._estimate_memory_usage(),
                "disk_entries": len(list(self.cache_dir.glob("*.cache")))
            }


class ProgressIndicator:
    """Progress indicator for long-running operations."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()

    def update(self, step: int = 1, message: str = "") -> None:
        """Update progress."""
        self.current_step = min(self.current_step + step, self.total_steps)

        if self.total_steps > 0:
            percentage = (self.current_step / self.total_steps) * 100
            elapsed = time.time() - self.start_time

            if self.current_step > 0:
                eta = (elapsed / self.current_step) * (self.total_steps - self.current_step)
                eta_str = f" ETA: {eta:.1f}s" if eta > 1 else ""
            else:
                eta_str = ""

            status = f"{self.description}: {percentage:.1f}% ({self.current_step}/{self.total_steps}){eta_str}"
            if message:
                status += f" - {message}"

            print(f"\r{status}", end="", flush=True)

            if self.current_step >= self.total_steps:
                print()  # New line when complete

    def finish(self, message: str = "Complete") -> None:
        """Mark progress as finished."""
        self.current_step = self.total_steps
        elapsed = time.time() - self.start_time
        print(f"\r{self.description}: {message} (took {elapsed:.1f}s)")


async def run_with_timeout_and_progress(
    coro,
    timeout: int,
    progress_callback=None,
    error_message: str = "Operation timed out"
) -> Any:
    """Run coroutine with timeout and optional progress callback."""
    try:
        if progress_callback:
            task = asyncio.create_task(coro)

            result = await asyncio.wait_for(task, timeout=timeout)
            return result
        else:
            return await asyncio.wait_for(coro, timeout=timeout)

    except asyncio.TimeoutError:
        return {"error": f"{error_message} after {timeout}s", "timeout": True}
    except Exception as e:
        return {"error": str(e), "exception": True}
