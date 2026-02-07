"""
Dask cluster utilities for VVM data processing.

This module provides helper functions to initialize a Dask client
optimized for VVM data loading with NetCDF/HDF5 files.
"""

from __future__ import annotations

import logging
import multiprocessing
from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)

__all__ = [
    'init_client',
]


def init_client(
    n_workers: int | None = None,
    memory_limit: str = 'auto',
    port: int = 8787,
) -> Client:
    """
    Initialize a Dask Client optimized for VVM data loading.
    
    This function enforces threads_per_worker=1 to avoid race conditions
    and segfaults that can occur with NetCDF4/HDF5 in multi-threaded environments.

    Parameters
    ----------
    n_workers : int, optional
        Number of workers. Defaults to min(CPU_count // 2, 16) to avoid
        I/O bottlenecks while leaving resources for the system.
    memory_limit : str, optional
        Memory limit per worker. Defaults to 'auto'.
    port : int, optional
        Dashboard port. Will automatically try the next port if conflict occurs.

    Returns
    -------
    dask.distributed.Client
        Initialized Dask client connected to a LocalCluster.
    """
    # Determine optimal worker count
    # Based on testing, 16 workers is typically a sweet spot for I/O-bound tasks.
    # Beyond ~16-24 workers, I/O becomes the bottleneck rather than computation.
    if n_workers is None:
        cpu_count = multiprocessing.cpu_count()
        n_workers = min(cpu_count // 2, 16)
        n_workers = max(1, n_workers)  # Ensure at least 1 worker
    
    # Create cluster with NetCDF-safe settings
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,         # Critical: prevents NetCDF/HDF5 race conditions
        memory_limit=memory_limit,
        dashboard_address=f':{port}',
        processes=True,               # Force process-based parallelism
        silence_logs=30,              # Reduce log noise
    )
    
    client = Client(cluster)
    
    # Log initialization info
    logger.info("PyVVM Dask Client Initialized")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Threads per worker: 1 (NetCDF Safe Mode)")
    logger.info(f"  Dashboard: {client.dashboard_link}")
    
    return client
