"""
VVM configuration and profile data I/O utilities.

This module provides functions to parse VVM model configuration files (vvm.setup)
and background profile data files (fort.98).
"""

from __future__ import annotations

import re
import logging
import numpy as np
import pandas as pd
from typing import Any
from pathlib import Path
from .calc.constants import omega

logger = logging.getLogger(__name__)

__all__ = [
    'parse_vvm_setup',
    'parse_fort98',
]

NUM = r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?'

def _read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')

def _normalize_header(line: str) -> str:
    return re.sub(r'\s+', '', line).upper()

def _to_float(token: str) -> float:
    return float(token.replace('D', 'E').replace('d', 'E'))

def _save_buffer(
    store: dict[str, pd.DataFrame],
    name: str,
    buffer: list[list[float]],
    sections: dict[str, dict[str, Any]],
) -> None:
    """Helper function: convert buffer to DataFrame and store it."""
    cols = sections[name]['columns']
    # Ensure column count matches (guard against malformed trailing rows)
    valid_rows = [r for r in buffer if len(r) == len(cols)]
    store[name] = pd.DataFrame(valid_rows, columns=cols)
    # Convert K to int
    store[name]['K'] = store[name]['K'].astype(int)


def parse_vvm_setup(path: str | Path) -> dict[str, Any]:
    """
    Parse vvm.setup to extract model configuration parameters.

    Parameters
    ----------
    path : str or Path
        Path to the vvm.setup file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing parsed configuration parameters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Setup file not found: {path}")

    content = _read_text(path)

    config: dict[str, Any] = {}

    # Define parameters and their regex patterns
    patterns = {
        'DT':     rf'\bDT\s*=\s*({NUM})',
        'NXSAVG': r'\bNXSAVG\s*=\s*(\d+)',
        'DX':     rf'\bDX\s*=\s*({NUM})',
        'DYNEW':  rf'\bDYNEW\s*=\s*({NUM})',
        'DZ':     rf'\bDZ\s*=\s*({NUM})',
        'DZ1':    rf'\bDZ1\s*=\s*({NUM})',
        'NK2':    r'vert_dimension/(\d+)/',
        'RLAT':   rf'\bRLAT\s*=\s*({NUM})',
        'RLON':   rf'\bRLON\s*=\s*({NUM})',
    }

    int_keys = {'NXSAVG', 'NK2'}
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            val_str = match.group(1)
            val = _to_float(val_str)
            if key in int_keys:
                config[key] = int(val)
            else:
                config[key] = val
        else:
            logger.warning(f"Parameter '{key}' not found in {path}")

    # Calculate output time interval (seconds)
    if 'DT' in config and 'NXSAVG' in config:
        config['output_interval'] = config['DT'] * config['NXSAVG']

    # Parse CPP defines from definesld.com block
    config['defines'] = _parse_defines(content)

    # Calculate Coriolis parameter if CORIOLIS is enabled
    if 'CORIOLIS' in config['defines'] and 'RLAT' in config:
        rlat_rad = np.radians(config['RLAT'])
        config['f'] = 2 * omega * np.sin(rlat_rad)
    else:
        config['f'] = None

    return config


def _parse_defines(content: str) -> set[str]:
    """
    Parse #define flags from the definesld.com block in vvm.setup.
    
    Only defines within the 'cat > definesld.com << END1' block are valid.
    Defines outside this block are just shell comments.
    
    Returns
    -------
    set[str]
        Set of enabled preprocessor flags (e.g., {'MPI', 'CORIOLIS', ...}).
    """
    # Match content between "cat > definesld.com << 'END1'" and "'END1'"
    pattern = r"cat\s*>\s*definesld\.com\s*<<\s*['\"]?END1['\"]?(.+?)['\"]?END1['\"]?"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not match:
        return set()
    
    block = match.group(1)
    
    # Extract all #define FLAG patterns
    defines = set()
    for line in block.splitlines():
        line = line.strip()
        if line.startswith('#define'):
            parts = line.split()
            if len(parts) >= 2:
                defines.add(parts[1])
    
    return defines


def parse_fort98(path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Parse fort.98 to extract vertical profile and background field data.
    Supports reading and merging multiple tables.

    Note
    ----
        Some VVM outputs omit VG(K) in the header line while still
        writing six numeric columns (K, UG, VG, Q1LS, Q2LS, WLS). We accept
        both header variants and map data to the 6-column schema.

    Parameters
    ----------
    path : str or Path
        Path to the fort.98 file.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with keys 'profile', 'forcing' and 'rhoz' containing the parsed data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    # Define header patterns and column names for different sections
    # Configured based on typical fort.98 file format
    sections: dict[str, dict[str, Any]] = {
        'grid': {
            'headers': ['K, ZZ(K),ZT(K),FNZ(K),FNT(K)'],
            'columns': ['K', 'ZZ', 'ZT', 'FNZ', 'FNT']
        },
        'thermo': {
            'headers': ['K, RHO(K),THBAR(K),PBAR(K),PIBAR(K),QVBAR(K)'],
            'columns': ['K', 'RHO', 'THBAR', 'PBAR', 'PIBAR', 'QVBAR']
        },
        'forcing': {
            'headers': [
                'K,UG(K),VG(K),Q1LS(K),Q2LS(K),WLS(m/s)',
                'K,UG(K),Q1LS(K),Q2LS(K),WLS(m/s)',
            ],
            'columns': ['K', 'UG', 'VG', 'Q1LS', 'Q2LS', 'WLS']
        },
        'rhoz': {
            'headers': ['K, RHOZ(K)'],
            'columns': ['K', 'RHOZ']
        }
    }

    data_store: dict[str, pd.DataFrame] = {}

    lines = _read_text(path).splitlines()

    # Simple state machine parser
    current_section: str | None = None
    buffer: list[list[float]] = []
    forcing_header_missing_vg = False

    for line in lines:
        line = line.strip()

        # 1. Check if entering a new section
        found_new_section = False
        norm_line = _normalize_header(line)
        for name, info in sections.items():
            headers = [_normalize_header(h) for h in info['headers']]
            if any(h in norm_line for h in headers):
                if name == 'forcing':
                    short_header = _normalize_header(
                        'K,UG(K),Q1LS(K),Q2LS(K),WLS(m/s)'
                    )
                    if short_header in norm_line:
                        forcing_header_missing_vg = True
                # Save the previous section if it exists
                if current_section and buffer:
                    _save_buffer(data_store, current_section, buffer, sections)

                # Start new section
                current_section = name
                buffer = []
                found_new_section = True
                break

        if found_new_section:
            continue

        # 2. Skip separator lines or header lines
        if line.startswith('=') or line.startswith('*') or not line:
            continue

        # 3. Read data lines (extract numbers with Fortran exponent support)
        if current_section:
            nums = re.compile(NUM).findall(line)
            if nums:
                try:
                    row = [_to_float(x) for x in nums]
                    buffer.append(row)
                except ValueError:
                    logger.debug("Skip malformed data line: %s", line)
                    continue

    # After loop ends, save the last section
    if current_section and buffer:
        _save_buffer(data_store, current_section, buffer, sections)

    # --- Post-processing and merging ---

    # 1. Basic merge: Grid + Thermo (both typically have NK3 levels, i.e., NK2 + 1)
    if 'grid' in data_store and 'thermo' in data_store:
        # Merge using 'K' as key
        df_profile = pd.merge(data_store['grid'], data_store['thermo'], on='K')
    elif 'grid' in data_store:
        df_profile = data_store['grid']
    else:
        raise ValueError("Could not parse Grid information from fort.98")

    # 2. Handle large scale forcing (typically NK3 levels)
    df_forcing = data_store.get('forcing', pd.DataFrame())
    if forcing_header_missing_vg and not df_forcing.empty:
        logger.warning(
            "Forcing header missing VG(K) but data rows include an extra column; "
            "parsed as ['K','UG','VG','Q1LS','Q2LS','WLS']."
        )

    # 3. Handle RHOZ (typically NK2 levels)
    # RHOZ is defined on edges (ZZ), usually one level fewer
    df_rhoz = data_store.get('rhoz', pd.DataFrame())

    return {
        'profile': df_profile,  # Contains ZZ, ZT, RHO, THBAR... (NK3 rows)
        'forcing': df_forcing,  # Contains UG, VG, Q1LS, Q2LS, WLS (NK3 rows)
        'rhoz': df_rhoz,        # Contains RHOZ (NK2 rows)
    }
