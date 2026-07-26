# pyvvm

Python VVM (Vector Vorticity Model) Dataset Reader

A package for loading and processing VVM simulation output with proper C-grid structure and coordinate handling.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import pyvvm

# Load VVM simulation data
loader = pyvvm.VVMDataLoader('/path/to/simulation')
ds = loader.ds

# Access raw variables
print(ds['th'])  # Potential temperature

# Access computed diagnostics via xarray accessor
print(ds.vvm.thv)   # Virtual potential temperature
print(ds.vvm.rhl)   # Relative humidity (liquid)
print(ds.vvm.pv)    # Potential vorticity

# Terrain-masked access
print(ds.vvm.masked.th)  # Masked potential temperature
```

## Features

### Data Loading

- Automatic C-grid restructuring for use with xgcm
- Flexible time step selection (slice, list, or single step)
- Lazy loading with Dask for large datasets
- Background profile integration (rho, thbar, pbar, etc.)
- Terrain data loading (TOPO.nc)

```python
# Load specific time steps
loader = pyvvm.VVMDataLoader('/path/to/case', steps=slice(0, 100, 10))

# Load with custom chunking
loader = pyvvm.VVMDataLoader('/path/to/case', chunks={'time': 1, 'lev': -1})

# Load specific variable groups
loader = pyvvm.VVMDataLoader('/path/to/case', groups=['L.Dynamic'])
```

### Thermodynamic Diagnostics

Access via `ds.vvm.<property>`:

| Property | Description | Units |
|----------|-------------|-------|
| **Temperature** | | |
| `t` | Air temperature | K |
| `tv` | Virtual temperature | K |
| `td` | Dew point temperature | K |
| `tl` | LCL temperature | K |
| **Potential temperature** | | |
| `thv` | Virtual potential temperature | K |
| `the` | Equivalent potential temperature (Bolton) | K |
| `thes` | Saturation equivalent potential temperature | K |
| **Moisture** | | |
| `e` | Vapor pressure | Pa |
| `esl` | Saturation vapor pressure (liquid) | Pa |
| `esi` | Saturation vapor pressure (ice) | Pa |
| `qvsl` | Saturation mixing ratio (liquid) | kg/kg |
| `qvsi` | Saturation mixing ratio (ice) | kg/kg |
| `rhl` | Relative humidity w.r.t. liquid | 1 |
| `rhi` | Relative humidity w.r.t. ice | 1 |
| **Static energy** | | |
| `sd` | Dry static energy | J/kg |
| `hm` | Moist static energy | J/kg |
| `hms` | Saturation moist static energy | J/kg |
| `hf` | Frozen moist static energy | J/kg |
| **Entropy** | | |
| `s` | Specific entropy | J/kg/K |
| **Stability** | | |
| `b` | Buoyancy | m/s² |
| `n2` | Brunt-Väisälä frequency squared | s⁻² |
| `cape_cin` | CAPE and CIN | J/kg |
| **Column-integrated** | | |
| `cwv` | Column water vapor | mm |
| `lwp` | Liquid water path | mm |
| `iwp` | Ice water path | mm |
| `crh` | Column relative humidity | 1 |

### Dynamics Diagnostics

| Property | Description | Units |
|----------|-------------|-------|
| `ws` | Wind speed | m/s |
| `wd` | Wind direction | deg |
| `ivtu` | Zonal integrated vapor transport | kg/m/s |
| `ivtv` | Meridional integrated vapor transport | kg/m/s |
| `ivt` | Integrated vapor transport magnitude | kg/m/s |
| `pv` | Ertel potential vorticity | K m² kg⁻¹ s⁻¹ |
| `psi` | Streamfunction | m²/s |

### Terrain Masking

```python
# Apply terrain mask to any variable
u_masked = ds.vvm.mask(ds['u'])

# Access pre-masked variables
th_masked = ds.vvm.masked.th
thv_masked = ds.vvm.masked.thv
```

### Tropical Cyclone Analysis

TC center tracking and cylindrical remapping are available through
`ds.vvm.tc`. Diagnostics consume explicit cylindrical or radial-profile
inputs:

```python
# Find and cache the TC center track before accessor-based remapping
track = ds.vvm.tc.find_center(field='psi', method='extremum', level=1000.0)
```

#### Center Finding

```python
# Streamfunction-based (default, robust)
track = ds.vvm.tc.find_center(field='psi', method='extremum', level=1000.0)

# Vorticity-based methods
track = ds.vvm.tc.find_center(field='zeta', method='centroid')
track = ds.vvm.tc.find_center(field='zeta', method='extremum')

# Custom smoothing and search radius
track = ds.vvm.tc.find_center(field='zeta', method='centroid', sigma=50e3, radius=100e3)

# Level range averaging
track = ds.vvm.tc.find_center(field='zeta', method='centroid', level=(500.0, 3000.0))
```

#### Cartesian-to-Cylindrical Remapping

```python
import numpy as np
import xarray as xr

from pyvvm.tc import (
    CylindricalGridSpec,
    remap_dataarray,
    rotate_vorticity,
    rotate_wind,
)

# Uniform cell-centered radii and azimuths
spec = CylindricalGridSpec.from_spacing(
    r_max=300e3,
    dr=2e3,
    ntheta=360,
    method='linear',
    boundary='periodic',
    nan_policy='propagate',
)

# Named raw/computed field or an arbitrary DataArray
th_cyl = ds.vvm.tc.remap('th', spec=spec)
derived = (ds['rhoz'] * ds['w']).rename('rhow')
derived_cyl = ds.vvm.tc.remap(derived, spec=spec)

# Variable names are optional; remap all horizontally gridded data variables
all_cyl = ds.vvm.tc.remap_dataset(spec=spec)
selected_cyl = ds.vvm.tc.remap_dataset(
    spec=spec,
    variables=['th', 'qv', 'sprec'],
)

# Remap vector components from their native C-grid locations first, then rotate
vectors_cyl = ds.vvm.tc.remap_dataset(
    spec=spec,
    variables=['u', 'v', 'xi', 'eta'],
)
wind_cyl = rotate_wind(vectors_cyl['u'], vectors_cyl['v'])
vorticity_cyl = rotate_vorticity(
    vectors_cyl['xi'],
    vectors_cyl['eta'],
)
vectors_cyl = xr.merge([vectors_cyl, wind_cyl, vorticity_cyl])
# Added variables:
# radial_wind, tangential_wind,
# radial_vorticity, tangential_vorticity
vectors_cyl['wind_speed'] = np.hypot(
    vectors_cyl['u'],
    vectors_cyl['v'],
).rename('wind_speed')

# The free function also accepts a fixed (x, y) center
snapshot_cyl = remap_dataarray(
    ds['th'].isel(time=0),
    (150e3, 200e3),
    spec=spec,
)
```

The source horizontal dimensions are replaced by `(theta, r)` while `time`,
`zc`, `zb`, and other leading dimensions are preserved. Dask-backed inputs
remain lazy. Vector rotation is performed at the cylindrical targets, using
`theta` measured counter-clockwise from the positive x-axis. For raw VVM
horizontal vorticity, `rotate_vorticity` applies the model convention that the
physical y component is `-eta`. Components at an explicitly requested `r=0`
are masked by default because the cylindrical basis is undefined there.

#### Theta Reductions

Theta reductions are explicit xarray operations, so callers choose the angular
sector and missing-value behavior appropriate for their analysis:

```python
# Complete mean over every sampled angle
mean = vectors_cyl.mean('theta')

# First-quadrant mean on an equally spaced theta grid
q1 = vectors_cyl.where(
    (vectors_cyl.theta >= 0)
    & (vectors_cyl.theta < 0.5 * np.pi),
    drop=True,
).mean('theta')
```

#### Wind Metrics

```python
from pyvvm.tc import wind_metrics

# wind_metrics consumes whichever radial profile the caller selected
metrics = wind_metrics(mean['wind_speed'])
q1_metrics = wind_metrics(
    q1['wind_speed'],
    thresholds=(15.0, 25.0, 35.0),
)
# Returns: vmax, rmw and threshold radii such as r15, r25, r35
```

#### Radial-Profile Diagnostics

```python
from pyvvm.tc import (
    angular_momentum,
    inertial_stability,
    mass_streamfunction,
)

f = ds.attrs.get('coriolis_parameter', 0.0)
aam = angular_momentum(mean['tangential_wind'], f)
i2 = inertial_stability(mean['tangential_wind'], f)
psi = mass_streamfunction(mean['radial_wind'], ds['rho'])
```

#### Performance Tips

- Center finding automatically loads full-chunk data when needed.
- Use `remap_dataset` for many variables so variables on the same C-grid
  location share their interpolation stencils.

### Dask Cluster

For parallel processing of large datasets:

```python
import pyvvm

# Initialize optimized Dask client
client = pyvvm.init_client(n_workers=8)

# Load and process data
loader = pyvvm.VVMDataLoader('/path/to/case')
result = loader.ds.vvm.cwv.compute()
```

## Dataset Structure

After loading, the dataset has:

**Dimensions:**
- `time`: Simulation time
- `xc`, `yc`, `zc`: Cell center coordinates
- `xb`, `yb`, `zb`: Cell edge coordinates (staggered)

**Coordinates:**
- `dx`, `dy`, `dz`: Grid spacing
- `rho`, `thbar`, `pbar`, `pibar`, `qvbar`: Background profiles
- `topo`: Terrain height index (if TOPO.nc exists)

## Dependencies

- xarray
- xgcm
- dask
- netCDF4
- scipy
- numpy

## License

MIT
