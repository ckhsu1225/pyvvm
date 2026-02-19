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
print(ds.vvm.rh)    # Relative humidity
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
| `t` | Air temperature | K |
| `tv` | Virtual temperature | K |
| `td` | Dew point temperature | K |
| `thv` | Virtual potential temperature | K |
| `the` | Equivalent potential temperature | K |
| `thes` | Saturation equivalent potential temperature | K |
| `e` | Vapor pressure | Pa |
| `es` | Saturation vapor pressure | Pa |
| `qvs` | Saturation mixing ratio | kg/kg |
| `rh` | Relative humidity | 1 |
| `sd` | Dry static energy | J/kg |
| `hm` | Moist static energy | J/kg |
| `hms` | Saturation moist static energy | J/kg |
| `b` | Buoyancy | m/s² |
| `n2` | Brunt-Väisälä frequency squared | s⁻² |
| `cwv` | Column water vapor | mm |
| `lwp` | Liquid water path | mm |
| `iwp` | Ice water path | mm |
| `crh` | Column relative humidity | 1 |
| `cape_cin` | CAPE and CIN | J/kg |

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

TC-specific diagnostics are accessed via `ds.vvm.tc`:

```python
# 1. Find TC center (required first step)
track = ds.vvm.tc.find_center(field='psi', method='extremum', level=1000.0)

# 2. Compute TC wind components
vr = ds.vvm.tc.vr    # Radial wind (positive outward)
vt = ds.vvm.tc.vt    # Tangential wind (positive cyclonic)

# 3. Azimuthal mean of any variable
th_az = ds.vvm.tc.azimuth('th')          # By variable name
th_az = ds.vvm.tc.azimuth(ds['th'])      # By DataArray

# 4. Wind intensity metrics (vmax, rmw, threshold radii)
ws = ds.vvm.ws.persist()
metrics = ds.vvm.tc.wind_metrics(ws)
# Returns: vmax, rmw, r17, r25, r33

# 5. Axisymmetric diagnostics (properties)
aam = ds.vvm.tc.aam   # Absolute angular momentum
i2  = ds.vvm.tc.i2    # Inertial stability
psi = ds.vvm.tc.psi   # Mass streamfunction
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

#### Azimuthal Mean

```python
# Configure default radius and resolution
ds.vvm.tc.set_params(radius=300e3, dr=2e3)

# By variable name (raw or computed)
th_az = ds.vvm.tc.azimuth('th')
vt_az = ds.vvm.tc.azimuth('vt')

# With terrain masking
th_az_masked = ds.vvm.tc.masked.azimuth('th')
```

#### Wind Metrics

```python
# Compute wind speed and persist for performance
ws = ds.vvm.ws.persist()

# Compute metrics (vmax, rmw, threshold radii)
metrics = ds.vvm.tc.wind_metrics(ws)

# Select levels before passing
ws_1km = ds.vvm.ws.sel(zc=1000, method='nearest').persist()
metrics_1km = ds.vvm.tc.wind_metrics(ws_1km)

# Custom thresholds
metrics = ds.vvm.tc.wind_metrics(ws, thresholds=(15.0, 25.0, 35.0))
```

#### Performance Tips

- Center finding automatically loads full-chunk data when needed.

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
