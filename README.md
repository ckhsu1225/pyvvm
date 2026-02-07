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
| `cape_cin` | Convective available potential energy and convective inhibition | J/kg |

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
