"""Benchmark Cartesian-to-cylindrical remapping on a real VVM case.

The benchmark separates metadata/graph construction, stencil construction,
the in-memory sparse kernel, and end-to-end Dask computation.  Run a small
subset first; this case is hundreds of gigabytes.

Examples
--------
Quick local run:

    python benchmarks/benchmark_cylindrical.py --scheduler threads

Production-like local distributed run:

    python benchmarks/benchmark_cylindrical.py \
        --scheduler distributed --workers 4 --repeats 2

Larger scaling run:

    python benchmarks/benchmark_cylindrical.py \
        --steps 120 121 122 123 --levels 44 \
        --variables th qv qc qr sprec --workers 8
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import dask
import numpy as np
import psutil
import xarray as xr
from dask.distributed import performance_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pyvvm  # noqa: E402
from pyvvm.tc import (  # noqa: E402
    CylindricalGridSpec,
    apply_cylindrical_stencil,
    build_cylindrical_stencil,
    remap_dataset,
)


DEFAULT_CASE = Path(
    "/data/ckhsu/tc_data/vvmflux/"
    "twv_rs400_zs2_zc2_th2_rh90_sst28"
)

VARIABLE_GROUPS = {
    "xi": "L.Dynamic",
    "eta": "L.Dynamic",
    "zeta": "L.Dynamic",
    "u": "L.Dynamic",
    "v": "L.Dynamic",
    "w": "L.Dynamic",
    "th": "L.Thermodynamic",
    "qv": "L.Thermodynamic",
    "qc": "L.Thermodynamic",
    "qr": "L.Thermodynamic",
    "qi": "L.Thermodynamic",
    "nc": "L.Thermodynamic",
    "nr": "L.Thermodynamic",
    "ni": "L.Thermodynamic",
    "qrim": "L.Thermodynamic",
    "brim": "L.Thermodynamic",
    "uw": "C.Surface",
    "wv": "C.Surface",
    "wth": "C.Surface",
    "wqv": "C.Surface",
    "sprec": "C.Surface",
    "tg": "C.Surface",
    "olr": "C.Surface",
}

GROUP_ORDER = (
    "L.Dynamic",
    "L.Thermodynamic",
    "L.Radiation",
    "C.Surface",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pyvvm cylindrical remapping on real VVM data."
    )
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--steps", type=int, nargs="+", default=[120, 121])
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["th", "qv", "sprec"],
        help="Variables to remap in one shared remap_dataset call.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=8,
        help="Maximum zc/zb levels retained per three-dimensional variable.",
    )
    parser.add_argument("--r-max", type=float, default=300e3)
    parser.add_argument("--dr", type=float, default=2e3)
    parser.add_argument("--ntheta", type=int, default=360)
    parser.add_argument(
        "--method",
        choices=("linear", "nearest"),
        default="linear",
    )
    parser.add_argument(
        "--nan-policy",
        choices=("propagate", "omit"),
        default="propagate",
    )
    parser.add_argument(
        "--center",
        choices=("fixed", "moving"),
        default="moving",
        help="Moving uses a deterministic one-grid-cell shift per time.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("distributed", "threads", "sync"),
        default="distributed",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument(
        "--vertical-chunk",
        type=int,
        default=1,
        help="Loader lev chunk; use -1 to batch all selected vertical levels.",
    )
    parser.add_argument(
        "--horizontal-chunk",
        type=int,
        default=-1,
        help="Loader lat/lon chunk. Current remapper ultimately rechunks it to -1.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="End-to-end repeats. The first and warm-cache times are reported.",
    )
    parser.add_argument("--kernel-repeats", type=int, default=5)
    parser.add_argument(
        "--performance-report",
        type=Path,
        default=None,
        help="Optional Dask HTML report for the first distributed compute.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable result.",
    )
    return parser.parse_args()


def timed_call(func, /, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def infer_groups(variables: list[str]) -> list[str]:
    unknown = [name for name in variables if name not in VARIABLE_GROUPS]
    if unknown:
        raise ValueError(
            f"Cannot infer file groups for {unknown}. Add them to "
            "VARIABLE_GROUPS in this benchmark script."
        )
    required = {VARIABLE_GROUPS[name] for name in variables}
    return [group for group in GROUP_ORDER if group in required]


def select_working_dataset(
    ds: xr.Dataset,
    variables: list[str],
    levels: int,
) -> xr.Dataset:
    missing = [name for name in variables if name not in ds]
    if missing:
        raise ValueError(f"Variables not found after loading: {missing}.")

    selected: dict[str, xr.DataArray] = {}
    for name in variables:
        variable = ds[name]
        indexers: dict[str, slice] = {}
        for z_dim in ("zc", "zb"):
            if z_dim in variable.dims:
                indexers[z_dim] = slice(0, min(levels, variable.sizes[z_dim]))
        selected[name] = variable.isel(indexers)
    return xr.Dataset(selected, attrs=dict(ds.attrs))


def make_center(
    work: xr.Dataset,
    mode: str,
) -> tuple[float, float] | xr.Dataset:
    reference = next(iter(work.data_vars.values()))
    x_dim, y_dim = horizontal_dims(reference)
    x0 = float(work[x_dim].values[work.sizes[x_dim] // 2])
    y0 = float(work[y_dim].values[work.sizes[y_dim] // 2])
    if mode == "fixed":
        return x0, y0

    if "time" not in work.dims:
        raise ValueError("Moving-center benchmark requires a time dimension.")
    dx = float(work.coords["dx"].values)
    dy = float(work.coords["dy"].values)
    offsets = np.arange(work.sizes["time"], dtype=np.float64)
    return xr.Dataset(
        {
            "x": ("time", x0 + offsets * dx),
            "y": ("time", y0 + offsets * dy),
        },
        coords={"time": work["time"]},
    )


def horizontal_dims(variable: xr.DataArray) -> tuple[str, str]:
    x_dims = [dim for dim in ("xc", "xb") if dim in variable.dims]
    y_dims = [dim for dim in ("yc", "yb") if dim in variable.dims]
    if len(x_dims) != 1 or len(y_dims) != 1:
        raise ValueError(
            f"Cannot infer horizontal dimensions for {variable.name!r}: "
            f"{variable.dims}."
        )
    return x_dims[0], y_dims[0]


def center_arrays(
    center: tuple[float, float] | xr.Dataset,
    work: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    nt = work.sizes.get("time", 1)
    if isinstance(center, xr.Dataset):
        return (
            np.asarray(center["x"].values, dtype=np.float64),
            np.asarray(center["y"].values, dtype=np.float64),
        )
    return (
        np.full(nt, center[0], dtype=np.float64),
        np.full(nt, center[1], dtype=np.float64),
    )


def csr_nbytes(stencil) -> int:
    operator = stencil.operator
    return int(
        operator.data.nbytes
        + operator.indices.nbytes
        + operator.indptr.nbytes
        + stencil.valid.nbytes
    )


def benchmark_stencil_builds(
    work: xr.Dataset,
    center: tuple[float, float] | xr.Dataset,
    spec: CylindricalGridSpec,
    repeats: int,
) -> dict[str, Any]:
    cx, cy = center_arrays(center, work)
    if (
        np.equal(cx, cx[0]).all()
        and np.equal(cy, cy[0]).all()
    ):
        cx = cx[:1]
        cy = cy[:1]
    grids: dict[tuple[str, str], xr.DataArray] = {}
    for variable in work.data_vars.values():
        grids.setdefault(horizontal_dims(variable), variable)

    grid_results: dict[str, Any] = {}
    for (x_dim, y_dim), variable in grids.items():
        x = np.asarray(variable[x_dim].values, dtype=np.float64)
        y = np.asarray(variable[y_dim].values, dtype=np.float64)
        timings: list[float] = []
        last_stencils = None
        for _ in range(repeats):
            start = time.perf_counter()
            last_stencils = [
                build_cylindrical_stencil(
                    x,
                    y,
                    center_x=float(center_x),
                    center_y=float(center_y),
                    spec=spec,
                )
                for center_x, center_y in zip(cx, cy, strict=True)
            ]
            timings.append(time.perf_counter() - start)

        assert last_stencils is not None
        key = f"{y_dim},{x_dim}"
        grid_results[key] = {
            "count_per_run": len(last_stencils),
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "median_seconds_per_stencil": (
                statistics.median(timings) / len(last_stencils)
            ),
            "bytes_per_stencil": csr_nbytes(last_stencils[0]),
            "nnz_per_stencil": int(last_stencils[0].operator.nnz),
        }
        del last_stencils
    return grid_results


def benchmark_kernel(
    work: xr.Dataset,
    center: tuple[float, float] | xr.Dataset,
    spec: CylindricalGridSpec,
    repeats: int,
) -> dict[str, Any]:
    variable = next(iter(work.data_vars.values()))
    x_dim, y_dim = horizontal_dims(variable)
    sample = variable.isel(time=0, drop=True) if "time" in variable.dims else variable
    sample = sample.transpose(
        *(dim for dim in sample.dims if dim not in (y_dim, x_dim)),
        y_dim,
        x_dim,
    )

    source, load_seconds = timed_call(sample.compute)
    cx, cy = center_arrays(center, work)
    stencil = build_cylindrical_stencil(
        np.asarray(sample[x_dim].values, dtype=np.float64),
        np.asarray(sample[y_dim].values, dtype=np.float64),
        center_x=float(cx[0]),
        center_y=float(cy[0]),
        spec=spec,
    )

    # Sparse matrix multiplication has one-time dispatch/allocation effects.
    # Exclude two warm-up calls so the reported distribution measures the
    # steady-state in-memory kernel.
    for _ in range(2):
        apply_cylindrical_stencil(source.values, stencil)

    timings: list[float] = []
    output = None
    for _ in range(repeats):
        start = time.perf_counter()
        output = apply_cylindrical_stencil(source.values, stencil)
        timings.append(time.perf_counter() - start)

    assert output is not None
    return {
        "variable": str(variable.name),
        "warmup_calls": 2,
        "source_shape": list(source.shape),
        "source_bytes": int(source.nbytes),
        "source_load_seconds_warm_cache": load_seconds,
        "seconds": timings,
        "median_seconds": statistics.median(timings),
        "output_shape": list(output.shape),
        "output_bytes": int(output.nbytes),
    }


def graph_key_name(key: object) -> str:
    return str(key[0] if isinstance(key, tuple) else key)


def compute_output(
    output: xr.Dataset,
    args: argparse.Namespace,
    repeat_index: int,
) -> xr.Dataset:
    report_context = nullcontext()
    if args.performance_report is not None and repeat_index == 0:
        if args.scheduler != "distributed":
            raise ValueError("--performance-report requires --scheduler distributed.")
        args.performance_report.parent.mkdir(parents=True, exist_ok=True)
        report_context = performance_report(filename=str(args.performance_report))

    with report_context:
        return output.compute()


def result_checksum(output: xr.Dataset) -> dict[str, float]:
    return {
        name: float(np.nanmean(variable.values))
        for name, variable in output.data_vars.items()
    }


def main() -> None:
    args = parse_args()
    if args.levels < 1:
        raise ValueError("--levels must be positive.")
    if args.repeats < 1 or args.kernel_repeats < 1:
        raise ValueError("Repeat counts must be positive.")
    for name in ("vertical_chunk", "horizontal_chunk"):
        value = getattr(args, name)
        if value == 0 or value < -1:
            raise ValueError(f"--{name.replace('_', '-')} must be -1 or positive.")

    groups = infer_groups(args.variables)
    print(f"case: {args.case}", flush=True)
    print(f"steps: {args.steps}", flush=True)
    print(f"groups: {groups}", flush=True)
    print(f"variables: {args.variables}", flush=True)

    loader, load_seconds = timed_call(
        pyvvm.VVMDataLoader,
        args.case,
        steps=args.steps,
        groups=groups,
        chunks={
            "time": 1,
            "lev": args.vertical_chunk,
            "lat": args.horizontal_chunk,
            "lon": args.horizontal_chunk,
        },
    )
    work = select_working_dataset(loader.ds, args.variables, args.levels)
    center = make_center(work, args.center)
    spec = CylindricalGridSpec.from_spacing(
        r_max=args.r_max,
        dr=args.dr,
        ntheta=args.ntheta,
        method=args.method,
        boundary="periodic",
        nan_policy=args.nan_policy,
    )

    print(f"working sizes: {dict(work.sizes)}", flush=True)
    input_mib = (
        sum(variable.nbytes for variable in work.data_vars.values()) / 2**20
    )
    print(
        f"input bytes: {input_mib:.1f} MiB",
        flush=True,
    )

    output, graph_seconds = timed_call(
        remap_dataset,
        work,
        center,
        spec=spec,
    )
    graph = output.__dask_graph__()
    graph_keys = list(graph.keys()) if graph is not None else []
    stencil_task_count = sum(
        graph_key_name(key).startswith("build_cylindrical_stencil-")
        for key in graph_keys
    )

    client = None
    scheduler_context = nullcontext()
    scheduler_start = time.perf_counter()
    if args.scheduler == "distributed":
        client = pyvvm.init_client(n_workers=args.workers, port=args.port)
        scheduler_context = nullcontext()
        dashboard = client.dashboard_link
    else:
        scheduler_name = "synchronous" if args.scheduler == "sync" else "threads"
        scheduler_context = dask.config.set(
            scheduler=scheduler_name,
            num_workers=args.workers,
        )
        dashboard = None
    scheduler_start_seconds = time.perf_counter() - scheduler_start

    process = psutil.Process()
    compute_seconds: list[float] = []
    rss_after_mib: list[float] = []
    checksums: dict[str, float] = {}
    try:
        with scheduler_context:
            for repeat_index in range(args.repeats):
                gc.collect()
                print(
                    f"compute repeat {repeat_index + 1}/{args.repeats} ...",
                    flush=True,
                )
                computed, elapsed = timed_call(
                    compute_output,
                    output,
                    args,
                    repeat_index,
                )
                compute_seconds.append(elapsed)
                rss_after_mib.append(process.memory_info().rss / 2**20)
                checksums = result_checksum(computed)
                print(f"  {elapsed:.3f} s", flush=True)
                del computed

            stencil_results = benchmark_stencil_builds(
                work,
                center,
                spec,
                args.kernel_repeats,
            )
            kernel_results = benchmark_kernel(
                work,
                center,
                spec,
                args.kernel_repeats,
            )
    finally:
        if client is not None:
            client.close()

    input_bytes = int(sum(variable.nbytes for variable in work.data_vars.values()))
    output_bytes = int(sum(variable.nbytes for variable in output.data_vars.values()))
    report: dict[str, Any] = {
        "case": str(args.case),
        "steps": args.steps,
        "variables": args.variables,
        "levels": args.levels,
        "source_sizes": dict(work.sizes),
        "spec": {
            "r_max": args.r_max,
            "dr": args.dr,
            "nr": spec.nr,
            "ntheta": spec.ntheta,
            "method": spec.method,
            "nan_policy": spec.nan_policy,
        },
        "center_mode": args.center,
        "scheduler": {
            "name": args.scheduler,
            "workers": args.workers,
            "startup_seconds": scheduler_start_seconds,
            "dashboard": dashboard,
        },
        "loader_chunks": {
            "time": 1,
            "lev": args.vertical_chunk,
            "lat": args.horizontal_chunk,
            "lon": args.horizontal_chunk,
        },
        "metadata_load_seconds": load_seconds,
        "graph_build_seconds": graph_seconds,
        "graph_tasks": len(graph_keys),
        "stencil_tasks": stencil_task_count,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "end_to_end": {
            "seconds": compute_seconds,
            "first_seconds": compute_seconds[0],
            "warm_median_seconds": (
                statistics.median(compute_seconds[1:])
                if len(compute_seconds) > 1
                else None
            ),
            "rss_after_mib": rss_after_mib,
            "checksum": checksums,
        },
        "stencil_build": stencil_results,
        "kernel_only": kernel_results,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote JSON: {args.json}", flush=True)


if __name__ == "__main__":
    main()
