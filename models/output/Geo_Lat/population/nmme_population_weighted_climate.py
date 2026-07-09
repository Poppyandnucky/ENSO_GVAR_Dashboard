#!/usr/bin/env python3
"""Population-weighted NMME climate anomalies by country.

This script converts two NMME forecast anomaly fields onto country-level
population-weighted monthly time series. Population is first accumulated from a
fine GeoTIFF onto the 1-degree NMME grid using pixel-center country assignment;
monthly climate anomalies are then averaged with those grid-cell weights.
"""

from __future__ import annotations

import argparse
import calendar
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.windows import Window


COUNTRIES = {
    "Brazil": "BRA",
    "Chile": "CHL",
    "Colombia": "COL",
    "Peru": "PER",
    "Philippines": "PHL",
    "Thailand": "THA",
    "India": "IND",
    "Indonesia": "IDN",
    "Kenya": "KEN",
    "South Africa": "ZAF",
    "Egypt": "EGY",
    "Mexico": "MEX",
}


@dataclass(frozen=True)
class InputPaths:
    data_dir: Path
    heat_nc: Path
    moisture_nc: Path
    countries_shp: Path
    population_tif: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create country-month population-weighted NMME heat and moisture "
            "anomaly time series."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing the NMME NetCDFs, country shapefile folder, and "
            "population GeoTIFF. If omitted, common local locations are tried."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to data-dir/nmme_country_population_weighted_climate.csv.",
    )
    parser.add_argument(
        "--weights-cache",
        type=Path,
        default=None,
        help=(
            "Optional .npz cache for country-by-NMME-cell population weights. "
            "Defaults to data-dir/nmme_population_weights_12countries.npz."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the population-weight cache.",
    )
    return parser.parse_args()


def find_default_data_dir() -> Path:
    env_dir = os.environ.get("NMME_CROP_YIELD_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir))

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent,
            here.parent / "NMME crop yield",
            here.parent.parent / "NMME crop yield",
            Path.cwd(),
            Path.cwd() / "NMME crop yield",
            Path.home() / "Downloads" / "NMME crop yield",
        ]
    )

    for candidate in candidates:
        if (candidate / "NMME.tmp2m.202606.ENSMEAN.anom.nc").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the NMME crop-yield data folder. Pass --data-dir or "
        "set NMME_CROP_YIELD_DIR."
    )


def build_input_paths(data_dir: Path) -> InputPaths:
    paths = InputPaths(
        data_dir=data_dir,
        heat_nc=data_dir / "NMME.tmp2m.202606.ENSMEAN.anom.nc",
        moisture_nc=data_dir / "NMME.prate.202606.ENSMEAN.anom.nc",
        countries_shp=data_dir
        / "ne_50m_admin_0_countries"
        / "ne_50m_admin_0_countries.shp",
        population_tif=data_dir / "global_pop_2027_CN_1km_R2025A_UA_v1.tif",
    )

    missing = [str(path) for path in paths.__dict__.values() if isinstance(path, Path) and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return paths


def add_months(origin: pd.Timestamp, months: int) -> pd.Timestamp:
    return origin + pd.DateOffset(months=int(months))


def decode_target_months(target_values: Iterable[float]) -> pd.PeriodIndex:
    origin = pd.Timestamp("1960-01-02 21:00:00")
    timestamps = [add_months(origin, int(value)) for value in target_values]
    return pd.PeriodIndex([ts.to_period("M") for ts in timestamps], freq="M")


def old_matlab_target_months(target_values: Iterable[float]) -> pd.PeriodIndex:
    origin = pd.Timestamp("1960-01-01")
    timestamps = [add_months(origin, int(value) - 1) for value in target_values]
    return pd.PeriodIndex([ts.to_period("M") for ts in timestamps], freq="M")


def print_target_date_check(target: np.ndarray, initial_time: np.ndarray) -> tuple[pd.Period, pd.PeriodIndex]:
    old_targets = old_matlab_target_months(target)
    corrected_targets = decode_target_months(target)
    corrected_init = decode_target_months(initial_time)[0]

    check = pd.DataFrame(
        {
            "target_value": target.astype(int),
            "old_matlab_decoded": old_targets.astype(str),
            "corrected_netcdf_units": corrected_targets.astype(str),
        }
    )

    print("\nTarget-month decoding check")
    print(check.to_string(index=False))
    print(f"Corrected init_month from NetCDF units: {corrected_init}")
    print()

    return corrected_init, corrected_targets


def require_anomaly(ds: xr.Dataset, path: Path) -> None:
    long_name = str(ds["fcst"].attrs.get("long_name", "")).lower()
    path_name = path.name.lower()
    if "anom" not in long_name and "anom" not in path_name:
        raise ValueError(
            f"{path} does not appear to be an anomaly file. A climatology is "
            "needed before this script can compute anomalies."
        )


def load_nmme_cube(path: Path, expected_units: str) -> tuple[xr.DataArray, xr.Dataset]:
    ds = xr.open_dataset(path, engine="scipy", decode_times=False)
    require_anomaly(ds, path)

    units = str(ds["fcst"].attrs.get("units", "")).strip()
    if units != expected_units:
        raise ValueError(f"{path.name} has units {units!r}; expected {expected_units!r}.")

    needed_dims = {"lat", "lon", "target"}
    if set(ds["fcst"].dims) != needed_dims:
        raise ValueError(f"{path.name} fcst dims are {ds['fcst'].dims}; expected lat/lon/target.")

    cube = ds["fcst"].transpose("lat", "lon", "target").sortby("lat")
    return cube, ds


def convert_prate_to_monthly_anomaly(prate: xr.DataArray, target_months: pd.PeriodIndex) -> xr.DataArray:
    days = np.array(
        [calendar.monthrange(period.year, period.month)[1] for period in target_months],
        dtype=np.float32,
    )
    scale = xr.DataArray(days * 86400.0, dims=("target",), coords={"target": prate.target})
    monthly = prate * scale
    monthly.attrs.update(
        {
            "long_name": "NMME monthly precipitation anomaly",
            "units": "mm/month",
            "conversion": "prate_mm_per_s * 86400 * days_in_target_month",
        }
    )
    return monthly


def load_target_countries(shapefile_path: Path, raster_crs) -> gpd.GeoDataFrame:
    countries = gpd.read_file(shapefile_path)
    if countries.crs is None:
        countries = countries.set_crs("EPSG:4326")
    if raster_crs is not None and countries.crs != raster_crs:
        countries = countries.to_crs(raster_crs)

    selected = countries[countries["ISO_A3"].isin(COUNTRIES.values())].copy()
    missing = sorted(set(COUNTRIES.values()) - set(selected["ISO_A3"]))
    if missing:
        raise ValueError(f"Could not find requested countries in shapefile: {missing}")

    order = pd.Categorical(selected["ISO_A3"], categories=list(COUNTRIES.values()), ordered=True)
    selected = selected.assign(_order=order).sort_values("_order").drop(columns="_order")
    return selected[["ADMIN", "ISO_A3", "geometry"]].rename(columns={"ADMIN": "country", "ISO_A3": "iso_a3"})


def pixel_centers_for_window(transform, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
        raise ValueError("Rotated population rasters are not supported.")

    xs = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    ys = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return ys, xs


def climate_indices_for_pixels(lat_centers: np.ndarray, lon_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.floor(lat_centers + 90.5).astype(np.int16)
    lat_idx = np.clip(lat_idx, 0, 180)

    lon360 = np.mod(lon_centers, 360.0)
    lon_idx = np.floor(lon360 + 0.5).astype(np.int16) % 360
    return lat_idx, lon_idx


def clipped_geometry_window(src: rasterio.io.DatasetReader, geometry) -> Window:
    bounds = geometry.bounds
    window = src.window(*bounds)
    row_start = max(0, int(np.floor(window.row_off)))
    col_start = max(0, int(np.floor(window.col_off)))
    row_stop = min(src.height, int(np.ceil(window.row_off + window.height)))
    col_stop = min(src.width, int(np.ceil(window.col_off + window.width)))

    if row_stop <= row_start or col_stop <= col_start:
        raise ValueError(f"Country geometry does not overlap population raster: {bounds}")

    return Window.from_slices((row_start, row_stop), (col_start, col_stop))


def population_weights_for_country(
    src: rasterio.io.DatasetReader,
    geometry,
    nlat: int,
    nlon: int,
) -> np.ndarray:
    window = clipped_geometry_window(src, geometry)
    pop = src.read(1, window=window, masked=True)
    transform = src.window_transform(window)

    inside = geometry_mask(
        [geometry],
        out_shape=pop.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    values = np.asarray(pop.filled(0.0), dtype=np.float64)
    valid = inside & ~np.ma.getmaskarray(pop) & np.isfinite(values) & (values > 0)

    weights = np.zeros((nlat, nlon), dtype=np.float64)
    if not np.any(valid):
        return weights

    lat_centers, lon_centers = pixel_centers_for_window(transform, pop.shape[0], pop.shape[1])
    lat_idx, lon_idx = climate_indices_for_pixels(lat_centers, lon_centers)

    row_idx = np.broadcast_to(lat_idx[:, None], pop.shape)[valid]
    col_idx = np.broadcast_to(lon_idx[None, :], pop.shape)[valid]
    np.add.at(weights, (row_idx, col_idx), values[valid])
    return weights


def build_population_weights(
    population_tif: Path,
    countries: gpd.GeoDataFrame,
    nlat: int,
    nlon: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    weight_grids = np.zeros((len(countries), nlat, nlon), dtype=np.float64)
    summaries = []

    with rasterio.open(population_tif) as src:
        print("Population GeoTIFF metadata")
        print(f"  CRS: {src.crs}")
        print(f"  resolution: {src.res}")
        print(f"  bounds: {src.bounds}")
        print(f"  nodata: {src.nodata}")
        print()

        for i, country in countries.reset_index(drop=True).iterrows():
            print(f"Aggregating population to NMME grid: {country.country} ({country.iso_a3})")
            weights = population_weights_for_country(src, country.geometry, nlat, nlon)
            pop_sum = float(weights.sum())
            if pop_sum <= 0:
                raise ValueError(f"No valid population pixels found for {country.country}.")
            weight_grids[i, :, :] = weights
            summaries.append(
                {
                    "country": country.country,
                    "iso_a3": country.iso_a3,
                    "population_sum": pop_sum,
                    "occupied_nmme_cells": int(np.count_nonzero(weights)),
                }
            )

    summary = pd.DataFrame(summaries)
    print("\nPopulation-weight summary")
    print(summary.to_string(index=False))
    print()
    return weight_grids, summary


def save_weights_cache(cache_path: Path, countries: gpd.GeoDataFrame, weights: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        weights=weights,
        country=np.array(countries["country"], dtype=object),
        iso_a3=np.array(countries["iso_a3"], dtype=object),
    )


def load_weights_cache(cache_path: Path, countries: gpd.GeoDataFrame) -> np.ndarray | None:
    if not cache_path.is_file():
        return None

    cached = np.load(cache_path, allow_pickle=True)
    cached_iso = list(cached["iso_a3"])
    expected_iso = list(countries["iso_a3"])
    if cached_iso != expected_iso:
        print(f"Ignoring weight cache with different country order: {cache_path}")
        return None

    print(f"Loaded population weights cache: {cache_path}")
    return np.asarray(cached["weights"], dtype=np.float64)


def weighted_average(climate_map: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(climate_map) & (weights > 0)
    denom = weights[valid].sum()
    if denom <= 0:
        return np.nan
    return float(np.sum(climate_map[valid] * weights[valid]) / denom)


def build_output_table(
    countries: gpd.GeoDataFrame,
    weights: np.ndarray,
    heat: xr.DataArray,
    moisture: xr.DataArray,
    init_month: pd.Period,
    target_months: pd.PeriodIndex,
) -> pd.DataFrame:
    records = []
    heat_np = np.asarray(heat.values, dtype=np.float64)
    moisture_np = np.asarray(moisture.values, dtype=np.float64)

    for country_idx, country in countries.reset_index(drop=True).iterrows():
        country_weights = weights[country_idx]
        pop_sum = float(country_weights.sum())

        for target_idx, target_month in enumerate(target_months):
            lead_time = (target_month.year - init_month.year) * 12 + (target_month.month - init_month.month)
            variables = {
                "heat": heat_np[:, :, target_idx],
                "moisture": moisture_np[:, :, target_idx],
            }

            for variable, climate_map in variables.items():
                records.append(
                    {
                        "country": country.country,
                        "iso_a3": country.iso_a3,
                        "init_month": str(init_month),
                        "target_month": str(target_month),
                        "lead_time": int(lead_time),
                        "variable": variable,
                        "pop_weighted_anomaly": weighted_average(climate_map, country_weights),
                        "population_sum": pop_sum,
                    }
                )

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve() if args.data_dir else find_default_data_dir()
    paths = build_input_paths(data_dir)
    output = args.output or (data_dir / "nmme_country_population_weighted_climate.csv")
    weights_cache = args.weights_cache or (data_dir / "nmme_population_weights_12countries.npz")

    heat, heat_ds = load_nmme_cube(paths.heat_nc, expected_units="K")
    prate, prate_ds = load_nmme_cube(paths.moisture_nc, expected_units="mm/s")

    target = np.asarray(heat_ds["target"].values)
    initial_time = np.asarray(heat_ds["initial_time"].values)
    init_month, target_months = print_target_date_check(target, initial_time)

    moisture = convert_prate_to_monthly_anomaly(prate, target_months)

    if heat.shape != moisture.shape:
        raise ValueError(f"Heat and moisture shapes differ: {heat.shape} vs {moisture.shape}")

    print("NMME metadata")
    print(f"  heat dims: {heat.dims}, shape: {heat.shape}, units: {heat.attrs.get('units')}")
    print(f"  moisture dims: {moisture.dims}, shape: {moisture.shape}, units: {moisture.attrs.get('units')}")
    print(f"  lat range: {float(heat.lat.min())} to {float(heat.lat.max())}, count: {heat.sizes['lat']}")
    print(f"  lon range: {float(heat.lon.min())} to {float(heat.lon.max())}, count: {heat.sizes['lon']}")
    print()

    with rasterio.open(paths.population_tif) as pop_src:
        countries = load_target_countries(paths.countries_shp, pop_src.crs)

    if args.no_cache:
        weights = None
    else:
        weights = load_weights_cache(weights_cache, countries)

    if weights is None:
        weights, _ = build_population_weights(
            paths.population_tif,
            countries,
            nlat=heat.sizes["lat"],
            nlon=heat.sizes["lon"],
        )
        if not args.no_cache:
            save_weights_cache(weights_cache, countries, weights)
            print(f"Saved population weights cache: {weights_cache}")

    table = build_output_table(countries, weights, heat, moisture, init_month, target_months)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)

    print(f"\nWrote {len(table)} rows to {output}")


if __name__ == "__main__":
    main()
