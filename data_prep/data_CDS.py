import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.config import COUNTRY_NAMES, REGIONMASK_TO_ISO3, ISO3
# ISO3 = list("BRA","IND",...) used as columns country in CSV files

# ----- PARAMETERS
quantile_heat_moist = 0.84 # defines severity at lat,lon pixels, complement used for moisture

flag_save_all_countries = False                 # False: don't overwrite files with all countries
flag_save_country_data = True                   #
flag_download_nc = False                        # already done for whole globe
flag_save_prithvi_proxy_heat_anom = False
flag_save_prithvi_proxy_heat_exceedance = True
flag_save_ENSO = False

# Single country
flag_single_country = False
if flag_single_country:
    ISO3 = ["AUS"]
    COUNTRY_NAMES = ["Australia"]
# else:
    # ISO3 = ISO3[13:] # Australia is 12
    # COUNTRY_NAMES = COUNTRY_NAMES[13:]

if flag_download_nc:
    import cdsapi

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": [
                "2m_temperature",
                "total_precipitation",
                "evaporation",
            ],
            "year": [str(y) for y in range(1981, 2025)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "format": "netcdf",
        },
        "era5_monthly_t2m_tp_e.nc",
    )

if flag_save_country_data:
    import regionmask
    import xarray as xr

    # 1. Build country masks (once)
    # Built-in Natural Earth (no geopandas)
    regions_all = regionmask.defined_regions.natural_earth_v5_0_0.countries_110

    keep_idx = [
        i for i, name in enumerate(regions_all.names)
        if name in COUNTRY_NAMES
    ]
    # Subset regions correctly
    countries = regions_all[keep_idx]
    print(countries.names)

    # 2. Load ERA5 monthly temperature
    # Load ERA5 monthly variables
    # t2m
    # ds = xr.open_dataset("era5_monthly_t2m_tp_e.nc")
    # ds = xr.open_dataset(
    #     "era5_monthly_t2m_tp_e.nc",
    #     engine="netcdf4"
    # )
    ds_temp = xr.open_dataset(
        "../tvp/data_stream-moda_stepType-avgua.nc",
        engine="netcdf4"
    )
    ds_flux = xr.open_dataset(
        "../tvp/data_stream-moda_stepType-avgad.nc",
        engine="netcdf4"
    )
    print(ds_temp.data_vars)
    print(ds_flux.data_vars)

    t2m_c = ds_temp["t2m"] - 273.15
    t2m_c = t2m_c.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    t2m_c = t2m_c.assign_coords(lon=((t2m_c.lon + 180) % 360) - 180).sortby("lon")

    # Meters of water - names can be 'tp' and 'e' depending on your download
    p = ds_flux["tp"]  # total precipitation
    et = ds_flux["e"]  # evaporation (typically negative over land)

    # Moisture balance (P - ET)
    # Note: if 'e' is negative, p - e increases dryness when e is large in magnitude
    moisture = p - et
    moisture = moisture.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
    moisture = moisture.assign_coords(lon=((moisture.lon + 180) % 360) - 180).sortby("lon")

    # 3. Apply masks to ERA5 grid (valid for any ERA5 variable on the same grid)
    t2m_sample = t2m_c.isel(time=0)  # dims: lat, lon
    mask = countries.mask(t2m_sample)  # dims: lat, lon
    print(mask)
    print("Unique region IDs:", np.unique(mask.values[~np.isnan(mask.values)]))

    flag_values = mask.attrs["flag_values"]
    flag_meanings = mask.attrs["flag_meanings"].split()
    region_map = dict(zip(flag_values, flag_meanings))

    print("Region code → country code:")
    for k, v in region_map.items():
        print(k, v)

    records_temp = []
    records_moisture = []

    for region_code, abbr in region_map.items():
        if abbr not in REGIONMASK_TO_ISO3:
            print(f"[SKIP] Unmapped region {abbr}")
            continue

        iso3 = REGIONMASK_TO_ISO3[abbr]

        country_temp = t2m_c.where(mask == region_code)
        country_moisture = moisture.where(mask == region_code)
        country_temp_2d = country_temp.stack(location=("lat", "lon")).dropna("location", how="all")
        country_moisture_2d = country_moisture.stack(location=("lat", "lon")).dropna("location", how="all")
        # Can still get lat, lon, e.g. lon = country_moisture_2d.lon.values
        df_moist = country_moisture_2d.to_dataframe(name="moisture")
        df_heat = country_temp_2d.to_dataframe(name="heat")
        # Reset the MultiIndex to turn lat/lon into regular columns, then drop them
        df_simple_moist = df_moist.reset_index().drop(columns=["number", "expver"])
        df_simple_heat = df_heat.reset_index().drop(columns=["number", "expver"])
        # Save to CSV (index=False prevents adding a redundant 0, 1, 2... column)
        df_simple_moist.to_csv(f"../data_large/heat_moisture/moisture_{iso3}.csv", index=False)
        df_simple_heat.to_csv(f"../data_large/heat_moisture/heat_{iso3}.csv", index=False)

if flag_save_prithvi_proxy_heat_anom:
    records_temp = []
    records_moisture = []

    for iso3 in ISO3:
        df_moist_all = pd.read_csv(f"../data_large/heat_moisture/moisture_{iso3}.csv",
                                   parse_dates=["time"])
        df_heat_all = pd.read_csv(f"../data_large/heat_moisture/heat_{iso3}.csv",
                                  parse_dates=["time"])

        # Compute mean over the 1D 'location' dimension (which contains both lat and lon)
        ts = df_heat_all.groupby("time")["heat"].mean()
        ms = df_moist_all.groupby("time")["moisture"].mean()

        df_temp = ts.to_frame(name="temp_c").reset_index()
        df_moisture = ms.to_frame(name="moisture").reset_index()
        df_temp["country"] = iso3
        df_moisture["country"] = iso3

        records_temp.append(df_temp)
        records_moisture.append(df_moisture)

    # Convert to DataFrame
    monthly_df_temp = pd.concat(records_temp, ignore_index=True)
    monthly_df_moisture = pd.concat(records_moisture, ignore_index=True)

    # 6. Compute baseline mean per country
    baseline_df_temp = (
        monthly_df_temp[
            (monthly_df_temp["time"] >= "1981-01-01") &
            (monthly_df_temp["time"] <= "2010-12-31")
            ]
        .groupby("country")["temp_c"]
        .mean()
        .rename("baseline_mean")
    )
    baseline_df_moisture = (
        monthly_df_moisture[
            (monthly_df_moisture["time"] >= "1981-01-01") &
            (monthly_df_moisture["time"] <= "2010-12-31")
            ]
        .groupby("country")["moisture"]
        .mean()
        .rename("baseline_mean")
    )

    # 7. Compute monthly anomaly
    #    First create a new column (using the original baseline name, "baseline_mean,"
    #    repeating baseline for every row in monthly of the same country
    monthly_df_temp = monthly_df_temp.merge(
        baseline_df_temp,
        on="country",
        how="left"
    )
    monthly_df_moisture = monthly_df_moisture.merge(
        baseline_df_moisture,
        on="country",
        how="left"
    )

    #    Then create another column with the anomaly
    monthly_df_temp["heat_anom"] = (
            monthly_df_temp["temp_c"] - monthly_df_temp["baseline_mean"]
    )
    monthly_df_moisture["moisture_anom"] = (
            monthly_df_moisture["moisture"] - monthly_df_moisture["baseline_mean"]
    )

    # 8. Aggregate to quarterly anomaly
    monthly_df_temp["quarter"] = (
        pd.to_datetime(monthly_df_temp["time"]).dt.to_period("Q").dt.to_timestamp(how="start")
    )
    monthly_df_moisture["quarter"] = (
        pd.to_datetime(monthly_df_moisture["time"]).dt.to_period("Q").dt.to_timestamp(how="start")
    )

    prithvi_heat_q = (
        monthly_df_temp.groupby(["country", "quarter"])["heat_anom"].mean()
        .reset_index().rename(columns={"heat_anom": "PRITHVI_HEAT_ANOM"})
    )
    prithvi_moisture_q = (
        monthly_df_moisture.groupby(["country", "quarter"])["moisture_anom"].mean()
        .reset_index().rename(columns={"moisture_anom": "PRITHVI_MOISTURE_ANOM"})
    )

    # “1-unit shock” = one standard deviation heat anomaly for that country
    prithvi_heat_q["PRITHVI_HEAT_STD"] = (
        prithvi_heat_q.groupby("country")["PRITHVI_HEAT_ANOM"].transform(lambda x: x / x.std())
    )
    prithvi_moisture_q["PRITHVI_MOISTURE_STD"] = (
        prithvi_moisture_q.groupby("country")["PRITHVI_MOISTURE_ANOM"].transform(lambda x: x / x.std())
    )

    # 9. Save final proxy
    if flag_save_all_countries:
        prithvi_heat_q.to_csv("data/prithvi_heat_anom.csv", index=False)
        prithvi_moisture_q.to_csv("data/prithvi_moisture_anom.csv", index=False)

    # Sanity checks (do these before merging)
    prithvi_heat_q.groupby("country")["PRITHVI_HEAT_ANOM"].describe()
    prithvi_heat_q[prithvi_heat_q.country == "India"].set_index("quarter")["PRITHVI_HEAT_ANOM"].plot()
    prithvi_moisture_q.groupby("country")["PRITHVI_MOISTURE_ANOM"].describe()
    prithvi_moisture_q[prithvi_moisture_q.country == "India"].set_index("quarter")[
        "PRITHVI_MOISTURE_ANOM"].plot()
    plt.show()

if flag_save_prithvi_proxy_heat_exceedance:
    records_temp = []
    records_moisture = []
    all_moisture = []
    all_temp = []

    for iso3 in ISO3:
        df_moist_all = pd.read_csv(f"../data_large/heat_moisture/moisture_{iso3}.csv",
                                   parse_dates=["time"])
        df_heat_all = pd.read_csv(f"../data_large/heat_moisture/heat_{iso3}.csv",
                                  parse_dates=["time"])
        # Subtract the average of each location (lat, lon pair) from its own values
        df_moist_all["anomaly"] = (df_moist_all.groupby(["lat", "lon"])["moisture"]
                                   .transform(lambda x: x - x.mean()))
        df_heat_all["anomaly"] = (df_heat_all.groupby(["lat", "lon"])["heat"]
                                   .transform(lambda x: x - x.mean()))

        # Compute grid‐cell drought thresholds (once) and store as a pandas Series
        #   df_moist_all is a DataFrame (2d table)
        #   df_moist_all["anomaly"] is a Series (1d labeled vector)
        #   df_moist_all[["anomaly"]] or df_moist_all["anomaly"].to_frame() is a DataFrame with 1 column
        #   Compute over the 1D 'location' dimension (which contains both lat and lon)
        #   Group by location and calculate the 0.10 quantile for each pixel
        # grid_moist_thresh = df_moist_all.groupby(["lat", "lon"])["anomaly"].quantile(0.10)
        # grid_heat_thresh = df_heat_all.groupby(["lat", "lon"])["anomaly"].quantile(0.10)
        moist_xr = df_moist_all.set_index(["time", "lat", "lon"])["anomaly"].to_xarray()
        heat_xr = df_heat_all.set_index(["time", "lat", "lon"])["anomaly"].to_xarray()
        grid_moist_thresh = moist_xr.quantile( 1 - quantile_heat_moist, dim="time", skipna=True )
        grid_heat_thresh = heat_xr.quantile( quantile_heat_moist, dim="time", skipna=True )
        # Identify drought grid cells at each time
        drought_cells = moist_xr <= grid_moist_thresh
        heat_cells = heat_xr >= grid_heat_thresh
        drought_count = drought_cells.sum(dim=("lat", "lon"), skipna=True)
        heat_count = heat_cells.sum(dim=("lat", "lon"), skipna=True)
        total_drought_count = drought_cells.count(dim=("lat", "lon"))
        total_heat_count = heat_cells.count(dim=("lat", "lon"))
        drought_extent = drought_count / total_drought_count
        heat_extent = heat_count / total_heat_count
        df_drought = drought_extent.to_dataframe(name="DROUGHT_EXTENT").reset_index()
        df_heat = heat_extent.to_dataframe(name="HEAT_EXTENT").reset_index()
        df_drought["country"] = iso3
        df_heat["country"] = iso3
        records_moisture.append(df_drought)
        records_temp.append(df_heat)
        df_moisture = moist_xr.to_dataframe(name="moisture").reset_index().dropna()
        df_heat = heat_xr.to_dataframe(name="heat").reset_index().dropna()
        df_moisture["country"] = iso3
        df_heat["country"] = iso3
        all_moisture.append(df_moisture)
        all_temp.append(df_heat)

    # Convert to DataFrame
    monthly_df_moisture = pd.concat(records_moisture, ignore_index=True)
    monthly_df_temp = pd.concat(records_temp, ignore_index=True)

    all_df_moisture = pd.concat(all_moisture, ignore_index=True)
    all_df_temp = pd.concat(all_temp, ignore_index=True)

    # SAVE IN MATLAB FORMAT
    # import xarray as xr
    import scipy.io
    import numpy as np

    def save_for_matlab(df, file_name):
        mat_data = {}
        for col_name in df.columns:
            mat_data[col_name] = df[col_name].values.reshape(-1, 1)
        scipy.io.savemat(f'../data_MATLAB/{file_name}.mat', mat_data)

    if not flag_save_all_countries:
        save_for_matlab(all_df_moisture, 'moisture_data_new')
        save_for_matlab(all_df_temp,'temp_data_new')
    else:
        save_for_matlab(all_df_moisture,'moisture_data')
        save_for_matlab(all_df_temp,'temp_data')

        # 8. Aggregate to quarterly anomaly
        monthly_df_moisture["quarter"] = (
            pd.to_datetime(monthly_df_moisture["time"]).dt.to_period("Q").dt.to_timestamp(how="start")
        )
        monthly_df_temp["quarter"] = (
            pd.to_datetime(monthly_df_temp["time"]).dt.to_period("Q").dt.to_timestamp(how="start")
        )

        prithvi_moisture_q = (
            monthly_df_moisture.groupby(["country", "quarter"])["DROUGHT_EXTENT"].mean()
            .reset_index().rename(columns={"DROUGHT_EXTENT": "PRITHVI_MOISTURE_EXTENT"})
        )
        prithvi_heat_q = (
            monthly_df_temp.groupby(["country", "quarter"])["HEAT_EXTENT"].mean()
            .reset_index().rename(columns={"HEAT_EXTENT": "PRITHVI_HEAT_EXTENT"})
        )

        prithvi_moisture_q.to_csv("data/prithvi_moisture_extent.csv", index=False)
        prithvi_heat_q.to_csv("data/prithvi_heat_extent.csv", index=False)

if flag_save_ENSO:
    # ----- Read ENSO
    enso_raw = pd.read_csv("../data/nina34.csv")
    enso = enso_raw.rename(columns={
        enso_raw.columns[0]: "date",
        enso_raw.columns[1]: "ENSO"
    })

    enso["date"] = pd.to_datetime(enso["date"], errors="coerce")
    enso["ENSO"] = pd.to_numeric(enso["ENSO"], errors="coerce")

    # Replace NOAA missing value code
    enso.loc[enso["ENSO"] <= -99, "ENSO"] = np.nan

    enso["quarter"] = (
        enso["date"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="start")
    )

    enso_q = (
        enso
        .groupby("quarter")["ENSO"]
        .mean()
        .reset_index()
    )

    # ----- Standardize ENSO
    enso_q["ENSO_STD"] = (
                                 enso_q["ENSO"] - enso_q["ENSO"].mean()
                         ) / enso_q["ENSO"].std()

    enso_q.to_csv(
        "data/enso_quarterly.csv",
        index=False
    )

    enso_q.set_index("quarter")["ENSO"].plot(title="Quarterly Niño 3.4 Index")
    plt.show()
