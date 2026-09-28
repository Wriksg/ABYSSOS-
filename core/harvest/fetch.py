import os
import numpy as np
import xarray as xr
from pystac_client import Client
import planetary_computer
from collections import Counter
import odc.stac
import pandas as pd

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

def fetch_s2_stack(bbox, start, end, max_cloud=60):
    print("📡 Querying AWS Element84 for Sentinel-2...")
    s2_client = Client.open("https://earth-search.aws.element84.com/v1")
    
    search = s2_client.search(
        collections=["sentinel-2-c1-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}}
    )
    items = list(search.items())
    if not items:
        raise ValueError("No S2 items found.")

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    items = items[:8]
    print(f"🛰️ Selected {len(items)} clearest passes.")

    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    
    print("📥 Downloading optical bands... (No Dask, single-threaded for stability)")
    ds = odc.stac.load(
        items,
        bbox=bbox,
        bands=["blue", "green", "red", "nir", "scl"],
        resolution=10,
        groupby="solar_day"
    )
    
    scaled_vars = {}
    for band in ["blue", "green", "red", "nir"]:
        scaled_vars[band] = ds[band].astype(np.float32) * 0.0001

    lr = xr.concat([scaled_vars[b] for b in ["blue", "green", "red", "nir"]], dim="band")
    lr = lr.transpose("time", "band", "y", "x").values
    scl = ds["scl"].values
    
    # FIX: Vectorized pandas datetime conversion (100% foolproof)
    dates = pd.to_datetime(ds.time.values).strftime('%Y-%m-%d').tolist()
    epsg = ds.rio.crs.to_epsg() if ds.rio.crs else 32600
    
    return lr, scl, dates, epsg, ds.geobox

def fetch_s1_sar(bbox, ref_date, s2_geobox):
    print(f"📡 Querying Planetary Computer for Sentinel-1 (Nearest to {ref_date})...")
    s1_client = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    
    t_ref = pd.to_datetime(ref_date)
    start, end = (t_ref - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), (t_ref + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    search = s1_client.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}")
    items = list(search.items())
    
    if not items:
        print("⚠️ No SAR coverage found in window.")
        return np.zeros((2, s2_geobox.shape[0], s2_geobox.shape[1]), dtype=np.float32), False
        
    items.sort(key=lambda i: abs(pd.to_datetime(i.datetime).tz_localize(None) - t_ref))
    
    print("📥 Downloading SAR data...")
    ds = odc.stac.load(
        [items[0]], geobox=s2_geobox, bands=["vv", "vh"], resampling="bilinear"
    )
    
    vv = ds["vv"].values[0]
    vh = ds["vh"].values[0]
    
    vv_db = 10 * np.log10(np.clip(vv, 1e-10, None))
    vh_db = 10 * np.log10(np.clip(vh, 1e-10, None))
    
    vv_norm = np.clip((vv_db + 30) / 30, 0, 1)
    vh_norm = np.clip((vh_db + 30) / 30, 0, 1)
    
    return np.stack([vv_norm, vh_norm], axis=0).astype(np.float32), True
