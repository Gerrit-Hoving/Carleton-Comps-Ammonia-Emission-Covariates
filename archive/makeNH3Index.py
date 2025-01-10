# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:26:52 2024

@author: Gerrit
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import xarray as xr
import os



raster_path = r'C:\Users\Gerrit\Documents\comps\data\EMIT\EMIT_L1B_RAD_001_20240804T181617_2421712_005.nc'
output_path = r'C:\Users\Gerrit\Documents\comps\data\Process'

data = xr.open_dataset(raster_path)
  
band_215 = data['radiance'].isel(bands=214).values
band_219 = data['radiance'].isel(bands=218).values
band_252 = data['radiance'].isel(bands=251).values
band_248 = data['radiance'].isel(bands=247).values
band_253 = data['radiance'].isel(bands=252).values
band_259 = data['radiance'].isel(bands=258).values


nh3_1 = (band_215 + band_219) / band_215
nh3_2 = (band_252 + band_248) / band_252
nh3_3 = (band_253 + band_259) / band_253

visible = data['radiance'].isel(bands=19).values
nh3_123sum = nh3_1 + nh3_2 + nh3_3
nh3_4 = (band_253 + band_252) / band_248 + band_259

band_245 = data['radiance'].isel(bands=244).values
band_262 = data['radiance'].isel(bands=261).values
band_222 = data['radiance'].isel(bands=221).values

nh3_4wide = (band_253 + band_252) / band_245 + band_262
nh3_1wide = (band_215 + band_222) / band_215

# Testing in/out plume spectra ratio result
band_2011nm = data['radiance'].isel(bands=219).values
band_2004nm = data['radiance'].isel(bands=218).values
band_2019nm = data['radiance'].isel(bands=220).values

band_1989nm = data['radiance'].isel(bands=216).values
band_1982nm = data['radiance'].isel(bands=215).values

band_2034nm = data['radiance'].isel(bands=222).values
band_2041nm = data['radiance'].isel(bands=223).values


test1 = band_2011nm + band_1982nm / band_2011nm
test2 = band_2011nm + band_2041nm / band_2011nm
test3 = (band_2004nm + band_2011nm + band_2019nm) / (band_1989nm + band_1982nm)
test4 = (band_2004nm + band_2011nm + band_2019nm) / (band_2034nm + band_2041nm)

#indexes = {"Index_NH3_1" : nh3_1, "Index_NH3_2" :nh3_2, "Index_NH3_3" : nh3_3}
#indexes = {"Index_NH3_4" : nh3_4, "Visible" : visible, "Index_NH3_1-3_Sum" : nh3_123sum}
#indexes = {"Index_NH3_4_Wide" : nh3_4wide, "Index_NH3_1_Wide" : nh3_1wide}
indexes = {"Test 1" : test1, "Test 2" : test2, "Test 3" : test3, "Test 4" : test4}

transform = from_origin(data.attrs['westernmost_longitude'], data.attrs['northernmost_latitude'], data.attrs['spatialResolution'], data.attrs['spatialResolution'])
crs = data.attrs.get('spatial_ref', 'EPSG:4326')  # Default to EPSG:4326 if not present

for name, index in indexes.items():
    output_geotiff = os.path.join(output_path, name + ".tif")
    # Write NDVI to GeoTIFF
    with rasterio.open(
        output_geotiff, 'w',
        driver='GTiff',
        height=index.shape[0],
        width=index.shape[1],
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(index, 1)



        
    #with rasterio.open(rasterPath) as inRaster:
    #    numBands = inRaster.count
    #df_combined.to_csv('bandMedians.csv', mode='x')
    #return df_combined