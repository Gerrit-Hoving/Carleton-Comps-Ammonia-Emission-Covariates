#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:42:55 2024

@author: Gerrit Hoving

Functions for extracting EMIT and AVIRIS spectra within polygons
"""

from rasterstats import zonal_stats
import fiona
import os
import pandas as pd
import numpy as np
import rasterio
import xarray as xr

import sys
sys.path.append('/data/Documents/Projects/comps/repo')
import emit_tools


def extract():
    # Extract statistics within polygons for a set of files
    
    fileDirectory = r'/data/Documents/Projects/comps/data/EMIT'
    polygonFile = r'/data/Documents/Projects/comps/data/Shapefiles/CAFOs.gpkg'
    polygonLayer = "CAFOs_EMIT_CorrectedV3"
    dropColumns = ['CAFO_sum_emission_auto', 'CAFO_sum_emission_uncertainty_auto', 'CAFO_Point_Count']
    
    allFiles = os.listdir(fileDirectory)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(fileDirectory, f))]
    rasterPaths = [os.path.join(fileDirectory, raster) for raster in rasterNames]
    rasterStat = 'median'
    
    print("Calculating statistic", rasterStat, "for rasters", rasterNames)
    
    results = []
    attributes = []
    
    # Open CAFO polygon file and add all the attributes to a list
    cafos = fiona.open(polygonFile, layer=polygonLayer)
    for feature in cafos:
        attributes.append(fiona.model.to_dict(feature['properties']))
    results.append(attributes)
    
    # Work through every image in the rasters folder and calculate the zonal statistics for it
    for img in rasterPaths:
        zs = nc_zonal_stats(img, cafos)
        results.append(zs)

    # Combine lists into a dictionary of DataFrames
    rasterNames.insert(0, "CAFO")
    dfs = {}
    for list_name, data in zip(rasterNames, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        print(df)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    result_df = pd.concat(dfs.values(), axis=1)
  
    result_df = result_df.drop(columns=dropColumns)
    
    print("Output attributes", result_df.columns)
    
    result_df.to_csv('indexInfo.csv', mode='w')
    
    return result_df
    


def nc_zonal_stats(nc_file, shp_file):
    nc_file=r'/data/Documents/Projects/comps/data/EMIT/EMIT_L2A_RFL_001_20240423T183333_2411412_004.nc'
    raster = xr.open_dataset(nc_file)
    
    wvl = xr.open_dataset(nc_file,group='sensor_band_parameters')
    
    loc = xr.open_dataset(nc_file,group='location')
 
    # Create coordinates and an index for the downtrack and crosstrack dimensions, then unpack the variables from the wvl and loc datasets and set them as coordinates for ds
    raster = raster.assign_coords({'downtrack':(['downtrack'], raster.downtrack.data),'crosstrack':(['crosstrack'],raster.crosstrack.data), **wvl.variables, **loc.variables})

    raster = raster.swap_dims({'bands':'wavelengths'})
 
    del wvl
    del loc

    # Open nc file orthorectified
    raster = emit_tools.emit_xarray(nc_file, ortho=True)
    
    # Limit to good wavelnegths
    raster['reflectance'].data[:,:,raster['good_wavelengths'].data==0] = np.nan
    
    # Transpose dimensions
    raster = raster.transpose('wavelengths','latitude','longitude')
    
    #Testing saving then reopening with rio
    raster.to_netcdf('../data/geo_ds_out.nc')
    ds = rasterio.open('../data/geo_ds_out.nc')
    
    
    # List all the variables (bands) in the NetCDF file
    band_names = list(raster['reflectance']['wavelengths'].values)
    print(f"Found {len(band_names)} bands: {band_names}")
    
    zonal_stats_list = []
    
    # Loop through each band and extract it as a raster using rasterio
    for band_name in band_names:
        print(f"Reading band: {band_name}")
        
        # Extract the band data
        band_data = raster.sel(wavelengths = band_name)
        
        stats = zonal_stats(shp_file, band_data, stats=['mean'])
        
        stats_df = pd.DataFrame(stats)
        
        # You can assign a new column name for the band statistics
        stats_df[band_name] = stats_df['mean']  # Here we add the 'mean' value for this band
       
        zonal_stats_list.append(stats_df)
        
        
           
    # Combine the statistics for each band into a single DataFrame
    zonal_stats_df = pd.concat(zonal_stats_list, axis=1)
        
    # Close the dataset when done
    raster.close()
    
    return zonal_stats_df
    
extract()