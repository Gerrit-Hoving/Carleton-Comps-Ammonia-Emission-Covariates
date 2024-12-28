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
from netCDF4 import Dataset
import geopandas as gpd
import pandas as pd



def extract():
    # Extract statistics within polygons for a set of files
    
    fileDirectory = r'/data/Documents/Projects/comps/data/EMIT'
    polygonFile = r'/data/Documents/Projects/comps/data/Shapefiles/CAFOs.gpkg'
    polygonLayer = "CAFOs_EMIT_CorrectedV3"
    dropColumns = ['CAFO_sum_emission_auto', 'CAFO_sum_emission_uncertainty_auto', 'CAFO_Point_Count']
    
    allFiles = os.listdir(fileDirectory)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(fileDirectory, f)) and '.' not in f]
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
        
        
        zs = nc_zonal_stats(img, cafos, stats=rasterStat)
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
    
    result_df.to_csv('indexInfo.csv', mode='x')
    
    return result_df
    


def nc_zonal_stats(ncFile, shpFile):
    dataset = Dataset(ncFile) 
    band_names = list(dataset.variables.keys())  # List of band names (variables)
    gdf = gpd.read_file(shpFile)
    
    # List to store the zonal stats results for each band
    zonal_stats_list = []
    
    # Loop through each band in the .nc file
    for band_name in band_names:
        # Get the band data from the .nc file
        band_data = dataset.variables[band_name]  # Get the band data
        
        # Extract the band data (this may need adjustment depending on how the data is stored)
        band_array = band_data[:]
        
        # Use rasterstats to calculate zonal statistics for this band
        stats = zonal_stats(gdf, band_array, stats=['mean'])
        
        # Extract the desired statistics for each region and store in the list
        stats_df = pd.DataFrame(stats)
        
        # You can assign a new column name for the band statistics
        stats_df[band_name] = stats_df['mean']  # Here we add the 'mean' value for this band
       
    # Combine the statistics for each band into a single DataFrame
    zonal_stats_df = pd.concat(zonal_stats_list, axis=1)
    
    # Optionally, you can rename the columns to match the band names
    zonal_stats_df.columns = band_names
    
    # Close the .nc file when done
    dataset.close()
    
    return zonal_stats_df
    
extract()