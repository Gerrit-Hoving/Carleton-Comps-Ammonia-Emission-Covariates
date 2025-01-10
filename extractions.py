# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:40:14 2025

@author: Gerrit Hoving

Functions for extracting spectral data from EMIT and AVIRIS raster files
"""

from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import rasterio
import os



# Uses zonal_stats to extract the specified stats from an orthorectified EMIT 
# raster in ENVI format for each of the polygons in a given shapefile

def extractEMITByRaster(rasterPath, vectorPath, rasterStats = ['median']):
    cafos = gpd.read_file(vectorPath)
    with rasterio.open(rasterPath) as inRaster:
        numBands = inRaster.count
    
    # Extract polygon attributes
    attributes = cafos.drop(columns=['geometry']).copy()

    # Initialize a DataFrame to store zonal statistics for each band
    band_results = []
    
    # Calculate zonal statistics for each band
    for band in range(1, numBands + 1):
        # Calculate zonal statistics for the current band
        stats = zonal_stats(cafos, rasterPath, band=band, stats=rasterStats)
        
        # Convert the stats to a DataFrame and add a column for the band
        stats_df = pd.DataFrame(stats)
        stats_df.rename(columns={'median': f'band_{band}_median'}, inplace=True)
        
        # Combine stats with attributes
        band_results.append(stats_df)
    
    # Concatenate all band results along columns
    df_bands = pd.concat(band_results, axis=1)
    
    # Combine DataFrames
    df_combined = pd.concat([attributes, df_bands], axis=1)
    
    # Write to csv
    df_combined.to_csv('bandMedians.csv', mode='w')
    
    return df_combined


def extractAvgAcrossRasters(rasterFolder, vectorPath, stats = ['median']):
    allFiles = os.listdir(rasterFolder)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(rasterFolder, f)) and '.' not in f]
    rasterPaths = [os.path.join(rasterFolder, raster) for raster in rasterNames]
    
    results = []
    
    # Work through every image in the rasters folder and calculate the zonal statistics for it
    for file in rasterPaths:
        zs = extractEMITByRaster(file, vectorPath, rasterStats=stats)
        results.append(zs)
        
    # Combine lists into a dictionary of DataFrames
    #rasterNames.insert(0, "CAFO")
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
    
    # Take the mean of the values for each band
    median_df = pd.DataFrame()
    
    for band in range(1, 286):
        # Create a list of columns that refer to the current band
        band_columns = [col for col in result_df.columns if f'band_{band}' in col]
        
        # Calculate the median of those columns and add it to the new DataFrame
        median_df[f'reflectance_band_{band}_median'] = result_df[band_columns].median(axis=1)
    
    
    return median_df


folder = r'D:\Documents\Projects\comps\data\process'
vector = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'

df = extractAvgAcrossRasters(folder, vector)

raster = r'D:\Documents\Projects\comps\data\process\EMIT_L2A_RFL_001_20240423T183345_2411412_005_reflectance'
vector = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'
 
df = extractEMITByRaster(raster, vector)

df.to_csv('testExtracted.csv')
