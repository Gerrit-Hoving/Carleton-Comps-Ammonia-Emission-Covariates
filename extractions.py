# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:40:14 2025

@author: Gerrit Hoving

Functions for extracting spectral data from EMIT and AVIRIS raster files
"""

from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import os


# Output file paths
emit_bands_avg_path = r'D:\Documents\Projects\comps\data\emitReflectance.csv'
aviris_bands_path = r'D:\Documents\Projects\comps\data\avirisReflectance.csv'

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
    dfs = []
    for list_name, data in zip(rasterNames, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        print(df)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    full_df = pd.concat(dfs.values(), axis=1)
    
    median_df = pd.DataFrame()
    full_df[full_df <= 0] = np.nan
    
    # Take the mean of the values for each band
    for band in range(1, 286):
        # Create a list of columns that refer to the current band
        band_columns = [col for col in full_df.columns if f'band_{band}' in col]
        
        # Calculate the median of those columns and add it to the new DataFrame
        median_df[f'reflectance_band_{band}_median'] = full_df[band_columns].median(axis=1)
        
    return median_df


def pullData(mode = 'EMIT'):
    nh3_path = r'D:\Documents\Projects\comps\data\cafoNH3Aerial.csv'
    cafos_path = r'D:\Documents\Projects\comps\data\cafoAttributes.csv'
    
    nh3_df = pd.read_csv(nh3_path)
    nh3_df = nh3_df.drop(columns=['Unnamed: 0'])
    nh3_df = nh3_df.rename(columns={'Lot_CAFOID': 'CAFO_ID'})

    cafos_df = pd.read_csv(cafos_path)
    cafos_df = cafos_df.rename(columns={'ID': 'CAFO_ID'})
    
    if(mode == 'EMIT'):
        raster_df = pd.read_csv(emit_bands_avg_path)
        raster_df = raster_df.rename(columns={'Unnamed: 0': 'CAFO_ID'})
        raster_df['CAFO_ID'] = raster_df['CAFO_ID'] + 1
        # Double checks to make sure nans set coreectly
        raster_df[raster_df <= 0] = np.nan

    # Combine CAFO attributes and NH3 emissions data to get feedlot data
    target_df = pd.merge(cafos_df, nh3_df, on='CAFO_ID', how='left')
    target_df = target_df.fillna(0)
    target_df['HyTES_NH3_Detect'] = target_df['NH3_mean'] != 0
    
    # Merge with raster data
    full_df = pd.merge(target_df, raster_df, on='CAFO_ID', how='left')
    
    # Create lists of targets and features for easier use with RF
    target_list = list(target_df.columns)
    feature_list = [item for item in full_df.columns if item not in target_list]
    
    return full_df, target_list, feature_list

data, targets, features = pullData()

folder = r'D:\Documents\Projects\comps\data\process'
vector = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'

#df = extractAvgAcrossRasters(folder, vector)
#df.to_csv(emit_bands_avg_path)



#raster = r'D:\Documents\Projects\comps\data\process\EMIT_L2A_RFL_001_20240423T183345_2411412_005_reflectance'
#vector = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'

#df = extractEMITByRaster(raster, vector)


