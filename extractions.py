# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:40:14 2025

@author: Gerrit Hoving

Functions for extracting spectral data from EMIT and AVIRIS raster files
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import rasterio
import os
import fiona

# File path parameters
CARB_CAFOS_PATH = r'D:\Documents\Projects\comps\data\CAFOs_CARB_Atttributes.csv'
CARB_EMISSION_PATH = r'D:\Documents\Projects\comps\data\CAFOs_CARB_Emissions.csv'
CARB_EFACTOR_PATH = r'D:\Documents\Projects\comps\data\CAFOs_CARB_EmissionFactors.csv'

HYTES_CAFOS_PATH = r'D:\Documents\Projects\comps\data\cafoAttributes.csv'
HYTES_NH3_PATH = r'D:\Documents\Projects\comps\data\cafoNH3Aerial.csv'

HYTES_CAFOS_EMIT_BANDAVG_PATH = r'D:\Documents\Projects\comps\data\emitReflectanceHyTESLots.csv'
HYTES_CAFOS_AVIRIS_BANDAVG_PATH = r'D:\Documents\Projects\comps\data\avirisReflectanceHyTESLots.csv'
CARB_CAFOS_EMIT_BANDAVG_PATH = r'D:\Documents\Projects\comps\data\emitReflectanceCARBLots.csv'
CARB_CAFOS_EMIT_ALL_PATH = r'D:\Documents\Projects\comps\data\emitReflectanceCARBLotsComplete.csv'
CARB_CAFOS_EMIT_SUBSET_PATH = r'D:\Documents\Projects\comps\data\emitReflectanceCARBLotsSubset.csv'
CARB_CAFOS_EMIT_RAD_PATH = r'D:\Documents\Projects\comps\data\emitRadianceCARBLots.csv'
CARB_CAFOS_AVIRIS_BANDAVG_PATH = r'D:\Documents\Projects\comps\data\avirisReflectanceCARBLots.csv'



# Uses zonal_stats to extract the specified stats from an orthorectified EMIT 
# raster in ENVI format for each of the polygons in a given shapefile

def extractEMITByRaster(rasterPath, vectorPath, layer = None, rasterStats = ['median']):
    if layer is None:
        cafos = gpd.read_file(vectorPath) 
    else:
        cafos = fiona.open(vectorPath, layer=layer)
    
    with rasterio.open(rasterPath) as inRaster:
        numBands = inRaster.count

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
    
    return df_bands


def extractAvgAcrossRasters(rasterFolder, vectorPath, layer = None, stats = ['median']):
    allFiles = os.listdir(rasterFolder)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(rasterFolder, f)) and '.' not in f]
    rasterPaths = [os.path.join(rasterFolder, raster) for raster in rasterNames]
    
    results = []
    
    # Work through every image in the rasters folder and calculate the zonal statistics for it
    for file in rasterPaths:
        zs = extractEMITByRaster(file, vector, layer, rasterStats=stats)
        results.append(zs)
        
    # Combine lists into a dictionary of DataFrames
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
    full_df = pd.concat(dfs.values(), axis=1)
    
    median_df = pd.DataFrame()
    full_df[full_df.select_dtypes(include=['number']) <= 0] = np.nan
    
    # Take the mean of the values for each band
    for band in range(1, 286):
        # Create a list of columns that refer to the current band
        band_columns = [col for col in full_df.columns if f'band_{band}' in col]
        
        # Calculate the median of those columns and add it to the new DataFrame
        median_df[f'reflectance_band_{band}_median'] = full_df[band_columns].median(axis=1)
        
    return median_df, full_df


def pullData(farms = 'CARB', mode = 'EMIT', file = CARB_CAFOS_EMIT_SUBSET_PATH):
    if (farms == 'CARB'):
        emission_df = pd.read_csv(CARB_EMISSION_PATH)
        emission_df = emission_df.rename(columns={'CAFO': 'CAFO_ID'})
        
        factor_df = pd.read_csv(CARB_EFACTOR_PATH)
        factor_df = factor_df.rename(columns={'Farm': 'CAFO_ID'})
        
        cafos_df = pd.read_csv(CARB_CAFOS_PATH)
        cafos_df = cafos_df.rename(columns={'CARB_ID': 'CAFO_ID'})
        
        if(mode == 'EMIT'):
            feature_df = pd.read_csv(file)
            feature_df = pd.merge(cafos_df, feature_df, left_index = True, right_index = True)
            feature_df = feature_df.drop(columns=['Unnamed: 0'])
            
            # Perfrom one-hot encoding on region variable for RF compatability 
            feature_df = pd.get_dummies(feature_df, columns=['Region'])
            
        target_df = pd.merge(emission_df, factor_df, on='CAFO_ID', how='left')
        target_df = target_df.fillna(0)

    elif (farms == 'HyTES'):
        nh3_df = pd.read_csv(HYTES_NH3_PATH)
        nh3_df = nh3_df.drop(columns=['Unnamed: 0'])
        nh3_df = nh3_df.rename(columns={'Lot_CAFOID': 'CAFO_ID'})
    
        cafos_df = pd.read_csv(HYTES_CAFOS_PATH)
        cafos_df = cafos_df.rename(columns={'ID': 'CAFO_ID'})
    
        if(mode == 'EMIT'):
            feature_df = pd.read_csv(HYTES_CAFOS_EMIT_BANDAVG_PATH)
            feature_df = feature_df.rename(columns={'Unnamed: 0': 'CAFO_ID'})
            feature_df['CAFO_ID'] = feature_df['CAFO_ID'] + 1
            # Double checks to make sure nans set coreectly
            feature_df[feature_df <= 0] = np.nan
            
        # Combine CAFO attributes and NH3 emissions data to get feedlot data
        target_df = pd.merge(cafos_df, nh3_df, on='CAFO_ID', how='left')
        target_df = target_df.fillna(0)
        target_df['HyTES_NH3_Detect'] = target_df['NH3_mean'] != 0
        
    # Merge with raster data
    full_df = pd.merge(target_df, feature_df, on='CAFO_ID', how='left')
        
    # Create lists of targets and features for easier use with RF
    target_list = list(target_df.columns)
    feature_list = [item for item in full_df.columns if item not in target_list]
    
    return full_df, target_list, feature_list

#data, targets, features = pullData()


folder = r'D:\Documents\Projects\comps\data\EMIT\processed\subset'
vector = 'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_CARB.gpkg'
layer = "CAFOs_Buffer45_WGS84"


#df, full_df = extractAvgAcrossRasters(folder, vector, layer)
#df.to_csv(CARB_CAFOS_EMIT_SUBSET_PATH)
#full_df.to_csv(CARB_CAFOS_EMIT_ALL_PATH)



#raster = r'D:\Documents\Projects\comps\data\process\EMIT_L2A_RFL_001_20240423T183345_2411412_005_reflectance'
#vector = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'

#df = extractEMITByRaster(raster, vector)


