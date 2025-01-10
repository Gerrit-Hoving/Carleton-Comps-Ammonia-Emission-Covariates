# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:11:26 2024

@author: sarp
"""


from rasterstats import zonal_stats
import pandas as pd
import geopandas as gpd
#import fiona
import rasterio



def extractFullSpectrum():
    rasterPath = r'D:\Documents\Projects\comps\data\process\EMIT_L2A_RFL_001_20240423T183345_2411412_005_reflectance'
    rasterStats = 'median'
    vectorPath = r'D:\Documents\Projects\comps\data\Shapefiles\CAFOs_EMIT_WGS84.shp'
    vectorLayer = "CAFOs_EMIT_CorrectedV3"
    dropColumns = ['sum_emission_auto', 'sum_emission_uncertainty_auto', 'Point_Count']
    
    with rasterio.open(rasterPath) as inRaster:
        numBands = inRaster.count
    
    nc = rasterio.open(rasterPath)
    
    cafos = gpd.read_file(vectorPath)
    
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
    
    #Drop extra columns
    #df_combined = df_combined.drop(columns=dropColumns)
    
    df_combined.to_csv('bandMedians.csv', mode='w')
    
    return df_combined

df = extractFullSpectrum()
#
    
    
'''
    
    
    list_names = ['Lot', 'NH3']
    
    # Combine lists into a dictionary of DataFrames
    dfs = {}
    for list_name, data in zip(list_names, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    nh3df = pd.concat(dfs.values(), axis=1)
    
    # Drop rows where the value in the first column is null
    nh3df = nh3df.dropna(subset=[nh3df.columns[0]])
    
    nh3df.to_csv('nh3.csv')
    
    
    
    
    
    import pandas as pd
    
    # List names for column naming
    list_names = ['CAFO', 'NH3_1', 'NH3_2', 'NH3_3', 'WVBuiltUp', 'WVImpVeg', 'WVNonHom', 'WVSoil', 'WVWater']
    
    # Combine lists into a dictionary of DataFrames
    dfs = {}
    for list_name, data in zip(list_names, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    result_df = pd.concat(dfs.values(), axis=1)
    
    result_df.to_csv('rasterInfo.csv')
    
    return result_df

extractByPolygons()



df = extractFullSpectrum()
print(df)

'''