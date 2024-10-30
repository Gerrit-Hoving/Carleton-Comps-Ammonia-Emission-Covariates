# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:33:25 2024

@author: sarp

"""


from rasterstats import zonal_stats
import fiona
import os
import pandas as pd

def extractByPolygons():
    # Extract statistics within polygons for a set of files
    
    fileDirectory = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes'
    polygonFile = r'C:\Users\sarp\Documents\sarp_research\code\CAFOs.gpkg'
    polygonLayer = "CAFOs_EMIT_CorrectedV3"
    dropColumns = ['CAFO_sum_emission_auto', 'CAFO_sum_emission_uncertainty_auto', 'CAFO_Point_Count']
    
    allFiles = os.listdir(fileDirectory)
    rasterNames = [f for f in allFiles if os.path.isfile(os.path.join(fileDirectory, f)) and '.' not in f]
    rasterPaths = [os.path.join(fileDirectory, raster) for raster in rasterNames]
    rasterStats = ['median', 'mean', 'sum']
    
    print("Calculating statistics", rasterStats, "for rasters", rasterNames)
    
    results = []
    attributes = []
    
    # Open CAFO polygon file and add all the attributes to a list
    cafos = fiona.open(polygonFile, layer=polygonLayer)
    for feature in cafos:
        attributes.append(fiona.model.to_dict(feature['properties']))
    results.append(attributes)
    
    # Work through every image in the rasters folder and calculate the zonal statistics for it
    for img in rasterPaths:
        zs = zonal_stats(cafos, img, stats=rasterStats)
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
    
    
extractByPolygons()
    









def junk():
    # Old code no longer in use
    nh3_1 = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NH3_1'
    nh3_2 = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NH3_2'
    nh3_3 = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NH3_3'
    WVBuiltUp = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\WVBuiltUp'
    WVImpVeg = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\WVImpVeg'
    WVNonHom = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\WVNonHom'
    WVSoil = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\WVSoil'
    WVWater = r'C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\WVWater'
    EOMI4 = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\EOMI4"
    EOMI3 = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\EOMI3"
    EOMI2 = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\EOMI2"
    EOMI1 = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\EOMI1"
    Clay = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\Clay"
    Cellulose = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\Cellulose"
    NDMI = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NDMI"
    NDLI = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NDLI"
    NBR2 = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NBR2"
    NBR = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\NBR"
    FoliageN = r"C:\Users\sarp\Documents\sarp_research\data\EMIT\Indexes\FoliageN"
    #indexes = [nh3_1, nh3_2, nh3_3, WVBuiltUp, WVImpVeg, WVNonHom, WVSoil, WVWater]
    #indexes = [EOMI4,EOMI3,EOMI2,EOMI1,Clay,Cellulose,NDMI,NDLI,NBR2,NBR,FoliageN]
    indexes = [nh3_1, nh3_2, nh3_3, WVBuiltUp, WVImpVeg, WVNonHom, WVSoil, WVWater,EOMI4,EOMI3,EOMI2,EOMI1,Clay,Cellulose,NDMI,NDLI,NBR2,NBR,FoliageN]
    
    results = []
    
    cafos = fiona.open(r'C:\Users\sarp\Documents\sarp_research\code\CAFOs.gpkg', layer="CAFOs_EMIT_CorrectedV2")
    attributes = []
    for feature in cafos:
        attributes.append(fiona.model.to_dict(feature['properties']))
    results.append(attributes)
    
    for img in indexes:
        zs = zonal_stats(cafos, img, stats=['median', 'mean', 'sum'])
        results.append(zs)
    
    
    import pandas as pd
    
    # List names for column naming
    list_names = ['CAFO', 'NH3_1', 'NH3_2', 'NH3_3', 'WVBuiltUp', 'WVImpVeg', 'WVNonHom', 'WVSoil', 'WVWater','EOMI4','EOMI3','EOMI2','EOMI1','Clay','Cellulose','NDMI','NDLI','NBR2','NBR','FoliageN']
    
    # Combine lists into a dictionary of DataFrames
    dfs = {}
    for list_name, data in zip(list_names, results):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        print(df)
        # Rename columns to include the list name
        df.columns = [f'{list_name}_{col}' for col in df.columns]
        # Add DataFrame to dictionary
        dfs[list_name] = df
    
    # Concatenate all DataFrames horizontally (axis=1)
    result_df = pd.concat(dfs.values(), axis=1)
    
    result_df.to_csv('indexInfo_CAFOCorr.csv')
    
    return result_df
