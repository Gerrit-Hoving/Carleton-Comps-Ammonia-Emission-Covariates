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
        zs = zonal_stats(cafos, img, stats=rasterStat)
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
    
    
extract()