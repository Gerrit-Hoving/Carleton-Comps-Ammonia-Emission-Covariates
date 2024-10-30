# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:22:51 2024

@author: sarp
"""

import pandas as pd

def pullData(target = 'index'):
    nh3Path = r"C:\Users\sarp\Documents\sarp_research\code\nh3.csv"
    ch4Path = r"C:\Users\sarp\Documents\sarp_research\code\ch4.csv"
    indexPath = r"C:\Users\sarp\Documents\sarp_research\code\indexInfo_CAFOCorrV3.csv"
    bandsPath = r"C:\Users\sarp\Documents\sarp_research\code\bandMedians_CAFOCorrV3.csv"
    cafosPath = r"C:\Users\sarp\Documents\sarp_research\code\CAFOsAttributesV1.csv"
    
    dropColumns = []
    
    nh3_df = pd.read_csv(nh3Path)
    nh3_df = nh3_df.drop(columns=['Unnamed: 0'])
    nh3_df = nh3_df.rename(columns={'Lot_CAFOID': 'CAFO_ID'})
    nh3_df = nh3_df.fillna(0)

    ch4_df = pd.read_csv(ch4Path)
    ch4_df = ch4_df.drop(columns=['Unnamed: 0'])
    ch4_df = ch4_df.rename(columns={'Lot_CAFOID': 'CAFO_ID'})
    ch4_df = ch4_df.fillna(0)

    cafos_df = pd.read_csv(cafosPath)
    cafos_df = cafos_df.rename(columns={'ID': 'CAFO_ID'})

    if target == 'index':
        raster_df = pd.read_csv(indexPath)
        raster_df = raster_df.drop(columns=['Unnamed: 0'])
        raster_df = raster_df.rename(columns={'ID': 'CAFO_ID'})
    elif target == 'bands':
        raster_df = pd.read_csv(bandsPath)
        raster_df = raster_df.drop(columns=['Unnamed: 0'])
        raster_df = raster_df.rename(columns={'ID': 'CAFO_ID'})
    else:
        print('Data pull error')
        return None

    df = pd.merge(raster_df, cafos_df, on='CAFO_ID', how='left')
    df = pd.merge(df, nh3_df, on='CAFO_ID', how='left')
    df = pd.merge(df, ch4_df, on='CAFO_ID', how='left')
    
    df = df.fillna(0)
    df = df.drop(columns=dropColumns)
    
    # Add categorical variables 
    df['HyTES_NH3_Detect'] = df['NH3_mean'] != 0
    df['HyTES_CH4_Detect'] = df['CH4_mean'] != 0
    df['CarbonMapper_CH4_Detect'] = df['mean_emission_uncertainty_auto'] != 0
    
    #print(df.columns)
    
    return df

#pullData()