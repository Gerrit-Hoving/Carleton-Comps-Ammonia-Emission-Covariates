# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:45:57 2025

@author: Gerrit Hoving
"""

import hapi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from emit_tools import emit_xarray
import rasterio
from scipy.linalg import inv
from sklearn.preprocessing import normalize

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 20  




def calcSpectrum(bands, do_fetch=False, use_bands=True):
    bands['wavenumber'] = (1 / (bands['wavelengths'] * 1e-7)).astype('float64')

    ### Calculate coefficents from EMIT band positions

    hapi.db_begin('../data/HITRAN')

    if(do_fetch):
        hapi.fetch('NH3',6,1,4000,27000)

    if(use_bands):
        nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_HT(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_Voigt(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_Doppler(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_SDVoigt(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})

    else:
        nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3',  Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        
        wavelength_nm = 1e7 / nu  # Convert to wavelength in nm

        # Define the wavelength range (380 nm to 2500 nm)
        wavelength_min = 380  # nm
        wavelength_max = 2500  # nm

        # Interpolate the coefficients to get 285 values within the desired wavelength range
        wavelength_range = np.linspace(wavelength_min, wavelength_max, 285)

        # Interpolate coef as a function of wavelength
        interp_func = interp1d(wavelength_nm, coef, kind='linear', fill_value=0, bounds_error=False)

        # Get the interpolated absorption coefficients for the new wavelength range
        interpolated_coef = interp_func(wavelength_range)

        nu = wavelength_range
        coef = interpolated_coef

    plt.plot(nu,coef)
    plt.show()

    nu_nm = 1e7 / nu

    plt.figure(figsize=(6.5, 4))
    plt.plot(nu_nm,coef)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Absorbtion (cm$^{-1}$)')
    plt.show()
    
    # Return wavenumber, wavelength, coefficents
    return nu, nu_nm, coef

def matchedFilter(rasterFile, nu, coef, save=False):
    with rasterio.open(rasterFile) as src:
        raster_data = src.read()  # Returns an array (bands, rows, cols)
        transform = src.transform
        crs = src.crs
        meta = src.meta

    # Apply the matched filter to get the concentration map
    concentration_map = np.zeros((raster_data.shape[1], raster_data.shape[2]))
        
    #### Matched filter implementation from isofit tutorials
    rows, cols, bands = raster_data.shape[1],raster_data.shape[2],285

    #mm = np.memmap(raster_file, dtype=np.float32, mode='r', shape=(rows, cols, bands))
    mm = raster_data
    X = np.asarray(mm).copy()

    #X = X.reshape(rows*cols, bands)

    # A subset of pixels for calculating Cov and mu
    #subset = np.arange(0,X.shape[0],100)
    #Xsub = X[subset,:]
    #Xsub = Xsub[~np.isnan(Xsub).any(axis=1)]
    #mu = Xsub.mean(axis=0)

    # Calculate the covariance
    #Cov = np.cov(Xsub, rowvar=False);
    #Cinv = inv(Cov + np.eye(len(nu_nm))*1e-8)

    # Loop through each pixel 
    for col in range(raster_data.shape[2]):
        # Extract all data for the current column (over all rows, for all bands)
        Xsub = X[:, :, col]  # Shape will be (bands, rows)
    
        # Remove rows that have NaN values
        Xsub = Xsub[~np.isnan(Xsub).any(axis=1)]
    
        # Calculate the mean (mu) of the subset (over rows)
        mu = Xsub.mean(axis=0)
    
        # Calculate the covariance matrix and invrse for the column
        Cov = np.cov(Xsub, rowvar=False)  # Covariance over rows, so it's (bands, bands)
        Cinv = inv(Cov + np.eye(len(mu)) * 1e-8)  # Adding a small regularization term to avoid singularity
 
        for row in range(raster_data.shape[1]):
            # Extract the spectrum for this pixel
            spectrum = raster_data[:, row, col]
            
            # Normalize the spectrum (optional: you may want to perform other pre-processing)
            spectrum /= np.max(spectrum)

            # Perform the matched filter - dot product between the spectrum and the absorption coefficients
            #matched_filter = np.dot(spectrum, coef)  # We assume both spectrum and coef have the same shape
            #matched_filter = ((spectrum-mu).dot(Cinv.dot(coef)))/(coef.dot(Cinv.dot(coef)))
            matched_filter = np.dot(np.dot((spectrum - mu).T, Cinv), coef) / np.dot(np.dot(coef.T, Cinv), coef)
            
            #matched_filter = matched_filter if matched_filter > (10^(-20)) else 0
            
            # Calculate concentration (you can modify this calculation based on the Beer-Lambert Law)
            # Simplified: assuming direct proportionality between matched filter and concentration
            concentration_map[row, col] = matched_filter

    plt.figure()
    plt.imshow(concentration_map, cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.title('Concentration Map')
    
    #plt.savefig('interpolated_absorption_coefficients.tif', format='tiff', dpi=800)
    plt.show()
    
    concentration_map = abs(concentration_map) / 1e18
    low = 0
    high = 3
    new_array = np.where(concentration_map < low, low, concentration_map)
    new_array = np.where(new_array > high, low, new_array)
    plt.figure()
    plt.imshow(new_array, cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.title('Concentration Map')
    
    #plt.savefig('interpolated_absorption_coefficients_threshold.tif', format='tiff', dpi=800)
    plt.show()

    if(save):
        ### Save to geotiff
        
        # Update the metadata for the output file
        meta.update({
            'count': 1,  # Single band (grayscale image)
            'dtype': 'float32',  # Data type for concentration map
        })
        with rasterio.open('concentration_map_nh3_006_01.tif', 'w', **meta) as dst:
            dst.write(concentration_map.astype('float32'), 1)  # Write the data to the first band  
        
    return concentration_map

def inOutPlumeGraph(point_df, ac_df, raster_path):
    # Plot using seaborn
    plt.figure(figsize=(6.5, 4))

    # Create the line plot with seaborn
    sns.lineplot(
        data=point_df,
        x='wavelengths',
        y='radiance',
        hue='ID',  # Group data by 'ID'
        palette='Dark2',  # Use Dark2 color palette
        marker=''  # Optional: add markers to lines
    )


    # Add titles and labels
    plt.title('Radiance Spectra, ID 0 is in-plume')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Radiance (W/m^2/sr/nm)')

    # Show the plot
    plt.show()
    
    in_nh3 = point_df[point_df['ID'] == 0]['radiance'].reset_index(drop=True)
    out_nh3 = point_df[point_df['ID'] == 1]['radiance'].reset_index(drop=True)
    far_nh3 = point_df[point_df['ID'] == 2]['radiance'].reset_index(drop=True)

    # Combine the two filtered DataFrames into one with two columns
    df = pd.DataFrame({
        'radiance_in': in_nh3,
        'radiance_out': out_nh3,
        'radiance_far': far_nh3
    })

    # Apply vector normalization using sklearn's normalize function
    df_normalized = pd.DataFrame(normalize(df, norm='l2', axis=1))
    
    df_normalized.columns = ['radiance_in', 'radiance_out', 'radiance_far']
    
    df_normalized['wavelengths'] = point_df[point_df['ID'] == 0]['wavelengths'].reset_index(drop=True)
    df['wavelengths'] = point_df[point_df['ID'] == 0]['wavelengths'].reset_index(drop=True)


    in_plume = point_df.loc[point_df['ID'] == 0].copy().reset_index(drop=True)
    out_plume = point_df.loc[point_df['ID'] == 1].copy().reset_index(drop=True)
    far = point_df.loc[point_df['ID'] == 2].copy().reset_index(drop=True)
    far_2 = point_df.loc[point_df['ID'] == 3].copy().reset_index(drop=True)
    out_plume['band_ratio'] = (in_plume['radiance']/out_plume['radiance'])
    far['band_ratio'] = (far_2['radiance']/far['radiance'])

    out_plume = out_plume.drop(out_plume[out_plume.band_ratio > 200].index)


    ### Radiance + absorbtion spectra @ 2200 nm
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    
    sns.lineplot(
         data=df,
         x='wavelengths',
         y='radiance_in',
         color='#ff7f0e',  
         label='Ammonia plume',    
         marker='o',
         ax = ax1
     )
    
    #df['radiance_out_s'] = df['radiance_out'] + 0.32
    df['radiance_out_s'] = df['radiance_out'] * 6.1
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_out_s',
        color='#9467bd',  
        label='Outside plume',  
        marker='o',
        ax = ax1
    )
    
    #df['radiance_far_s'] = df['radiance_far'] + 0.3
    df['radiance_far_s'] = df['radiance_far'] * 3.8
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_far_s',
        color='#1f77b4',  
        label='Clean air',  
        marker='o',
        ax = ax1
    )
     
    sns.lineplot(
        data=ac_df,
        x='wavelengths',
        y='coefficients',
        marker='o',  
        color='#2ca02c',  
        label='Simulated absorbtion',  
        ax = ax2
     )
    

    plt.xlim(2280, 2360)
    ax1.set_ylim(0.18, 0.48)
    #ax2.set_ylim(0, 2e-21)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Radiance (uW/nm/sr/cm2)')
    ax2.set_ylabel('Simulated NH3 Absorbtion (cm$^{-1}$)')
    
    ax1.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.9415, 0.925))
    ax2.legend(loc='upper right', frameon=False)
    
    plt.savefig("../figures/Fig16.png",bbox_inches='tight',dpi=300) 
    plt.show()

    ### Radiance + absorbtion spectra @ 2200 nm, unscaled radiance
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    
    sns.lineplot(
         data=df,
         x='wavelengths',
         y='radiance_in',
         color='#ff7f0e',  
         label='Ammonia plume',    
         marker='o',
         ax = ax1
     )
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_out',
        color='#9467bd',  
        label='Outside plume',  
        marker='o',
        ax = ax1
    )
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_far',
        color='#1f77b4',  
        label='Clean air',  
        marker='o',
        ax = ax1
    )
     
    sns.lineplot(
        data=ac_df,
        x='wavelengths',
        y='coefficients',
        marker='o',  
        color='#2ca02c',  
        label='Simulated absorbtion',  
        ax = ax2
     )
    

    plt.xlim(2280, 2360)
    ax1.set_ylim(0, 0.5)
    #ax2.set_ylim(0, 2e-21)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Radiance (uW/nm/sr/cm2)')
    ax2.set_ylabel('Simulated NH3 Absorbtion (cm$^{-1}$)')
    
    ax1.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.9415, 0.925))
    ax2.legend(loc='upper right', frameon=False)
    
    plt.savefig("../figures/FigA4.png",bbox_inches='tight',dpi=300)
    plt.show()
    

    ### Radiance + absorbtion spectra @ 1640 nm
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax2 = ax1.twinx()
    
    sns.lineplot(
         data=df,
         x='wavelengths',
         y='radiance_in',
         color='#ff7f0e',  
         label='Ammonia plume',    
         marker='o',
         ax = ax1
     )
    
    df['radiance_out_s'] = df['radiance_out'] * 3.16
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_out_s',
        color='#9467bd',  
        label='Outside plume',  
        marker='o',
        ax = ax1
    )
    
    df['radiance_far_s'] = df['radiance_far'] * 3.01
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_far_s',
        color='#1f77b4',  
        label='Clean air',  
        marker='o',
        ax = ax1
    )
     
    sns.lineplot(
        data=ac_df,
        x='wavelengths',
        y='coefficients',
        marker='o',  
        color='#2ca02c',  
          
        ax = ax2
     )
     
    plt.xlim(1590, 1690)
    ax1.set_ylim(2.6, 3.4)
    #ax2.set_ylim(0, 2e-21)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Radiance (uW/nm/sr/cm$^{2}$)')
    ax2.set_ylabel('Simulated NH3 Absorbtion (cm$^{-1}$)')
    
    ax1.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.04, 1.05))
    #ax2.legend(loc='upper right', frameon=False)

    plt.savefig("../figures/Fig15.png",bbox_inches='tight',dpi=300)    
    plt.show()
    
    
    ### Radiance + absorbtion spectra @ 1640 nm, unscaled
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax2 = ax1.twinx()
    
    sns.lineplot(
         data=df,
         x='wavelengths',
         y='radiance_in',
         color='#ff7f0e',  
         label='Ammonia plume',    
         marker='o',
         ax = ax1
     )
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_out',
        color='#9467bd',  
        label='Outside plume',  
        marker='o',
        ax = ax1
    )
    
    sns.lineplot(
        data=df,
        x='wavelengths',
        y='radiance_far',
        color='#1f77b4',  
        label='Clean air',  
        marker='o',
        ax = ax1
    )
     
    sns.lineplot(
        data=ac_df,
        x='wavelengths',
        y='coefficients',
        marker='o',  
        color='#2ca02c',  
        label='Simulated absorbtion',  
        ax = ax2
     )
     
    plt.xlim(1590, 1690)
    ax1.set_ylim(0.5, 4.5)
    #ax2.set_ylim(0, 2e-21)
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Radiance (uW/nm/sr/cm2)')
    ax2.set_ylabel('Simulated NH3 Absorbtion (cm$^{-1}$)')
    
    ax1.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.9415, 0.925))
    ax2.legend(loc='upper right', frameon=False)

    plt.savefig("../figures/FigA3.png",bbox_inches='tight',dpi=300)
    plt.show()


def generateRetrievalGraphs():
    ### Load EMIT raster
    raster_path = r'D:\Documents\Projects\comps\data\EMIT\raw\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006.nc'
    #raster_path = r'D:\Documents\Projects\comps\data\EMIT\raw\radiance\EMIT_L1B_RAD_001_20230822T192543_2323413_004.nc'
    #raster_path = r'D:\Documents\Projects\comps\data\EMIT\raw\radiance\EMIT_L1B_RAD_001_20230927T214543_2327014_003.nc'
    
    
    rad = emit_xarray(raster_path, ortho=True)
    
    
    # 0 is in plume
    #old model
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.2286746, "longitude": -119.1756712}, {"ID": 1, "latitude": 36.2279773, "longitude": -119.1756845}])
    #new model
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.224390, "longitude": -119.175144}, {"ID": 1, "latitude": 36.223793, "longitude": -119.176743}])
    #new model max concentration difference
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.229778, "longitude": -119.161069}, {"ID": 1, "latitude": 36.234525, "longitude": -119.166433}])
    #pond-feedlot diff - P1
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.062170, "longitude": -119.378442}, {"ID": 1, "latitude": 36.061509, "longitude": -119.378456}])
    #feedlot-off feedlot edge - P2
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.0529654, "longitude": -119.3969367}, {"ID": 1, "latitude": 36.0523497, "longitude": -119.3958112}])
    
    # P2 + far points pred high
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.0529654, "longitude": -119.3969367}, {"ID": 1, "latitude": 36.0523497, "longitude": -119.3958112}, {"ID": 2, "latitude": 36.255588, "longitude": -119.063412}, {"ID": 3, "latitude": 36.241934, "longitude": -119.045614}])
    
    # Points in same field with different concentrations (P3) + far
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.114725, "longitude": -119.285175}, {"ID": 1, "latitude": 36.114282, "longitude": -119.280355}, {"ID": 2, "latitude": 36.255588, "longitude": -119.063412}, {"ID": 3, "latitude": 36.241934, "longitude": -119.045614}])
    
    # Same field slightly differet (P3.5) and different adjacent far - okay results
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.115002, "longitude": -119.285173}, {"ID": 1, "latitude": 36.113682, "longitude": -119.278212}, {"ID": 2, "latitude": 36.0947921, "longitude": -118.7094658}, {"ID": 3, "latitude": 36.0946982, "longitude": -118.7086598}])
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.115002, "longitude": -119.285173}, {"ID": 1, "latitude": 36.113654, "longitude": -119.280307}, {"ID": 2, "latitude": 36.0947921, "longitude": -118.7094658}, {"ID": 3, "latitude": 36.0946982, "longitude": -118.7086598}])

    # Emitting vs non-emitting lot surface
    #points = pd.DataFrame([{"ID": 0, "latitude": 36.110436, "longitude": -119.298685}, {"ID": 1, "latitude": 36.111565, "longitude": -119.295516}, {"ID": 2, "latitude": 36.0947921, "longitude": -118.7094658}, {"ID": 3, "latitude": 36.0946982, "longitude": -118.7086598}])

    
    points = pd.DataFrame([{"ID": 0, "latitude": 36.110436, "longitude": -119.298685}, {"ID": 1, "latitude": 36.110399, "longitude": -119.291208}, {"ID": 2, "latitude": 36.0947921, "longitude": -118.7094658}, {"ID": 3, "latitude": 36.0946982, "longitude": -118.7086598}])


    points = points.set_index(['ID'])
    
    point_ds = rad.sel(latitude=points.to_xarray().latitude, longitude=points.to_xarray().longitude, method='nearest')
    
    df = point_ds.to_dataframe().reset_index()
     
    emit_bands = df[df['ID'] == 0].copy()
    
    nu, nu_nm, coef = calcSpectrum(emit_bands)
    
    ac = {'wavelengths':nu_nm, 'coefficients':coef}
    
    ac_df = pd.DataFrame(ac)
    
    raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006_radiance'  
    #raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230822T192543_2323413_004_radiance'  
    #raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230927T214543_2327014_003_radiance'  
    
    #matchedFilter(raster_file, nu, coef, save=True)
    
    inOutPlumeGraph(df, ac_df, raster_path)
    
    
    
generateRetrievalGraphs()
