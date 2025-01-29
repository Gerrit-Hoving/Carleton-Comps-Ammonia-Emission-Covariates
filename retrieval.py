# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:45:57 2025

@author: Gerrit Hoving
"""

import hapi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from emit_tools import emit_xarray
import rasterio
from scipy.linalg import inv



def calcSpectrum(bands, do_fetch=False, use_bands=True):
    bands['wavenumber'] = (1 / (bands['wavelengths'] * 1e-7)).astype('float64')

    ### Calculate coefficents from EMIT band positions

    hapi.db_begin('../data/HITRAN')

    if(do_fetch):
        hapi.fetch('NH3',6,1,4000,27000)

    if(use_bands):
        #nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_HT(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        #nu,coef = hapi.absorptionCoefficient_Voigt(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
        nu,coef = hapi.absorptionCoefficient_Doppler(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
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

    plt.plot(nu_nm,coef)
    plt.show()
    
    # Return wavenumber, wavelength, coefficents
    return nu, nu_nm, coef

def matchedFilter(rasterFile, absorbtions):
    with rasterio.open(rasterFile) as src:
        raster_data = src.read()  # Returns an array (bands, rows, cols)
        transform = src.transform
        crs = src.crs
        meta = src.meta

    # Apply the matched filter to get the concentration map
    concentration_map = np.zeros((raster_data.shape[1], raster_data.shape[2]))
        

    #### Matched filter implementation from isofit tutorials
    rows, cols, bands = 2018,2239,285

    #mm = np.memmap(raster_file, dtype=np.float32, mode='r', shape=(rows, cols, bands))
    mm = raster_data
    X = np.asarray(mm).copy()

    X = X.reshape(rows*cols, bands)

    # A subset of pixels is sufficient, say one out of every 100
    subset = np.arange(0,X.shape[0],100)
    Xsub = X[subset,:]
    Xsub = Xsub[~np.isnan(Xsub).any(axis=1)]
    mu = Xsub.mean(axis=0)

    # Calculate the covariance
    Cov = np.cov(Xsub, rowvar=False);

    Cinv = inv(Cov + np.eye(len(nu_nm))*1e-8)

    # Loop through each pixel 
    for row in range(raster_data.shape[1]):
        for col in range(raster_data.shape[2]):
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




    #### Matched filter implementation from isofit tutorials
    rows, cols, bands = 2018,2239,285

    #mm = np.memmap(raster_file, dtype=np.float32, mode='r', shape=(rows, cols, bands))
    mm = raster_data
    X = np.asarray(mm).copy()

    X = X.reshape(rows*cols, bands)

    # A subset of pixels is sufficient, say one out of every 100
    subset = np.arange(0,X.shape[0],100)
    Xsub = X[subset,:]
    Xsub = Xsub[~np.isnan(Xsub).any(axis=1)]
    mu = Xsub.mean(axis=0)

    # Calculate the covariance
    Cov = np.cov(Xsub, rowvar=False);

    Cinv = inv(Cov + np.eye(len(nu_nm))*1e-8)
    mf = ((X-mu).dot(Cinv.dot(coef)))/(coef.dot(Cinv.dot(coef)))

    ## TODO: Fix reshape to get comprehensible output
    out = mf.reshape(cols, rows, 1)
    plt.imshow(out * 10000.0)
    plt.colorbar()
    plt.clim([0,200])
    plt.title('NH3 plume (ppm m)')
    plt.imsave('MF2.png',mf*10000.0)











### Load EMIT raster
raster_path = r'D:\Documents\Projects\comps\data\EMIT\raw\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006.nc'

rad = emit_xarray(raster_path, ortho=True)

#old model
#points = pd.DataFrame([{"ID": 0, "latitude": 36.2286746, "longitude": -119.1756712}, {"ID": 1, "latitude": 36.2279773, "longitude": -119.1756845}])
#new model
#points = pd.DataFrame([{"ID": 0, "latitude": 36.224390, "longitude": -119.175144}, {"ID": 1, "latitude": 36.223793, "longitude": -119.176743}])
#new model max concentration difference
points = pd.DataFrame([{"ID": 0, "latitude": 36.229778, "longitude": -119.161069}, {"ID": 1, "latitude": 36.234525, "longitude": -119.166433}])

points = points.set_index(['ID'])

point_ds = rad.sel(latitude=points.to_xarray().latitude, longitude=points.to_xarray().longitude, method='nearest')

df = point_ds.to_dataframe().reset_index()

emit_bands = df[df['ID'] == 0].copy()


nu, nu_nm, coef = calcSpectrum(emit_bands)



raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006_radiance'  
matchedFilter()



