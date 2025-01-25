# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:24:10 2025

@author: Gerrit Hoving
"""

from hapi import db_begin, fetch
import hapi
import isofit
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio


db_begin('../data/HITRAN')

#fetch('NH3',11,1,4000,27000)

nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3', Diluent={'air':1.0})

plt.plot(nu,coef)




### Test conversion to EMIT wavelengths
wavelength_nm = 1e7 / nu  # Convert to wavelength in nm

# Define the wavelength range (380 nm to 2500 nm)
wavelength_min = 380  # nm
wavelength_max = 2500  # nm

# Interpolate the coefficients to get 285 values within the desired wavelength range
wavelength_range = np.linspace(wavelength_min, wavelength_max, 285)

# Interpolation of absorption coefficients to match the new wavelength range
# We need to interpolate the original absorption coefficients (coef) to the new wavelength range
# First, we map the wavelength values to the original nu-wavelength space

# Interpolation function (linear interpolation in this case)
from scipy.interpolate import interp1d

# Interpolate coef as a function of wavelength
interp_func = interp1d(wavelength_nm, coef, kind='linear', bounds_error=False, fill_value=0)

# Get the interpolated absorption coefficients for the new wavelength range
interpolated_coef = interp_func(wavelength_range)

# Visualize the result
plt.figure()
plt.plot(wavelength_range, interpolated_coef)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption Coefficient')
plt.title('Interpolated Absorption Coefficients (285 values between 380nm and 2500nm)')
plt.show()

nu = wavelength_range
coef = interpolated_coef



### Test matched filter implementation

def load_raster(file_path):
    with rasterio.open(file_path) as src:
        raster_data = src.read()  # Returns an array (bands, rows, cols)
        transform = src.transform
        crs = src.crs
        return raster_data, transform, crs


# File paths (replace these with your actual file paths)
raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006_radiance'  # Hyperspectral raster with 285 bands

# Load the raster data
raster_data, transform, crs = load_raster(raster_file)

# Apply the matched filter to get the concentration map
concentration_map = np.zeros((raster_data.shape[1], raster_data.shape[2]))
    
# Loop through each pixel (each pixel has a spectrum across bands)
for row in range(raster_data.shape[1]):
    for col in range(raster_data.shape[2]):
        # Extract the spectrum for this pixel
        spectrum = raster_data[:, row, col]
        
        # Normalize the spectrum (optional: you may want to perform other pre-processing)
        spectrum /= np.max(spectrum)
        
        # Perform the matched filter - dot product between the spectrum and the absorption coefficients
        matched_filter = np.dot(spectrum, coef)  # We assume both spectrum and coef have the same shape
        
        # Calculate concentration (you can modify this calculation based on the Beer-Lambert Law)
        # Simplified: assuming direct proportionality between matched filter and concentration
        concentration_map[row, col] = matched_filter

# Visualize the concentration map (e.g., for a particular row/column)
plt.figure()
plt.imshow(concentration_map, cmap='viridis')
plt.colorbar(label='Concentration')
plt.title('Concentration Map')
plt.show()