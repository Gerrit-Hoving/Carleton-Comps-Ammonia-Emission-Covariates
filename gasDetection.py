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
from scipy.linalg import norm, inv

from skimage import exposure
import seaborn as sns
import hvplot
import hvplot.xarray
from emit_tools import emit_xarray
import pandas as pd


db_begin('../data/HITRAN')

fetch('CH4',6,1,4000,27000)

nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='CH4', Diluent={'air':1.0}, Environment = {'p':0.98,'T':305},)

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
        meta = src.meta
        return raster_data, transform, crs, meta


# File paths (replace these with your actual file paths)
raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006_radiance'  # Hyperspectral raster with 285 bands

# Load the raster data
raster_data, transform, crs, meta = load_raster(raster_file)

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
        #matched_filter = np.dot(spectrum, coef)  # We assume both spectrum and coef have the same shape
        matched_filter = ((spectrum-mu).dot(Cinv.dot(coef)))/(coef.dot(Cinv.dot(coef)))
        
        
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

Cinv = inv(Cov + np.eye(len(nu))*1e-8)
mf = ((X-mu).dot(Cinv.dot(coef)))/(coef.dot(Cinv.dot(coef)))

## TODO: Fix reshape to get comprehensible output
out = mf.reshape(cols, rows, 1)
plt.imshow(out * 10000.0)
plt.colorbar()
plt.clim([0,200])
plt.title('NH3 plume (ppm m)')
plt.imsave('MF2.png',mf*10000.0)



#### Visualize the concentration map (e.g., for a particular row/column)
plt.figure()
plt.imshow(concentration_map, cmap='viridis')
plt.colorbar(label='Concentration')
plt.title('Concentration Map')

plt.savefig('interpolated_absorption_coefficients.tif', format='tiff', dpi=800)
plt.show()

#concentration_map = abs(concentration_map) / 1e17
low = 0
high = 2
new_array = np.where(concentration_map < low, low, concentration_map)
new_array = np.where(new_array > high, low, new_array)
plt.figure()
plt.imshow(new_array, cmap='viridis')
plt.colorbar(label='Concentration')
plt.title('Concentration Map')

plt.savefig('interpolated_absorption_coefficients_threshold.tif', format='tiff', dpi=800)
plt.show()


### Save to geotiff

# Update the metadata for the output file
meta.update({
    'count': 1,  # Single band (grayscale image)
    'dtype': 'float32',  # Data type for concentration map
})
with rasterio.open('concentration_map_ch4.tif', 'w', **meta) as dst:
    dst.write(concentration_map.astype('float32'), 1)  # Write the data to the first band




### Visualize in/out of plume spectra



rad = emit_xarray(raster_file, ortho=True)

#In manure pond
#points = pd.DataFrame([{"ID": 0, "latitude": 35.167664437574814, "longitude": -119.10087986567486}, {"ID": 1, "latitude": 35.1676775784815, "longitude": -119.09837215132448}])
# In hills
points = pd.DataFrame([{"ID": 0, "latitude": 35.31848809204527, "longitude": -119.42502781410386}, {"ID": 1, "latitude": 35.31897748845836, "longitude": -119.42502800538341}])
#Feedlot surface, accounting for wind
#points = pd.DataFrame([{"ID": 0, "latitude": 35.16709534468988, "longitude": -119.09588385907719}, {"ID": 1, "latitude": 35.17023191518769, "longitude": -119.09584450552866}])
#Feedlot surface 2, accounting for wind
#points = pd.DataFrame([{"ID": 0, "latitude": 35.17118076215722, "longitude": -119.10268430272738}, {"ID": 1, "latitude": 35.17202346329284, "longitude": -119.10853490784534}])
#Western Sky Dairy
#points = pd.DataFrame([{"ID": 0, "latitude": 35.18950382181828, "longitude": -119.11785339563848}, {"ID": 1, "latitude": 35.19335877937062, "longitude": -119.11997608881786}])
#T&W Farms
#points = pd.DataFrame([{"ID": 0, "latitude": 35.191961378381194, "longitude": -119.10283239289654}, {"ID": 1, "latitude": 35.19367198662642, "longitude": -119.10701881555586}])


points = points.set_index(['ID'])

point_ds = rad.sel(latitude=points.to_xarray().latitude, longitude=points.to_xarray().longitude, method='nearest')


# Create an RGB from Radiance
rgb = rad.sel(wavelengths=[650,560,470], method='nearest')
rgb.radiance.data[rgb.radiance.data == -9999] = 0
rgb.radiance.data = exposure.rescale_intensity(rgb.radiance.data, in_range='image', out_range=(0,1))




df = point_ds.to_dataframe().reset_index()

# Plot using seaborn
plt.figure(figsize=(8, 6))  # Equivalent to frame_height=400, frame_width=600

# Create the line plot with seaborn
sns.lineplot(
    data=df,
    x='wavelengths',
    y='radiance',
    hue='ID',  # Group data by 'ID'
    palette='Dark2',  # Use Dark2 color palette
    marker='o'  # Optional: add markers to lines
)

# Add titles and labels
plt.title('Radiance Spectra, ID 0 is in-plume')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W/m^2/sr/nm)')

# Show the plot
plt.show()


in_plume = df.loc[df['ID'] == 0].copy()
out_plume = df.loc[df['ID'] == 1].copy()
out_plume['band_ratio'] = (in_plume.loc[0, 'radiance']/out_plume['radiance'])


# Plot using seaborn
plt.figure(figsize=(8, 6))  # Equivalent to frame_height=400, frame_width=600

# Create the line plot with seaborn
sns.lineplot(
    data=out_plume,
    x='wavelengths',
    y='band_ratio',
    hue='ID',  # Group data by 'ID'
    palette='Dark2',  # Use Dark2 color palette
    marker='o'  # Optional: add markers to lines
)

# Add titles and labels
plt.title('In Plume/Out of Plume Ratio - Hills')
plt.xlabel('Wavelength (nm)')
plt.xlim(1700, 2400)
plt.ylabel('In Plume/Out of Plume Ratio')
plt.ylim(0, 100)

"""
for i in range(498, 510, 1):
    plt.text(out_plume['wavelengths'][i], 
             out_plume['band_ratio'][i], 
             str(i) + ':' + str(out_plume['wavelengths'][i]), 
             fontsize=6, ha='right', va='bottom')

"""

# Show the plot
plt.show()










