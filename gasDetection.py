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
import hvplot.pandas
from emit_tools import emit_xarray
import pandas as pd
import holoviews as hv



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

#emit_bands['wavenumber'] = 1e7 / emit_bands[1]
emit_bands['wavenumber'] = (1 / (emit_bands['wavelengths'] * 1e-7)).astype('float64')

### Calculate coefficents from EMIT band positions

db_begin('../data/HITRAN')

#fetch('NH3',6,1,4000,27000)

nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3', WavenumberGrid=np.array(emit_bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})

plt.plot(nu,coef)
plt.show()

nu_nm = 1e7 / nu

plt.plot(nu_nm,coef)
plt.show()



### Join coefficents and radiance dfs
ac = {'wavelengths':nu_nm, 'coefficients':coef}
ac_df = pd.DataFrame(ac)

# Round both wavelength columns to the nearest ten
df['wavelength'] = df['wavelengths'].astype('float64').round(1)
ac_df['wavelength'] = ac_df['wavelengths'].astype('float64').round(1)

# Merge the DataFrames on the rounded columns
full_df = pd.merge(df, ac_df, left_on='wavelength', right_on='wavelength', how='left')

# Drop the temporary rounded columns if not needed
full_df.drop(['wavelengths_x', 'wavelengths_y'], axis=1, inplace=True)




### Plot in/out of plume spectra

# Plot using seaborn
plt.figure(figsize=(8, 6))  # Equivalent to frame_height=400, frame_width=600

# Create the line plot with seaborn
sns.lineplot(
    data=full_df,
    x='wavelength',
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
out_plume = out_plume.drop(out_plume[out_plume.radiance < 0.1].index)

out_plume['band_ratio'] = (in_plume.loc[0, 'radiance']/out_plume['radiance'])

#out_plume = out_plume.drop(out_plume[out_plume.band_ratio > 10].index)




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

full_df['scaled_coef'] = full_df['coefficients']*1e21

sns.lineplot(
    data=full_df,
    x='wavelength',
    y='scaled_coef',
    marker='o'
)


# Add titles and labels
plt.title('In Plume/Out of Plume Ratio')
plt.xlabel('Wavelength (nm)')
plt.xlim(2000, 2500)
plt.ylabel('In Plume/Out of Plume Ratio')
#plt.ylim(0, 20)

plt.show()

sns.lineplot(
    data=full_df,
    x='wavelength',
    y='coefficients',
    marker='o'
)


# Add titles and labels
plt.title('NH3 Spectrum')
plt.xlabel('Wavelength (nm)')
plt.xlim(1400, 2500)
plt.ylabel('Absorbtion Coefficent')
#plt.ylim(0, 100)

# Show the plot
plt.show()
















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
interp_func = interp1d(wavelength_nm, coef, kind='linear', fill_value=0, bounds_error=False)

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


# File paths 
raster_file = r'D:\Documents\Projects\comps\data\EMIT\processed\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006_radiance'  

# Load the raster data
raster_data, transform, crs, meta = load_raster(raster_file)

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
        matched_filter = np.dot(spectrum, coef)  # We assume both spectrum and coef have the same shape
        #matched_filter = ((spectrum-mu).dot(Cinv.dot(coef)))/(coef.dot(Cinv.dot(coef)))
        
        
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



#### Visualize the concentration map (e.g., for a particular row/column)
plt.figure()
plt.imshow(concentration_map, cmap='viridis')
plt.colorbar(label='Concentration')
plt.title('Concentration Map')

#plt.savefig('interpolated_absorption_coefficients.tif', format='tiff', dpi=800)
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


#plt.savefig('interpolated_absorption_coefficients_threshold.tif', format='tiff', dpi=800)
plt.show()


### Save to geotiff

# Update the metadata for the output file
meta.update({
    'count': 1,  # Single band (grayscale image)
    'dtype': 'float32',  # Data type for concentration map
})
with rasterio.open('concentration_map_nh3_2.tif', 'w', **meta) as dst:
    dst.write(concentration_map.astype('float32'), 1)  # Write the data to the first band



### Visualize in/out of plume spectra
raster_path = r'D:\Documents\Projects\comps\data\EMIT\raw\radiance\EMIT_L1B_RAD_001_20230818T210107_2323014_006.nc'

rad = emit_xarray(raster_path, ortho=True)

#old model
#points = pd.DataFrame([{"ID": 0, "latitude": 36.2286746, "longitude": -119.1756712}, {"ID": 1, "latitude": 36.2279773, "longitude": -119.1756845}])
#new model
points = pd.DataFrame([{"ID": 0, "latitude": 36.224390, "longitude": -119.175144}, {"ID": 1, "latitude": 36.223793, "longitude": -119.176743}])


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

out_plume = out_plume.drop(out_plume[out_plume.band_ratio > 10].index)


ac = {'wavelengths':nu, 'coefficients':coef}

ac_df = pd.DataFrame(ac)


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
plt.title('In Plume/Out of Plume Ratio')
plt.xlabel('Wavelength (nm)')
plt.xlim(1400, 2500)
plt.ylabel('In Plume/Out of Plume Ratio')
plt.ylim(0, 100)


plt.show()

sns.lineplot(
    data=ac_df,
    x='wavelengths',
    y='coefficients',
    hue='ID',  # Group data by 'ID'
    palette='Dark2',  # Use Dark2 color palette
    marker='o'  # Optional: add markers to lines
)


# Add titles and labels
plt.title('NH3 Spectrum')
plt.xlabel('Wavelength (nm)')
plt.xlim(1400, 2500)
plt.ylabel('Absorbtion Coefficent')
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







## Plotting code from lpdaac
ac_plot = ac_df.hvplot(x='wavelengths',y='coefficients', frame_height=400, frame_width=400, line_color='black', line_width=2, xlim=(2150,2450), ylim=(-1.5,0), xlabel='Wavelength (nm)', title='Ammonia Absorption Coefficient', ylabel='')

from bokeh.models import GlyphRenderer, LinearAxis, LinearScale, Range1d

def overlay_hook(plot, element):
    # Adds right y-axis
    p = plot.handles["plot"]
    p.extra_y_scales = {"right": LinearScale()}
    p.extra_y_ranges = {"right": Range1d(-1.5,0)}
    p.add_layout(LinearAxis(y_range_name="right"), "right")

   # find the last line and set it to right
    lines = [p for p in p.renderers if isinstance(p, GlyphRenderer)]
    lines[-1].y_range_name = "right"



out_plume['out-out'] = (out_plume['radiance'] / out_plume.loc[2,'radiance'])

in_out_plot = out_plume.hvplot(x='wavelengths',y='band_ratio', by=['ID'], color=hv.Cycle('Dark2'), frame_height=400, frame_width=400, xlim=(2150,2450), ylim=(0.85,1.05), ylabel='In Plume/Out of Plume Ratio', xlabel='Wavelength (nm)', title='In Plume/Out of Plume Ratio')

(in_out_plot.opts(ylim=(0.85,0.95)) * ac_plot.opts(color="k")).opts(hooks=[overlay_hook]).opts(title='In Plume/Out of Plume and Absorption Coefficient') 


out_out_plot = out_plume.hvplot(x='wavelengths',y='out-out', by=['ID'], color=hv.Cycle('Dark2'), frame_height=400, frame_width=400, xlim=(2150,2450), ylim=(0.85,1.05), ylabel='Out/Out 2 Ratio', xlabel='Wavelength (nm)', title='Out of Plume/Out of Plume 2 Ratio')

from bokeh.models import GlyphRenderer, LinearAxis, LinearScale, Range1d

def overlay_hook(plot, element):
    # Adds right y-axis
    p = plot.handles["plot"]
    p.extra_y_scales = {"right": LinearScale()}
    p.extra_y_ranges = {"right": Range1d(-1.5,0)}
    p.add_layout(LinearAxis(y_range_name="right"), "right")

   # find the last line and set it to right
    lines = [p for p in p.renderers if isinstance(p, GlyphRenderer)]
    lines[-1].y_range_name = "right"

# Create Figure
(out_out_plot * ac_plot.opts(color="k")).opts(hooks=[overlay_hook]).opts(title='Out of Plume/Out of Plume and Absorption Coefficient') 




# Create an RGB from Radiance
rgb = rad.sel(wavelengths=[650,560,470], method='nearest')

rgb.radiance.data[rgb.radiance.data == -9999] = 0

vmin, vmax = np.percentile(rgb.radiance.data, (2, 98))
rgb.radiance.data = exposure.rescale_intensity(rgb.radiance.data, in_range=(vmin, vmax), out_range=(0,1))

rgb_image = rgb.radiance.data

# Plot the RGB image using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(rgb_image)
plt.title('RGB Radiance')

sns.scatterplot(x=df['longitude'], y=df['latitude'], s=100, edgecolor='w', legend=None)

plt.show()

