# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:32:21 2024

@author: Gerrit
"""
from emit_tools import emit_xarray
import os
import sys
import numpy as np
import pandas as pd
from osgeo import gdal
import earthaccess
import rasterio as rio
import rioxarray as rxr
import holoviews as hv
import hvplot
import hvplot.xarray
import hvplot.pandas
from skimage import exposure
import seaborn as sns
import matplotlib.pyplot as plt


raster_path = r'C:\Users\Gerrit\Documents\comps\data\EMIT\EMIT_L1B_RAD_001_20240804T181617_2421712_005.nc'

rad = emit_xarray(raster_path, ortho=True)

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

#rgb_map = rgb.hvplot.rgb(x='longitude',y='latitude',bands='wavelengths',title='RGB Radiance', geo=True, crs='EPSG:4326')
#point_map = point_ds.hvplot.points(x='longitude',y='latitude', color='in-plume', cmap='HighContrast', geo=True, crs='EPSG:4326', hover=False, colorbar=False)


#hvplot.show(rgb_map*point_map)



#point_ds.hvplot.line(x='wavelengths',y='radiance', by=['ID'], color=hv.Cycle('Dark2'), frame_height=400, frame_width=600, title = 'Radiance Spectra, ID 0 is in-plume' , xlabel='Wavelength (nm)', ylabel='Radiance (W/m^2/sr/nm)')



#35.1676775784815, -119.09837215132448