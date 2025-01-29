# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:45:57 2025

@author: Gerrit Hoving
"""

from hapi import db_begin, fetch
import hapi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from emit_tools import emit_xarray



def calcSpectrum(bands, do_fetch=False):
    bands['wavenumber'] = (1 / (bands['wavelengths'] * 1e-7)).astype('float64')

    ### Calculate coefficents from EMIT band positions

    db_begin('../data/HITRAN')

    if(do_fetch):
        fetch('NH3',6,1,4000,27000)

    #nu,coef = hapi.absorptionCoefficient_Lorentz(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
    #nu,coef = hapi.absorptionCoefficient_HT(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
    #nu,coef = hapi.absorptionCoefficient_Voigt(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
    #nu,coef = hapi.absorptionCoefficient_Doppler(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})
    nu,coef = hapi.absorptionCoefficient_SDVoigt(SourceTables='NH3', WavenumberGrid=np.array(bands['wavenumber']), Diluent={'air':1.0}, Environment = {'p':0.98,'T':305})



    plt.plot(nu,coef)
    plt.show()

    nu_nm = 1e7 / nu

    plt.plot(nu_nm,coef)
    plt.show()
    
    # Return wavenumber, wavelength, coefficents
    return nu, nu_nm, coef





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


calcSpectrum(emit_bands)