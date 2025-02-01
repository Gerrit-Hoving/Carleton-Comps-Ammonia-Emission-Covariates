# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:52:33 2025

@author: Gerrit Hoving

Code for running analysis and eventually replicating figures
"""

import pandas as pd
from sklearn.decomposition import PCA

from analysis import randomForestReg, partialLeastSquaresReg
from analysis import graphRFRegStability
from analysis import graphRFEst
from analysis import graphPLSRComp

from extractions import pullData



### Testing RF on reduction of bands via PCA
in_df, targets, features = pullData()

# Drop bad bands
clean_df = in_df.dropna(axis=1, how='all')

# Drop those bands from features list
features = [item for item in clean_df.columns if item in features]

# Optionally drop rows without emission factors
#clean_df = clean_df[clean_df['NH3 (g/head/h) Avg'] != 0]

# Drop empty rows
clean_df = clean_df.dropna(axis=0, how='any')

bands_df = clean_df[features]
attributes_df = clean_df[targets]

pca = PCA(n_components=10)

pca.fit(bands_df)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced_df = pd.DataFrame(pca.fit_transform(bands_df))

#input_df = pd.concat([attributes_df['NH3 (kg/h)'].reset_index(drop=True), reduced_df], axis=1)

input_df = pd.concat([attributes_df['NH3 (kg/h)'], bands_df], axis=1)

#input_df = input_df.iloc[:-3]

#randomForestReg('NH3 (kg/h)', 300, df=input_df, details=True, testSize=0.3)
#partialLeastSquaresReg('NH3 (kg/h)', 8, df=input_df, details=True, testSize=0.3)

#findParams('NH3 (kg/h)', 'RFR', df=input_df)
graphPLSRComp('NH3 (kg/h)', 5, 10, 1)
#graphRFEst('NH3 (kg/h)', 5, 500, 5, input_df)

#accuracy, r2, featureImportance, matrix = randomForestClass('HyTES_NH3_Detect', 50, df=input_df)
#graphRFRegStability('NH3 (kg/h)', 200, df=input_df, iterations = 1000, dimensionality='reduced')

### Decent figures

#graphRFEst('NH3 (kg/h)', 5, 300, 1)


