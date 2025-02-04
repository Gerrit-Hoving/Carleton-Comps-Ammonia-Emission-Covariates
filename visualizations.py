# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:52:33 2025

@author: Gerrit Hoving

Code for running analysis and eventually replicating figures
"""

import pandas as pd
from sklearn.decomposition import PCA

from analysis import randomForestReg, partialLeastSquaresReg
from analysis import graphRFRegStability, graphCompareModels, graphModelPredictions
from analysis import graphRFEst
from analysis import graphPLSRComp

from extractions import pullData



### Testing RF on reduction of bands via PCA
in_df, all_targets, all_features = pullData()
min_in_df, min_targets, min_features = pullData(extra_vars=False)

# Drop bad bands
clean_df = in_df.dropna(axis=1, how='all')

# Optionally drop rows without cow number or drop cow number
#clean_df = clean_df[clean_df['Cows (head)'] != 0]
clean_df = clean_df.drop('Cows (head)', axis=1)

# Drop those bands from features list
all_features = [item for item in clean_df.columns if item in all_features]
min_features = [item for item in clean_df.columns if item in min_features]

# Drop empty rows
clean_df = clean_df.dropna(axis=0, how='any')

# Select features 
bands_df = clean_df[min_features]
bands_plus_df = clean_df[all_features]

attributes_df = clean_df[all_targets]

# Run PCA
pca = PCA(n_components=10)
pca.fit(bands_df)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced_df = pd.DataFrame(pca.fit_transform(bands_df))

#reduced_df = pd.concat([attributes_df['NH3 (kg/h)'].reset_index(drop=True), reduced_df], axis=1)

# Creat df with all predictors, with only bands, and with pca reduced bands
input_df_full = pd.concat([attributes_df['NH3 (kg/h)'], bands_plus_df], axis=1)
input_df_bands = pd.concat([attributes_df['NH3 (kg/h)'], bands_df], axis=1)
input_df_pca = pd.concat([attributes_df['NH3 (kg/h)'], reduced_df], axis=1)
input_df_fullysampled = 0 # Work on and test eventually 

input_df_random = input_df_full.copy()
input_df_random['NH3 (kg/h)'] = input_df_random['NH3 (kg/h)'].sample(frac=1).reset_index(drop=True)


#r2, mape, importance, y_test, y_pred = randomForestReg('NH3 (kg/h)', 300, df=input_df, returnPredictions=True, testSize=0.3)
#randomForestReg('NH3 (kg/h)', 300, df=input_df, details=True, testSize=0.3)
#partialLeastSquaresReg('NH3 (kg/h)', 8, df=input_df, details=True, testSize=0.3)

#findParams('NH3 (kg/h)', 'RFR', df=input_df)
#graphPLSRComp(input_df_full, 'NH3 (kg/h)', 3, 10, 1)
#graphRFEst('NH3 (kg/h)', 1, 200, 1, input_df_full)
#graphRFEst('NH3 (kg/h)', 1, 200, 1, input_df_bands)

#accuracy, r2, featureImportance, matrix = randomForestClass('HyTES_NH3_Detect', 50, df=input_df)
#graphRFRegStability('NH3 (kg/h)', 200, df=input_df_full, iterations = 100, dimensionality='reduced')
#graphRFRegStability('NH3 (kg/h)', 200, df=input_df_bands, iterations = 100, dimensionality='reduced')

#graphModelPredictions(target = 'NH3 (kg/h)', df=input_df_full, iterations = 100, model='PLS')


### Decent figures
comparison_dfs = {'full':input_df_full, 'bands':input_df_bands, 'random':input_df_random}
graphCompareModels(target = 'NH3 (kg/h)', df=comparison_dfs, iterations=100)


