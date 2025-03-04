# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:52:33 2025

@author: Gerrit Hoving

Code for running analysis and eventually replicating figures
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12  

from analysis import randomForestReg, partialLeastSquaresReg, linReg
from analysis import graphRFRegStability, graphCompareModels, graphModelPredictions
from analysis import graphRFEst, graphFeatureImportance
from analysis import graphPLSRComp

from extractions import pullData




### Testing RF on reduction of bands via PCA
in_df, all_targets, all_features = pullData(normalize=False)
band_in_df, band_targets, band_features = pullData(extra_vars=False)

# Drop bad bands
clean_df = in_df.dropna(axis=1, how='all')

# Drop unsampled farms for full variable model
clean_df_all_vars = clean_df[clean_df['Cows (head)'] != 0]

# Drop incompletely sampled variables
clean_df = clean_df.drop('Cows (head)', axis=1)
clean_df = clean_df.drop('CH4 Cover', axis=1)


# Drop unused vars from feature lists
band_features = [item for item in clean_df.columns if item in band_features]
fully_sampled_features = [item for item in clean_df.columns if item in all_features]

all_features = [item for item in clean_df_all_vars.columns if item in all_features]


# Drop empty rows
clean_df = clean_df.dropna(axis=0, how='any')
clean_df_all_vars = clean_df_all_vars.dropna(axis=0, how='any')
clean_df_all_vars['CH4 Cover'] = clean_df_all_vars['CH4 Cover'] == 'Y'


# Select features 
bands_df = clean_df[band_features]
bands_plus_allfarms_df = clean_df[fully_sampled_features]
bands_plus_allvars_df = clean_df_all_vars[all_features]

attributes_df = clean_df[all_targets]

# Run PCA
#pca = PCA(n_components=10)
#pca.fit(bands_df)

#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)

#reduced_df = pd.DataFrame(pca.fit_transform(bands_df))

#reduced_df = pd.concat([attributes_df['NH3 (kg/h)'].reset_index(drop=True), reduced_df], axis=1)

# Creat df with all predictors, with only bands, and with pca reduced bands
input_df_full = pd.concat([attributes_df['NH3 (kg/h)'], bands_plus_allfarms_df], axis=1)
input_df_bands = pd.concat([attributes_df['NH3 (kg/h)'], bands_df], axis=1)
#input_df_pca = pd.concat([attributes_df['NH3 (kg/h)'], reduced_df], axis=1)
input_df_fullysampled = pd.concat([attributes_df['NH3 (kg/h)'], bands_plus_allvars_df], axis=1)
input_df_fullysampled = input_df_fullysampled.dropna(axis=0, how='any')

input_df_random = input_df_full.copy()
input_df_random['NH3 (kg/h)'] = input_df_random['NH3 (kg/h)'].sample(frac=1).reset_index(drop=True)

#r2, mape, importance, y_test, y_pred = randomForestReg('NH3 (kg/h)', 300, df=input_df, returnPredictions=True, testSize=0.3)
#randomForestReg('NH3 (kg/h)', 300, df=input_df, details=True, testSize=0.3)
#partialLeastSquaresReg('NH3 (kg/h)', 20, df=input_df_full, details=True, testSize=0.3)

#findParams('NH3 (kg/h)', 'RFR', df=input_df)
#graphPLSRComp(input_df_bands, 'NH3 (kg/h)', 3, 16, 1, n_runs=1000)
#graphPLSRComp(input_df_full, 'NH3 (kg/h)', 3, 16, 1, n_runs=1000)

#graphRFEst('NH3 (kg/h)', 1, 200, 1, input_df_bands)

#accuracy, r2, featureImportance, matrix = randomForestClass('HyTES_NH3_Detect', 50, df=input_df)

#graphRFRegStability('NH3 (kg/h)', 200, df=input_df_bands, iterations = 100, dimensionality='reduced')

#graphModelPredictions(target = 'NH3 (kg/h)', df=input_df_full, iterations = 100, model='PLS')

print(linReg(input_df_full[['NH3 (kg/h)', 'OverallArea (m2)']], 'NH3 (kg/h)') )
print(linReg(input_df_full[['NH3 (kg/h)', 'PenArea (m2)', ]], 'NH3 (kg/h)') )
print(linReg(input_df_full[['NH3 (kg/h)', 'OverallArea (m2)', 'PenArea (m2)']], 'NH3 (kg/h)') )
print(linReg(input_df_full, 'NH3 (kg/h)') )



comparison_dfs = {'all':input_df_full, 'bands':input_df_bands, 'random':input_df_random}
#graphCompareModels(target = 'NH3 (kg/h)', df=comparison_dfs, iterations=500)




'''
import umap

labels = pd.concat([input_df_bands.iloc[:, 0], 'subset', input_df_bands.iloc[:, 0]], axis=0)  # First column for color
labels['dataset'] = ['subset'] * 22 + ['average'] * 22
data = input_df_bands.iloc[:, 1:]  # Rest of the columns for UMAP

avg_bands, a, b = pullData(file=r'D:\Documents\Projects\comps\data\emitReflectanceCARBLots.csv', extra_vars=False)
avg_bands = avg_bands.dropna(axis=1, how='all')
avg_bands = avg_bands.drop('CH4 Cover', axis=1)

b = [item for item in avg_bands.columns if item in b]

avg_bands = pd.concat([data, avg_bands[b]], axis=0)

# Initialize UMAP
umap_model = umap.UMAP(n_components=2)

# Fit and transform the data
umap_result = umap_model.fit_transform(avg_bands)

# Plotting the UMAP result
plt.figure(figsize=(10, 8))
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('UMAP projection vs emission rate')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()
'''

'''
# Calculate R2 and MSE avg values for SLR for an apples to apples comparison
rows = []
for x in range(0, 10000, 1):
    r2, mae = linReg(input_df_full[['NH3 (kg/h)', 'OverallArea (m2)']], 'NH3 (kg/h)')
    rows.append({'Index': x,
                 'R2': r2,
                 'MAE':mae})
        

# Graph box plot of accuracy for multiple iterations of SLR
df = pd.DataFrame(rows)
plt.figure(figsize=(5.5, 4))
sns.set(font_scale=1.5)
sns.boxplot(y='R2', data=df)
plt.title('Linear Regression Model Performance, Average of 100 Iterations')
plt.xlabel('Model, Input data')
plt.ylabel('Accuracy ($R^2$)')
plt.ylim(bottom=-2, top=1)
plt.xticks(rotation=90)
plt.show()

print(df['R2'].median(axis=0))
print(df['MAE'].median(axis=0))
'''


### Final figures

# Fig 7
plt.figure(figsize=(6.5,4))
sns.set_theme(style="ticks", font="Times New Roman", font_scale=1.2)
#plt.scatter('OverallArea (m2)', 'NH3 (kg/h)', data=input_df_full)
sns.regplot(x='OverallArea (m2)', y='NH3 (kg/h)', data=input_df_full, scatter_kws={'s': 20}, line_kws={'color': 'red', 'linestyle': '--'})
plt.xlabel('Area (m^2)')
plt.ylabel('Emission Rate (kg/h)')
plt.savefig("../figures/Fig7.png",bbox_inches='tight',dpi=300)
plt.show()

# Fig 8
imp_df = graphRFRegStability('NH3 (kg/h)', n_estimators=100, df=input_df_full, iterations=250, importance_tt_level=0.3) #500 iter-100 enough?

# Fig 9
graphRFEst('NH3 (kg/h)', 1, 100, 1, input_df_full, n_runs=1000) #1000 iter
graphPLSRComp(input_df_full, 'NH3 (kg/h)', 3, 16, 1, n_runs=1000) #1000 iter

# Fig 10+11
comparison_dfs = {'full':input_df_full, 'bands':input_df_bands, 'random':input_df_random}
graphCompareModels(target = 'NH3 (kg/h)', df=comparison_dfs, iterations=200) #200 iter

# Fig 12
graphFeatureImportance(imp_df)

# Fig 13
graphModelPredictions(target = 'NH3 (kg/h)', df=input_df_full, iterations = 100, model='RF') #100 iter

