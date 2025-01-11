# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:41:36 2025

@author: Gerrit
"""

from sklearn.decomposition import PCA
import pandas as pd

test_path = 'testExtracted.csv'
test_df = pd.read_csv(test_path)

# Drop bad bands
clean_df = test_df.dropna(axis=1, how='all')
clean_df.drop(clean_df.columns[0], axis=1, inplace=True)

# Drop empty rows
clean_df.dropna(axis=0, how='any', inplace=True)

pca = PCA(n_components=10)

pca.fit(clean_df)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced_df = pca.fit_transform(clean_df)