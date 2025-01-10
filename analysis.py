# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:41:36 2025

@author: Gerrit
"""

from sklearn.decomposition import PCA
import pandas as pd

test_path = 'testExtracted.csv'
test_df = pd.read_csv(test_path)
pca = PCA(n_components=10)

pca.fit(test_df)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced_df = pca.fit_transform(test_df)