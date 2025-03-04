# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:41:36 2025

@author: Gerrit Hoving

Functions for implementing RF regression and classification and PLSR, along 
with benchmarking of these models
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from extractions import pullData

EMIT_BAND_META_PATH = r'D:\Documents\Projects\comps\data\EMIT\band_metadata.csv'

from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12  

sns.set_theme(style="ticks")


def randomForestReg(target, estimators, df = None, details=False, returnPredictions=False, testSize=0.2):
    if df is None:
        df, targets, features = pullData()
    else:
        targets = target
    
    # Separate features and target
    X = df.drop(columns=targets)
    y = df[target]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    
    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=estimators, bootstrap=True, min_samples_leaf=1, min_samples_split=2)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    featureImportance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
    })
    
    if details:
        print('\nRandom forest results: ')
        print(f'Mean Squared Error: {mse:.2f}')
        print(f'R-squared: {r2:.2f}')
        
        # Print feature importances
        print('Feature Importances:')
        for feature, importance in zip(X.columns, model.feature_importances_):
            print(f'{feature}: {importance:.4f}')
         
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.title("Random Forest Regressor Test :" + target)
        plt.xlabel('Test ' + target)                
        plt.ylabel('Predicted ' + target)
        
        # Calculate trendline and r2
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
        trendline = slope * y_test + intercept
        plt.plot(y_test, trendline, color='red')
        r_squared = r_value**2
        plt.text(0.8, 0.1, f'$R^2_s = {r_squared:.2f}$', transform=plt.gca().transAxes, fontsize=12, color='green')
        plt.text(0.76, 0.01, f'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes, fontsize=12, color='green')
        
        graphFeatureImportance(featureImportance)
        
    if returnPredictions:
        return r2, mae, mape, featureImportance, y_test, y_pred
        
    return r2, mae, mape, featureImportance

def partialLeastSquaresReg(target, components, df=None, details=False, returnPredictions=False, testSize=0.3):
    if df is None:
        df, targets, features = pullData(normalize=True)
    else:
        targets = target
    
    # Separate features and target
    X = df.drop(columns=targets)
    y = df[target]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    
    # Initialize the PLS Regression model
    # n_components is the number of PLS components to use
    pls_model = PLSRegression(n_components=components)
    
    # Fit the model to the training data
    pls_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = pls_model.predict(X_test)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    if details:
        print('\nPLS results: ')
        print(f'Mean Squared Error: {mse:.2f}')
        print(f'R-squared: {r2:.2f}')
        print('PLS Coefficients:')
        print(pls_model.coef_)
        
        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.title("Partial Least Squares Regression Test: " + target)
        plt.xlabel('Test ' + target)                
        plt.ylabel('Predicted ' + target)
        
        # Calculate trendline and r2
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
        trendline = slope * y_test + intercept
        plt.plot(y_test, trendline, color='red')
        r_squared = r_value**2
        plt.text(0.8, 0.1, f'$R^2_s = {r_squared:.2f}$', transform=plt.gca().transAxes, fontsize=12, color='green')
        plt.text(0.76, 0.01, f'$R^2 = {r2:.2f}$', transform=plt.gca().transAxes, fontsize=12, color='green')
        
    if returnPredictions:
        return r2, mse, y_test, y_pred
    
    return r2, mae, mape

def linReg(df, target):
    X = df.drop(columns=target)  
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the values
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return r2, mae

def randomForestClass(target, estimators, df=None, details=False, testSize=0.2):
    if df is None:
        df, targets, features = pullData()
    else:
        targets = target
    
    # Convert 'has_zero' to integer (0 or 1) for compatibility with RandomForest
    df[target] = df[target].astype(int)
    
    # Separate features and target
    X = df.drop(columns=targets)
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, stratify=y)
    
    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=estimators, bootstrap=True, max_depth=None, min_samples_leaf=1, min_samples_split=2) #NH3 Bands
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    featureImportance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
    })
    
    matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
    
    if details:
        print("Random Forest Classification Results: ")
        print(f"Accuracy: {accuracy:.2f}")
        
        # Nice confusion matrix plot
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(16,7))
        sns.set(font_scale=3)
        sns.heatmap(matrix, annot=True, annot_kws={'size':30},
                    cmap=plt.cm.Greens, linewidths=0.2)
        class_names = ['No plume', 'Plume']
        tick_marks = np.arange(len(class_names))
        tick_marks2 = tick_marks + 0.5
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks2, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix for Random Forest Model')
        plt.show()
        print(confusion_matrix(y_test, y_pred))
        
        
        print(classification_report(y_test, y_pred))
        
        # Print feature importances
        print('Feature Importances:')
        for feature, importance in zip(X.columns, clf.feature_importances_):
            print(f'{feature}: {importance:.4f}')
              
        
        graphFeatureImportance(featureImportance)
        
    return accuracy, r2, featureImportance, matrix

def findParams(target, checkModel, df = None):
    if df is None:
        df, targets, features = pullData()
    else:
        targets = target

    # Define the model
    if checkModel == 'RFC': 
        model = RandomForestClassifier(random_state=42)
        scoreMet = 'accuracy'
        # Convert 'has_zero' to integer (0 or 1) for compatibility with RandomForest
        df[target] = df[target].astype(int)
    elif checkModel == 'RFR': 
        model = RandomForestRegressor()
        scoreMet = 'r2'
        df = df[df[target] != 0]
    elif checkModel == 'PLSR':
        model = PLSRegression()
        scoreMet = 'r2'
        df = df[df[target] != 0]
    else:
        print("Invalid model input")
        return None
    
    # Separate features and target
    X = df.drop(columns=targets)
    y = df[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [10, 40, 55, 100, 150, 200, 1000],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 50],  # Maximum depth of the trees
        'min_samples_split': [2, 3, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 4, 6, 8],  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3,  # Number of cross-validation folds
                               scoring=scoreMet,  # Evaluation metric
                               n_jobs=-1,  # Use all available cores
                               verbose=1)  # Verbosity level
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

def graphRFEst(target, start, stop, step=1, df=None, n_runs=1):
    r2s = []
    for n_estimators in range(start, stop, step):
        r2_values = []
        
        # Run the RFR multiple times and collect R2 values
        for i in range(n_runs):
            r2, mae, mape, importance = randomForestReg(target, n_estimators, df, testSize=0.3)
            r2_values.append(r2)
            
        avg_r2 = np.mean(r2_values)
        r2s.append(avg_r2)
        
    plt.figure(figsize=(3, 3))
    plt.scatter(range(start, stop, step), r2s)
    plt.title('Random Forest', fontsize=12, family='Times New Roman')    
    plt.xlabel('n_estimators', fontsize=12, family='Times New Roman')      
    plt.ylabel('R2', fontsize=12, family='Times New Roman')    
    plt.xticks(fontsize=12, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')
    
def graphPLSRComp(df, target, start, stop, step=1, n_runs=1):
    r2s = []
    for n_components in range(start, stop, step):
        r2_values = []
        
        # Run the PLSR multiple times and collect R2 values
        for i in range(n_runs):
            r2, mae, mape = partialLeastSquaresReg(target, n_components, df)
            r2_values.append(r2)
        
        # Compute the average R2 for the current n_components
        avg_r2 = np.mean(r2_values)
        r2s.append(avg_r2)
       
    plt.figure(figsize=(3, 3))
    plt.rcParams.update({'font.size': 12})
    plt.scatter(range(start, stop, step), r2s)
    plt.title('PLSR', fontsize=12, family='Times New Roman')    
    plt.xlabel('n_components', fontsize=12, family='Times New Roman')
    plt.ylabel('R2', fontsize=12, family='Times New Roman') 
    plt.xticks(fontsize=12, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')
     
def graphRFClass(target, start, stop, step=1):
    r2s = []
    accuracyScores = []
    for n_estimators in range(start, stop, step):
        accuracy, r2, imp, mat = randomForestClass(target, n_estimators, False, 0.2)
        r2s.append(r2)
        accuracyScores.append(accuracy)
        
    #print(r2s)
    plt.figure()
    #plt.scatter(range(start, stop, step), r2s)
    plt.scatter(range(start, stop, step), accuracyScores)
    plt.title('R2 vs n_estimators for Random Forest at test size = 0.2, HyTES NH3')    
    plt.xlabel('n_estimators')                
    plt.ylabel('Accuracy')    
    
def graphRFClassStability(target = 'HyTES_NH3_Detect', n_estimators = 60, df=None, iterations = 100, dimensionality = 'reduced'):
    rows= []
    #testValues = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    #testValues = [0.05, 0.1, 0.15, 0.2, 0.5]
    testValues = [0.2]
    
    if df is None:
        df, targets, features = pullData()
    
    for test in testValues:
        for x in range(0, iterations, 1):
            accuracy, r2, importance, matrix = randomForestClass(target, n_estimators, df, False)
            rows.append({'Index': x,
                     'Category': test,
                     'Accuracy': accuracy, 
                     'Matrix': matrix,
                     'Importance': importance})
            


    # Graph box plot of accuracy at different test values
    df = pd.DataFrame(rows)
    plt.figure()
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=2.5)
    sns.boxplot(x='Category', y='Accuracy', data=df)
    plt.title('Random Forest Model Performance')
    plt.xlabel('Proportion of Data Reserved for Testing')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0, top=1)
    plt.show()
    
    
    
    # Plots the confusion matrix
    matrix_list = []
    for row in rows:
        print(type(row['Matrix']))
        if row['Category'] == 0.2:
            matrix_list.append(row['Matrix'])
            
    stack = np.stack(matrix_list)

    # Compute the mean along the first axis
    mean_matrix = np.mean(stack, axis=0)
    sum_matrix = np.sum(stack, axis=0)
    
    matrix = mean_matrix
    
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16,7))
    sns.set(font_scale=3)
    sns.heatmap(matrix, annot=True, annot_kws={'size':30},
                cmap=plt.cm.Greens, linewidths=0.2)
    class_names = ['No plume', 'Plume']
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix for Random Forest Model')
    plt.show()
   
    
    # Get importance values and statistics
    imp_df = pd.DataFrame()
    for row in rows:
        if row['Category'] == 0.2:
            imp_values = row['Importance']['Importance'].values
            features = row['Importance']['Feature']
            
            imps_as_row = pd.DataFrame([imp_values], columns=features)
            imp_df = pd.concat([imp_df, imps_as_row], ignore_index=True)

    mean_values = imp_df.mean()
    std_values = imp_df.std()
    
    # Combine means and standard deviations into a DataFrame
    stats_df = pd.DataFrame({
        'Mean': mean_values,
        'Std Dev': std_values
    })
    
    stats_df = stats_df * 100
    
    stats_df = stats_df.reset_index()
    stats_df.columns = ['Band Number', 'Mean', 'Std Dev']
    
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)
    
    
    
    # If set to index mode, make a bar chart for importances, else make a line graph
    if dimensionality == 'reduced':
        
        # Remove suffixes from index names
        pattern = r'_(median|mean|sum)$'
        stats_df['Band Number'] = stats_df['Band Number'].str.replace(pattern, '', regex=True)
        
        # Get top 8 means with their standard deviations
        top_8_stats = stats_df.nlargest(10, 'Mean')
        top_8_means = top_8_stats['Mean']
        top_8_std = top_8_stats['Std Dev']
        top_8_columns = top_8_stats['Band Number']
        
        # Plotting
        plt.figure(figsize=(10, 8))
        
        # Add plot title and labels
        plt.title('Top 8 Variables with ±1 Standard Deviation')
        plt.ylabel('Spectral Index')
        plt.xlabel('Average Variable Importance (%)')
        
        
        # Create a barplot
        ax = sns.barplot(x=top_8_means, y=top_8_columns, palette='viridis')
        sns.set(font_scale=2)

        # Add error bars manually
        ax.errorbar(x=top_8_means, y=top_8_columns, xerr=top_8_std, fmt='none', capsize=5, color='black', linestyle='none')
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
        return
    

    
    
    # Line plot for mean with ±1 standard deviation
    stats_df = stats_df.iloc[:-1]
    
    sns.lineplot(data=stats_df, x=stats_df.index, y='Mean', label='Mean', marker='o', color='b')
    plt.fill_between(stats_df.index, stats_df['Mean'] - stats_df['Std Dev'], stats_df['Mean'] + stats_df['Std Dev'], color='b', alpha=0.2, label='±1 Std Dev')
    
    # Add smoothed line (LOWESS)
    lowess = sm.nonparametric.lowess
    print(stats_df.index)
    smooth = lowess(stats_df['Mean'], stats_df.index, frac=0.05)
    plt.plot(stats_df.index, smooth[:, 1], color='red', label='Smoothed')
    
    plt.title('Importance vs Band Name')
    plt.xticks(ticks=[0, len(stats_df)-1], labels=['380', '2500'])
    plt.xlabel('Wavelength (nm)')
    
    # Set the minimum y-axis value to 0
    plt.ylim(bottom=0)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    
    # Add plot title and labels
    plt.title('Mean Importance and ±1 Standard Deviation')
    plt.xlabel('Wavelength')
    plt.ylabel('Average Importance, ' + str(iterations) + ' models')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def graphRFRegStability(target = 'NH3 (kg/h)', n_estimators = 100, df=None, iterations = 100, importance_tt_level=None):
    rows= []
    #testValues = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    testValues = [0.1, 0.2, 0.3, 0.5, 0.8]
    #testValues = [0.2]
    
    if df is None:
        df, targets, features = pullData()
    
    for test in testValues:
        for x in range(0, iterations, 1):
            r2, mae, mape, importance = randomForestReg(target, n_estimators, df, False, testSize=test)
            rows.append({'Index': x,
                     'Category': test,
                     'R2': r2, 
                     'MAPE': mape,
                     'Importance': importance})
            

    # Graph box plot of accuracy at different test values
    df = pd.DataFrame(rows)
    plt.figure()
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(x='Category', y='R2', data=df)
    #plt.title('Random Forest Model Performance')
    plt.xlabel('Proportion of Data Reserved for Testing')
    plt.ylabel('Accuracy (R^2)')
    plt.ylim(bottom=-2, top=1)
    plt.show()
    
    if importance_tt_level is not None:
        # Get importance values and statistics
        imp_df = pd.DataFrame()
        for row in rows:
            if row['Category'] == 0.3:
                imp_values = row['Importance']['Importance'].values
                features = row['Importance']['Feature']
                
                imps_as_row = pd.DataFrame([imp_values], columns=features)
                imp_df = pd.concat([imp_df, imps_as_row], ignore_index=True)
    
        mean_values = imp_df.mean()
        std_values = imp_df.std()
        
        # Combine means and standard deviations into a DataFrame
        stats_df = pd.DataFrame({
            'Mean': mean_values,
            'Std Dev': std_values
        })
        
        stats_df = stats_df * 100
        
        stats_df = stats_df.reset_index()
        stats_df.columns = ['Feature', 'Mean', 'Std Dev']
        
        return stats_df
    
def graphFeatureImportance(imp_df):
    bands_df = imp_df[imp_df['Feature'].str.contains('band', case=False)]
    #other_df = imp_df[~imp_df['Feature'].str.contains('band', case=False)]   

    band_meta = pd.read_csv(EMIT_BAND_META_PATH)

    bands_df['ID'] = bands_df['Feature'].str.extract(r'reflectance_band_(\d+)_median')
    bands_df['ID'] = bands_df['ID'].astype(int)
    
    # Add metadata on band positions to importance information
    bands_df = pd.merge(bands_df, band_meta, on="ID", how="right")
    bands_df = bands_df.fillna(0)
    
    # Plot line chart of importance over wavelength
    plt.figure(figsize=(6.5, 4))
    
    sns.lineplot(data=bands_df, x='wavelengths', y='Mean', label='Mean', marker='o', color='b')
    plt.fill_between(bands_df['wavelengths'], bands_df['Mean'] - bands_df['Std Dev'], bands_df['Mean'] + bands_df['Std Dev'], color='b', alpha=0.2, label='±1 Std Dev')
    
    # Add smoothed line (LOWESS)
    lowess = sm.nonparametric.lowess
    #print(bands_df.index)
    smooth = lowess(bands_df['Mean'], bands_df['wavelengths'], frac=0.05)
    plt.plot(bands_df['wavelengths'], smooth[:, 1], color='red', label='Smoothed')
    
    # Set the y-axis to 0-4%
    plt.ylim(0, 4)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter())
    
    # Add plot title and labels
    #plt.title('Mean Importance and ±1 Standard Deviation')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Importance, 100 models')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot Bar Chart for Top n Values
    top_df = imp_df.nlargest(6, 'Mean')
    
    sns.barplot(x='Mean', y='Feature', data=top_df, palette='viridis')
    plt.xlabel('% Importance')

    plt.show()
    
def graphCompareModels(target = 'NH3 (kg/h)', df=None, iterations = 100, dimensionality = 'reduced'):
    rows= []
    dfs = {}
    #testValues = [0.2, 0.3, 0.5]
    test = 0.3
    
    # Allow user to pass multiple dfs to compare model results
    if df is None:
        df, targets, features = pullData()
        dfs.append(df)
    else:
        dfs = df
    
    for df in dfs:
        # Benchmark RF models
        for x in range(0, iterations, 1):
            r2, mae, mape, importance = randomForestReg(target, 150, dfs.get(df), False, testSize=test)
            rows.append({'Index': x,
                     'Category': 'RFR,\n' + df,
                     'R2': r2, 
                     'MAE': mae,
                     'MAPE': mape,
                     'Importance': importance})
        
        # Benchmark PLSR models
        for x in range(0, iterations, 1):
            r2, mae, mape = partialLeastSquaresReg(target, 5, dfs.get(df), False, testSize=test)
            rows.append({'Index': x,
                     'Category': 'PLSR,\n' + df,
                     'R2': r2, 
                     'MAE': mae,
                     'MAPE': mape,
                     'Importance': importance})
            

    # Graph box plot of accuracy at different test values
    df = pd.DataFrame(rows)
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(x='Category', y='R2', data=df)
    #plt.title('Regression Model Performance, Average of ' + str(iterations) + ' iterations')
    plt.xlabel('Model, Input data')
    plt.ylabel('Accuracy (R^2)')
    plt.ylim(bottom=-2, top=1)
    #plt.xticks(rotation=90)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
    plt.axhline(y=0.63, color='blue', linestyle='--', linewidth=1)
    plt.show()
    
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(x='Category', y='MAE', data=df)
    #plt.title('Regression Model Performance, Average of ' + str(iterations) + ' iterations')
    plt.xlabel('Model, Input data')
    plt.ylabel('Accuracy (MAE)')
    plt.ylim(bottom=0)
    #plt.xticks(rotation=90)
    plt.axhline(y=23.6, color='blue', linestyle='--', linewidth=1)
    plt.show()
        
def graphModelPredictions(target = 'NH3 (kg/h)', df=None, iterations = 100, model = 'RF'):
    rows = []
    test = 0.25
    
    if model == 'RF':
        for x in range(0, iterations, 1):
            r2, mae, mape, importance, y_test, y_pred = randomForestReg(target, 100, df, details=False, returnPredictions=True, testSize=test)
            
            y_test = y_test.to_numpy()
            
            for y in range(0, len(y_pred), 1):
                rows.append({'Test': y_test[y],
                         'Predicted': y_pred[y]})
                
    if model == 'PLS':
        for x in range(0, iterations, 1):
            r2, mae, mape, y_test, y_pred = partialLeastSquaresReg(target, 3, df, details=False, returnPredictions=True, testSize=test)
            
            y_test = y_test.to_numpy()
            
            for y in range(0, len(y_pred), 1):
                rows.append({'Test': y_test[y],
                         'Predicted': y_pred[y]})
            
    df = pd.DataFrame(rows)

    plt.figure(figsize=(6.5, 5))
    sns.scatterplot(x='Test', y='Predicted', data=df)
    plt.xlim(left=0, right=200)
    plt.ylim(bottom=0, top=200)
    #plt.title(model + ' Regression Model Predictions')
    plt.xlabel('Test')
    plt.ylabel('Predicted')
    # Add 1-1 Line
    plt.plot([0, 200], [0, 200], color='red', linestyle='--', label='1:1 line')
    # Add regression line
    sns.regplot(x='Test', y='Predicted', scatter=False, color='blue', label='Linear regression line', data=df)
    plt.show()

