# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 22:41:36 2025

@author: Gerrit
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

from extractions import pullData


def randomForestReg(target, estimators, details=False, testSize=0.2, mode='index'):
    
    df = pullData(mode)
    
    #target = 'NH3_mean'
    df = df[df[target] != 0]
    dropColumns = ['CAFO_ID','NH3_mean', 'NH3_sum', 'NH3_median', 'CH4_mean', 'CH4_sum', 'CH4_median', 'mean_emission_auto', 'mean_emission_uncertainty_auto', 'sum_emission_auto', 'sum_emission_uncertainty_auto', 'Point_Count', 'HyTES_NH3_Detect', 'HyTES_CH4_Detect', 'CarbonMapper_CH4_Detect'] 
    
    # Separate features and target
    X = df.drop(columns=dropColumns)
    y = df[target]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    
    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=estimators, bootstrap=True, min_samples_leaf=1, min_samples_split=2)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
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
        
        
    return r2, mse

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

def findParams(target, checkModel, mode='bands'):

    df = pullData(mode)
    #print(df.columns)
    
    dropColumns = ['CAFO_ID','NH3_mean', 'NH3_sum', 'NH3_median', 'CH4_mean', 'CH4_sum', 'CH4_median', 'mean_emission_auto', 'mean_emission_uncertainty_auto', 'sum_emission_auto', 'sum_emission_uncertainty_auto', 'Point_Count', 'HyTES_NH3_Detect', 'HyTES_CH4_Detect', 'CarbonMapper_CH4_Detect'] 
    
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
    X = df.drop(columns=dropColumns)
    y = df[target]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
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
                               cv=5,  # Number of cross-validation folds
                               scoring=scoreMet,  # Evaluation metric
                               n_jobs=-1,  # Use all available cores
                               verbose=1)  # Verbosity level
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

def graphRFEst(target, start, stop, step=1):
    r2s = []
    for n_estimators in range(start, stop, step):
        r2, mse = randomForestReg(target,n_estimators)
        r2s.append(r2)
        
    plt.figure()
    plt.scatter(range(start, stop, step), r2s)
    plt.title('R2 vs n_estimators for Random Forest at test size = 0.2, HyTES NH3')    
    plt.xlabel('n_estimators')                
    plt.ylabel('R2')    
     
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
    
def graphFeatureImportance(df):

    # Plot Line Graph
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.5)
    '''
    plt.subplot(1, 2, 1)  # (rows, cols, panel number)
    sns.lineplot(x='Feature', y='Importance', data=df, marker='o')
    
    # Add smoothed line (LOWESS)
    lowess = sm.nonparametric.lowess
    smooth = lowess(df['Importance'], df.index, frac=0.1)
    plt.plot(df.index, smooth[:, 1], color='red', label='Smoothed')
    
    plt.title('Importance vs Band Name')
    plt.xticks([0, 284])
    plt.xlabel('EMIT Band Number')
    '''
    # Plot Bar Chart for Top 10 Values
    top_10_df = df.nlargest(8, 'Importance')
    
    #plt.subplot(1, 2, 2)
    sns.barplot(x='Importance', y='Feature', data=top_10_df, palette='viridis')
    plt.title('Top 10 Importance Values')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()



### Testing RF on reduction of bands via PCA
in_df, targets, features = pullData()

# Drop bad bands
clean_df = in_df.dropna(axis=1, how='all')

# Drop those bands from features list
features = [item for item in clean_df.columns if item in features]

# Drop empty rows
clean_df = clean_df.dropna(axis=0, how='any')

bands_df = clean_df[features]
attributes_df = clean_df[targets]

pca = PCA(n_components=10)

pca.fit(bands_df)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

reduced_df = pd.DataFrame(pca.fit_transform(bands_df))

input_df = pd.concat([attributes_df['HyTES_NH3_Detect'].reset_index(drop=True), reduced_df], axis=1)

accuracy, r2, featureImportance, matrix = randomForestClass('HyTES_NH3_Detect', 50, df=input_df)

graphRFClassStability('HyTES_NH3_Detect', 50, df=input_df, iterations = 1000, dimensionality='reduced')


