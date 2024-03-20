#!/usr/bin/env python
# coding: utf-8

'''
Authors: Daniel M. Low
License: See license in github repository
'''

# !pip install scikit-optimize
# !pip install importlib-metadata
# !pip install --upgrade scipy


import warnings
from skopt import BayesSearchCV # had to replace np.int for in in transformers.py
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import soundfile as sf
import os 
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
import time
import numpy as np
from scipy import signal
import io
import math
import numpy.matlib

# Config
scoring = 'roc_auc'
toy = False
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
on_colab = False

pd.set_option("display.max_columns", None)
# pd.options.display.width = 0




if on_colab:
  from google.colab import drive
  project_name = 'project_name'
  drive.mount('/content/drive')
  input_dir = f'/content/drive/MyDrive/datum/{project_name}/data/input/'
  output_dir = f'/content/drive/MyDrive/datum/{project_name}/data/output/'
else:
  input_dir = './data/input/'
  output_dir = './data/output/'

os.makedirs(output_dir, exist_ok=True)

audio_files = os.listdir(input_dir+'audios_16khz/')
audio_files.sort()


x, fs = sf.read(input_dir+'audios_16khz/'+audio_files[0])

# egemaps features
egemaps_filenames = ['egemaps_vector_both.csv',
                    'egemaps_vector_speech.csv',
                    'egemaps_vector_vowel.csv'
                   ]

egemaps_features_df = {}
for i in egemaps_filenames:
    df_i = pd.read_csv(input_dir+'features/'+i, index_col = 0)
    egemaps_features_df[i]=df_i
    

# # Bootstrapping classification

models = [
    # LogisticRegressionCV(solver='liblinear', penalty = 'l1', max_iter = 100),
    LogisticRegression(solver='liblinear', penalty = 'l1', max_iter = 100),
    SGDClassifier(loss='log', penalty="elasticnet", early_stopping=True, max_iter = 5000),
    MLPClassifier(alpha = 1, max_iter= 1000),
    RandomForestClassifier(n_estimators= 100)
]

names = ['LogisticRegression', 'SGDClassifier', "MLPClassifier","RandomForestClassifier"]





ridge_alphas = [0.01, 0.1, 1, 10, 100]
ridge_alphas_toy = [0.1, 10]
def get_params(model_name = 'Ridge', toy=False):
    
    if model_name in ['LogisticRegressionCV']:
        if toy:
            warnings.warn('WARNING, running toy version')
            param_grid = {
                'model__Cs': ridge_alphas_toy,
            }
        else:
            param_grid = {
                'model__Cs': ridge_alphas,
            }
    
    if model_name in ['LogisticRegression']:
    
        if toy:
            warnings.warn('WARNING, running toy version')
            param_grid = {
                'model__C': ridge_alphas_toy,
            }
        else:
            param_grid = {
                'model__C': ridge_alphas,
            }

    elif model_name in ['SGDClassifier']:
    
        if toy:
            warnings.warn('WARNING, running toy version')
            param_grid = {
                'model__alpha': ridge_alphas_toy,
            }
        else:
            param_grid = {
                'model__alpha': ridge_alphas,
                'model__l1_ratio': [0.1, 0.4, 0.5, 0.7, 0.9],
            }


    elif model_name in [ 'RandomForestClassifier']:
        if toy:
            warnings.warn('WARNING, running toy version')
            param_grid = {
               # 'vectorizer__max_features': [256,2048],
                # 'model__colsample_bytree': [0.5, 1],
                'model__max_depth': [10,20], #-1 is the default and means No max depth
        
            }
        else:
            param_grid = {
                # Number of trees in the forest
                'model__n_estimators' : [50, 100, 200],

                # Maximum number of levels in each decision tree
                'model__max_depth' : [10, 20, 30, None],

                # Number of features to consider at every split
                'model__max_features': ['auto', None],
                
            }

    
    elif model_name in [ 'MLPClassifier']:
        if toy:
            warnings.warn('WARNING, running toy version')
            param_grid = {
                'model__hidden_layer_sizes': [(50,)],

        
            }
        else:			
            param_grid = {
                # The size of the hidden layers
                'model__hidden_layer_sizes': [(100,), (200,100), (100,50, )],
                # 'model__max_iter': [200, 300, 500, 1000]
                # Activation function for the hidden layer

                # The learning rate schedule for weight updates
                # 'model__learning_rate': ['constant', 'invscaling', 'adaptive']
            }

    return param_grid

variables = ['F0semitoneFrom27.5Hz_sma3nz_amean',
 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm',
 'F0semitoneFrom27.5Hz_sma3nz_percentile20.0',
 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
 'F0semitoneFrom27.5Hz_sma3nz_percentile80.0',
 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
 'F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope',
 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
 'F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope',
 'loudness_sma3_amean',
 'loudness_sma3_stddevNorm',
 'loudness_sma3_percentile20.0',
 'loudness_sma3_percentile50.0',
 'loudness_sma3_percentile80.0',
 'loudness_sma3_pctlrange0-2',
 'loudness_sma3_meanRisingSlope',
 'loudness_sma3_stddevRisingSlope',
 'loudness_sma3_meanFallingSlope',
 'loudness_sma3_stddevFallingSlope',
 'spectralFlux_sma3_amean',
 'spectralFlux_sma3_stddevNorm',
 'mfcc1_sma3_amean',
 'mfcc1_sma3_stddevNorm',
 'mfcc2_sma3_amean',
 'mfcc2_sma3_stddevNorm',
 'mfcc3_sma3_amean',
 'mfcc3_sma3_stddevNorm',
 'mfcc4_sma3_amean',
 'mfcc4_sma3_stddevNorm',
 'jitterLocal_sma3nz_amean',
 'jitterLocal_sma3nz_stddevNorm',
 'shimmerLocaldB_sma3nz_amean',
 'shimmerLocaldB_sma3nz_stddevNorm',
 'HNRdBACF_sma3nz_amean',
 'HNRdBACF_sma3nz_stddevNorm',
 'logRelF0-H1-H2_sma3nz_amean',
 'logRelF0-H1-H2_sma3nz_stddevNorm',
 'logRelF0-H1-A3_sma3nz_amean',
 'logRelF0-H1-A3_sma3nz_stddevNorm',
 'F1frequency_sma3nz_amean',
 'F1frequency_sma3nz_stddevNorm',
 'F1bandwidth_sma3nz_amean',
 'F1bandwidth_sma3nz_stddevNorm',
 'F1amplitudeLogRelF0_sma3nz_amean',
 'F1amplitudeLogRelF0_sma3nz_stddevNorm',
 'F2frequency_sma3nz_amean',
 'F2frequency_sma3nz_stddevNorm',
 'F2bandwidth_sma3nz_amean',
 'F2bandwidth_sma3nz_stddevNorm',
 'F2amplitudeLogRelF0_sma3nz_amean',
 'F2amplitudeLogRelF0_sma3nz_stddevNorm',
 'F3frequency_sma3nz_amean',
 'F3frequency_sma3nz_stddevNorm',
 'F3bandwidth_sma3nz_amean',
 'F3bandwidth_sma3nz_stddevNorm',
 'F3amplitudeLogRelF0_sma3nz_amean',
 'F3amplitudeLogRelF0_sma3nz_stddevNorm',
 'alphaRatioV_sma3nz_amean',
 'alphaRatioV_sma3nz_stddevNorm',
 'hammarbergIndexV_sma3nz_amean',
 'hammarbergIndexV_sma3nz_stddevNorm',
 'slopeV0-500_sma3nz_amean',
 'slopeV0-500_sma3nz_stddevNorm',
 'slopeV500-1500_sma3nz_amean',
 'slopeV500-1500_sma3nz_stddevNorm',
 'spectralFluxV_sma3nz_amean',
 'spectralFluxV_sma3nz_stddevNorm',
 'mfcc1V_sma3nz_amean',
 'mfcc1V_sma3nz_stddevNorm',
 'mfcc2V_sma3nz_amean',
 'mfcc2V_sma3nz_stddevNorm',
 'mfcc3V_sma3nz_amean',
 'mfcc3V_sma3nz_stddevNorm',
 'mfcc4V_sma3nz_amean',
 'mfcc4V_sma3nz_stddevNorm',
 'alphaRatioUV_sma3nz_amean',
 'hammarbergIndexUV_sma3nz_amean',
 'slopeUV0-500_sma3nz_amean',
 'slopeUV500-1500_sma3nz_amean',
 'spectralFluxUV_sma3nz_amean',
 'loudnessPeaksPerSec',
 'VoicedSegmentsPerSec',
 'MeanVoicedSegmentLengthSec',
 'StddevVoicedSegmentLengthSec',
 'MeanUnvoicedSegmentLength',
 'StddevUnvoicedSegmentLength',
 'equivalentSoundLevel_dBp']

len(variables)




if toy: 
    tasks = ['speech', 'vowel']
    dfs =     ['egemaps_vector_speech.csv','egemaps_vector_vowel.csv']


else:
    tasks = ['speech', 'vowel', 'both']
    
    dfs =     ['egemaps_vector_speech.csv',
    	'egemaps_vector_vowel.csv', 
    	'egemaps_vector_both.csv']
    

for df_name, task_type_df in zip(tasks, dfs):

    print(df_name, task_type_df)
    df_i = pd.read_csv(input_dir+f'features/{task_type_df}', index_col = 0)
    
    print(df_name, '\n====')

    for null_model in [False]: #[True, False]
        print('\npermute', null_model)
    
        if toy:
            n_bootstraps = 3
            cv = 3
        else:
            n_bootstraps = 50
            cv = 5

        

        X = df_i[variables].values
        y = df_i['target'].values
        if null_model:
            y = np.random.permutation(y)
        groups = df_i['sid'].values

        y_pred_all = {}
        roc_auc_all = {}
        run_time_all = {}
        for model, name in zip(models, names):
          
            y_pred_all[name] = []
            roc_auc_all[name] = []
            run_time_all[name] = []

            
            model_name  = str(model).split('(')[0]
            assert name == model_name

            model.set_params(random_state = 123)

            pipe = Pipeline(steps=[
                    ('scaler', StandardScaler()), 
                    ('model', model)])


            ## Performing bootstrapping
            splitter = GroupShuffleSplit(n_splits=n_bootstraps, test_size=0.2, random_state=0)
            runtimes_one_model = []
            for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Hyperparameter tuning
                start_time = time.time()
                if null_model == False:
                    parameters = get_params(model_name=model_name, toy=toy)  		
                    pipeline = RandomizedSearchCV(pipe, parameters, cv=cv, scoring=scoring, return_train_score=False,
                        n_iter=10, random_state=123)    		
                    pipeline.fit(X_train,y_train)

                    
                    best_params = pipeline.best_params_
                    best_model = pipeline.best_estimator_

                    y_pred = best_model.predict(X_test)
                else:
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)

                # # Evaluate
                end_time = time.time()
                runtime = end_time - start_time
                runtimes_one_model.append(runtime)

                roc_auc = metrics.roc_auc_score(y_test, y_pred)  # ROC AUC takes probabilities but here we match what pydra-ml does: https://github.com/nipype/pydra-ml/issues/56

                y_pred_all[name].append(y_pred)
                y_pred_df = pd.DataFrame(y_pred, columns = [f'y_pred_{i}'])
                
                

                roc_auc_all[name].append(roc_auc)
                run_time_all[name].append(runtime)
            
            y_pred_df['y_true'] = y_test
            y_pred_df.to_csv(output_dir+f'vfp_hyperparameter_tuning/y_pred_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}_{n_bootstraps}.csv')
            runtime_df = pd.DataFrame(runtimes_one_model, columns = [model_name])
            runtime_df.to_csv(output_dir+f'vfp_hyperparameter_tuning/runtime_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}_{n_bootstraps}.csv')

            results_i = []
            
            scores = roc_auc_all.get(name)
            roc_auc_median = np.round(np.median(scores),2)
            roc_auc_5 = np.round(np.percentile(scores, 5),2)
            roc_auc_95 = np.round(np.percentile(scores, 95),2)
            results_str = f'{roc_auc_median} ({roc_auc_5}–{roc_auc_95}; )'
            results_str = results_str.replace('0.', '.')
            results_i.append([name, results_str])
            

            if null_model:
                print(name, str(roc_auc_median).replace('0.', '.'))
            if not null_model:
                results_i_df = pd.DataFrame(results_i, ).T
                # display(results_i_df)
                results_i_df.to_csv(output_dir+f'vfp_hyperparameter_tuning/results_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}.csv')
                




    results_i = []
    for name in names:
        scores = roc_auc_all.get(name)
        roc_auc_median = np.round(np.median(scores),2)
        roc_auc_5 = np.round(np.percentile(scores, 5),2)
        roc_auc_95 = np.round(np.percentile(scores, 95),2)
        results_str = f'{roc_auc_median} ({roc_auc_5}–{roc_auc_95}; )'
        results_str = results_str.replace('0.', '.')
        results_i.append([name, results_str])
        
        if null_model:
            print(name, str(roc_auc_median).replace('0.', '.'))
    if not null_model:
        results_i_df = pd.DataFrame(results_i, ).T
        display(results_i_df)
        results_i_df.to_csv(output_dir+f'vfp_hyperparameter_tuning/results_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}.csv')

            
# Test n n_bootstraps		
# ========================================================

models = [
	LogisticRegression(solver='liblinear', penalty = 'l1', max_iter = 100),
    SGDClassifier(loss='log', penalty="elasticnet", early_stopping=True, max_iter = 5000),
    MLPClassifier(alpha = 1, max_iter= 1000),
    RandomForestClassifier(n_estimators= 100)
]


names = ['LogisticRegression', 'SGDClassifier', "MLPClassifier","RandomForestClassifier"
		 ]


output_dir_i = output_dir+f'n_bootraps/'
os.makedirs(output_dir_i, exist_ok=True)

toy = False


if toy: 
    tasks = ['speech', 'vowel']
    dfs =     ['egemaps_vector_speech.csv','egemaps_vector_vowel.csv']


else:
    tasks = ['speech', 'vowel', 'both']
    
    dfs =     ['egemaps_vector_speech.csv',
        'egemaps_vector_vowel.csv', 
        'egemaps_vector_both.csv']
    
scores_all_tasks = {}
for df_name, task_type_df in zip(tasks, dfs):

    print(df_name, task_type_df)
    df_i = pd.read_csv(input_dir+f'features/{task_type_df}', index_col = 0)
    
    print(df_name, '\n====')

    for null_model in [False]: #[True, False]
        print('\npermute', null_model)
    
        if toy:
            n_bootstraps = 3
        else:
            n_bootstraps = 300



        X = df_i[variables].values
        y = df_i['target'].values
        if null_model:
            y = np.random.permutation(y) #CHECK
        groups = df_i['sid'].values

        y_pred_all = {}
        roc_auc_all = {}
        run_time_all = {}
        for model, name in zip(models, names):
          
            y_pred_all[name] = []
            roc_auc_all[name] = []
            run_time_all[name] = []

            
            model_name  = str(model).split('(')[0]
            assert name == model_name

            model.set_params(random_state = 123)

            pipe = Pipeline(steps=[
                    ('scaler', StandardScaler()), 
                    ('model', model)])


            ## Performing bootstrapping
            splitter = GroupShuffleSplit(n_splits=n_bootstraps, test_size=0.2, random_state=0)
            runtimes_one_model = []
            for i, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
                if i % 50 == 0:
                    print(i, '============================================')
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Hyperparameter tuning
                start_time = time.time()
                
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                # # Evaluate
        
                end_time = time.time()
                runtime = end_time - start_time
                runtimes_one_model.append(runtime)

                roc_auc = metrics.roc_auc_score(y_test, y_pred)  # ROC AUC takes probabilities but here we match what pydra-ml does: https://github.com/nipype/pydra-ml/issues/56

                y_pred_all[name].append(y_pred)
                y_pred_df = pd.DataFrame(y_pred, columns = [f'y_pred_{i}'])
                
                

                roc_auc_all[name].append(roc_auc)
                run_time_all[name].append(runtime)
            
            y_pred_df['y_true'] = y_test
            y_pred_df.to_csv(output_dir_i+f'y_pred_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}_{n_bootstraps}.csv')
            runtime_df = pd.DataFrame(runtimes_one_model, columns = [model_name])
            runtime_df.to_csv(output_dir_i+f'runtime_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}_{n_bootstraps}.csv')

            results_i = []
            
            scores = roc_auc_all.get(name)
            roc_auc_median = np.round(np.median(scores),2)
            roc_auc_5 = np.round(np.percentile(scores, 5),2)
            roc_auc_95 = np.round(np.percentile(scores, 95),2)
            results_str = f'{roc_auc_median} ({roc_auc_5}–{roc_auc_95}; )'
            results_str = results_str.replace('0.', '.')
            results_i.append([name, results_str])
            

            if null_model:
                print(name, str(roc_auc_median).replace('0.', '.'))
            if not null_model:
                results_i_df = pd.DataFrame(results_i, ).T
                # display(results_i_df)
                results_i_df.to_csv(output_dir_i+f'results_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}_{model_name}.csv')
                


        scores_all_tasks[df_name]= scores

    results_i = []
    for name in names:
        scores = roc_auc_all.get(name)
        roc_auc_median = np.round(np.median(scores),2)
        roc_auc_5 = np.round(np.percentile(scores, 5),2)
        roc_auc_95 = np.round(np.percentile(scores, 95),2)
        results_str = f'{roc_auc_median} ({roc_auc_5}–{roc_auc_95}; )'
        results_str = results_str.replace('0.', '.')
        results_i.append([name, results_str])
        
        if null_model:
            print(name, str(roc_auc_median).replace('0.', '.'))
    if not null_model:
        results_i_df = pd.DataFrame(results_i, ).T
        display(results_i_df)
        results_i_df.to_csv(output_dir_i+f'results_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}.csv')


        results_i = []
        for name in names:
          scores = roc_auc_all.get(name)
          roc_auc_median = np.round(np.median(scores),2)
          roc_auc_5 = np.round(np.percentile(scores, 5),2)
          roc_auc_95 = np.round(np.percentile(scores, 95),2)
          results_str = f'{roc_auc_median} ({roc_auc_5}–{roc_auc_95}; )'
          results_str = results_str.replace('0.', '.')
          results_i.append([name, results_str])
            
          if null_model:
            print(name, str(roc_auc_median).replace('0.', '.'))
        if not null_model:
            results_i_df = pd.DataFrame(results_i, ).T
            display(results_i_df)
            results_i_df.to_csv(output_dir_i+f'results_vfp_hyperparametertuning_{df_name}_permute-{null_model}_{ts}.csv')

            
                

scores = roc_auc_all.get(name)
len(scores)

plt.figure(figsize=(10, 6))
for task in tasks:
    scores = scores_all_tasks[task]
    len_scores = len(scores)

    ticks = range(5, len_scores, 5)
    roc_auc_median_all = []
    for i in ticks:
        # plot median roc auc using i as the number of bootstraps 
        scores_i = scores[:i]
        roc_auc_median = np.round(np.median(scores_i),2)
        roc_auc_median_all.append(roc_auc_median)
    plt.plot(ticks, roc_auc_median_all, label = task)
    # plt.plot(x =ticks, y = roc_auc_median_all, label = task)

    # plot every other xtick
    plt.xticks(ticks[::3], rotation = 90)
    plt.ylabel('Median ROC AUC')
    plt.xlabel('Amount of bootstrapping samples')
    # add vertical line at x 50
    plt.axvline(50, color = 'black', linestyle = '--')
    plt.ylim(0, 1)
    

plt.legend()

plt.savefig(output_dir_i+f'roc_auc_median_vfp_hyperparametertuning_2022-13-03.png', dpi=300, bbox_inches='tight')
plt.show()



