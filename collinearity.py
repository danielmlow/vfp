


'''
# Reduce multicollinearity through VIF or Independence Factor
Authors: Daniel M. Low, Satra Ghosh (MIT)
License: Apache 2.0
'''

import json
import pandas as pd
import os
import numpy as np
import dcor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor




input_dir = './data/outputs/vfp_v7_indfact/'
n_features = 88
run_vif = False #traditional method for multicollinearity, but is only linear
run_independence_factor = True # Out method
run_nested_crossvalidation_test = False


class ReduceVIF(BaseEstimator, TransformerMixin):
	def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
		# From looking at documentation, values between 5 and 10 are "okay". 5 = r2 of 0.8 because VIF = 5 = 1/(1-r2)
		# Above 10 is too high and so should be removed.
		self.thresh = thresh

		# The statsmodel function will fail with NaN values, as such we have to impute them.
		# By default we impute using the median value.
		# This imputation could be taken out and added as part of an sklearn Pipeline.
		if impute:
			self.imputer = SimpleImputer(strategy=impute_strategy)

	def fit(self, X, y=None):
		print('ReduceVIF fit')
		if hasattr(self, 'imputer'):
			self.imputer.fit(X)
		return self

	def transform(self, X, y=None):
		print('ReduceVIF transform')
		columns = X.columns.tolist()
		if hasattr(self, 'imputer'):
			X = pd.DataFrame(self.imputer.transform(X), columns=columns)
		return ReduceVIF.calculate_vif(X, self.thresh)

	@staticmethod
	def calculate_vif(X, thresh=5.0):
		# Taken from https://stats.stackexchange.com/a/253620/53565 and modified
		variables_drop = []
		dropped = True
		while dropped:
			variables = X.columns
			dropped = False
			vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

			max_vif = max(vif)
			if max_vif > thresh:
				maxloc = vif.index(max_vif)
				print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
				variables_drop.append(X.columns[maxloc])
				X = X.drop([X.columns.tolist()[maxloc]], axis=1)
				dropped = True
		print(f'dropped {len(variables_drop)} variables, kept {X.shape[1]}')
		return X, variables_drop



class FindIndependentFactors(BaseEstimator, TransformerMixin):
	def __init__(self, thresh=0.8, impute=True, impute_strategy='median'):
		# distance correlation default 0.8
		# Above 10 is too high and so should be removed.
		self.thresh = thresh

		# The statsmodel function will fail with NaN values, as such we have to impute them.
		# By default we impute using the median value.
		# This imputation could be taken out and added as part of an sklearn Pipeline.
		if impute:
			self.imputer = SimpleImputer(strategy=impute_strategy)

	def fit(self, X, y=None):
		print('ReduceIF fit')
		if hasattr(self, 'imputer'):
			self.imputer.fit(X)
		return self

	def transform(self, X, y=None):
		print('ReduceIF transform')
		columns = X.columns.tolist()
		if hasattr(self, 'imputer'):
			X = pd.DataFrame(self.imputer.transform(X), columns=columns)
		return FindIndependentFactors.calculate_if(X, self.thresh)


	@staticmethod
	def calculate_if(X, thresh=0.8):
		variables = list(X.columns)
		variables.sort()
		variables_keep = []
		variables_drop = []
		variables_remaining = variables.copy()
		for var in variables:
			X_remaining = X[variables_remaining]
			if var in variables_drop:
				continue
			print(f'target variable: {var}')
			idx = X_remaining.columns.get_loc(var)
			x_i = X_remaining.iloc[:, idx].values
			k_vars = X_remaining.shape[1]
			mask = np.arange(k_vars) != idx
			x_noti= X_remaining.iloc[:, mask]
			ds = [dcor.distance_correlation(x_i, x_noti_i) for x_noti_i in x_noti.T.values]
			remaining_variables = x_noti.columns
			for d, remaining_var in zip(ds, remaining_variables ):
				if d >= thresh:
					variables_drop.append(remaining_var)
					variables_remaining.remove(remaining_var)

					print(f'dropping {remaining_var} with dcor {np.round(d, 2)}')
			variables_remaining.remove(var)
			variables_keep.append(var)
		X = X[variables_keep]
		print(f'dropped {len(variables_drop)} variables, kept {X.shape[1]}')
		return X, variables_drop


def vif_calculator(r2):
	return 1 / (1 - r2)

if __name__ == "__main__":

	df2 = pd.read_csv(input_dir + 'northwestern_compare16_vector_freeresp_anh.csv', index_col= 0)
	features = df.drop('name', axis=1).columns.values

	X_if = df2[features].copy()
	# X_if = X_if.iloc[:30,:30]

	transformer = ReduceVIF(thresh=5)
	X_if, variables_drop = transformer.fit_transform(X_if, y=None)


	if run_vif:

		# VIF: Removes collinear cols using VIF
		# ============================================================
		for data_type in ['vowel', 'speech', 'both']:
			df = pd.read_csv(input_dir+f'egemaps_vector_{data_type}.csv')
			df_features = df.drop(['sid', 'token', 'target', 'filename'], axis=1)



			thresholds = [np.round(vif_calculator(r2), 2) for r2 in np.arange(0.2, 1, 0.1)]
			print(thresholds)
			thresholds_Nvars = []
			feature_names = {}
			for i,thresh in enumerate(thresholds):
				i += 1 #for job arrays which is 1 indexing
				# if i in [1,2,3,4,5,6]:
				# 	continue
				print(thresh,'======================')
				X = df_features.copy()
				transformer = ReduceVIF(thresh=thresh) #vowel: 1.33 (r2 = 0.25), drops 71 and keeps 17; speech drops 74, keeps 14
				X, variables_drop = transformer.fit_transform(X, y=None)
				thresholds_Nvars.append([thresh, X.shape[1]])
				feature_names[thresh] = list(X.columns)

				with open(input_dir + f'/specs/vfp_spec_4models_{data_type}.json', 'r') as f:
					array = json.load(f)

				array['x_indices'] = list(X.columns)
				with open(input_dir + f'/specs/vfp_spec_4models_{data_type}_vif_{i}.json', 'w') as f:
					json.dump(array, f)

			with open(input_dir + f'/thresholds_vif_Nvars_{data_type}.txt', 'w') as f:
				f.write(str(thresholds_Nvars))
				f.write('\n\n')
				f.write(str(feature_names))


	if run_independence_factor:
		# Removes collinear cols using Independence Factor (we created this)
		for data_type in ['vowel', 'speech', 'both']:
			# Independence Factor
			# ============================================================
			df = pd.read_csv(input_dir+f'egemaps_vector_{data_type}.csv',index_col = 0)
			df_features = df.drop(['sid', 'token', 'target', 'filename'], axis=1)
			assert df_features.shape[1] == n_features

			thresholds = [np.round(n, 2) for n in np.arange(0.2, 1.1, 0.1)]
			thresholds_Nvars = []
			feature_names = {}
			for i, thresh in enumerate(thresholds):
				i += 1  # for job arrays which is 1 indexing
				print(thresh,'======================')
				X = df_features.copy()
				transformer = FindIndependentFactors(thresh=thresh) #vowel: 0.25 drops 78 and keeps 10
				X, variables_drop = transformer.fit_transform(X, y=None)
				thresholds_Nvars.append([thresh, X.shape[1]])
				feature_names[thresh] = list(X.columns)

				# Open template with all features and replace x_indeces by subset
				with open(input_dir + f'/specs/vfp_spec_4models_{data_type}_if_1.json', 'r') as f:
					array = json.load(f)

				array['x_indices'] = list(X.columns)
				with open(input_dir + f'/specs/vfp_spec_4models_{data_type}_if_{i}.json', 'w') as f:
					json.dump(array, f)


			with open(input_dir + f'/thresholds_if_Nvars_{data_type}.txt', 'w') as f:
				f.write(str(thresholds_Nvars))
				f.write('\n\n')
				f.write(str(feature_names))

	if run_nested_crossvalidation_test:
		# 50 runs to see if removed features changes a lot
		# ====================================================================
		runs_n = 50
		input_dir = './../../datum/vfp/vfp/data/input/features/'
		n_features = 88


		for data_type in ['vowel', 'speech', 'both']:
			# Independence Factor
			# ============================================================
			df = pd.read_csv(input_dir+f'egemaps_vector_{data_type}.csv',index_col = 0)
			df_features = df.drop(['sid', 'token', 'target', 'filename'], axis=1)
			assert df_features.shape[1] == n_features

			if data_type == 'speech':
				# ix :      1,  2,  3,  4,      5
				# thresh: 0.2, 0.3, 0.4, 0.5, 0.6
				thresh = 0.5
			elif data_type == 'vowel':
				thresh = 0.3
			elif data_type == 'both':
				thresh = 0.4

			dropped_vars = []
			for i in range(runs_n):
				print(i)
				transformer = FindIndependentFactors(thresh=thresh)  # vowel: 0.25 drops 78 and keeps 10
				X, dropped_vars_i = transformer.fit_transform(df_features.sample(frac=0.8), y=None)
				dropped_vars.append(dropped_vars_i)



			dropped_vars_sorted = [np.sort(n) for n in dropped_vars ]
			dropped_vars_sorted = pd.DataFrame(dropped_vars_sorted).T
			dropped_vars_sorted.to_csv(input_dir+f'dropped_variables_if_{runs_n}runs_{data_type}_sorted.csv')



		# Load and make Table S1

		from collections import Counter

		df = pd.read_csv(input_dir + f'egemaps_vector_{data_type}.csv', index_col=0)
		features = df.drop(['sid', 'token', 'target', 'filename'], axis=1).columns.values

		variables_kept = {}
		variables_count = {}
		for data_type in ['vowel', 'speech', 'both']:
			dropped_variables = pd.read_csv(input_dir+f'dropped_variables_if_{runs_n}runs_{data_type}_sorted.csv', index_col = 0)
			variables_kept_count_data_type = []
			variables_kept = []
			example = variables_kept.copy()
			for i in range(runs_n):
				dropped_variables_i = dropped_variables[str(i)].values
				dropped_variables_i = [n for n in dropped_variables_i if str(n) != 'nan']

				variables_kept_i = list(set(features) - set(dropped_variables_i))
				variables_kept.append(variables_kept_i)
				variables_kept_count_data_type.append(len(variables_kept_i))

			variables_kept = np.array([n for i in variables_kept for n in i]) #flatten list of lists
			c = Counter(variables_kept)
			# Are the most frequent the same?
			if data_type=='speech':
				json_file = 'vfp_spec_4models_speech_if_4-39_explanations.json'
			elif data_type=='vowel':
				json_file = 'vfp_spec_4models_vowel_if_2-13_explanations.json'
			elif data_type == 'both':
				json_file = 'vfp_spec_4models_both_if_3-19_explanations.json'

			specs_dir = f'./../../datum/vfp/vfp/data/output/vfp_v7_indfact/specs/'
			with open(specs_dir + json_file, 'r') as f:
				spec_file = json.load(f)

			feature_names = spec_file['x_indices']

			# Most common the same as features used?
			len_features = len(feature_names)
			most_common_frequency = c.most_common(len_features)
			most_common = [n[0] for n in most_common_frequency ]
			print(data_type, ', used features not in most common: ', set(feature_names) - set(most_common))

			proportion_usedfeature_in_n_runs = []
			for feature in feature_names:
				proportion = c.get(feature)/50
				proportion_usedfeature_in_n_runs .append(proportion )

			print(f'{data_type}: mean proportion_usedfeature_in_n_runs: {np.round(np.mean(proportion_usedfeature_in_n_runs),2)}')
			# kept_ci = [np.percentile(proportion_usedfeature_in_n_runs , 5), np.percentile(proportion_usedfeature_in_n_runs , 95)]
