#!/usr/bin/env python3

'''
Authors: Daniel M. Low
License: Apache 2.0
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json
import datetime
from scipy import stats

# For method 2
# def closest(lst, K):
# 	return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]

def permutation_test_pvalue(mean_score,distribution, method = 1):
	'''
	Old version used Wilcoxon test
	'''

	'''
	the permutation-based empirical p-value from Test 1 in:
	Ojala and Garriga. Permutation Tests for Studying Classifier Performance. The Journal of Machine Learning Research (2010) vol. 11
	Based off of: https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/model_selection/_validation.py#L1062 
	'''

	n_distribution = len(distribution)
	pvalue = (np.sum(np.array(distribution) >= mean_score) + 1.0) / (n_distribution + 1)
	return pvalue
	#
	# if method == 2:
	# 	# similar method except return pvalue for rank of closest value
	# 	n_distribution = len(distribution)
	# 	distribution_ranked = sorted(distribution)
	# 	value_closest_in_null = closest(distribution_ranked , mean_score)
	# 	rank = distribution_ranked.index(value_closest_in_null)
	# 	pvalue = (n_distribution-rank)/(n_distribution)
	# 	return pvalue


# TEST
# print('mean score of null, distribution: ', np.round(np.mean(distribution),2))
# for mean_score in np.arange(0.5,1,0.05):
# 	print('\nmean_score', np.round(mean_score,2))
# 	print('method1', permutation_test_pvalue(mean_score, distribution, method=1))
# 	print('method2',permutation_test_pvalue(mean_score, distribution, method=2))






	# return pvalue



def compute_pairwise_stats(df):
	"""Run empirical p-value from Good, 2002.
	When comparing a classifier to itself, compare to its null distribution.
	A one sided test is used.
	Assumes that the dataframe has three keys: Classifier, type, and score
	with type referring to either the data distribution or the null distribution
	"""
	N = len(df.Classifier.unique())
	effects = np.zeros((N, N)) * np.nan
	pvalues = np.zeros((N, N)) * np.nan
	for idx1, group1 in enumerate(df.groupby("Classifier")):
		filter = group1[1].apply(lambda x: x.type == "data", axis=1).values
		group1df = group1[1].iloc[filter, :]
		filter = group1[1].apply(lambda x: x.type == "null", axis=1).values
		group1nulldf = group1[1].iloc[filter, :]
		for idx2, group2 in enumerate(df.groupby("Classifier")):
			filter = group2[1].apply(lambda x: x.type == "data", axis=1).values
			group2df = group2[1].iloc[filter, :]
			if group1[0] != group2[0]:
				mean_score = np.mean(group1df["score"].values)
				distribution = group2df["score"].values
				pval = permutation_test_pvalue(distribution, mean_score)
				stat = 0 #ToDo: compute effect size independent of permutation size

			else:
				mean_score = np.mean(group1df["score"].values)
				distribution = group1nulldf["score"].values
				pval = permutation_test_pvalue(distribution, mean_score)
				stat = 0  # ToDo: compute effect size independent of amount of permutations

			effects[idx1, idx2] = stat
			pvalues[idx1, idx2] = pval
	return effects, pvalues


pd.options.display.width = 0



# os.getcwd()
if __name__ == "__main__":
	# Paths
	input_dir = './data/output/vfp_v7_indfact/'
	output_dir = './data/output/'


	# TODO 1: Open this file:
	'results-20200910T054922.382256.pkl'
	# TODO 2: re compute p-value by taking percentile of mean in null distribution.


	models = 1
	permute_order = [False, True]
	permute_order = permute_order * models


	dirs = [n for n in os.listdir(input_dir+'outputs/') if 'out-vfp' in n]
	dirs.sort()
	for results_dir in dirs:
		json_file = f"specs/{results_dir.split('json')[0]+'json'}".replace('out-vfp', 'vfp') #this is the specification file that tells pydra-ml how to run models #'northwestern_spec_text_liwc_extremes.json' #'northwestern_spec_text_liwc.json'
		results_dir = f'outputs/{results_dir}/' # results_dir = 'outputs/out-vfp_spec_4models_both_if_3-19_explanations.json-20200910T072101.085324/'
		with open(input_dir+json_file, 'r') as f:
			spec_file = json.load(f)

		feature_names = spec_file['x_indices']
		score_names = ["roc_auc_score"] #["f1_score", "roc_auc_score"] #todo obtain from json

		# for results_dir in dirs:
		files = os.listdir(input_dir+results_dir)
		results_pkl = [n for n in files if 'results' in n][0]
		with open(os.path.join(input_dir,results_dir, results_pkl), 'rb') as f:
			results = pickle.load(f)

		output_dir = input_dir + results_dir



		prefix = 'ml_wf'
		metrics = ['roc_auc_score']
		if len(results) == 0:
			raise ValueError("results is empty")
		df = pd.DataFrame(columns=["metric", "score", "Classifier", "type"])
		for val in results:
			score = val[1].output.score
			if not isinstance(score, list):
				score = [score]

			clf = val[0][prefix + ".clf_info"]
			if isinstance(clf[0], list):
				clf = clf[-1][1]
			else:
				clf = clf[1]
			if "Classifier" in clf:
				name = clf.split("Classifier")[0]
			else:
				name = clf.split("Regressor")[0]
			name = name.split("CV")[0]
			permute = val[0][prefix + ".permute"]
			for scoreval in score:
				for idx, metric in enumerate(metrics):
					df = df.append(
						{
							"Classifier": name,
							"type": "null" if permute else "data",
							"metric": metrics[idx],
							"score": scoreval[idx] if scoreval[idx] is not None else np.nan,
						},
						ignore_index=True,
					)
		order = [group[0] for group in df.groupby("Classifier")]
		for name, subdf in df.groupby("metric"):
			if "score" in name:
				effects, pvalues, = compute_pairwise_stats(subdf)

				# plot pvalues
				sns.set(style="whitegrid", palette="pastel", color_codes=True)
				sns.set_context("talk")
				plt.figure(figsize=(2 * len(order), 2 * len(order)))
				# plt.figure(figsize=(8, 8))





				ax = sns.heatmap(
					pvalues<=0.05,
					annot=pvalues.round(3),
					yticklabels=order,
					xticklabels=order,

					cbar=False,
					# cbar_kws={"shrink": 0.7},
					square=True,
				)

				ax.xaxis.set_ticks_position("top")
				ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
				ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
				ax.tick_params(axis="both", which="both", length=0)
				plt.tight_layout()


				timestamp = datetime.datetime.utcnow().isoformat()
				timestamp = timestamp.replace(":", "").replace("-", "")
				plt.savefig(input_dir+results_dir+f"stats-{name}-{timestamp}.png")
				plt.close()
				save_obj(
					dict(effects=effects, pvalues=pvalues, order=order),
					f"stats-{name}-{timestamp}.pkl",
				)

for model_mean_score in range(n_models):
		for model_distribution in range(n_models):
			if model_mean_score==model_distribution:
				# Loop through data and null models
				distribution

				scores_null = model_null[1].output.score
				for i, permute in enumerate(permute_order):
					# permute_order will be [True, False] or [False, True]
					if permute:
						model_null = results[i]
						scores_null = model_null[1].output.score

						# scores_null_median = np.median(scores_null)
						# scores_null_median_all.append(scores_null_median)
					else:
						model = results[i]
						model_name = list(model[0].values())[0][1]
						# columns.append(model_name)
						scores_data = np.array(model[1].output.score)[:,score_i] #which metric
						# scores_data_all.append(scores_data)
						scores_data_mean = np.mean(scores_data)

				pvalue_i = compute_pvalue(scores_null, scores_data_mean)

			# scores_data_median = np.median(scores_data)
			# scores_data_median_all.append(scores_data_median)
			ci = [np.percentile(scores_data,5),np.percentile(scores_data,95)]
			# scores_data_ci_all.append(ci)



		# Save median score with median null score in parenthesis as strings
	if False in permute_order:
		for data, null, ci in zip(scores_data_median_all, scores_null_median_all, scores_data_ci_all ):
			data = format(np.round(data,round),'.2f')
			null = format(np.round(null,round),'.2f')
			ci_lower = format(np.round(ci[0],round),'.2f')
			ci_upper = format(np.round(ci[1], round), '.2f')
			df_null.append(f'{data} ({ci_lower}–{ci_upper}; {null})')
		df_null = pd.DataFrame(df_null).T
		df_null.columns = columns
		columns.sort() # we put cols in alphabetical order to match the test and stats plot
		df_null = df_null[columns]
		df_null.to_csv(os.path.join(output_dir, f'test_performance_with_null_{score_name}.csv')) # Todo add timestep
		print(df_null.values)
		print('=====')

	# Save all results
	for all_score in scores_data_all:
		df_all.append(all_score)

	df_all = pd.DataFrame(df_all).T
	df_all.columns = columns
	columns.sort()  # we put cols in alphabetical order to match the test and stats plot
	df_all = df_all[columns]
	df_all.to_csv(os.path.join(output_dir, f'test_performance_{score_name}.csv'))# Todo add timestep
	df_median = df_all.median()
	df_median.to_csv(os.path.join(output_dir, f'test_performance_median_{score_name}.csv'))  # Todo add timestep


