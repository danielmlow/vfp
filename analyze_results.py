

import os
import pickle
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def performance_table(results, permute_order, output_dir, score_i = 0,score_name='roc_auc_score', round = 2):
	# score = {N}, choose which metric to compute, if run has 30 splits with 2 score metrics, scores_data.shape = (30,2)
	# Performance df

	df_null = []
	df_all = []
	columns = []
	scores_data_all = []
	scores_data_median_all = [ ]
	scores_null_median_all = [ ]
	scores_data_ci_all = []

	# Loop through models
	for i, permute in enumerate(permute_order):
		if permute:
			model_null = results[i]
			scores_null = model_null[1].output.score
			scores_null_median = np.median(scores_null)
			scores_null_median_all.append(scores_null_median)
		else:
			model = results[i]
			model_name = list(model[0].values())[0][1]
			columns.append(model_name)
			scores_data = np.array(model[1].output.score)[:,score_i] #which metric
			scores_data_all.append(scores_data)
			scores_data_median = np.median(scores_data)
			scores_data_median_all.append(scores_data_median)
			ci = [np.percentile(scores_data,2.5),np.percentile(scores_data,97.5)] # 95% CI
			scores_data_ci_all.append(ci)



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

	return df_median, df_all


def feature_importance_to_summary(results, permute_order, feature_names, output_dir):
	# output from sklearn pipeline.coef_, .coefs_ or .feature_importances_
	new_output_dir = output_dir + 'feature_importance/'
	try:
		os.mkdir(new_output_dir)
	except:
		pass
	for i, permute in enumerate(permute_order):
		if permute:
			# if this is the null distribution with permuted labels, it won't output feature importance
			continue
		else:
			model = results[i]
			model_name = list(model[0].values())[0][1]
			if 'MLP' in model_name:
				# it is computed but it will have N weights for each input feature. One could take the sum, but it is harder to interpret.
				continue

			feature_importance = model[1].output.feature_importance
			columns = ['split_'+str(n) for n in range(len(feature_importance))]
			df = pd.DataFrame(feature_importance, index= columns, columns = feature_names).T
			df["mean"] = df.mean(axis=1)
			df["std"] = df.std(axis=1)
			df["min"] = df.min(axis=1)
			df["max"] = df.max(axis=1)
			df_sorted = df.sort_values("mean")[::-1]
			df_sorted.to_csv(f"{new_output_dir}feature_importance_{model_name}.csv")
	return


def permutation_importance_to_summary(results, permute_order, feature_names,output_dir):
	# output from sklearn.inspection.permutation_importance()
	new_output_dir = output_dir + 'permutation_importance/'
	try: os.mkdir(new_output_dir )
	except: pass
	for i, permute in enumerate(permute_order):
		if permute:
			# if this is the null distribution with permuted labels, it won't output feature importance
			continue
		else:
			model = results[i]
			model_name = list(model[0].values())[0][1]
			permutation_importance = model[1].output.permutation_importance
			columns = ['split_'+str(n) for n in range(len(permutation_importance))]
			df = pd.DataFrame(permutation_importance, index= columns, columns = feature_names).T
			df["mean"] = df.mean(axis=1)
			df["std"] = df.std(axis=1)
			df["min"] = df.min(axis=1)
			df["max"] = df.max(axis=1)
			df_sorted = df.sort_values("mean")[::-1]
			df_sorted.to_csv(f"{new_output_dir}permutation_importance_{model_name}.csv")
	return




def plot_summary(summary, output_dir=None, filename="shap_plot", plot_top_n_shap=16):
	plt.clf()
	plt.figure(figsize=(8, 12))
	# plot without all bootstrapping values
	summary = summary[["mean", "std", "min", "max"]]
	num_features = len(list(summary.index))
	if (plot_top_n_shap != 1 and type(plot_top_n_shap) == float) or type(
		plot_top_n_shap) == int:
		# if plot_top_n_shap != 1.0 but includes 1 (int)
		if plot_top_n_shap <= 0:
			raise ValueError(
				"plot_top_n_shap should be a float between 0 and 1.0 or an integer >= 1. You set to zero or negative."
			)
		elif plot_top_n_shap < 1:
			plot_top_n_shap = int(np.round(plot_top_n_shap * num_features))
		summary = summary.iloc[:plot_top_n_shap, :]
		filename += f"_top_{plot_top_n_shap}"
		#     todo remove
		filename = filename.replace('_values', '')


	hm = sns.heatmap(
		summary.round(3), annot=True, xticklabels=True, yticklabels=True, cbar=False, square=True, annot_kws={"size": 10}
	)
	hm.set_xticklabels(summary.columns, rotation=45)
	hm.set_yticklabels(summary.index, rotation=0)
	plt.ylabel("Features")
	# plt.savefig(output_dir + f"summary_{filename}.png", dpi=100, bbox_inches='tight')
	plt.savefig(output_dir + f"{filename.replace('.csv', '')}.png", dpi=100, bbox_inches='tight')
	plt.show(block=False)


# # Redo plots
# # =============================================
# # Obtain results pkl
# input_dir = './../vfp_v6_collinearity/'
# results_dir = 'out-voto_spec.json-20200822T104154.938604/'
# json_file = 'northwestern_spec_text_liwc_extremes.json' #'northwestern_spec_text_liwc.json'
#
# # Redo plots
#
# input_dir = input_dir+dirs[0]+'/shap-20200822T104159.176859/'
# output_dir = input_dir
#
# for file in os.listdir(input_dir):#todo pasar nuevo report a pydra cluster
# 	if file.endswith('.csv'):
# 		summary = pd.read_csv(input_dir+file, index_col=0)
# 		plot_summary(summary, output_dir=output_dir, filename=file, plot_top_n_shap=16)
# =============================================




if __name__ == "__main__":


	input_dir = './data/output/vfp_v8_wo_top5/'

	models = 4
	permute_order = [False, True]
	permute_order = permute_order * models


	dirs = [n for n in os.listdir(input_dir+'outputs/') if 'out-vfp' in n]
	dirs.sort()


	'''
	import glob
	dirs = glob.glob(input_dir + '*/out-vfp*', recursive=True)
	dirs.sort()
	dirs = [n.replace(input_dir+'outputs/','') for n in dirs]
	'''

	for results_dir in dirs:
		json_file = f"specs/{results_dir.split('json')[0]+'json'}".replace('out-vfp', 'vfp') #'northwestern_spec_text_liwc_extremes.json' #'northwestern_spec_text_liwc.json'
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
		for score_i, score_name in enumerate(score_names):
			print(results_dir)
			performance_table(results, permute_order, output_dir, score_i=score_i, score_name = score_names[score_i], round = 2)
			# feature_importance_to_summary(results, permute_order, feature_names, output_dir)
			# permutation_importance_to_summary(results, permute_order, feature_names, output_dir)


	# ====================================================
	# Obtain results pkl
	models = 4

	# other
	permute_order = [False, True]
	score_names = ["roc_auc_score"] #todo obtain from json
	permute_order = permute_order * models

	input_dir = './../vfp_v7_indfact/outputs/'
	spec_dir = './../vfp_v7_indfact/specs/'
	data_types = ['both', 'speech', 'vowel']
	collinearity_methods = ['if']
	thresholds_n = 9
	dirs = os.listdir(input_dir)
	dirs = [n for n in dirs if not n.startswith('.')]
	dirs.sort()
	# dirs = ['out-northwestern_spec_text_liwc_extremes.json-20200819T100712.772595']
	# 				# 'out-vfp_spec_4models_vowel.json-20200814T100938.800433',
	# 				# 'out-vfp_spec_4models_both.json-20200814T085556.295861']




	# this would have been done by pydraml

	for collinearity_method in collinearity_methods :
		for data_type in data_types:
			performance_all = []
			performance_median = []

			vars_count_by_thesh_id = []
			vars_by_thesh_id = []
				# if not 'out-' in results_dir or not data_type in results_dir or not '_'+collinearity_method in results_dir:
				# 	continue
				# data_type_file = results_dir.split('.json')[0].split('_')[-3]
				# collinearity_method_file = results_dir.split('.json')[0].split('_')[-2]
				#
				# # assert data_type ==data_type_file and collinearity_method == collinearity_method_file
				# job_array_id = results_dir.split('.json')[0].split('_')[-1]


			for job_array_id in range(1,thresholds_n+1):
				# Load from spec file

				with open(spec_dir + f'vfp_spec_4models_{data_type}_{collinearity_method}_{job_array_id}.json', 'r') as f:
					spec_file = json.load(f)

				feature_names = spec_file['x_indices']


				vars_count_by_thesh_id.append([job_array_id, len(feature_names)])
				vars_by_thesh_id.append([job_array_id, feature_names])
				models = len(spec_file['clf_info'])
				permute_order = spec_file['permute']  # [False, True]
				permute_order = permute_order * models
				score_names = spec_file['metrics']  # ["f1_score", "roc_auc_score"]

				# Load results
				results_dir = [n for n in os.listdir(input_dir) if (data_type in n and '_'+collinearity_method in n and '_'+str(job_array_id)+'.' in n)]
				if len(results_dir) == 1:
					# make sure it only found 1
					results_dir = results_dir[0]
				else:
					print('multiple')
					break
				files = os.listdir(input_dir+results_dir)
				results_pkl = [n for n in files if 'results' in n][0]
				with open(os.path.join(input_dir,results_dir, results_pkl), 'rb') as f:
					results = pickle.load(f)

				feature_names_results = results[0][1].output.feature_names
				assert len(feature_names) == len(feature_names_results)
				print(job_array_id, len(feature_names), len(feature_names_results))

				# create table of performance
				output_dir = input_dir + results_dir + '/'

				# Obtain median
				for score_i, score_name in enumerate(score_names):
					df_median, df_all = performance_table(results, permute_order, output_dir, score_i=score_i, score_name = score_names[score_i], round = 2)

					df_median['run_name'] = f'{data_type}_{collinearity_method}_{job_array_id}'
					df_median['job_id'] = f'{job_array_id}'

					df_all['run_name'] = [f'{data_type}_{collinearity_method}_{job_array_id}'] * df_all.shape[0]
					df_all['job_id'] = [f'{job_array_id}'] * df_all.shape[0]

					performance_median.append(df_median)
					performance_all.append(df_all)


				# feature_importance_to_summary(results, permute_order, feature_names, output_dir)
				# permutation_importance_to_summary(results, permute_order, feature_names, output_dir)



			performance_median_df = pd.concat(performance_median, axis=1).T
			performance_median_df = performance_median_df.sort_index().round(2)
			performance_median_df['vars_count'] = [n[1] for n in vars_count_by_thesh_id]
			performance_median_df = performance_median_df.reset_index(drop=True)
			performance_median_df.to_csv(input_dir+f'test_performance_median_{data_type}_{collinearity_method}.csv')

			performance_all_df = pd.concat(performance_all, axis=0, ignore_index=True)
			performance_all_df = performance_all_df.sort_index().round(2)
			performance_all_df['vars_count'] = [n[1] for n in vars_count_by_thesh_id] * performance_all[0].shape[0]
			performance_all_df = performance_all_df.reset_index(drop=True)
			performance_all_df.to_csv(input_dir+f'test_performance_all_{data_type}_{collinearity_method}.csv')






	# Plot
	# =====
	boxpoints = 'outliers'

	import plotly.graph_objects as go
	for collinearity_method in collinearity_methods:
		for data_type in data_types:
		# for data_type in ['speech', 'vowel', 'both']:
			performance_all_df = pd.read_csv(input_dir + f'test_performance_all_{data_type}_{collinearity_method}.csv', index_col = 0)
			x = performance_all_df.job_id.values

			fig = go.Figure()

			fig.add_trace(go.Box(
			    y=performance_all_df.LogisticRegressionCV.values,
			    x=x,
			    name='Logistic Regression',
				boxpoints=boxpoints,
			    # marker_color='#3D9970'
			))
			fig.add_trace(go.Box(
			    y=performance_all_df.SGDClassifier.values,
			    x=x,
			    name='SGD',
				boxpoints=boxpoints,
			    # marker_color='#FF4136'
			))
			fig.add_trace(go.Box(
			    y=performance_all_df.MLPClassifier.values,
			    x=x,
			    name='MLP',
				boxpoints=boxpoints,

			    # marker_color='blue'
			))

			fig.add_trace(go.Box(
			    y=performance_all_df.RandomForestClassifier.values,
			    x=x,
			    name='Random Forest',
				boxpoints=boxpoints,
				# marker_color = '#FECB52'
			))



			fig.update_layout(
				template='ggplot2',
			    yaxis_title='ROC AUC score',
				xaxis_title='Feature set size',
				title = f'{data_type} {collinearity_method}',
				boxmode='group', # group together boxes of the different traces for each value of x
				yaxis = dict(

					range=[0.3, 1],autorange=False

				),

				xaxis = dict(
					tickvals = list(range(1,thresholds_n+1)),
					ticktext = performance_all_df.vars_count[:thresholds_n].values),

			)



			# fig.show()
			fig.to_image(format="png", engine="orca")
			fig.write_image(input_dir+f'{data_type}_{collinearity_method}.png', scale=6)


	# Performance with all features
	input_dir = './../vfp_v7_indfact/outputs/'

	# add performance to fig 5.
	input_dir = './../../datum/vfp/vfp/data/output/vfp_v8_top1outof5/outputs/'

	dirs = os.listdir(input_dir)
	dirs.remove('.DS_Store')
	dirs.sort()
	for d in dirs:
		df = pd.read_csv(input_dir+d+'/test_performance_with_null_roc_auc_score.csv', index_col = 0).values[0][0]
		print('=====')
		print(d)
		print(df.split('(')[0])
		print('('+df.split('(')[1].split(';')[0]+')')

