import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from matplotlib import rcParams
from matplotlib.pyplot import figure
import shap
import json
import matplotlib

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
current_dir = os.getcwd()

from plot_tools import set_size

with open('./feature_list.txt', 'r') as file:
	feats = json.load(file)

feature_dict = feats
feature_list = list(feature_dict)
feature_list_cleaned = [feature_dict[i] for i in list(feature_dict.keys())]

plt.rcParams["font.family"] = "arial"

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

def make_plot(df,save=False):


	# Extract features
	x = df[feature_list].to_numpy()
	y = df.apply(lambda a: int(a['cell'] == 'hl60'), axis=1).to_numpy()

	# Normalize and standardize first

	x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)

	scalar = sklearn.preprocessing.StandardScaler()
	x_train = scalar.fit_transform(x_train)
	x_val = scalar.transform(x_val)
	
	x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

	plt.figure(figsize=set_size(469.7,fraction=1,hratio=1.5,subplots=(1,2)))
	plt.subplot(1,2,1)
	

	# Iterate over models and plot roc
	model_dict = {}
	i=0

	for rel_dir in ['knn', 'logistic_regression', 'random_forest','svm']:

		
		names = ['k-NN', 'logistic regression', 'random forest','svm']
		model_dir = os.path.join(current_dir, rel_dir, rel_dir+'.pkl')
		# Load model
		with open(model_dir, 'rb') as pkl_file:
			model = pickle.load(pkl_file)

		# Obtain fpr, tpr
		y_pred = model.predict_proba(x_test)[:, 1]
		fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
		auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
		plt.plot(fpr, tpr, label=names[i] + ' ('+ str(round(auc, 2)) + ')')
		model_dict[rel_dir] = sklearn.metrics.accuracy_score(y_test, y_pred.round())*100
		print(sklearn.metrics.accuracy_score(y_test, y_pred.round()))
		i = i +1

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	ax = plt.gca()
	plt.text(-0.2, 1.15, 'a)',transform=ax.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')


	plt.title('ROC Curve')
	plt.legend(loc='best')

	plt.subplot(1,2,2)
	# Load trained random forest model
	with open('./random_forest/random_forest.pkl', 'rb') as pkl_file:
		model = pickle.load(pkl_file)

	# Create object to calculate shap values
	explainer = shap.TreeExplainer(model)

	# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
	shap_values = explainer.shap_values(x_test)
	shap.summary_plot(shap_values[1], x_test, feature_names=feature_list_cleaned, plot_type='bar', show=False,plot_size=None)
	#plt.xlabel('Average impact on model output magnitude')

	ax = plt.gca()
	matplotlib.rcParams.update({'font.family': 'arial'})

	ax.tick_params(axis='both', labelsize=SMALL_SIZE)
	
	for label in ax.xaxis.get_majorticklabels():
		label.set_fontproperties('arial')
		label.set_color('k')
	for label in ax.yaxis.get_majorticklabels():
		label.set_fontproperties('arial')
		label.set_color('k')

		
	ax.set_xlabel('Average impact on model output magnitude',fontsize=MEDIUM_SIZE,fontname='arial')
	ax.text(-0.3, 1.15, 'b)', transform=ax.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	
	plt.tight_layout()

	if save:
		plt.savefig('sklearn_roc_final.png', format='png',dpi=600)
	else:
		plt.show()
	
