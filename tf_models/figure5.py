import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from matplotlib import rcParams
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib
import pandas as pd

rcParams['font.family'] = 'arial'
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
current_dir = os.getcwd()

from plot_tools import set_size

font = {'family' : 'Arial'}

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
matplotlib.rc('font', **font)

def make_plot(save=False):
	# Load gru fpr/tpr
	gru_fpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_fpr.csv'))
	gru_tpr = np.loadtxt(os.path.join(current_dir, 'gru', 'gru_tpr.csv'))
	gru_auc = float(np.loadtxt(os.path.join(current_dir, 'gru', 'gru_auc.csv')))
	gru_pred = np.load(os.path.join(current_dir, 'gru', 'gru_pred.npy'))
	seq_y_true = np.load(os.path.join(current_dir, 'seq_y_true.npy'))
	gru_accuracy = sklearn.metrics.accuracy_score(seq_y_true, gru_pred[0].round())

	# Load CNN-gru fpr/tpr
	cnn_gru_fpr = np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_fpr.csv'))
	cnn_gru_tpr = np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_tpr.csv'))
	cnn_gru_auc = float(np.loadtxt(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_auc.csv')))
	cnn_gru_pred_true = np.load(os.path.join(current_dir, 'CNN_GRU', 'cnn_gru_pred.npy'))
	cnn_gru_accuracy = sklearn.metrics.accuracy_score(cnn_gru_pred_true[1], cnn_gru_pred_true[0].round())

	# Load SVM perf
	svm_pred = np.load(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_pred.npy'))
	svm_y_true = np.load(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_ground_truth.npy'))
	svm_accuracy = sklearn.metrics.accuracy_score(svm_y_true, svm_pred.round())
	svm_fpr = np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_fpr.csv'))
	svm_tpr = np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_tpr.csv'))
	svm_auc = float(np.loadtxt(os.path.join(ROOT_DIR, 'sklearn_models/svm', 'svm_auc.csv')))

	width = 469.7/72.27
	height = 3./4.*width
	fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(width,height*.5))

	img = plt.imread(current_dir+'/network.png')
	ax1.imshow(img,aspect='auto')
	ax1.axis('off')

	ax2.plot(svm_fpr, svm_tpr, label='SVM' + ' ('+str(round(svm_auc, 2)) + ')')
	ax2.plot(gru_fpr, gru_tpr, label='GRU' + ' (' + str(round(gru_auc, 2)) + ')')
	ax2.plot(cnn_gru_fpr, cnn_gru_tpr, label='CNN GRU' + ' (' + str(round(cnn_gru_auc, 2)) + ')')
	ax2.set_xlabel('False positive rate')
	ax2.set_ylabel('True positive rate')
	#ax2.set_title('Deep Learning ROC Curve')
	ax2.legend(loc='best',prop={'size': 6})

	df = pd.DataFrame([svm_accuracy, gru_accuracy, cnn_gru_accuracy], index=['SVM', 'GRU', 'CNN GRU'])
	df.plot(kind='bar', legend=False,rot=0,ax=ax3)
	ax3.set_ylim(0.6,1)
	ax3.set_ylabel('Accuracy (%)')

	ax1.text(-0.05, 1.15, 'a)', transform=ax1.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	ax2.text(-0.4, 1.15, 'b)', transform=ax2.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	ax3.text(-0.3, 1.15, 'c)', transform=ax3.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')

	#asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
	#ax2.set_aspect(asp)

	#asp = np.diff(ax3.get_xlim())[0] / np.diff(ax3.get_ylim())[0]
	#ax3.set_aspect(asp)
	

	plt.tight_layout()

	if save:
		plt.savefig('figure5_update.eps', format='eps',dpi=600)
	else:
		plt.show()
