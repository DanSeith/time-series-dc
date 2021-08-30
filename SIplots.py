import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import splitfolders
import pickle
import sys
from plot_tools import set_size
import matplotlib


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

def morph_cnn_plot(filename='./figures/SI/loss_acc_plot.png',save=False):

	with open('./trainHistoryDict', 'rb') as f:
	    history = pickle.load(f)


	fig, (ax1, ax2) = plt.subplots(1, 2,figsize=set_size(469.7,fraction=1,hratio=1.4,subplots=(1,2)))
	plt.rcParams["font.family"] = "Arial"
	plt.rc('font', size=12)
	#fig.suptitle('Horizontally stacked subplots')

	ax1.plot(history['loss'],label='train')
	ax1.plot(history['val_loss'],label='validation')

	ax2.plot(history['accuracy'],label='train')
	ax2.plot(history['val_accuracy'],label='validation')

	ax1.set_xlabel('epoch')
	ax2.set_xlabel('epoch')

	ax1.set_ylabel('Loss')
	ax2.set_ylabel('Accuracy')

	ax1.set_ylim(.3,1)
	ax1.legend(loc='lower left',fontsize=6)
	ax2.legend(loc='best',fontsize=6)

	ax1.text(-0.1, 1.15, 'a)', transform=ax1.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	ax2.text(-0.1, 1.15, 'b)', transform=ax2.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')

	plt.tight_layout()

	if save:
		plt.savefig(filename,dpi=600)
	else:
		plt.show() 