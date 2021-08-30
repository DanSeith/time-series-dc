import pandas as pd
import os
import numpy as np
import pickle
import json
import scipy
import matplotlib.pyplot as plt
import shutil
from skimage.measure import regionprops
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import sys
import string
from PIL import Image
from scipy import signal
from scipy.interpolate import interp1d
from skimage import measure
import seaborn as sns
from scipy import stats
from skimage.draw import rectangle
import importlib
import splitfolders
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
from plot_tools import set_size
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter
from skimage.transform import resize
import matplotlib


import plots
import df_utils
import features

plt.rcParams["font.family"] = "Arial"

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


def aspect_dvdx_x(df,output_path,event_num=193,save = False):



	df_cm = pd.read_csv('C://Users//codyt//Desktop//comsol//sheath_particle//bare_channel//velocity.csv',skiprows=8)[0:-2]
	df_cm = df_cm.drop(df_cm.columns[-1],axis=1)
	df_cm = df_cm.rename(columns={df_cm.columns[4]: 'u',df_cm.columns[5]: 'x', df_cm.columns[6]: "dudx",df_cm.columns[3]:'sr'})
	df_cm = df_cm.sort_values(by ='x' )

	df_cm['v_avg'] = df_cm.u.rolling(5).mean()
	df_cm['dv_avg'] = df_cm.dudx.rolling(5).mean()
	df_cm['sr_avg'] = df_cm.sr.rolling(5).mean()

	fig, axs = plt.subplots(2,sharex=False,sharey=False,figsize=(10,5))
	#fig.suptitle('Vertically stacked subplots')

	row = df[df.idx==event_num].iloc[0]

	axs[0].scatter(row.xcm_um,row.aspect,color='black',s=32)
	axs[0].set_xlim([-20,170])
	axs[0].set_ylim([0.9,1.7])

	axs[0].set_ylabel('Aspect Ratio',size=12)

	axs[0].xaxis.set_tick_params(labelsize=12)
	axs[0].yaxis.set_tick_params(labelsize=12)


	# trend = df.iloc[event_num].x_poly1
	# trendpoly = np.poly1d(trend) 
	# xs = df.iloc[event_num].xcm_um[df.iloc[event_num].nar1_max_arg-1:df.iloc[event_num].cav1_min_arg+1]
	# axs[0].plot(xs,trendpoly(xs))

	fig.text(0.5, 0.04, 'X position ($\mu$m)', ha='center',size=12)


	xposition = [0,150]
	for xc in xposition:
	    axs[0].axvline(x=xc, color='r', linestyle='--',lw=3)
	    axs[1].axvline(x=xc, color='r', linestyle='--',lw=3)

	axs[1].plot(df_cm.x*1000,df_cm.dv_avg/1000,color='black',linewidth=2)
	axs[1].set_xlim([-20,170])
	axs[1].set_ylim([-210,210])

	axs[1].set_ylabel('dv/dx (1/s)',size=12)

	axs[1].xaxis.set_tick_params(labelsize=12)
	axs[1].yaxis.set_tick_params(labelsize=12)


	axs[0].axvspan(-20, 15, alpha=0.1, color='red')
	axs[1].axvspan(-20, 15, alpha=0.1, color='red')

	axs[0].axvspan(15, 75, alpha=0.1, color='green')
	axs[1].axvspan(15, 75, alpha=0.1, color='green')

	axs[0].axvspan(75, 125, alpha=0.1, color='yellow')
	axs[1].axvspan(75, 125, alpha=0.1, color='yellow')

	axs[0].axvspan(125, 170, alpha=0.1, color='blue')
	axs[1].axvspan(125, 170, alpha=0.1, color='blue')

	if save:
		plt.savefig(output_path)
	else:
		plt.show()
def aspect_x(df,output_path,event_num=193,save = False):


	plt.figure(figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	row = df[df.cell=='hl60d'].iloc[event_num]

	plt.scatter(row.xcm_um,row.aspect,color='black',s=32)
	plt.xlim([-20,170])
	plt.ylim([0.8,1.6])

	plt.ylabel('Aspect Ratio',size=20)
	plt.xlabel('X position ($\mu$m)',size=20)

	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)


	# trend = df.iloc[event_num].x_poly1
	# trendpoly = np.poly1d(trend) 
	# xs = df.iloc[event_num].xcm_um[df.iloc[event_num].nar1_max_arg-1:df.iloc[event_num].cav1_min_arg+1]
	# axs[0].plot(xs,trendpoly(xs))


	xposition = [0,150]

	for xc in xposition:
	    plt.axvline(x=xc, color='r', linestyle='--',lw=3)
	    
	plt.axvspan(-20, 15, alpha=0.1, color='red')
	plt.axvspan(15, 75, alpha=0.1, color='green')
	plt.axvspan(75, 125, alpha=0.1, color='yellow')
	plt.axvspan(125, 170, alpha=0.1, color='blue')

	if save:
		plt.savefig(output_path)
	else:
		plt.show()
def aspect_shear_x(df,output_path,event_num=193,save=False):


	df_cm = pd.read_csv('C://Users//codyt//Desktop//comsol//sheath_particle//bare_channel//velocity.csv',skiprows=8)[0:-2]
	df_cm = df_cm.drop(df_cm.columns[-1],axis=1)
	df_cm = df_cm.rename(columns={df_cm.columns[4]: 'u',df_cm.columns[5]: 'x', df_cm.columns[6]: "dudx",df_cm.columns[3]:'sr'})
	df_cm = df_cm.sort_values(by ='x' )

	df_cm['v_avg'] = df_cm.u.rolling(5).mean()
	df_cm['dv_avg'] = df_cm.dudx.rolling(5).mean()
	df_cm['sr_avg'] = df_cm.sr.rolling(5).mean()

	fig, axs = plt.subplots(2,sharex=False,sharey=False,figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	row = df[df.cell=='hl60d'].iloc[193]

	axs[0].scatter(row.xcm_um,row.aspect)
	axs[0].set_xlim([-20,170])
	axs[0].set_ylim([0.8,1.6])

	axs[0].set_ylabel('Aspect Ratio',size=20)

	fig.text(0.5, 0.04, 'X position ($\mu$m)', ha='center',size=20)


	xposition = [0,150]
	for xc in xposition:
	    axs[0].axvline(x=xc, color='r', linestyle='--',lw=3)
	    axs[1].axvline(x=xc, color='r', linestyle='--',lw=3)

	axs[1].plot(df_cm.x*1000,df_cm.sr_avg/1000,color='black',linewidth=2)
	axs[1].set_xlim([-20,170])
	axs[1].set_ylim([0,400])

	axs[1].set_ylabel('shear rate (1/s)',size=20)

	plt.show()


	axs[0].axvspan(-20, 15, alpha=0.1, color='red')
	axs[1].axvspan(-20, 15, alpha=0.1, color='red')

	axs[0].axvspan(15, 75, alpha=0.1, color='green')
	axs[1].axvspan(15, 75, alpha=0.1, color='green')

	axs[0].axvspan(75, 125, alpha=0.1, color='yellow')
	axs[1].axvspan(75, 125, alpha=0.1, color='yellow')

	axs[0].axvspan(125, 170, alpha=0.1, color='blue')
	axs[1].axvspan(125, 170, alpha=0.1, color='blue')

	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def feat_x(df,output_path,cell,y_label,feature='aspect',save=False):
	
	axis_size = 18
	for i,row in df[df.cell==cell].iterrows():
	    plt.scatter(row.xcm_um,row[feature],color='black',alpha=0.2,s=16)
	xposition = [0,150]
	for xc in xposition:
	    plt.axvline(x=xc, color='r', linestyle='--',lw=3)
	    
	plt.xlabel('x$_{c}$ ($\mu$m)',size=axis_size)
	plt.ylabel(y_label,size=axis_size)

	plt.tick_params(labelsize=14)
	plt.xlim((-25,175))

	plt.ylim((.6,2))
	plt.tight_layout()

	if save:
		plt.imsave(output_path)
	else:
		plt.show()

def feat_x_sub(df,output_path,y_label,feature='aspect',trend=False,save=False):

	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=set_size(469.7,fraction=1,hratio=1.4,subplots=(1,2)))

	axis_size = 12
	for i,row in df[df.cell=='hl60'].iterrows():
		ax1.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	for i,row in df[df.cell=='hl60d'].iterrows():
	    ax2.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	
	if trend == True:

		trend1_hl60 = np.mean(pd.DataFrame(df[df.cell=='hl60'].x_poly1.to_list(), columns=['slope', 'yint']),axis=0)
		trend2_hl60 = np.mean(pd.DataFrame(df[df.cell=='hl60'].x_poly2.to_list(), columns=['slope', 'yint']),axis=0)

		trendpoly1_hl60 = np.poly1d(trend1_hl60) 
		x = range(-50,200)
		ax1.plot(x,trendpoly1_hl60(x),label='Region 1',linestyle='--',lw=3)

		trendpoly2_hl60 = np.poly1d(trend2_hl60) 
		x = range(-50,200)
		ax1.plot(x,trendpoly2_hl60(x),label = 'Region 2',linestyle='--',lw=3)

		trend1_hl60d = np.mean(pd.DataFrame(df[df.cell=='hl60d'].x_poly1.to_list(), columns=['slope', 'yint']),axis=0)
		trend2_hl60d = np.mean(pd.DataFrame(df[df.cell=='hl60d'].x_poly2.to_list(), columns=['slope', 'yint']),axis=0)

		trendpoly1_hl60d = np.poly1d(trend1_hl60d) 
		x = range(-50,200)
		ax2.plot(x,trendpoly1_hl60d(x),label = 'Region 1',linestyle='--',lw=3)

		trendpoly2_hl60d = np.poly1d(trend2_hl60d) 
		x = range(-50,200)
		ax2.plot(x,trendpoly2_hl60d(x),label = 'Region 2',linestyle='--',lw=3)

		ax1.legend()
		ax2.legend()

	xposition = [0,150]
	for xc in xposition:
	    ax1.axvline(x=xc, color='r', linestyle='--',lw=2)
	    ax2.axvline(x=xc, color='r', linestyle='--',lw=2)
	    
	ax1.set_xlabel('X postion ($\mu$m)',size=axis_size)
	ax1.set_ylabel(y_label,size=axis_size)

	ax2.set_xlabel('X postion ($\mu$m)',size=axis_size)	

	ax1.set_xlim((-25,175))
	ax2.set_xlim((-25,175))

	ax1.xaxis.set_tick_params(labelsize=12)
	ax1.yaxis.set_tick_params(labelsize=12)

	ax2.xaxis.set_tick_params(labelsize=12)
	ax2.yaxis.set_tick_params(labelsize=12)
	ax1.set_ylim((.8,1.8))
	ax1.set_title('HL60', fontsize=12)
	ax2.set_title('HL60d', fontsize=12)

	ax1.text(-0.15, 1.15, 'a)', transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
	ax2.text(-0.05, 1.15, 'b)', transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

	"""
	img1 = plt.imread('./figures/figure2/ellipse.png')
	img2 = plt.imread('./figures/figure2/sphere.png')

	
	img1 = resize(img1, (47, 50))
	img2 = resize(img2, (47, 50))
	# [left, bottom, width, height]
	image_axis = fig.add_axes([0.1, 0.1, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img1)
	image_axis.axis('off')

	image_axis = fig.add_axes([0.1,-.425, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img2)
	image_axis.axis('off')
	"""
	plt.tight_layout()

	if save:
		plt.savefig(output_path,dpi=600)
	else:
		plt.show()

def feat_x_sub_fit(df,output_path,y_label,feature='aspect',trend=False,save=False):

	#fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=set_size(469.7,fraction=1,hratio=1.4,subplots=(1,2)))
	fig = plt.figure(figsize=set_size(469.7,fraction=1,hratio=2,subplots=(1,2)))
	ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
	ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1,sharey=ax1)
	ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
	axis_size = 12

	for i,row in df[df.cell=='hl60'].iterrows():
		ax1.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	for i,row in df[df.cell=='hl60d'].iterrows():
	    ax2.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	
	row = df[df.idx==1445].iloc[0]

	ax3.scatter(row.xcm_um,row.aspect,color='black',s=32)
	ax3.set_xlim([-20,170])
	ax3.set_ylim([0.9,1.7])

	ax3.set_ylabel('AR',size=12)

	ax3.xaxis.set_tick_params(labelsize=12)
	ax3.yaxis.set_tick_params(labelsize=12)




	x = range(-20,170)
	m1 = row.r1_slope
	b1 = row.r1_int
	poly1 = np.poly1d([m1,b1])

	m2 = row.r2_slope
	b2 = row.r2_int
	poly2 = np.poly1d([m2,b2])

	ax3.plot(x,poly1(x),label = 'R2 Slope',linestyle='--')
	ax3.plot(x,poly2(x),label = 'R3 Slope',linestyle='--')

	x1 = row.xcm_um[np.argmax(row.aspect)]
	y1 = 1.0
	x2 = x1
	y2 = row.nar1_asp
	ax3.plot((x1,x2),(y1,y2),'k-')
	linewidth = 2
	ax3.plot((x1-linewidth,x2+linewidth),(y1,y1),'k-')
	ax3.plot((x1-linewidth,x2+linewidth),(y2,y2),'k-')

	ax3.legend(loc='upper right')
	#ax3.text(x1, (row.nar1_asp+1)/2, 'Begin text', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
	ax3.text(x1+2, (row.nar1_asp+1)/2, 'AR1', horizontalalignment='left', 
		verticalalignment='center',fontsize=12,family='Arial')

	x1 = row.xcm_um[np.where(row.aspect==row.nar2_asp)][0]
	y1 = 1.0
	x2 = x1
	y2 = row.nar2_asp
	ax3.plot((x1,x2),(y1,y2),'k-')
	linewidth = 2
	ax3.plot((x1-linewidth,x2+linewidth),(y1,y1),'k-')
	ax3.plot((x1-linewidth,x2+linewidth),(y2,y2),'k-')

	ax3.text(x1+2, (row.nar2_asp+1)/2, 'AR2', horizontalalignment='left', 
		verticalalignment='center',fontsize=12,family='Arial')
	ax3.set_xlabel('X position ($\mu$m)',size=12)
	xposition = [0,150]
	for xc in xposition:
		ax1.axvline(x=xc, color='r', linestyle='--',lw=2)
		ax2.axvline(x=xc, color='r', linestyle='--',lw=2)
		ax3.axvline(x=xc, color='r', linestyle='--',lw=2)

	ax1.set_xlabel('X position ($\mu$m)',size=axis_size)
	ax1.set_ylabel(y_label,size=axis_size)

	ax2.set_xlabel('X position ($\mu$m)',size=axis_size)	

	ax1.set_xlim((-25,175))
	ax2.set_xlim((-25,175))

	ax1.xaxis.set_tick_params(labelsize=12)
	ax1.yaxis.set_tick_params(labelsize=12)

	ax2.xaxis.set_tick_params(labelsize=12)
	ax2.yaxis.set_tick_params(labelsize=12)
	ax1.set_ylim((.8,1.8))
	ax1.set_title('HL60', fontsize=12)
	ax2.set_title('HL60d', fontsize=12)

	ax1.text(-0.05, 1.15, 'a)', transform=ax1.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
	ax2.text(-0.05, 1.15, 'b)', transform=ax2.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
	ax3.text(-0.025, 1.15, 'c)', transform=ax3.transAxes,fontsize=16, fontweight='bold', va='top', ha='right')
	
	ax3.scatter(row.xcm_um,row.aspect,color='black',s=32)
	ax3.set_xlim([-20,170])
	ax3.set_ylim([0.9,1.7])

	ax3.set_ylabel('AR',size=12)

	ax3.xaxis.set_tick_params(labelsize=12)
	ax3.yaxis.set_tick_params(labelsize=12)
	"""
	img1 = plt.imread('./figures/figure2/ellipse.png')
	img2 = plt.imread('./figures/figure2/sphere.png')

	
	img1 = resize(img1, (47, 50))
	img2 = resize(img2, (47, 50))
	# [left, bottom, width, height]
	image_axis = fig.add_axes([0.1, 0.1, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img1)
	image_axis.axis('off')

	image_axis = fig.add_axes([0.1,-.425, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img2)
	image_axis.axis('off')
	"""
	plt.tight_layout()

	if save:
		plt.savefig(output_path,dpi=600)
	else:
		plt.show()

def feat_x_sub_fit_font(df,output_path,y_label,feature='aspect',trend=False,save=False):
	
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

	#fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=set_size(469.7,fraction=1,hratio=1.4,subplots=(1,2)))
	fig = plt.figure(figsize=set_size(469.7,fraction=1,hratio=2,subplots=(1,2)))
	ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
	ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1,sharey=ax1)
	ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
	axis_size = 12

	for i,row in df[df.cell=='hl60'].iterrows():
		ax1.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	for i,row in df[df.cell=='hl60d'].iterrows():
	    ax2.scatter(row.xcm_um,row[feature],color='black',alpha=0.15,s=1)
	
	row = df[df.idx==1445].iloc[0]

	ax3.scatter(row.xcm_um,row.aspect,color='black',s=32)
	ax3.set_xlim([-20,170])
	ax3.set_ylim([0.9,1.7])

	ax3.set_ylabel('AR')

	x = range(-20,170)
	m1 = row.r1_slope
	b1 = row.r1_int
	poly1 = np.poly1d([m1,b1])
	print(m1)
	m2 = row.r2_slope
	b2 = row.r2_int
	poly2 = np.poly1d([m2,b2])
	print(m2)
	ax3.plot(x,poly1(x),label = 'R2 Slope',linestyle='--')
	ax3.plot(x,poly2(x),label = 'R3 Slope',linestyle='--')

	x1 = row.xcm_um[np.argmax(row.aspect)]
	y1 = 1.0
	x2 = x1
	y2 = row.nar1_asp
	ax3.plot((x1,x2),(y1,y2),'k-')
	linewidth = 2
	ax3.plot((x1-linewidth,x2+linewidth),(y1,y1),'k-')
	ax3.plot((x1-linewidth,x2+linewidth),(y2,y2),'k-')

	ax3.legend(loc='upper right')
	#ax3.text(x1, (row.nar1_asp+1)/2, 'Begin text', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
	ax3.text(x1+2, (row.nar1_asp+1)/2, 'AR1', horizontalalignment='left', 
		verticalalignment='center')

	x1 = row.xcm_um[np.where(row.aspect==row.nar2_asp)][0]
	y1 = 1.0
	x2 = x1
	y2 = row.nar2_asp
	ax3.plot((x1,x2),(y1,y2),'k-')
	linewidth = 2
	ax3.plot((x1-linewidth,x2+linewidth),(y1,y1),'k-')
	ax3.plot((x1-linewidth,x2+linewidth),(y2,y2),'k-')

	ax3.text(x1+2, (row.nar2_asp+1)/2, 'AR2', horizontalalignment='left', 
		verticalalignment='center')
	ax3.set_xlabel('X position ($\mu$m)')
	xposition = [0,150]
	for xc in xposition:
		ax1.axvline(x=xc, color='r', linestyle='--',lw=2)
		ax2.axvline(x=xc, color='r', linestyle='--',lw=2)
		ax3.axvline(x=xc, color='r', linestyle='--',lw=2)

	ax1.set_xlabel('X position ($\mu$m)')
	ax1.set_ylabel(y_label)

	ax2.set_xlabel('X position ($\mu$m)')	

	ax1.set_xlim((-25,175))
	ax2.set_xlim((-25,175))

	ax1.set_ylim((.8,1.8))
	ax1.set_title('HL60',size=MEDIUM_SIZE)
	ax2.set_title('HL60d',size=MEDIUM_SIZE)

	ax1.text(-0.05, 1.25, 'a)', transform=ax1.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	ax2.text(-0.05, 1.25, 'b)', transform=ax2.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	ax3.text(-0.025, 1.15, 'c)', transform=ax3.transAxes,fontsize=BIGGER_SIZE, fontweight='bold', va='top', ha='right')
	
	ax3.scatter(row.xcm_um,row.aspect,color='black',s=32)
	ax3.set_xlim([-20,170])
	ax3.set_ylim([0.9,1.7])

	ax3.set_ylabel('AR')

	"""
	img1 = plt.imread('./figures/figure2/ellipse.png')
	img2 = plt.imread('./figures/figure2/sphere.png')

	
	img1 = resize(img1, (47, 50))
	img2 = resize(img2, (47, 50))
	# [left, bottom, width, height]
	image_axis = fig.add_axes([0.1, 0.1, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img1)
	image_axis.axis('off')

	image_axis = fig.add_axes([0.1,-.425, 0.04, 0.85], zorder=10, anchor="N")
	image_axis.imshow(img2)
	image_axis.axis('off')
	"""
	plt.tight_layout()

	if save:
		plt.savefig(output_path,dpi=600)
	else:
		plt.show()


def pop_means_sem(df,y_label,feat,output_path='D://',save=False):
	
	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'

	x = [1,2]
	y = [df[hlidx][feat].mean(),df[hldidx][feat].mean()]
	e = [stats.sem(df[feat][hlidx]),stats.sem(df[feat][hldidx])]

	temp_df = pd.DataFrame({'cell' : x, 'aspect' : y, 'sd_err' : e})

	g = sns.FacetGrid(data=temp_df, aspect=1, height=6)        
	g.map(plt.errorbar, "cell", "aspect", "sd_err", marker="o",linestyle='',markersize='6', capsize=4, elinewidth=2)


	# statistical annotation
	#x1, x2 = 1, 2   
	#y1, h, col = 1.45, .01, 'k'
	#plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=1.5, c=col)
	#plt.text((x1+x2)*.5, y1+h, "***", ha='center', va='bottom', color=col)


	# place the ticks at center by widening the plot
	plt.xlim((0, 3))
	# fix ticks at the number encoding for each class
	g.fig.axes[0].xaxis.set_ticks([1, 2])
	# name the numbers
	g.fig.axes[0].xaxis.set_ticklabels(['hl60', 'hl60d'])
	plt.ylabel(y_label,size=20)
	plt.xlabel('Cell',size=20)

	#plt.ylim(1.2,1.5)
	#plt.show()
	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def temp_sem(df,y_label,feat,output_path='D://',save=False):
	
	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'

	x = ['HL 60','HL 60d']
	y = [df[hlidx][feat].mean(),df[hldidx][feat].mean()]
	e = [stats.sem(df[feat][hlidx]),stats.sem(df[feat][hldidx])]

	temp_df = pd.DataFrame({'cell' : x, 'aspect' : y, 'sd_err' : e})

	       
	plt.errorbar([1,2], y, e, marker="o",linestyle='',markersize='6', capsize=4)




	# place the ticks at center by widening the plot
	plt.xlim((0, 3))
	# fix ticks at the number encoding for each class
	plt.xticks([1,2],['hl60','hl60d'])
	# name the numbers
	
	plt.ylabel(y_label,size=20)
	plt.xlabel('Cell',size=20)

	#plt.ylim(1.2,1.5)
	#plt.show()
	if save:
		plt.savefig(output_path)
	else:
		plt.show()


def pop_violin(df,output_path, feat, save=False):

	ax = sns.violinplot(x="cell", y=feat, hue="cell",
	                data=df, palette="muted")
	plt.xlabel(fontsize=10)

	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def pop_hist(df,output_path,feat,save=False):

	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'


	sns.distplot(df[hlidx][feat],label = 'hl60')
	sns.distplot(df[hldidx][feat],label='hl60d')

	plt.legend()

	if save:
		plt.savefig(output_path)
	else:
		plt.show()
	

def heatmat_pop(df,xfeat,yfeat,ylabel,xlabel,output_path='D://',save=False):

	fig, (ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	x = df[df.cell=='hl60'][xfeat].to_numpy()
	y = df[df.cell=='hl60'][yfeat].to_numpy()

	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)

	ax1.scatter(x, y, c=z, s=100, edgecolor=None)
	ax1.set_xlabel(xlabel,fontsize=20)
	#axs[0].set_xlim([-20,170])
	#axs[0].set_ylim([0.8,1.6])

	ax1.set_ylabel(ylabel,fontsize=20)
	ax1.set_title('HL60', fontsize=20)

	x = df[df.cell=='hl60d'][xfeat].to_numpy()
	y = df[df.cell=='hl60d'][yfeat].to_numpy()

	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)

	ax2.scatter(x, y, c=z, s=100, edgecolor=None)
	ax2.set_xlabel(xlabel,fontsize=20)
	ax2.set_title('HL60d',fontsize=20)

	#ax2.set_ylim([.8,2.2])
	ax2.set_xlim([5,8])


	ax1.annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16,
	            xytext=(-5, 5), textcoords='offset points',
	            ha='right', va='bottom')

	ax2.annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16,
	            xytext=(-5, 5), textcoords='offset points',
	            ha='right', va='bottom')

	if save:
		plt.imsave(output_path)
	else:
		plt.show()

def heatmap_pop_full(df,xfeat,yfeat,ylabel,xlabel,output_path='D://',save=False):

	#fig, ax = plt.subplots(nrows=len(yfeat), ncols=3,sharex='none',sharey='row',figsize=(20,30))

	fig,ax = plt.subplots(4,3,figsize=set_size(469.7,fraction=1,hratio=1.5,subplots=(4,3)))

	label = ['a)','b)','c)','d)']
	for row in range(len(yfeat)):

		#ax[row,0] = fig.add_subplot(len(yfeat),3,(1+row*3))
		#ax[row,1] = fig.add_subplot(len(yfeat),3,(2+row*3) ,sharey = ax1)
		#ax[row,2] = fig.add_subplot(len(yfeat),3,(3+row*3))

		ax[row,0].sharey(ax[row,1])

		x = df[df.cell=='hl60'][xfeat].to_numpy()
		y = df[df.cell=='hl60'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax[row,0].scatter(x, y, c=z, s=10, edgecolor=None,cmap='jet')
		#ax1.set_xlabel(xlabel)
		
		
		ax[row,0].set_ylabel(ylabel[row],fontsize=12)
		ax[row,0].text(-0.3, 1.15, label[row], transform=ax[row,0].transAxes,fontsize=16, fontweight='bold', va='top', ha='right')

		x = df[df.cell=='hl60d'][xfeat].to_numpy()
		y = df[df.cell=='hl60d'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax[row,1].scatter(x, y, c=z, s=10, edgecolor=None,cmap='jet')
		#ax2.set_xlabel(xlabel)
		#ax2.set_ylabel(ylabel,fontsize=20)


		if (row==2):
			ax[row,0].annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .1), xycoords='axes fraction', fontsize=12,
				xytext=(-5, 5), 
				textcoords='offset points',ha='right', va='bottom')
			ax[row,1].annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .1), xycoords='axes fraction', fontsize=12, 
		    	xytext=(-5, 5), textcoords='offset points',
		    	ha='right', va='bottom')
		else:
			ax[row,0].annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .8), xycoords='axes fraction', fontsize=12, 
				xytext=(-5, 5), textcoords='offset points',ha='right', va='bottom')
			ax[row,1].annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .8), xycoords='axes fraction', fontsize=12, 
				xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
		if (row == len(yfeat)-1):

			ax[row,0].tick_params(axis = 'x', which = 'major', labelsize = 12)
			ax[row,1].tick_params(axis = 'x', which = 'major', labelsize = 12)
			ax[row,2].tick_params(axis = 'x', which = 'major', labelsize = 12)

		else:
			ax[row,0].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)
			ax[row,1].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)
			ax[row,2].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)

		ax[row,0].tick_params(axis = 'y', which = 'major', labelsize = 12)
		ax[row,1].tick_params(axis = 'y', which = 'major', labelsize = 12)
		ax[row,2].tick_params(axis = 'y', which = 'major', labelsize = 12)

		hlidx = df.cell == 'hl60'
		hldidx = df.cell == 'hl60d'

		x = ['HL60','HL60d']
		y = [df[hlidx][yfeat[row]].mean(),df[hldidx][yfeat[row]].mean()]
		if (row==1)|(row==0):
			e = [np.round(stats.sem(df[hlidx][yfeat[row]]),2),np.round(stats.sem(df[hldidx][yfeat[row]]),2)]
		else:
			e = [np.round(stats.sem(df[hlidx][yfeat[row]]),5),np.round(stats.sem(df[hldidx][yfeat[row]]),5)]

		ax[row,2].errorbar([1,3], y, e, marker="o",linestyle='',markersize='6', capsize=4)

		# place the ticks at center by widening the plot
		ax[row,2].set_xlim((0, 4))
		# fix ticks at the number encoding for each class
		ax[row,2].set_xticks([1,3])
		ax[row,2].set_xticklabels(['HL60','HL60d'],fontsize=12)
		# name the numbers
		#ax[row,2].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

		#ax3.set_ylabel(y_label,size=20)
		#ax3.set_xlabel('Cell',fontsize=20)
		if (row == len(yfeat)-1)|(row == len(yfeat)-2):
			ax[row,0].ticklabel_format(axis='y', style='sci',scilimits=(0,0))
			ax[row,0].yaxis.offsetText.set_fontsize(10)
			ax[row,1].yaxis.offsetText.set_fontsize(10)
			ax[row,2].yaxis.offsetText.set_fontsize(10)
			ax[row,2].ticklabel_format(axis='y', style='sci',scilimits=(0,0))

		class ScalarFormatterForceFormat(ScalarFormatter):
			def _set_format(self):  # Override function that finds format to use.
				self.format = "%1.2f"  # Give format here

		yfmt = ScalarFormatterForceFormat()
		yfmt.set_powerlimits((0,0))
		ax[row,2].yaxis.set_major_formatter(yfmt)


		ax[row,0].yaxis.set_major_locator(plt.MaxNLocator(5))
		ax[row,2].yaxis.set_major_locator(plt.MaxNLocator(5))
		if row==0:
			ax[row,0].set_title('HL60', fontsize=12)
			ax[row,1].set_title('HL60d', fontsize=12)

		if row == len(yfeat)-1:
			ax[row,1].set_xlabel(xlabel,fontsize=12)
			ax[row,0].set_xlabel(xlabel,fontsize=12)
			ax[row,0].set_ylim((-0.005,.025))

		

	fig.tight_layout()
	#plt.xticks(fontsize=20)
	if save:
		plt.savefig(output_path,dpi=600)
	else:
		plt.show()

def heatmap_pop_full_font(df,xfeat,yfeat,ylabel,xlabel,output_path='D://',save=False):


	font = {'family' : 'Arial'}

	SMALL_SIZE = 8
	MEDIUM_SIZE = 10
	BIGGER_SIZE = 12

	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)
	matplotlib.rc('font', **font)

	#fig, ax = plt.subplots(nrows=len(yfeat), ncols=3,sharex='none',sharey='row',figsize=(20,30))

	fig,ax = plt.subplots(4,3,figsize=set_size(469.7,fraction=1,hratio=1.5,subplots=(4,3)))

	label = ['a)','b)','c)','d)']
	for row in range(len(yfeat)):

		#ax[row,0] = fig.add_subplot(len(yfeat),3,(1+row*3))
		#ax[row,1] = fig.add_subplot(len(yfeat),3,(2+row*3) ,sharey = ax1)
		#ax[row,2] = fig.add_subplot(len(yfeat),3,(3+row*3))

		ax[row,0].sharey(ax[row,1])

		x = df[df.cell=='hl60'][xfeat].to_numpy()
		y = df[df.cell=='hl60'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax[row,0].scatter(x, y, c=z, s=10, edgecolor=None,cmap='jet')
		#ax1.set_xlabel(xlabel)
		
		
		ax[row,0].set_ylabel(ylabel[row])
		ax[row,0].text(-0.3, 1.15, label[row], transform=ax[row,0].transAxes, fontsize=12,fontweight='bold', va='top', ha='right')

		x = df[df.cell=='hl60d'][xfeat].to_numpy()
		y = df[df.cell=='hl60d'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax[row,1].scatter(x, y, c=z, s=10, edgecolor=None,cmap='jet')
		#ax2.set_xlabel(xlabel)
		#ax2.set_ylabel(ylabel,fontsize=20)


		if (row==2):
			ax[row,0].annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .1), xycoords='axes fraction', 
				xytext=(-5, 5), 
				textcoords='offset points',ha='right', va='bottom')
			ax[row,1].annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .1), xycoords='axes fraction', 
		    	xytext=(-5, 5), textcoords='offset points',
		    	ha='right', va='bottom')
		else:
			ax[row,0].annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .8), xycoords='axes fraction', 
				xytext=(-5, 5), textcoords='offset points',ha='right', va='bottom')
			ax[row,1].annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .8), xycoords='axes fraction',  
				xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
		if (row == len(yfeat)-1):

			ax[row,0].tick_params(axis = 'x', which = 'major')
			ax[row,1].tick_params(axis = 'x', which = 'major')
			ax[row,2].tick_params(axis = 'x', which = 'major')

		else:
			ax[row,0].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)
			ax[row,1].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)
			ax[row,2].tick_params(axis = 'x', which = 'both',bottom=False,labelbottom=False)

		ax[row,0].tick_params(axis = 'y', which = 'major')
		ax[row,1].tick_params(axis = 'y', which = 'major')
		ax[row,2].tick_params(axis = 'y', which = 'major')

		hlidx = df.cell == 'hl60'
		hldidx = df.cell == 'hl60d'

		x = ['HL60','HL60d']
		y = [df[hlidx][yfeat[row]].mean(),df[hldidx][yfeat[row]].mean()]
		if (row==1)|(row==0):
			e = [np.round(stats.sem(df[hlidx][yfeat[row]]),2),np.round(stats.sem(df[hldidx][yfeat[row]]),2)]
		else:
			e = [np.round(stats.sem(df[hlidx][yfeat[row]]),5),np.round(stats.sem(df[hldidx][yfeat[row]]),5)]

		ax[row,2].errorbar([1,3], y, e, marker="o",linestyle='',markersize='6', capsize=4)

		# place the ticks at center by widening the plot
		ax[row,2].set_xlim((0, 4))
		# fix ticks at the number encoding for each class
		ax[row,2].set_xticks([1,3])
		ax[row,2].set_xticklabels(['HL60','HL60d'],size=MEDIUM_SIZE)
		# name the numbers
		#ax[row,2].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

		#ax3.set_ylabel(y_label,size=20)
		#ax3.set_xlabel('Cell',fontsize=20)
		if (row == len(yfeat)-1)|(row == len(yfeat)-2):
			ax[row,0].ticklabel_format(axis='y', style='sci',scilimits=(0,0))
			ax[row,2].ticklabel_format(axis='y', style='sci',scilimits=(0,0))

		class ScalarFormatterForceFormat(ScalarFormatter):
			def _set_format(self):  # Override function that finds format to use.
				self.format = "%1.2f"  # Give format here

		yfmt = ScalarFormatterForceFormat()
		yfmt.set_powerlimits((0,0))
		ax[row,2].yaxis.set_major_formatter(yfmt)


		ax[row,0].yaxis.set_major_locator(plt.MaxNLocator(5))
		ax[row,2].yaxis.set_major_locator(plt.MaxNLocator(5))
		if row==0:
			ax[row,0].set_title('HL60')
			ax[row,1].set_title('HL60d')

		if row == len(yfeat)-1:
			ax[row,1].set_xlabel(xlabel)
			ax[row,0].set_xlabel(xlabel)
			ax[row,0].set_ylim((-0.005,.025))

		

	fig.tight_layout()
	#plt.xticks(fontsize=20)
	if save:
		plt.savefig(output_path,dpi=600)
	else:
		plt.show()
def Figure2(df,output_path='D://',save=False):


	fig, axes = plt.subplots(3,3,sharex=True,sharey=True,figsize=(20,10))
