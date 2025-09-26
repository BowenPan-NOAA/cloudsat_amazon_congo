import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.measurements import label
from collections import Counter

def define_cluster(VORT_new,threshold):
    #explicitly for first time step:
    #1. Find out vorticity regions above threshold continous vorticity values
    #2. Eliminate close to boundary values using main value
    #3. Store:
    #.     VORT_select-lon,lat,time,cluster-label
    #Find out the continous region:
    #   a. radar reflectivity > -28 dBZ (threshold)
    #   b. mark continuously region with certain labels
    #   c. return the number of clusters and labeled cloud field 
    s   = [[1,1,1],[1,1,1],[1,1,1]]
    Ny, Nx = VORT_new.shape
    tmp = np.zeros([Ny,Nx])
    tmp = np.where(VORT_new>threshold,1,0) 
    #label VORT that satisfy VORT threshold
    #num_VORT - number of patches/cluster
    labeled_VORT, num_VORT  = label(tmp,structure=s)
    label_VORT_flat=labeled_VORT.flatten()
    count_num = Counter(label_VORT_flat)
    return(labeled_VORT,count_num)

def anvil_cutoff(pn1,pn2,frz):
    #Find the minimum vertical index that pn1<0
    #It should not be 85...since the index range changes in our dataset
    #Temporary using freezing level:
    #Input:
    ## frz = freezing level index
    ## pn1 = first derivative
    ## pn2 = second derivative
    checkindx = frz #NEED TO CHANGE
    #Find where the first derivative less than 0
    first_idx_to_check  = np.nanmin(np.where(pn1<0)[0])
    #Select indices with pn2 >0
    indices  = np.where(pn2[first_idx_to_check:checkindx]>0)[0]+first_idx_to_check
    values_where_positive = pn2[indices]
    #Calculate K-cutoff height
    #print('Second derivative positive:',np.sum(values_where_positive))
    cutoff = np.nansum(indices*values_where_positive)/np.nansum(values_where_positive)
    return(cutoff)


def def_cutoff(refl,frz_idx,fig):
    #Find out the cutoff height
    #Establish the derivative for cutoff height
    #Modify from the matlab code by Aryeh
    #Input:
    ##    reflectivity of one cloud object
    ##    freezing level of the cloud object indx
    ##    fig=1, output figures
    #Return:
    ##    cutoff height of the cloud object
    WSZ = 7 #Window size for smoothing is 7
    #Find cloudy pixel, REFL>-28dBZ
    tmppixel = np.where(refl>-28,1,0)
    #Smooth out of the fuzz
    smot1 = smooth(np.nansum(tmppixel,axis=0),WSZ)
    smot2 = smooth(smot1,WSZ)
    
    #Find first differential
    diff1 = np.diff(smot2)
    d1smot1= smooth(diff1,WSZ)
    d1smot2= smooth(d1smot1,WSZ)
    
    #Find second derivative
    diff2smot1 = np.diff(d1smot1)
    d2smot1sm1= smooth(diff2smot1,WSZ)
    
    diff2smot0  = np.diff(diff1)
    d2smot0sm1 = smooth(diff2smot0,WSZ)
    
    cutoff1 = anvil_cutoff(d1smot2,d2smot1sm1,frz_idx)
    cutoff2 = anvil_cutoff(d1smot2,d2smot0sm1,frz_idx)
    cutoff3 = anvil_cutoff(d1smot2,diff2smot1,frz_idx)
    cutoff4 = anvil_cutoff(d1smot2,diff2smot0,frz_idx)
    cutoff  = (cutoff1+cutoff2+cutoff3+cutoff4)/4
    if fig==1:
        fig, ax1 = plt.subplots()
        plt.contourf(tmppixel,cmap='binary')
        plt.vlines(cutoff,ymin=0,ymax=len(tmppixel),color='r')
        plt.vlines(frz_idx,ymin=0,ymax=len(tmppixel),color='b')

        ax1.set_xlabel('Height index')
        ax1.set_ylabel('Pn1')
        ax1.plot(smot1,color='lime')
        ax1.plot(smot2,color='green')
                 
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Pn2', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(diff2smot1,color='r',linestyle='--')
        ax2.plot(d2smot1sm1,color='orange',linestyle='--')
        ax2.plot(d2smot0sm1,color='maroon',linestyle='--')
        ax2.plot(diff2smot0,color='magenta',linestyle='--')
        ax2.hlines(0,xmin=0,xmax=76)
    return(np.round(cutoff))

def smooth(a,WSZ):
    #https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-np-python
    #Moving average - tough must be odd number... 
    # a: np 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))