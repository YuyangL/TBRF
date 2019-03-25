#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:17:22 2018

@author: mikael
"""
from __future__ import division
import numpy as np

from scipy import stats
import scipy.interpolate as interp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator


import cPickle
from tempfile import mkstemp
from shutil import move
from shutil import copy
from os import remove, close
import os


def plotFeatures(features, Nx, Ny, meshRANS,Re,trainTest,index_x=1,index_y=2):
    """
    Plot the features used for the machine learning algorithm
    """
    n_feat = features.shape[0]
    inv = features.reshape([n_feat,Nx,Ny])
    indices = np.linspace(0,n_feat-1,n_feat,dtype='int')
            
    n_rows = int(np.ceil(np.float(n_feat)/np.float(3)))
    
    f, axarr = plt.subplots(n_rows, 3, sharex='col', sharey='row')
    for i in range(indices.shape[0]):
        row = int(indices[i]/3)
        col = int(indices[i]%3)
        index = indices[i]
        
        if i == 10 or i == 15:
            maxval = np.max([inv[index,:,:]])
            minval = np.min([inv[index,:,:]])
        elif i == 8 or i == 13 or i == 14:
            maxval = 0.01*np.std([np.abs(inv[index,:,:])])
            if maxval < 1e-12:
                maxval = 1e-12
            minval = -maxval
        elif i == 9:
            maxval = 0.1*np.std([np.abs(inv[index,:,:])])
            if maxval < 1e-12:
                maxval = 1e-12
            minval = -maxval
        else:
            maxval = 2*np.std([np.abs(inv[index,:,:])])
            if maxval < 1e-12:
                maxval = 1e-12
            minval = -maxval
        contour_levels = np.linspace(1.0*minval, 1.0*maxval, 50)
#        contour_levels = np.linspace(minval,maxval, 50)
        cmap=plt.cm.coolwarm
        cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
        cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])
    
        contPlot = axarr[row,col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], inv[index,:,:],contour_levels,cmap=cmap,extend="both")
        div = make_axes_locatable(axarr[row,col])
        cax = div.append_axes("right", size="20%", pad=0.05)
        cax.set_visible(False)
        axarr[row,col].set_title('Feature %i' % (index+1), fontsize=10)
        cbar = plt.colorbar(contPlot)
        cbar.ax.tick_params(labelsize=8) 
        plt.suptitle('%s data, Re = %i' % (trainTest, Re))
#    plt.tight_layout()
    plt.show()

def plotFeatures2(features, Nx, Ny, meshRANS,Re,trainTest,index_x=1,index_y=2):
    """
    Plot the features used for the machine learning algorithm
    """
    n_feat = features.shape[0]
    inv = features.reshape([n_feat,Nx,Ny])
    indices = np.linspace(0,n_feat-1,n_feat,dtype='int')
    labels = ['FS1,1','FS1,2','FS1,5','FS2,1','FS2,2','FS2,3','FS2,9','FS2,13','FS3,1','FS3,2','FS3,3','FS3,4','FS3,5','FS3,6','FS3,7','FS3,8','FS3,9']
    
    n_rows = int(np.ceil(np.float(n_feat)/np.float(3)))
    
    f, axarr = plt.subplots(n_rows, 3, sharex='col', sharey='row')
    for i in range(indices.shape[0]):
        row = int(indices[i]/3)
        col = int(indices[i]%3)
        index = indices[i]
#        
#        if i == 10 or i == 15:
#            maxval = np.max([inv[index,:,:]])
#            minval = np.min([inv[index,:,:]])
#        elif i == 8 or i == 13 or i == 14:
#            maxval = 0.01*np.std([np.abs(inv[index,:,:])])
#            if maxval < 1e-12:
#                maxval = 1e-12
#            minval = -maxval
#        elif i == 9:
#            maxval = 0.1*np.std([np.abs(inv[index,:,:])])
#            if maxval < 1e-12:
#                maxval = 1e-12
#            minval = -maxval
#        else:
#            maxval = 2*np.std([np.abs(inv[index,:,:])])
#            if maxval < 1e-12:
#                maxval = 1e-12
#            minval = -maxval
        maxval = np.max([inv[index,:,:]])
        minval = np.min([inv[index,:,:]])
        
        if maxval < 1e-12:
            maxval = 1e-12
        minval = -maxval        
        
        contour_levels = np.linspace(1.0*minval, 1.0*maxval, 50)
#        contour_levels = np.linspace(minval,maxval, 50)
        cmap=plt.cm.coolwarm
        cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
        cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])
    
        contPlot = axarr[row,col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], inv[index,:,:],contour_levels,cmap=cmap,extend="both")
        div = make_axes_locatable(axarr[row,col])
        cax = div.append_axes("right", size="20%", pad=0.05)
        cax.set_visible(False)
        axarr[row,col].set_title('Feature %s' % (labels[i]), fontsize=10)
        cbar = plt.colorbar(contPlot)
        cbar.ax.tick_params(labelsize=8) 
        cbar.locator = MaxNLocator( nbins = 3)
        cbar.update_ticks()
        plt.suptitle('%s data, Re = %i' % (trainTest, Re))
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
#    plt.tight_layout()
    plt.show()

def plotAnisotopy(bij, Nx, Ny, meshRANS,Re,trainTest,index_x=1,index_y=2,tickssize = 14,titlesize=16,labelsize=14):
    """
    Plot the anisotropy tensor components
    """
    
    bij_mesh = bij.reshape([3,3,Nx,Ny])
    plt_indices = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]])
    
    n_rows = 2
    
    f, axarr = plt.subplots(n_rows, 3, sharex='col', sharey='row')
    
    for i in range(plt_indices.shape[0]):
        row = int(i/3)
        col = int(i%3)
        index = plt_indices[i]
        
        maxval = np.max(np.abs([bij_mesh[plt_indices[i,0],plt_indices[i,1],:,:]]))
        
#        if i == 3:
#            maxval = 0.33
#        minval = np.min([bij_mesh[plt_indices[i,0],plt_indices[i,1],:,:]])
#        maxval = 2*np.std([np.abs(bij_mesh[index[0],index[1],:,:])])
        if maxval < 1e-12:
            maxval = 1e-12
#        contour_levels = np.linspace(-1.05*maxval, 1.05*maxval, 50)
        contour_levels = np.linspace(-maxval,maxval, 50)
        cmap=plt.cm.coolwarm
        cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
        cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])
    
        contPlot = axarr[row,col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], bij_mesh[index[0],index[1],:,:],contour_levels,cmap=cmap,extend="both")
        div = make_axes_locatable(axarr[row,col])
        cax = div.append_axes("right", size="20%", pad=0.05)
        cax.set_visible(False)
        axarr[row,col].set_title('$b_{%i%i}$' % (index[0]+1, index[1]+1))
        cbar = plt.colorbar(contPlot)
        cbar.ax.tick_params(labelsize=8) 
        cbar.locator = MaxNLocator( nbins = 5)
        cbar.update_ticks()
    plt.suptitle('Anisotropy Tensor')
        
#    plt.xticks(fontsize=tickssize)
#    plt.yticks(fontsize=tickssize)    
    plt.show()
    
def plotBaryMap(baryMap,title='Barycentric map'):
    """
    Plot barycentric map locations
    """
    plt.figure()
    plt.plot(baryMap[0,:,:],baryMap[1,:,:],'b*')
    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
    plt.axis('equal')
    plt.title(title)
    plt.show()
    
def plotMahalanobisDistance(X_training,X_test,meshRANS,index_x=1,index_y=2):
    """
    Calculate and plot the Mahalanobis distance of the test data as a function
    of the input training data
    """
    # normal Mahalanobis distance:
    D_test = np.zeros(X_test.shape[1])
    for i in range(X_test.shape[1]):
        D_test[i] = np.sqrt( np.dot( (X_test[:,i] - np.mean(X_training,axis=1)).T, np.dot(  np.cov(X_training),  (X_test[:,i] - np.mean(X_training,axis=1)))))
    
    # normalized version, see Wu et al. (2016): 'PML, A Priori Assessment of Prediction Confidence' 
    D_training = np.zeros(X_training.shape[1])
    for i in range(X_training.shape[1]):
        D_training[i] = np.sqrt( np.dot( (X_training[:,i] - np.mean(X_training,axis=1)).T, np.dot(  np.cov(X_training),  (X_training[:,i] - np.mean(X_training,axis=1)))))

    D_normalized = np.zeros(D_test.shape[0])
    for i in range(D_test.shape[0]):
        mask_larger = D_training > D_test[i]
        frac_larger = float(np.sum(mask_larger))/float(D_training.shape[0])
        D_normalized[i] = 1-frac_larger
   
    contour_levels = np.linspace(0, 1, 50)
    cmap=plt.cm.inferno
    cmap.set_over([0.98836199999999996, 0.99836400000000003, 0.64492400000000005, 1.0])
    cmap.set_under([0.001462, 0.000466, 0.013866, 1.0])  
    
    plt.figure()
    contPlot = plt.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], D_normalized.reshape([meshRANS.shape[1],meshRANS.shape[2]]),
                            contour_levels,cmap=cmap,extend="both")
    plt.title('Mahalanobis distance')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(contPlot)
    plt.show()
    
    return D_normalized

def plotKDEDistance(X_training,X_test,meshRANS,index_x=1,index_y=2,kernelwidth='scott'):
    """
    Calculate and plot the KDE distance of the test data as a function
    of the input training data
    """
    # create the kernel density estimate using the scipy stats package
    kernel = stats.gaussian_kde(X_training,kernelwidth)
    # calculate volume hypervolume covered by training data
    mins = X_training.min(axis=1)
    maxs = X_training.max(axis=1)
    A_unif = np.product(maxs-mins)
    #calculate D_kde, see Wu et al. (2016): 'PML, A Priori Assessment of Prediction Confidence' 
    D_kde = np.zeros(X_test.shape[1])
    for i in range(X_test.shape[1]):
        D_kde[i] = 1 - (kernel(X_test[:,i])/(kernel(X_test[:,i]) + 1/A_unif))
        
    contour_levels = np.linspace(min(D_kde), max(D_kde), 50)
    cmap=plt.cm.inferno
    cmap.set_over([0.98836199999999996, 0.99836400000000003, 0.64492400000000005, 1.0])
    cmap.set_under([0.001462, 0.000466, 0.013866, 1.0])
        
    plt.figure()
    contPlot = plt.contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], D_kde.reshape([meshRANS.shape[1],meshRANS.shape[2]]),contour_levels,cmap=cmap,extend="both")
    plt.title('KDE distance')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.colorbar(contPlot)
    plt.show()
    
    return D_kde


def plotTreePredictions(bij, bij_index, Nx, Ny, meshRANS,Re,index_x=1,index_y=2):
    """
    Plot individual predictions of decision trees, use in case of RF algorithm
    """
    index1 = (bij_index%3)+1
    index2 = (bij_index/3)+1
    b_xx = bij[bij_index,:,:]
    n_feat = b_xx.shape[1]
    
    b_plot = b_xx.reshape([Nx,Ny,n_feat])
    indices = np.linspace(0,n_feat-1,n_feat,dtype='int')
    
    n_rows = int(np.ceil(np.float(n_feat)/np.float(3)))
    
    maxval = 3*np.std([np.abs(b_plot[:,:,:])])
    f, axarr = plt.subplots(n_rows, 3, sharex='col', sharey='row')
    for i in range(indices.shape[0]):
        row = int(indices[i]/3)
        col = int(indices[i]%3)
        index = indices[i]
        
#        maxval = np.max([np.abs(inv[index,:,:])])

        if maxval < 1e-12:
            maxval = 1e-12
        contour_levels = np.linspace(-1.05*maxval, 1.05*maxval, 50)
        cmap=plt.cm.coolwarm
        cmap.set_over([0.70567315799999997, 0.015556159999999999, 0.15023281199999999, 1.0])
        cmap.set_under([0.2298057, 0.298717966, 0.75368315299999999, 1.0])
    
#        print(b_plot.shape)
#        print(axarr.shape)
#        print(n_rows)
        if len(axarr.shape) == 1:
            contPlot = axarr[col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], b_plot[:,:,index],contour_levels,cmap=cmap,extend="both")
            div = make_axes_locatable(axarr[col])
            axarr[col].set_title('Tree %i' % (index+1), fontsize=10)
        else:
            contPlot = axarr[row,col].contourf(meshRANS[index_x,:,:], meshRANS[index_y,:,:], b_plot[:,:,index],contour_levels,cmap=cmap,extend="both")
            div = make_axes_locatable(axarr[row,col])
            axarr[row,col].set_title('Tree %i' % (index+1), fontsize=10)
        cax = div.append_axes("right", size="20%", pad=0.05)
        cax.set_visible(False)
#        axarr[row,col].set_title('Tree %i' % (index+1), fontsize=10)
        cbar = plt.colorbar(contPlot)
        cbar.ax.tick_params(labelsize=8) 
        plt.suptitle('Predictions individual decision trees, $b_{%i%i}$, Re = %i' % (index1,index2,Re))
    plt.show()

def copyAndTranspose(Tensor):
    """
     I had to create this function, as there is something strange going on with
     the strides of the OpenFOAM dicts (e.g. dataRANS_training['aij'].strides), which
     theano (used for the TBNN) does not like.
     Function copies and transposes the data such that the strides are correct 
     for the training/test data
    """
    if len(Tensor.shape) == 3:
        output = np.zeros(np.shape(np.transpose(Tensor)))
        for i1 in range(Tensor.shape[0]):
            for i2 in range(Tensor.shape[2]):
                output[i2,:,i1] = Tensor[i1,:,i2]
    elif len(Tensor.shape) == 2:
        output = np.zeros(np.shape(np.transpose(Tensor)))
        for i1 in range(Tensor.shape[0]):
            for i2 in range(Tensor.shape[1]):
                output[i2,i1] = Tensor[i1,i2]
    return output



def writeSymmTensorField(fieldName,Tensor,home,flowCase,Re,turbModel,nx_RANS,ny_RANS,time_end,suffix):
    """
    write the calculated Reynolds stress field (y_predict) to OpenFOAM format file
    prerequisites: a file named 'R' or 'turbulenceProperties:R' in the time_end directory, which will be used 
    as a template for the adjusted reynolds stress
    """
    if flowCase == 'PeriodicHills' or flowCase == 'SquareDuct':
        case  = home + flowCase + '/' + ('Re%i_%s_%i' % (Re,turbModel,nx_RANS))
    else:
        case  = home + flowCase + '/' + ('Re%i_%s_%i' % (Re,turbModel,ny_RANS))
    time = time_end
    var = fieldName + suffix
    

    tau_0 = np.swapaxes(Tensor[0,0,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_1 = np.swapaxes(Tensor[0,1,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_2 = np.swapaxes(Tensor[0,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_3 = np.swapaxes(Tensor[1,1,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_4 = np.swapaxes(Tensor[1,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    tau_5 = np.swapaxes(Tensor[2,2,:,:],0,1).reshape(nx_RANS*ny_RANS)
    
    # if R exists, back it up, otherwise use R as template for SymmTensor
    if os.path.exists(case + '/' + str(time) + '/' + var) == True:
        copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
    else:
        copy(case + '/' + str(time) + '/' + 'R', case + '/' + str(time) + '/' + var)
    
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
    print (file_path)
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as file:
            for i,line in enumerate(file):  
                if 'object' in line:
                    new_file.write('    object      %s;\n' % fieldName)
                elif cc==False and 'internalField' not in line:
                    new_file.write(line)
                
                elif 'internalField' in line:
                    tmp = i + 1
                    tmp2 = i + 3
                    cc = True
                    new_file.write(line)
                    print(tmp, tmp2)
                    
        
                elif i==tmp:
                    print (line.split())
                    maxLines = int(line.split()[0])
                    maxIter  = tmp2 + maxLines
                    new_file.write(line)
                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    #print line
                    new_file.write('(' + str(tau_0[j]) + ' ' +  str(tau_1[j]) + ' ' + str(tau_2[j]) + ' ' +  str(tau_3[j]) + ' ' +  str(tau_4[j]) + ' ' +  str(tau_5[j]) +') \n'  )
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)

def writeScalarField(fieldName,Scalar,home,flowCase,Re,turbModel,nx_RANS,ny_RANS,time_end,suffix):
   
    if flowCase == 'PeriodicHills' or flowCase == 'SquareDuct':
        case  = home + flowCase + '/' + ('Re%i_%s_%i' % (Re,turbModel,nx_RANS))
    else:
        case  = home + flowCase + '/' + ('Re%i_%s_%i' % (Re,turbModel,ny_RANS))
    time = time_end
    var = fieldName + suffix
    
    time = time_end
    data = Scalar
    
    # if pd exists, back it up, otherwise use p as template for pd
    if os.path.exists(case + '/' + str(time) + '/' + var) == True:
        copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
    else:
        copy(case + '/' + str(time) + '/' + 'p', case + '/' + str(time) + '/' + var)
    
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
    print (file_path)
    fh, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as file:
            for i,line in enumerate(file):  
                if 'object' in line:
                    new_file.write('    object      p;\n')
                elif cc==False and 'internalField' not in line:
                    new_file.write(line)
                
                elif 'internalField' in line:
                    tmp = i + 1
                    tmp2 = i + 3
                    cc = True
                    new_file.write(line)
                    print (tmp, tmp2)
                    
        
                elif i==tmp:
                    print (line.split())
                    maxLines = int(line.split()[0])
                    maxIter  = tmp2 + maxLines
                    new_file.write(line)
                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    new_file.write( str(data[j]) + ' \n'  )            
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)
    
    

def writeTensorData(suffix,tensor):
    tensor = np.swapaxes(tensor,2,3)
    tensor = tensor.reshape([3,3,tensor.shape[2]*tensor.shape[3]])
    
    fileName = './bij_BFS_ML_TBRF_' + suffix + '.dat'
    with open(fileName,'w') as newFile:
        for i in range(tensor.shape[2]):
            newFile.write('(' + str(tensor[0,0,i]) + ' ' +  str(tensor[0,1,i]) + ' ' + str(0*tensor[0,0,i]) + ' ' +  str(tensor[1,1,i]) + ' ' +  str(0*tensor[0,0,i]) + ' ' +  str(tensor[2,2,i]) +') \n'  )
        
def plotStressType(X,Y,nx,ny,C1c,C2c,C3c,title='Stress type in the flow domain',default='white',tickssize = 14,titlesize=16,labelsize=14):
    """
    Plotting stress types from barycentric map in RGB
    X,Y: meshgrid of the flow field
    nx: amount of points to sample in x-dir
    ny: amount of points to sample in y-dir
    C1c,C2c,C3c: arrays which contain the barycentric map coordinates, corresponding
                 to the meshgrid X,Y
    """
    print('Plotting stress types from barycentric map in RGB...')
    
    clrs = np.vstack((C1c.flatten(),C2c.flatten(),C3c.flatten())).T
    clrs[clrs<0] = 0 #cap the values in case they are not realizable
    clrs[clrs>1] = 1
    
    x_min = X.min()
    x_max = X.max()
    y_min = Y.min()
    y_max = Y.max()
    X_n,Y_n = np.meshgrid(np.linspace(x_min, x_max, nx),np.linspace(y_min, y_max, ny))

    # interpolate colors onto regular grid
    clrs_n = np.zeros([X_n.shape[0],X_n.shape[1],3])
    for i in range(3):
        clrs_n[:,:,i] = interp.griddata((X.flatten(),Y.flatten()),
              clrs[:,i],(X_n,Y_n), method='linear')
    
    f2 = interp.interp1d(X[:,0], Y[:,0], kind='linear',fill_value='extrapolate')
    for i1 in range(X_n.shape[0]):
        for i2 in range(X_n.shape[1]):
            
            if default == 'white':
                clrs_n[i1,i2,:] = clrs_n[i1,i2,:]/np.max(clrs_n[i1,i2,:])
                         
            if Y_n[i1,i2] < f2(X_n[i1,i2]):
                clrs_n[i1,i2,:] = np.ones(3)
            
    mask = np.isnan(clrs_n)
    clrs_n[mask] = 0.0
    plt.figure()
    plt.imshow(clrs_n,origin='lower', extent=[x_min,x_max,y_min,y_max])
    plt.xlabel('x-axis',size=labelsize)
    plt.ylabel('y-axis',size=labelsize)
    plt.xticks(fontsize=tickssize)
    plt.yticks(fontsize=tickssize)
    plt.title(title,size=titlesize)
    plt.show()    
def writeDataFile(flowCase,Re,Regressor,nFeat,turbModel,meshRANS,bijDNS,bijRANS,bijML,saveData,folder):
    """
    Write data file; with mesh and b_ij for later use 
    """
    
    fileName = './%s/%s_%s_%iFeat_%s' % (flowCase,Regressor,nFeat,turbModel)
    with open(fileName,'w') as newFile:
        
        newFile.write('------------Mesh------------\n')
        newFile.write('Nx: %i\n' % meshRANS.shape[1])
        newFile.write('Ny: %i\n' % meshRANS.shape[2])
        meshVars = ['x','y','z']
        for i in range(meshRANS.shape[0]):
            tmp_mesh = meshRANS[i,:,:].flatten()
            newFile.write('mesh: %s\n' % meshVars[i])
            for i2 in range(tmp_mesh.shape[0]):
                newFile.write(str(tmp_mesh[i2]) + '\n')
        
        newFile.write('------------bijDNS------------\n')
        bijVars = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]
        for i in range(len(bijVars)):
            tmp_bij = bijDNS[bijVars[i][0],bijVars[i][1],:,:].flatten()
            newFile.write('bij: %i %i\n' % (bijVars[i][0],bijVars[i][1]))
            for i2 in range(tmp_bij.shape[0]):
                newFile.write(str(tmp_bij[i2]) + '\n')
    
        newFile.write('------------bijRANS------------\n')
        bijVars = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]
        for i in range(len(bijVars)):
            tmp_bij = bijRANS[bijVars[i][0],bijVars[i][1],:,:].flatten()
            newFile.write('bij: %i %i\n' % (bijVars[i][0],bijVars[i][1]))
            for i2 in range(tmp_bij.shape[0]):
                newFile.write(str(tmp_bij[i2]) + '\n')
    
        newFile.write('------------bijML------------\n')

        bijVars = [[0,0],[0,1],[0,2],[1,1],[1,2],[2,2]]
        for i in range(len(bijVars)):
            tmp_bij = bijML[bijVars[i][0],bijVars[i][1],:,:].flatten()
            newFile.write('bij: %i %i\n' % (bijVars[i][0],bijVars[i][1]))
            for i2 in range(tmp_bij.shape[0]):
                newFile.write(str(tmp_bij[i2]) + '\n')    
    
    fileName_pickle = './DataFiles/%s%i_%s_%iFeat_%s.pickle' % (flowCase,Re,Regressor,nFeat,turbModel)
    cPickle.dump(saveData, open(fileName_pickle, 'wb'))
    print('data written to %s' % fileName_pickle)

def writeDataFile2(testParam,trainingParam,Regressor,nFeat,saveData,folder,suffix=''):
    """
    Write data file; with mesh and b_ij for later use 
    """

    flowCase = testParam['flowCase']
    Re = testParam['Re']
    turbModel = testParam['turbModel']
    
    trainingString = ''
    for i in range(len(trainingParam['flowCase'])):
        trainingString = trainingString + trainingParam['flowCase'][i] + str(trainingParam['Re'][i]) + '_' 
    
    fileName_pickle = './%s/%s%i_Train%s%s_%iFeat_%s%s.pickle' % (folder,flowCase,Re,trainingString,Regressor,nFeat,turbModel,suffix)
    cPickle.dump(saveData, open(fileName_pickle, 'wb'))
    print('data written to %s' % fileName_pickle)
                
    
def plotBarySection(meshRANS,baryMap,x_line,y_line,col='k*'):
#    x_line = np.array([0.05,0.98])
#    y_line = np.array([0.05,0.98])
    line_coord = np.array([np.linspace(x_line[0],x_line[1],50),np.linspace(y_line[0],y_line[1],50)]).T
    
    RANS_coord = np.array([np.reshape(meshRANS[0,:,:],[meshRANS.shape[1]*meshRANS.shape[2]]),np.reshape(meshRANS[1,:,:],[meshRANS.shape[1]*meshRANS.shape[2]])]).T
    #
    line_bary={}
    DATA =  np.reshape(baryMap[0],[meshRANS.shape[1]*meshRANS.shape[2]])
    line_bary['x_bary'] = interp.griddata(RANS_coord, DATA, line_coord, method='linear')
    DATA =  np.reshape(baryMap[1],[meshRANS.shape[1]*meshRANS.shape[2]])
    line_bary['y_bary'] = interp.griddata(RANS_coord, DATA, line_coord, method='linear')
    
#    plt.figure()
    plt.plot(line_bary['x_bary'],line_bary['y_bary'],col,markersize=6)
#    plt.plot(line_bary['x_RANS'],line_bary['y_RANS'],'b+-',label='RANS',markersize=6)    
#    plt.plot(line_bary['x_TBRF'],line_bary['y_TBRF'],'ro-',label='TBRF',markersize=6)
#    plt.plot(line_bary['x_TBNN'],line_bary['y_TBNN'],'gx-',label='TBNN',markersize=6)

    #plt.plot(line_bary['x_RANS'],line_bary['y_RANS'],'bx',label='RANS')
#    plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
#    plt.axis('equal')
#    plt.legend()
#    plt.title('Barycentric map: (y,z) = (%2.1f,%2.1f) to (y,z) = (%2.1f,%2.1f)' % (x_line[0],y_line[0],x_line[1],y_line[1]))
#    plt.annotate('1C', xy=(1.03, 0),size=15)
#    plt.annotate('2C', xy=(-0.07, 0),size=15)
#    plt.annotate('3C', xy=(0.4, 0.83),size=15)
#    plt.show()