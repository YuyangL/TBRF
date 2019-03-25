#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:04:13 2017
Versions:
    v5: add BackwardFacingStep
    v6: CBFS and CDChannel added, added Wang features
@author: mikael
"""
import sys
sys.path.append("..")
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import PyFOAM as pyfoam
import scipy.interpolate as interp
from tempfile import mkstemp
from shutil import move
from shutil import copy
from os import remove, close
import os
import collections
import struct

#===============================================================================
# S U B R O U T I N E S
#===============================================================================


def loadData_avg(dataset,flowCase):

    if flowCase == 'SquareDuct':
        with open(dataset, 'rb') as f:
            reader = csv.reader(f)
            names = reader.next()
            ubulk = reader.next()
#            print names
            data_list = np.zeros([len(names), 100000])
#            print data_list.shape
            for i,row in enumerate(reader):
                if row:
                    data_list[:,i] = np.array([float(ii) for ii in row])
#        print i
        data_list = data_list[:,:i+1] 
        data = {}
        for j,var in enumerate(names):
            data[var] = data_list[j,:]     
    
    elif flowCase == 'PeriodicHills':
        data_list = np.zeros([10, 100000])
        with open(dataset, 'r') as f:
            reader = csv.reader(f)
            
            #check python version (2 or 3)
            if sys.version_info[:2][0]==2:
                names = reader.next()[:-1]
            elif sys.version_info[:2][0]==3:
                names = reader.__next__()[:-1]
            
            for i,row in enumerate(reader):
                if row:
                    data_list[:,i] = np.array([float(ii) for ii in row[:-1]])
        
        data_list = data_list[:,:i]
        
        data = {}
        for i,var in enumerate(names):
            data[var] = data_list[i,:]
            
    elif flowCase == 'ConvDivChannel':
        #amount of lines between different blocks
        ny = 193
        nx = 1152
        
        interval = int(np.ceil((float(ny)*float(nx))/5.0))
        
        dataFile = {}
        dataFile['vars'] = ["x","y","U","V","W","dx_mean_u","dx_mean_v","dx_mean_w","dy_mean_u","dy_mean_v",
        "dy_mean_w","dz_mean_u","dz_mean_v","dz_mean_w","uu","uv","uw","vv","vw","ww"]
        dataFile['vals'] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        
        l_start = np.zeros(len(dataFile['vars']),dtype=int)
        l_end = np.zeros(len(dataFile['vars']),dtype=int)
        start_data = 26
        
        # datafile is separated in blocks, each belonging to one of the variables.
        # set-up the start and end values of each block
        for i in range(len(dataFile['vars'])):
            l_start[i] = start_data + i*interval
            l_end[i] = (interval+start_data-1) + i*interval
        
        with open(dataset, 'rb') as f:
            
            # go through all lines of file
            for i,line in enumerate(f):  
                
                # go through all variables
                for j in range(l_start.shape[0]):
                    
                    # check the variable the current block belongs to
                    if i >= l_start[j] and i <= l_end[j]:
                        dataFile['vals'][j].append([float(x) for x in line.split()])
        
        data_DNS = {}
        data = {}
        # flatten the obtained lists with data            
        for i in range(len(dataFile['vals'])):
            data_DNS[dataFile['vars'][i]] = np.array([item for sublist in dataFile['vals'][i] for item in sublist])
            data[dataFile['vars'][i]] = np.reshape(data_DNS[dataFile['vars'][i]],[nx,ny])
    
    elif flowCase == 'CurvedBackwardFacingStep':
        #amount of lines between different blocks
        ny = 160
        nx = 768
        interval = int(np.ceil((float(ny)*float(nx))/5.0))
        
        dataFile = {}
        dataFile['vars'] = ["x","y","p","U","V","W","uu","vv","ww","uv","uw","vw","k"]
        dataFile['vals'] = [[],[],[],[],[],[],[],[],[],[],[],[],[]]
        
        l_start = np.zeros(len(dataFile['vars']),dtype=int)
        l_end = np.zeros(len(dataFile['vars']),dtype=int)
        start_data = 20
        
        # datafile is separated in blocks, each belonging to one of the variables.
        # set-up the start and end values of each block
        for i in range(len(dataFile['vars'])):
            l_start[i] = start_data + i*interval
            l_end[i] = (interval+start_data-1) + i*interval
        
        with open(dataset, 'rb') as f:
            
            # go through all lines of file
            for i,line in enumerate(f):  
                
                # go through all variables
                for j in range(l_start.shape[0]):
                    
                    # check the variable the current block belongs to
                    if i >= l_start[j] and i <= l_end[j]:
                        dataFile['vals'][j].append([float(x) for x in line.split()])
        
        data_DNS = {}
        data = {}
        # flatten the obtained lists with data            
        for i in range(len(dataFile['vals'])):
            data_DNS[dataFile['vars'][i]] = np.array([item for sublist in dataFile['vals'][i] for item in sublist])
            data[dataFile['vars'][i]] = np.reshape(data_DNS[dataFile['vars'][i]],[ny,nx])
            
    
    elif flowCase == 'BackwardFacingStep_before' or flowCase == 'BackwardFacingStep_after':
        
        def loadDataSets(dataset):
            data_list = np.zeros([66, 104427])
            with open(dataset, 'rb') as f:
                reader = csv.reader(f)
                names = reader.next()[:-1]
                #print len(names)
                for i,row in enumerate(reader):
                    if row:
                        data_list[:,i] = np.array([float(ii) for ii in row[:-1]])
            
            data_list = data_list[:,:i]
            
            data = {}
            for i,var in enumerate(names):
                data[var] = data_list[i,:]
                        
            return data
        
        datasets_shape  = {'plane1':[48,49], 'plane2':[48,500], 'plane3':[97,500]}

        tmp_data = {}
        for f in dataset.keys():
            dataDNS = loadDataSets(dataset[f])
            tmp = {}
            for v in dataDNS.keys():
                tmp[v] = dataDNS[v].reshape(datasets_shape[f])
            tmp_data[f] = tmp   
        
        dataDNS= {}
        xi = np.concatenate((tmp_data['plane1']['x'][0,:], tmp_data['plane2']['x'][0,1:]))
        yi = np.concatenate((tmp_data['plane3']['y'][:,0], tmp_data['plane2']['y'][1:,0]))
        X,Y = np.meshgrid(xi,yi)
        
        dataDNS['x'] = X
        dataDNS['y'] = Y
        
        data = {}
        if flowCase == 'BackwardFacingStep_before':
            data['x'] = X[96:,:48]
            data['y'] = Y[96:,:48]-1
        elif flowCase == 'BackwardFacingStep_after':
            data['x'] = X[:,48:]
            data['y'] = Y[:,48:]-1
        
        
        for key in tmp_data['plane1'].keys():
            
            if key is not 'x' and key is not 'y':
#                print key
            
                tmp = np.zeros([144, 548])*np.nan    
                tmp[96:,:48]   = tmp_data['plane1'][key][:,:-1]
                tmp[96:,48:] = tmp_data['plane2'][key][:,:]
                tmp[:96,48:]   = tmp_data['plane3'][key][:-1,:]
                    
                dataDNS[key] = tmp
                
                if flowCase == 'BackwardFacingStep_before':
                    data[key] = tmp[96:,:48]
                elif flowCase == 'BackwardFacingStep_after':
                    data[key] = tmp[:,48:]
                    
        
    elif flowCase == 'BackwardFacingStep2_before' or flowCase == 'BackwardFacingStep2_after':
        nHeaderLines = 138

        f = open(dataset, 'rb')
        
        for i in range(nHeaderLines):
            f.readline()
            
        #read 6 size parameters
        nx = struct.unpack('>I', f.read(4))[0]
        ny = struct.unpack('>I', f.read(4))[0]
        ixstep = struct.unpack('>I', f.read(4))[0]
        iystep = struct.unpack('>I', f.read(4))[0]
        nel = struct.unpack('>I', f.read(4))[0]
        dx = struct.unpack('>f', f.read(4))[0]
        Re = struct.unpack('>f', f.read(4))[0]
        
        #compute the x values
        xn = np.zeros(nx)
        for i in range(1, nx+1):
            xn[i-1] = (i - ixstep)*dx
        
        #read the y values
        yn = np.zeros(ny)
        for i in range(ny):
            yn[i] = struct.unpack('>f', f.read(4))[0]
           
        #read 242 zeros
        for i in range(242):
            #print struct.unpack('>f', f.read(4))[0]
            f.read(4)
        
        #read the stats
        stats = {}
        qoi = ['U', 'V', 'P', 'uu', 'vv', 'ww', 'pp', 'uv']
        for n in range(nel):
            stats[qoi[n]] = np.zeros([nx, ny])
        
        for n in range(nel):
            for j in range(ny):
                for i in range(nx):
                    tmp = f.read(4)
                    stats[qoi[n]][i, j] = struct.unpack('>f', tmp)[0]      
        f.close()

#        U_ref = 7.9
#        
#        for j in range(ny):
#            for i in range(nx):
#                Rij = np.zeros([3, 3])
#                Rij[0,0] = stats['uu'][i, j]*U_ref**2
#                Rij[1,1] = stats['vv'][i, j]*U_ref**2
#                Rij[2,2] = stats['ww'][i, j]*U_ref**2
#                Rij[0,1] = stats['uv'][i, j]*U_ref**2
#                Rij[1,0] = stats['uv'][i, j]*U_ref**2
        
        X,Y = np.meshgrid(xn,yn)
        X = X.T
        Y = Y.T
        
        # shift 1 downwards
        Y = Y-1
        
        if 'before' in flowCase:
            mask = (X < 0) & (Y > 0) & (Y<5.99)
            
        if 'after' in flowCase:
            mask = (X > 0) & (Y<4.99) & (Y>-1)
        
        #store results
        data = {'x':X[mask], 'y':Y[mask], 'U':stats['U'][mask], 'V':stats['V'][mask], \
                'P':stats['P'][mask], 'uu':stats['uu'][mask], 'vv':stats['vv'][mask], \
                'ww':stats['ww'][mask], 'pp':stats['pp'][mask], 'uv':stats['uv'][mask]}
#        print(data['uv'])
    
    elif flowCase == 'BackwardFacingStep3_before' or flowCase == 'BackwardFacingStep3_after':
        
        def loadDataDNS(dataset):
            data_list = np.zeros([22, 104427])
            with open(dataset, 'rb') as f:
                reader = csv.reader(f)
                names = reader.next()[:-1]
                for i,row in enumerate(reader):
                    if row:
                        data_list[:,i] = np.array([float(ii) for ii in row[:-1]])
            
            data_list = data_list[:,:i]
            
            dataDNS = {}
            for i,var in enumerate(names):
                dataDNS[var] = data_list[i,:]
                
            return(dataDNS)
    
        dataset_shape  = {'plane1':[92,186], 'plane2':[136,642]}

        data_tmp = {}
        for f in dataset.keys():
            dataDNS = loadDataDNS(dataset[f])
            tmp = {}
            for v in dataDNS.keys():
                tmp[v] = dataDNS[v].reshape(dataset_shape[f])
            data_tmp[f] = tmp
        
        if 'before' in flowCase:
            data = data_tmp['plane1']
            data['x'] = data['x'] - 0.1
        if 'after' in flowCase:
            data = data_tmp['plane2']
            data['x'] = data['x'] - 0.1            
            
    return data


def interpDNSOnRANS(dataDNS, meshRANS, flowCase):
    print('Interpolating DNS data on RANS mesh...')
    
    if flowCase == 'PeriodicHills' or flowCase == 'ConvDivChannel' or flowCase == 'CurvedBackwardFacingStep':
        var1 = 'x'
        var2 = 'y'
        n_1 = 0
        n_2 = 1
    elif flowCase == 'SquareDuct':
        var1 = 'Z'
        var2 = 'Y'
        n_1 = 1
        n_2 = 2
    elif '_before' in flowCase or '_after' in flowCase:
        var1 = 'x'
        var2 = 'y'
        n_1 = 0
        n_2 = 1     
    else:
        print('flow case unknown')

    names = dataDNS.keys()    
    data = {}
    
    if flowCase == 'BackwardFacingStep_before' or flowCase == 'BackwardFacingStep_after':
              
        #discard most of the variables
        keep_vars = ['rho*u_rms','rho*v_rms','rho*w_rms','rho*uv_rms','rho*uw_rms','rho*vw_rms',
                     'u_bar','v_bar','w_bar','rho_bar','rho*KTE'] 
        
        xy = np.array([dataDNS[var1].T.flatten(), dataDNS[var2].T.flatten()]).T
        for var in names:
            if var in keep_vars:        
                data[var] = interp.griddata(xy, dataDNS[var].T.flatten(), (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')        

                nanbools = np.isnan(data[var])
                if nanbools.any():
                    print('nanbools')
                    data[var][nanbools] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[0][nanbools], meshRANS[1][nanbools]), method='nearest')
                    
    elif flowCase == 'BackwardFacingStep2_before' or flowCase == 'BackwardFacingStep2_after':
        for var in names:
            if not var=='x' and not var=='y':        
                data[var] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')
                
                # check for nan values
                nanbools = np.isnan(data[var])
                if nanbools.any():
                    print('nanbools')
                    data[var][nanbools] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[0][nanbools], meshRANS[1][nanbools]), method='nearest')

    elif flowCase == 'BackwardFacingStep3_before' or flowCase == 'BackwardFacingStep3_after':
        
        keep_vars = ['vx','vy','vz','u2','v2','w2','uv','uw','vw','p'] 
        
        for var in names:
            if var in keep_vars:        
                data[var] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')
                
                # check for nan values
                nanbools = np.isnan(data[var])
                if nanbools.any():
                    print('RANS points outside of DNS mesh, using nearest neighbor extrapolation')
                    data[var][nanbools] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[0][nanbools], meshRANS[1][nanbools]), method='nearest')
            
    elif flowCase == 'ConvDivChannel':
        
        keep_vars = ["U","V","W","uu","uv","uw","vv","vw","ww"]
        
        for var in names:
            if var in keep_vars:        
                data[var] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')

    elif flowCase == 'CurvedBackwardFacingStep':
        for var in names:
            if not var=='x' and not var=='y':        
                data[var] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')
                
                # check for nan values
                nanbools = np.isnan(data[var])
                if nanbools.any():
                    print('nanbools')
                    data[var][nanbools] = interp.griddata((dataDNS['x'].flatten(),dataDNS['y'].flatten()), dataDNS[var].flatten(), (meshRANS[0][nanbools], meshRANS[1][nanbools]), method='nearest')
    else:
        xy = np.array([dataDNS[var1], dataDNS[var2]]).T
        for var in names:
            if not var==var1 and not var==var2:        
                data[var] = interp.griddata(xy, dataDNS[var], (meshRANS[n_1,:,:], meshRANS[n_2,:,:]), method='linear')
    
    return data
    
def writeP(case, time, var, data):
    #file = open(case + '/' + str(time) + '/' + var,'r').readlines()
    copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var)
    #file = open('R~','r+').readlines()
    #file = open('wavyWall_Re6850_komegaSST_4L_2D/20000/gradU','r').readlines()
    #lines=0
    tmp = []
    tmp2 = 10**12
    maxIter = -1
    #v= np.zeros([3,70000])
    cc = False
    j = 0
    file_path=case + '/' + str(time) + '/' + var
#    print( file_path)
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
#                    print( tmp, tmp2)
                    
        
                elif i==tmp:
#                    print (line.split())
                    maxLines = int(line.split())
                    maxIter  = tmp2 + maxLines
                    #v = np.zeros([3,3,maxLines])
                    new_file.write(line)
#                    print (maxLines, maxIter)
                
                elif i>tmp and i<tmp2:              
                    new_file.write(line)
                
                elif i>=tmp2 and i<maxIter:
                    #print line
                    new_file.write( str(data[0,0,j]) + '\n'  )
                    j += 1
                
                elif i>=maxIter:
                    new_file.write(line)
                    
    close(fh)
    remove(file_path)
    move(abs_path, file_path)



def fn_importAndInterpolate(home,flowCase,Re,turbModel,time_end,nx,ny,write,plot,SecondaryFeatures):

    
    DO_INTERP   = 1
    DO_WRITE    = write
    DO_PLOTTING = plot
    
    
    # file directories
    nx_RANS       = nx
    ny_RANS       = ny
    time_end = time_end
    Re = Re


    # Load DNS dataset
    if flowCase == 'PeriodicHills':
        home = home + flowCase + '/'
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,nx_RANS))
        dir_DNS   = home + ('DATA_CASE_LES_BREUER/Re_%i/' % Re)    
        dataset = dir_DNS + ('Hill_Re_%i_Breuer.csv' % Re)
    elif flowCase == 'SquareDuct':
        home = home + flowCase + '/'
        dir_DNS   = home + ('DATA/')
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,nx_RANS))
        dataset = dir_DNS + ('0%i.csv' % Re)
        
    elif flowCase == 'BackwardFacingStep_before' or flowCase == 'BackwardFacingStep_after':
        home = home + 'BackwardFacingStep' + '/'
        
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,90))
        dir_DNS   = home + ('DATA/')
        dataset    = {'plane1': dir_DNS+'STAT_p1_b1.csv',
                   'plane2': dir_DNS+'STAT_p1_b2.csv',
                   'plane3': dir_DNS+'STAT_p1_b3.csv'}
    elif flowCase == 'BackwardFacingStep2_before' or flowCase == 'BackwardFacingStep2_after':
        home = home + 'BackwardFacingStep2' + '/'
        
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,90))
        dir_DNS   = home + ('DATA/')
        dataset = dir_DNS + ('stats1pt.bin')
    elif flowCase == 'BackwardFacingStep3_before' or flowCase == 'BackwardFacingStep3_after':
        home = home + 'BackwardFacingStep3' + '/'
        
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,90))
        dir_DNS   = home + ('DATA/')
        dataset   = {'plane1':dir_DNS + 'merge_mean_kplane1.csv', 'plane2':dir_DNS + 'merge_mean_kplane2.csv'}
        
    elif flowCase == 'ConvDivChannel':
        
        home = home + flowCase + '/'
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,ny_RANS))
        dir_DNS   = home + ('DATA/')
        dataset = dir_DNS + ('conv-div-mean-half.dat')
    elif flowCase == 'CurvedBackwardFacingStep':
        home = home + flowCase + '/'
        dir_RANS  = home + ('Re%i_%s_%i' % (Re,turbModel,ny_RANS))
        dir_DNS   = home + ('DATA/')
        dataset = dir_DNS + ('curvedbackstep_vel_stress.dat')
    else:
        print('error: flow case %s unknown' % flowCase)
    
    print('RANS directory:')
    print(dir_RANS) 
    
    print('DNS file:')
    print(dataset)
    
    print('Loading RANS data and DNS data...')
    dataDNS = loadData_avg(dataset, flowCase)
    
    # Load RANS data
    dataRANS = {}
    
    # get lists of all relevant RANS variables
    meshRANSlist        = pyfoam.getRANSVector(dir_RANS, time_end, 'cellCentres')    
    U_RANSlist          = pyfoam.getRANSVector(dir_RANS, time_end, 'U')
    p_RANSlist          = pyfoam.getRANSScalar(dir_RANS, time_end, 'p')
    tau_RANSlist        = pyfoam.getRANSSymmTensor(dir_RANS, time_end, 'R')
    gradU_RANSlist      = pyfoam.getRANSTensor(dir_RANS, time_end, 'grad(U)')
    
    if 'k' in SecondaryFeatures:
        gradTke_RANSlist        = pyfoam.getRANSVector(dir_RANS, time_end, 'grad(k)')
    if 'Wang' in SecondaryFeatures:
        wallDist_RANSlist       = pyfoam.getRANSScalar(dir_RANS, time_end, 'yWall')
        gradP_RANSlist          = pyfoam.getRANSVector(dir_RANS, time_end, 'grad(p)')
        gradU2_RANSlist         = pyfoam.getRANSTensor(dir_RANS, time_end, 'grad(Usqr)')
    
    # convert lists to mesh format. In case of BFS, split into two domains
    if '_before' in flowCase or '_after' in flowCase:

        if 'BackwardFacingStep3' in flowCase:
            l_before = 50*(20+30+40)
        else:
            l_before = 150*(20+30+40)
            l_after = 150*(30+20+20+30+40)        
        
        if 'before' in flowCase:
            meshRANS            = pyfoam.getRANSPlane(meshRANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'vector')
            dataRANS['U']       = pyfoam.getRANSPlane(U_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'vector')    
            dataRANS['p']       = pyfoam.getRANSPlane(p_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'scalar')
            dataRANS['tau']     = pyfoam.getRANSPlane(tau_RANSlist[:,:,:l_before],'2D', nx_RANS, ny_RANS, 'tensor')
            dataRANS['gradU']   = pyfoam.getRANSPlane(gradU_RANSlist[:,:,:l_before],'2D', nx_RANS, ny_RANS, 'tensor')
            if 'k' in SecondaryFeatures:
                dataRANS['gradTke']     = pyfoam.getRANSPlane(gradTke_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'vector')
            if 'Wang' in SecondaryFeatures:
                dataRANS['wallDist']    = pyfoam.getRANSPlane(wallDist_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'scalar')
                dataRANS['gradP']       = pyfoam.getRANSPlane(gradP_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'vector')
                dataRANS['gradU2']      = pyfoam.getRANSPlane(gradU2_RANSlist[:,:,:l_before],'2D', nx_RANS, ny_RANS, 'tensor')
                
        elif 'after' in flowCase:
            meshRANS            = pyfoam.getRANSPlane(meshRANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'vector')
            dataRANS['U']       = pyfoam.getRANSPlane(U_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'vector')    
            dataRANS['p']       = pyfoam.getRANSPlane(p_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'scalar')
            dataRANS['tau']     = pyfoam.getRANSPlane(tau_RANSlist[:,:,l_before:],'2D', nx_RANS, ny_RANS, 'tensor')
            dataRANS['gradU']   = pyfoam.getRANSPlane(gradU_RANSlist[:,:,l_before:],'2D', nx_RANS, ny_RANS, 'tensor')
            if 'k' in SecondaryFeatures:
                dataRANS['gradTke']     = pyfoam.getRANSPlane(gradTke_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'vector')    
            if 'Wang' in SecondaryFeatures:
                dataRANS['wallDist']    = pyfoam.getRANSPlane(wallDist_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'scalar')
                dataRANS['gradP']       = pyfoam.getRANSPlane(gradP_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'vector')  
                dataRANS['gradU2']      = pyfoam.getRANSPlane(gradU2_RANSlist[:,:,l_before:],'2D', nx_RANS, ny_RANS, 'tensor')
                
    else:
        meshRANS            = pyfoam.getRANSPlane(meshRANSlist,'2D', nx_RANS, ny_RANS, 'vector')
        dataRANS['U']       = pyfoam.getRANSPlane(U_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')    
        dataRANS['p']       = pyfoam.getRANSPlane(p_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')
        dataRANS['tau']     = pyfoam.getRANSPlane(tau_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')
        dataRANS['gradU']   = pyfoam.getRANSPlane(gradU_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')
        if 'k' in SecondaryFeatures:
            dataRANS['gradTke']     = pyfoam.getRANSPlane(gradTke_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')
        if 'Wang' in SecondaryFeatures:
            dataRANS['wallDist']    = pyfoam.getRANSPlane(wallDist_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')
            dataRANS['gradP']       = pyfoam.getRANSPlane(gradP_RANSlist,'2D', nx_RANS, ny_RANS, 'vector')
            dataRANS['gradU2']      = pyfoam.getRANSPlane(gradU2_RANSlist,'2D', nx_RANS, ny_RANS, 'tensor')
            
    # interpolate DNS on RANS grid
    dataDNS_i = interpDNSOnRANS(dataDNS, meshRANS,flowCase)
    
    
    if flowCase == 'BackwardFacingStep_before' or flowCase == 'BackwardFacingStep_after':
        dataDNS_i['k'] = 0.5* ((dataDNS_i['rho*u_rms'] + dataDNS_i['rho*v_rms'] + dataDNS_i['rho*w_rms'])/dataDNS_i['rho_bar'])
    elif flowCase == 'BackwardFacingStep3_before' or flowCase == 'BackwardFacingStep3_after':
        dataDNS_i['k'] = 0.5 * (dataDNS_i['u2'] + dataDNS_i['v2'] + dataDNS_i['w2'])
    else:
        dataDNS_i['k'] = 0.5 * (dataDNS_i['uu'] + dataDNS_i['vv'] + dataDNS_i['ww'])
    
    l1 = np.shape(dataRANS['U'])[1]
    l2 = np.shape(dataRANS['U'])[2]
    
    
    dataDNS_i['tau'] = np.zeros([3,3,l1,l2])
    
    if flowCase == 'PeriodicHills' or flowCase == 'ConvDivChannel' or flowCase == 'CurvedBackwardFacingStep':
        dataDNS_i['tau'][0,0,:,:] = dataDNS_i['uu']
        dataDNS_i['tau'][1,1,:,:] = dataDNS_i['vv']
        dataDNS_i['tau'][2,2,:,:] = dataDNS_i['ww']
        dataDNS_i['tau'][0,1,:,:] = dataDNS_i['uv']
        dataDNS_i['tau'][1,0,:,:] = dataDNS_i['uv']
        
    if flowCase == 'BackwardFacingStep2_before' or flowCase == 'BackwardFacingStep2_after':    
        U_ref = 7.9
        dataDNS_i['tau'][0,0,:,:] = dataDNS_i['uu']#/(U_ref**2)
        dataDNS_i['tau'][1,1,:,:] = dataDNS_i['vv']#/(U_ref**2)
        dataDNS_i['tau'][2,2,:,:] = dataDNS_i['ww']#/(U_ref**2)
        dataDNS_i['tau'][0,1,:,:] = dataDNS_i['uv']#/(U_ref**2)
        dataDNS_i['tau'][1,0,:,:] = dataDNS_i['uv']#/(U_ref**2)   

    if flowCase == 'BackwardFacingStep3_before' or flowCase == 'BackwardFacingStep3_after':    
        U_ref = 50.0
        dataDNS_i['tau'][0,0,:,:] = dataDNS_i['u2']#/(U_ref**2)
        dataDNS_i['tau'][1,1,:,:] = dataDNS_i['v2']#/(U_ref**2)
        dataDNS_i['tau'][2,2,:,:] = dataDNS_i['w2']#/(U_ref**2)
        dataDNS_i['tau'][0,1,:,:] = dataDNS_i['uv']#/(U_ref**2)
        dataDNS_i['tau'][1,0,:,:] = dataDNS_i['uv']#/(U_ref**2)
        
    if flowCase == 'SquareDuct': 
        dataDNS_i['tau'][0,0,:,:] = dataDNS_i['uu']
        dataDNS_i['tau'][1,1,:,:] = dataDNS_i['vv']
        dataDNS_i['tau'][2,2,:,:] = dataDNS_i['ww']
        dataDNS_i['tau'][0,1,:,:] = dataDNS_i['uv']
        dataDNS_i['tau'][1,0,:,:] = dataDNS_i['uv']
        dataDNS_i['tau'][0,2,:,:] = dataDNS_i['uw']
        dataDNS_i['tau'][2,0,:,:] = dataDNS_i['uw']         
        dataDNS_i['tau'][1,2,:,:] = dataDNS_i['vw']
        dataDNS_i['tau'][2,1,:,:] = dataDNS_i['vw']
    
    if flowCase == 'BackwardFacingStep_before' or flowCase == 'BackwardFacingStep_after':
        dataDNS_i['tau'][0,0,:,:] = dataDNS_i['rho*u_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][1,1,:,:] = dataDNS_i['rho*v_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][2,2,:,:] = dataDNS_i['rho*w_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][0,1,:,:] = dataDNS_i['rho*uv_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][1,0,:,:] = dataDNS_i['rho*uv_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][0,2,:,:] = dataDNS_i['rho*uw_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][2,0,:,:] = dataDNS_i['rho*uw_rms']/dataDNS_i['rho_bar']   
        dataDNS_i['tau'][1,2,:,:] = dataDNS_i['rho*vw_rms']/dataDNS_i['rho_bar']
        dataDNS_i['tau'][2,1,:,:] = dataDNS_i['rho*vw_rms']/dataDNS_i['rho_bar']           
        
         
    dataDNS_i['bij'] = np.zeros([3,3,l1,l2])
    dataRANS['k'] = np.zeros([l1,l2])
    dataRANS['bij'] = np.zeros([3,3,l1,l2])
    
    for i in range(l1):
        for j in range(l2):
            dataDNS_i['bij'][:,:,i,j] = dataDNS_i['tau'][:,:,i,j]/(2.*dataDNS_i['k'][i,j]) - np.diag([1/3.,1/3.,1/3.])
            dataRANS['k'][i,j] = 0.5 * np.trace(dataRANS['tau'][:,:,i,j])
            dataRANS['bij'][:,:,i,j] = dataRANS['tau'][:,:,i,j]/(2.*dataRANS['k'][i,j]) - np.diag([1/3.,1/3.,1/3.])
#        nanbools = np.isnan(dataDNS_i['tau'])
#    if nanbools.any():
#        print('yooooooooooooooooooooooo')
#
#    plt.figure()
#    plt.contourf(meshRANS[0,:,:],meshRANS[1,:,:],dataDNS_i['tau'][0,0,:,:])
#    plt.show()
#    plt.figure()
#    plt.contourf(meshRANS[0,:,:],meshRANS[1,:,:],dataDNS_i['tau'][1,1,:,:])
#    plt.show()
#    plt.figure()
#    plt.contourf(meshRANS[0,:,:],meshRANS[1,:,:],dataDNS_i['tau'][2,2,:,:])
#    plt.show()
#    plt.figure()
#    plt.contourf(meshRANS[0,:,:],meshRANS[1,:,:],dataDNS_i['tau'][0,1,:,:])
#    plt.show()
#    
#    print(dataDNS_i['tau'])
    nanbools = np.isnan(dataDNS_i['tau'])
    if nanbools.any():
        print('yooooooooooooooooooooooo')
    infbools = np.isinf(dataDNS_i['tau'])
    if infbools.any():
        print('yooooooooooooooooooooooo inf')        
    nanbools = np.isnan(dataDNS_i['k'])    
    if nanbools.any():
        print('yooooooooooooooooooooooo k')
    infbools = np.isinf(dataDNS_i['k'])
    if infbools.any():
        print('yooooooooooooooooooooooo k inf')  
        
    dataDNS_i['eigVal'] = pyfoam.calcEigenvalues(dataDNS_i['tau'], dataDNS_i['k'])
    dataDNS_i['baryMap'] = pyfoam.barycentricMap(dataDNS_i['eigVal'])
    
    dataRANS['eigVal'] = pyfoam.calcEigenvalues(dataRANS['tau'], dataRANS['k'])
    dataRANS['baryMap'] = pyfoam.barycentricMap(dataRANS['eigVal'])
    
    if turbModel == 'kEps':
        eps_RANSlist = pyfoam.getRANSScalar(dir_RANS, time_end, 'epsilon')
        
        if '_before' in flowCase or '_after' in flowCase: #BFS
            if 'before' in flowCase:
                dataRANS['epsilon'] = pyfoam.getRANSPlane(eps_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'scalar') 
            elif 'after' in flowCase:
                dataRANS['epsilon'] = pyfoam.getRANSPlane(eps_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'scalar')
        else:        
            dataRANS['epsilon'] = pyfoam.getRANSPlane(eps_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')
            
    elif turbModel == 'kOmega':
        omega_RANSlist = pyfoam.getRANSScalar(dir_RANS, time_end, 'omega')
        
        if '_before' in flowCase or '_after' in flowCase:
            if 'before' in flowCase:
                dataRANS['omega'] = pyfoam.getRANSPlane(omega_RANSlist[:,:l_before],'2D', nx_RANS, ny_RANS, 'scalar')
            elif 'after' in flowCase:    
                dataRANS['omega'] = pyfoam.getRANSPlane(omega_RANSlist[:,l_before:],'2D', nx_RANS, ny_RANS, 'scalar')
        else:
            dataRANS['omega'] = pyfoam.getRANSPlane(omega_RANSlist,'2D', nx_RANS, ny_RANS, 'scalar')
        
        dataRANS['epsilon'] = 0.09*dataRANS['omega']*dataRANS['k']
    
    #%% write out OpenFOAM data files with DNS interpolated on RANS
    if DO_WRITE:
        # pressure p
    
        case = dir_RANS 
        time = time_end
        var = 'pd'
        data = np.swapaxes(dataDNS_i['pm'],0,1).reshape(nx_RANS*ny_RANS)
        
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
    
    
    # velocity
        time = time_end
        var = 'Ud'
    
        data_x = np.swapaxes(dataDNS_i['um'],0,1).reshape(nx_RANS*ny_RANS)
        data_y = np.swapaxes(dataDNS_i['vm'],0,1).reshape(nx_RANS*ny_RANS)
        data_z = np.swapaxes(dataDNS_i['wm'],0,1).reshape(nx_RANS*ny_RANS)
        
    
        # if Ud exists, back it up, otherwise use U as template for Ud
        if os.path.exists(case + '/' + str(time) + '/' + var) == True:
            copy(case + '/' + str(time) + '/' + var, case + '/' + str(time) + '/' + var + '_old')
        else:
            copy(case + '/' + str(time) + '/' + 'U', case + '/' + str(time) + '/' + var)
        
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
                        new_file.write('    object      U;\n')
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
                        new_file.write('(' + str(data_x[j]) + ' ' +  str(data_y[j]) + ' ' + str(data_z[j]) + ') \n'  )
                        j += 1
                    
                    elif i>=maxIter:
                        new_file.write(line)
                        
        close(fh)
        remove(file_path)
        move(abs_path, file_path)
    
    
    
    #%% PLOTTING
    if DO_PLOTTING:
        
        plt.close('all')
        
        plt.figure()
        plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['um'],20)
        plt.show()
        
        plt.figure()
        plt.contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS['U'][0,:,:],20)
        plt.show()
        
        plt.figure()
        f, axarr = plt.subplots(2, sharex=True)
        maxval = np.max([dataDNS_i['bij'][0,0,:,:],dataRANS['bij'][0,0,:,:]])
        minval = np.min([dataDNS_i['bij'][0,0,:,:],dataRANS['bij'][0,0,:,:]])
        subPlot1 = axarr[0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS['bij'][0,0,:,:],20,cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        subPlot2 = axarr[1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['bij'][0,0,:,:],20,cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        axarr[0].set_title('RANS $a_{11}$')
        axarr[1].set_title('DNS $a_{11}$')
        f.subplots_adjust(right=0.8)  
        f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()
        
        plt.figure()
        f, axarr = plt.subplots(2, sharex=True)
        maxval = np.max([dataDNS_i['bij'][0,1,:,:],dataRANS['bij'][0,1,:,:]])
        minval = np.min([dataDNS_i['bij'][0,1,:,:],dataRANS['bij'][0,1,:,:]])
        subPlot1 = axarr[0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS['bij'][0,1,:,:],20,cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        subPlot2 = axarr[1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['bij'][0,1,:,:],20, cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        axarr[0].set_title('RANS $a_{12}$')
        axarr[1].set_title('DNS $a_{12}$')
        f.subplots_adjust(right=0.8)  
        f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()
        
        plt.figure()
        f, axarr = plt.subplots(2, sharex=True)
        maxval = np.max([dataDNS_i['bij'][1,1,:,:],dataRANS['bij'][1,1,:,:]])
        minval = np.min([dataDNS_i['bij'][1,1,:,:],dataRANS['bij'][1,1,:,:]])
        subPlot1 = axarr[0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS['bij'][1,1,:,:],20,cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        subPlot2 = axarr[1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['bij'][1,1,:,:],20, cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        axarr[0].set_title('RANS $a_{22}$')
        axarr[1].set_title('DNS $a_{22}$')
        f.subplots_adjust(right=0.8)  
        f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()
        
        plt.figure()
        f, axarr = plt.subplots(2, sharex=True)
        maxval = np.max([dataDNS_i['bij'][2,2,:,:],dataRANS['bij'][2,2,:,:]])
        minval = np.min([dataDNS_i['bij'][2,2,:,:],dataRANS['bij'][2,2,:,:]])
        subPlot1 = axarr[0].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataRANS['bij'][2,2,:,:],20, cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        subPlot2 = axarr[1].contourf(meshRANS[0,:,:], meshRANS[1,:,:], dataDNS_i['bij'][2,2,:,:],20, cmap=plt.cm.coolwarm, vmin=minval, vmax=maxval)
        axarr[0].set_title('RANS $a_{33}$')
        axarr[1].set_title('DNS $a_{33}$')
        f.subplots_adjust(right=0.8)  
        f.colorbar(subPlot2,f.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()
    
    #    
        plt.figure()
        plt.plot(dataDNS_i['baryMap'][0,:,:],dataDNS_i['baryMap'][1,:,:],'b*')
        plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
        plt.axis('equal')
        plt.show()
        
        plt.figure()
        plt.plot(dataRANS['baryMap'][0,:,:],dataRANS['baryMap'][1,:,:],'b*')
        plt.plot([0,1,0.5,0],[0,0,np.sin(60*(np.pi/180)),0],'k-')
        plt.axis('equal')
        plt.show()
        
        plt.figure()
        i3 = 1
        for i1 in range(3):
            for i2 in range(3):
                plt.plot(i3*np.ones(l1*l2),np.ravel(dataRANS['bij'][i1,i2,:,:]),'r*')
                i3 = i3 + 1
        
        plt.figure()
        i3 = 1
        for i1 in range(3):
            for i2 in range(3):
                plt.plot(i3*np.ones(l1*l2),np.ravel(dataDNS_i['bij'][i1,i2,:,:]),'r*')
                i3 = i3 + 1
    return(dataRANS,dataDNS_i,meshRANS)               
                