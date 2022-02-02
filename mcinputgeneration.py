#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:33:48 2022

@author: henryisrael
"""
import numpy as np
import random
class fakeData:
    #Makes random gaussian for MCMC purposes    
    def __init__(self, _spacedim,_nfakedata=1000,verbose=False): #number of steps
        if not isinstance(verbose,bool):
            raise TypeError("Verbosity must be of type bool")
            
        self.verbose=verbose
        self.spacedim=self.getDim(_spacedim) #size of space
        self.mean=np.random.rand(self.spacedim)
        self.nfakedata=_nfakedata
        #generate cov matrix at init
        partial_cov=np.random.rand(self.spacedim,self.spacedim)
    
        self.cov=np.dot(partial_cov, partial_cov.transpose()) #Pos-def by construction
        self.fakedata=self.generateFakeData()
        if self.verbose:
            print("Constructed new fakeData object")
    
    def updateVerbose(self, newverbose):
        if not isinstance(newverbose,bool):
            raise TypeError("Verbosity must be of type bool")
        self.verbose=newverbose
        print(f"verbosity now set to {newverbose}")
        return self.verbose
    
    def getDim(self,spacedim):
        if spacedim<=0:
            raise ValueError(f"Total spatial dimensions must be >0, not {spacedim}")
        return spacedim
    
    #user inputted fake data size
    def updateFakeDataSize(self, updatedvalue):
        if updatedvalue<=0:
            raise ValueError(f"total size of fake data cannot be {updatedvalue}, must be >0")
        self.nfakedata=updatedvalue
        return self.nfakedata
    
    def fakeDataPoint(self):
        #Makes an spacedim-dimensional gaussian centered on rand([-1,1) with std (0,01,1)
        return np.random.multivariate_normal(self.mean,self.cov,self.spacedim)
    
    
    def generateFakeData(self):
        #array to be filled with fake data point 
        self.fakedata=[]
        i=0
        print(f"Making {self.nfakedata} fake data points")
        while i<self.nfakedata:
            self.fakedata.append(self.fakeDataPoint())
            i+=1
        return self.fakedata
            
    def fakeDataArr(self):
        return self.fakedata
        

    
class MetropolisHastings(fakeData):
    #Whilst coding this myself is unecessary (AND SLOW), I just want to 
    #get a better understanding of the algorithm
    def __init__(self, nsteps, spacedim, stepsizes):
        '''
        Does mcmc for some space for nsteps steps given an array of stepsizes

        Parameters
        ----------
        nsteps : Total MCMC steps.
        spacedim : Spatial Dimension.
        stepsizes : Array of step sizes (must be same size as spacedim)

        Returns
        -------
        None.

        '''
        if len(stepsizes!=spacedim):
            raise ValueError("Total length of step sizes must match spatial dimension")
            
        
        fakeData.__init__(self, spacedim,10000)
        #Start from some point
        self.prevstep=np.ones(spacedim)
        self.propstep=np.ones(spacedim)
        
        self.stepMatrix=np.diag(stepsizes)
        
        
    
    def accept_reject(self, prev_llh, curr_llh):
        acc_prob=random.uniform(0,1)
        return ((acc_prob<min(1,np.exp(curr_llh-prev_llh))))
    
    
    def proposeStep(self,stepsize):
        #proposes a new step
        return np.random.multivariate_normal(self.prevstep, self.stepMatrix, self.spacedim)
    
    def llh_calc(self):
        
    
    
    
    
    