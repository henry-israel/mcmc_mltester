import string
import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing as mp
import keras
from keras.models import Sequential
from keras.layers import Dense
import time

class process_data:
    def __init__(self, dataset=None, llhbins=20, autocorr_integral_split=20, autocorr_reduce_size=50,append_data=False):
        self._appenddata=False #Append to previous array when running
        self._llhbins=llhbins
        self._autocorr_integral_split=autocorr_integral_split
        self._autocorr_reducesize=autocorr_reduce_size
        self._appenddata=append_data
        self._dataset=dataset
        self._processed_data=pd.DataFrame()     

    #Training Data Processing
    def updateData(self,updatedata):
        self._dataset=updatedata

    def updateLLHBins(self,updatedbinsize):
        self._llhbins=updatedbinsize
        return 1
    
    def updateAutoCorrIntegralSplit(self,updatedsplit):
        self._autocorr_integral_split=updatedsplit
        return 1
    
    def updateAutoCorrReduceSize(self,updatedsize):
        self._autocorr_reducesize=updatedsize

    def setInputParams(self, dataset=None, llhbins=20, autocorr_integral_split=20, autocorr_reduce_size=50):
        self.updateData(dataset)
        self.updateLLHBins(llhbins)
        self.updateAutoCorrIntegralSplit(autocorr_integral_split)
        self.updateAutoCorrReduceSize(autocorr_reduce_size)

    
    def binMeansDataSet(self, indata, nbins=20): 
        #Splits data into nbins, then takes mean
        return [np.mean(x) for x in np.split(indata,nbins)]

    def binMaxDataSet(self,indata,nbins=20):
        #Splits data into nbins, then takes max of each bin
        return [np.max(x) for x in np.split(indata, 20)]

    def integrateData(self, indata, nbins=30):
        #Integrate data numerically by splitting in number of bins (also normalises it!)
        hbins=self.binMaxDataSet(indata,nbins)
        return np.sum(hbins)/nbins

    def getNthElements(self,indata,nelement):
        #Grabs every nth entry
        return [x[0] for x in np.split(indata,nelement)]
    #Can add more metrics where necessary!

    def updateProcessedData(self,updateddata):
        if self._appenddata:
            if updateddata.top() != self._processed_data.top:
                raise ValueError("headers for updated data do not match those of already present data!")
            pd.concat(self._processed_data,updateddata)
        else:
            self._processed_data=updateddata
        return 1
    
    def getProcessedData(self):
        return self._processed_data

    def processTrainData(self, llhbins=20, autocorr_integral_split=20, autocorr_reduce_size=50,append_data=False):

        self._appenddata=append_data
        trset=np.array(self._trainarr,dtype=object)
#        print(trset)
#        print(len(trset))
        mc_acc=trset[:,0]
        mc_llh=trset[:,1]
        mc_auto=trset[:,2]

        #For now let's do mean-binning for llhs
        processed_llh=[self.binMeansDataSet(i,llhbins) for i in mc_llh]
        #Autocorrs require a different treatment
        integrateautocorr=[self.integrateData(i,autocorr_integral_split) for i in mc_auto]
        #Reduce number of points (Mean pooling may smear useful features, will need investigating!)
        reduced_autocorr=[self.getNthElements(i, autocorr_reduce_size) for i in mc_auto]

        #Let's make a massive dataframe!
        process_df=pd.DataFrame()
        process_df["Acceptance Rate"]=mc_acc
        process_df["Integrated Autocorrelation"]=integrateautocorr
        process_df["Reduced Auto Corr"]=reduced_autocorr
        process_df["Binned LLH"]=processed_llh

        self.updateProcessedData(process_df)
        return process_df


class mcmc:
    def __init__(self, spacedim=10, data=None):

        self._data=data    
        self._spacedim=spacedim


        if data is not None:
            self._mu=data[0]
            self._cov=data[1]
            self._spacedim=len(self._mu)
       
        else:
            self._mu=np.random.randn(spacedim)
            #Make cov matrix
            sqrt_cov=np.random.randn(spacedim,spacedim)


        self._cov=np.dot(sqrt_cov,sqrt_cov.T)
        self._acceptedsteps=[]
        self._numberaccepted=0
        self._acceptedllhs=[]
        self._stepmatrix=[]
        self._nsteps=0
    
    def getData(self):
        return self._data

    def updateInputData(self,updated_data):
        self.__init__(self._spacedim,updated_data)
        return 1

    def getMeanVal(self):
        return self._mu

    def updateMeanVal(self,update_mu):
        self._mu=update_mu
        return 1
    
    def getCov(self):
        return self._cov
    
    def updateCov(self,updated_cov):
        self._cov=updated_cov
        return 1

    def getSpaceDim(self):
        return self._spacedim

    def updateSpaceDim(self,updated_dim):
        self._spacedim=updated_dim
        return 1

    def getAcceptedSteps(self):
        return self._acceptedsteps

    def getStepMatrix(self):
        return self._stepmatrix

    def getAcceptedLLHS(self):
        return self._acceptedllhs

    def getNSteps(self):
        return self._nsteps

    def loglikelihood(self,x):
        '''

        Parameters
        ----------
        x : Proposed point

        Returns
        -------
        TYPE
            Does LLH calc for this boi

        '''
        if np.all(x < 50) and np.all(x > -50):
            return scipy.stats.multivariate_normal(mean=self._mu, cov=self._cov).logpdf(x)
        else:
            return -1e6
    
    def acceptFunc(self, curr_step, prop_step):
        alpha=np.random.uniform(0,1)
        fact=min(1,np.exp(prop_step-curr_step))
        if alpha<fact:
            return True
        else:
            return False
        
    def proposeStep(self,curr_step):
        prop_step = curr_step + np.dot(self._stepmatrix, np.random.randn(len(curr_step)))
        return prop_step
    
    def __call__(self, startpos, stepsize=None, nsteps=10000):
        
        self._nsteps=nsteps
        
        if stepsize is None:
            stepsize=np.ones(self._spacedim)
        self._stepmatrix=np.diag(stepsize)
        
        #setup mcmc
        curr_step=startpos
        curr_llh=self.loglikelihood(startpos)
        
        
        self._acceptedsteps=np.zeros((nsteps,self._spacedim))
        self._acceptedllhs=np.zeros(nsteps)
        
        self._acceptedsteps[0]=curr_step
        self._acceptedllhs[0]=curr_llh
        
        stepcount=1
        self._numberaccepted=1
        while stepcount<nsteps:
            # if stepcount%np.floor(nsteps/10)==0:
            #     print(f"Completed {stepcount}/{nsteps} steps, accepted {self._numberaccepted}")
            
            
            prop_step=self.proposeStep(curr_step)
            prop_llh=self.loglikelihood(prop_step)
            if self.acceptFunc(curr_llh, prop_llh):
                curr_step=prop_step
                curr_llh=prop_llh
                self._numberaccepted+=1

            self._acceptedsteps[stepcount]=curr_step
            self._acceptedllhs[stepcount]=curr_llh
            


            stepcount+=1
        return self._acceptedsteps, self._acceptedllhs
    
    def autocorrs(self,totlag=1000):
        print("Making Autocorrelations")
        if "daemon" not in mp.current_process()._config:
            a_pool=mp.Pool()
            autocorrarr=a_pool.map(self.autocalc, range(totlag))
        else:
            autocorrarr=np.empty((totlag, self._spacedim))
            for k in range(totlag):
                autocorrarr[k]=self.autocalc(k)
        return autocorrarr
   
    def autocalc(self, k):
        parammeans=self._acceptedsteps.mean(0)
        
        num_k=np.zeros(self._spacedim)
        denom_k=np.zeros(self._spacedim)
        
    
        for i in range(self._nsteps):
            #((x_i-xbar)
            x_i = self._acceptedsteps[i]-parammeans
            x_i2=x_i**2
            
            if i<self._nsteps-k:
                x_ik=self._acceptedsteps[i+k]-parammeans
                num_k+=x_ik*x_i
                
            denom_k+=x_i2
        return num_k/denom_k
        
class mcmc_training_gen(process_data):
    def __init__(self,trainsize=10, mcmcdim=10, mcmcsteps=10000, autocorrlag=1000, mcmc_stepsizes=None):
        process_data.__init__(self)
        self._trainsize=trainsize #amount of training data we want
        self._mcmcdim=mcmcdim #mcmc dimension
        self._mcmcsteps=mcmcsteps #mcmc steps to run
        self._autocorrlag=autocorrlag #lag on autocorrelations
        self._mcmc_stepsizes=mcmc_stepsizes
        self._appenddata=False #Append to previous array when running
        self._trainarr=[]
        
        self._processed_data=pd.DataFrame()     

    def getMCMCStepSizes(self):
        return self._mcmc_stepsizes
    
    def updateMCMCStepSizes(self,updated_vals):
        self._mcmc_stepsizes=updated_vals
        return 1

    def getMCMCDim(self):
        return self._mcmcdim
    
    def updateMCMCDim(self,updated_mcmcdimval):
        self._mcmcdim=updated_mcmcdimval
        return 1
    
    def getMCMCSteps(self):
        return self._mcmcsteps

    def updateMCMCSteps(self,updated_mcmcstepsval):
        self._mcmcsteps=updated_mcmcstepsval
        return 1
    
    def updateAutoCorrLag(self, updated_lagval):
        self._autocorrlag=updated_lagval
        return 1
    
    def getAutoCorrLag(self):
        return self._autocorrlag

    def updateTrainSize(self,updateval):
        self._trainsize=updateval
        return 1


    def updateTrainData(self, update_arr):
        #Updates the array of training data
        #can either append values or overwrite
        if self._appenddata:
            self.updateTrainSize(self._trainsize+len(update_arr))
            self._trainarr.append(update_arr)
        else:
            self._trainarr=update_arr
        return 1

    def getTrainData(self):
        return self._trainarr

    def createTrain(self, i):
        #generate mcmc instance
        #i just a placeholder!
        mc=mcmc(self._mcmcdim)
        mc_acc, mc_llh=mc(np.random.randn(self._mcmcdim),nsteps=self._mcmcsteps,stepsize=self._mcmc_stepsizes)
        mcauto=mc.autocorrs(self._autocorrlag)
        return [mc_acc, mc_llh, mcauto]
    
    def generateTrainingData(self):
        b_pool=mp.Pool()
        train_arr=b_pool.map(self.createTrain, range(self._trainsize))
        # train_arr=[]
        # for i in range(self.trainsize):
        #     train_arr.append(self.createTrain(i))
        self.updateTrainData(train_arr)
        return train_arr

    def __call__(self, llhbins=20, autocorr_integral_split=20, autocorr_reduce_size=50,append_data=False):
        #llhbins = number of llh bins
        #autocorr_integral_split=number of bins to split autocorr data into before integration
        #autocorr_reduce_size= how much to reduce autocorrelation by
        #append_data -> append data?
        self.generateTrainingData()
        #Set inputs for processing training data!
        self.setInputParams(self.getTrainData(), llhbins, autocorr_integral_split, autocorr_reduce_size)
        self.processTrainData(llhbins, autocorr_integral_split, autocorr_reduce_size, append_data)
        print("Processed data set has been created!")
        

    #Here's the idea:
        # Generate step size
        # 1 run standard NN training 
        # Whoever win the NN decides the step size
        # Need way to shrink variance of each step in a smart way
if __name__== "__main__":
    traindata=mcmc_training_gen(2,autocorrlag=200)
    traindata()
    print(traindata.getProcessedData())