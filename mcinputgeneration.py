import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import multiprocessing as mp
import keras
from keras.models import Sequential
from keras.layers import Dense


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
            if stepcount%np.floor(nsteps/10)==0:
                print(f"Completed {stepcount}/{nsteps} steps, accepted {self._numberaccepted}")
            
            
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
        
class mcmc_training_gen():
    def __init__(self,trainsize=10000, mcmcdim=10, mcmcsteps=10000, autocorrlag=1000):
        
        self._trainsize=trainsize #amount of training data we want
        self._mcmcdim=mcmcdim #mcmc dimension
        self._mcmcsteps=mcmcsteps #mcmc steps to run
        self._autocorrlag=autocorrlag #lag on autocorrelations
        self._trainarr=[]
    

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

    def updateTrainData(self, update_arr,append_vals=False):
        #Updates the array of training data
        #can either append values or overwrite
        if append_vals:
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
        mc_acc, mc_llh=mc(np.random.randn(self._mcmcdim),nsteps=self._mcmsteps)
        mcauto=mc.autocorrs(self._autocorrlag)
        return [mc_acc, mc_llh, mcauto]
    
    def __call__(self,append_vals=True):
        b_pool=mp.Pool()
        train_arr=b_pool.map(self.createTrain, range(self._trainsize))
        # train_arr=[]
        # for i in range(self.trainsize):
        #     train_arr.append(self.createTrain(i))
        self.updateTrainData(train_arr,append_vals)
        return train_arr
        
    #Here's the idea:
        # Generate step size
        # 1 run standard NN training 
        # Whoever win the NN decides the step size
        # Need way to shrink variance of each step in a smart way
      



if __name__== "__main__":
    x=mcmc_ml(2)
    trset = x()