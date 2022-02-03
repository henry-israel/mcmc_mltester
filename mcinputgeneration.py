import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import multiprocessing as mp


class mcmc:
    def __init__(self, spacedim=10):
        
        self.spacedim=spacedim
        self.mu=np.random.randn(spacedim)
        
        #Make cov matrix
        sqrt_cov=np.random.randn(spacedim,spacedim)
        self.cov=np.dot(sqrt_cov,sqrt_cov.T)
        self.acceptedsteps=[]
        self.numberaccepted=0
        self.acceptedllhs=[]
        self.stepmatrix=[]
        self.nsteps=0
        
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
            return scipy.stats.multivariate_normal(mean=self.mu, cov=self.cov).logpdf(x)
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
        prop_step = curr_step + np.dot(self.stepmatrix, np.random.randn(len(curr_step)))
        return prop_step
    
    def __call__(self, startpos, stepsize=None, nsteps=10000):
        
        self.nsteps=nsteps
        
        if stepsize==None:
            stepsize=np.ones(self.spacedim)
        self.stepmatrix=np.diag(stepsize)
        
        #setup mcmc
        curr_step=startpos
        curr_llh=self.loglikelihood(startpos)
        
        
        self.acceptedsteps=np.zeros((nsteps,self.spacedim))
        self.acceptedllhs=np.zeros(nsteps)
        
        self.acceptedsteps[0]=curr_step
        self.acceptedllhs[0]=curr_llh
        
        stepcount=1
        self.numberaccepted=1
        while stepcount<nsteps:
            if stepcount%np.floor(nsteps/10)==0:
                print(f"Completed {stepcount}/{nsteps} steps, accepted {self.numberaccepted}")
            
            
            prop_step=self.proposeStep(curr_step)
            prop_llh=self.loglikelihood(prop_step)
            if self.acceptFunc(curr_llh, prop_llh):
                curr_step=prop_step
                curr_llh=prop_llh
                self.numberaccepted+=1

            self.acceptedsteps[stepcount]=curr_step
            self.acceptedllhs[stepcount]=curr_llh
            
            stepcount+=1
        return self.acceptedsteps, self.acceptedllhs
    
    def autocorrs(self,totlag=1000):
        print("Making Autocorrelations")
        a_pool=mp.Pool()
        autocorrarr=a_pool.map(self.autocalc, range(totlag))
        return autocorrarr
   
    def autocalc(self, k):
        parammeans=self.acceptedsteps.mean(0)
        
        num_k=np.zeros(self.spacedim)
        denom_k=np.zeros(self.spacedim)
        
    
        for i in range(self.nsteps):
            #((x_i-xbar)
            x_i = self.acceptedsteps[i]-parammeans
            x_i2=x_i**2
            
            if i<self.nsteps-k:
                x_ik=self.acceptedsteps[i+k]-parammeans
                num_k+=x_ik*x_i
                
            denom_k+=x_i2
        return num_k/denom_k
        

if __name__== "__main__":
    x=mcmc(10)
    xacc, xllhs= x(np.zeros(10),nsteps=100000)
    x_auto=x.autocorrs(1000)
    plt.plot(x_auto)
    plt.show()
        
        