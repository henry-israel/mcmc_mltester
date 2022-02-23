import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
import multiprocessing as mp
from keras.models import Sequential
from keras.layers import Dense, Normalization, Conv1D, MaxPooling1D, Dropout
#from numba import config, njit, threading_layer
import tensorflow as tf
from tqdm.contrib.concurrent import process_map

######################################

class mcmc:
    def __init__(self, spacedim: int=10, data=None):

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
        self._total_steps=0
        self._local_state=np.random.RandomState(None)

    def getData(self)->pd.DataFrame():
        return self._data

    def getMeanVal(self)->float:
        return self._mu

    def updateMeanVal(self,update_mu: float)->float:
        self._mu=update_mu
        return 0
    
    def getCov(self):
        return self._cov
    
    def updateCov(self,updated_cov)->None:
        self._cov=updated_cov

    def getSpaceDim(self)->int:
        return self._spacedim

    def updateSpaceDim(self,updated_dim: int)->None:
        self._spacedim=updated_dim

    def getAcceptedSteps(self)->list:
        return self._acceptedsteps

    def getStepMatrix(self):
        return self._stepmatrix

    def getAcceptedLLHS(self)->list:
        return self._acceptedllhs

    def getNSteps(self):
        return self._nsteps
    
    def getAcceptanceRate(self):
        if self._total_steps:
            return self._numberaccepted/self._total_steps
        else:
            return 0

    def loglikelihood(self, x: list)->float:
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
    
    def acceptFunc(self, curr_step: list, prop_step: list)->bool:
        alpha=self._local_state.uniform(0,1)
        fact=min(1,np.exp(prop_step-curr_step))
        if alpha<fact:
            return True
        else:
            return False
        
    def proposeStep(self,curr_step: list)->list:
        prop_step = curr_step + np.dot(self._stepmatrix, self._local_state.randn(len(curr_step)))
        return prop_step

    def autocorrs(self,totlag: int=1000)->list:
        print("Making Autocorrelations")
        if "daemon" not in mp.current_process()._config:
            a_pool=mp.Pool()
            autocorrarr=a_pool.map(self.autocalc, range(totlag))
        else:
            autocorrarr=np.empty((totlag, self._spacedim))
            for k in range(totlag):
                autocorrarr[k]=self.autocalc(k)
        return autocorrarr
   
    def autocalc(self, k: int)->float:
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
    
    def __call__(self, startpos: list, stepsize: int=None, nsteps: int=10000)->None:
        
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
        

        while self._total_steps<nsteps:
            # if stepcount%np.floor(nsteps/10)==0:
            #     print(f"Completed {stepcount}/{nsteps} steps, accepted {self._numberaccepted}")
            
            
            prop_step=self.proposeStep(curr_step)
            prop_llh=self.loglikelihood(prop_step)
            if self.acceptFunc(curr_llh, prop_llh):
                curr_step=prop_step
                curr_llh=prop_llh
                self._numberaccepted+=1

            self._acceptedsteps[self._total_steps]=curr_step
            self._acceptedllhs[self._total_steps]=curr_llh
            
            self._total_steps+=1
        return self._acceptedsteps, self._numberaccepted/self._total_steps

######################################
class multi_mcmc():
    def __init__(self, nchains: int=100, spacedim: int=10, nsteps: int=10000, data=None)->None:
        self._nchains=nchains
        self._spacedim=spacedim
        self._nsteps=nsteps
        if data is None: #We'll just use a fake data set here
            mu=np.random.randn(spacedim)
            #Make cov matrix
            sqrt_cov=np.random.randn(spacedim,spacedim)
            cov=np.dot(sqrt_cov,sqrt_cov.T)
            self._data=[mu,cov]
        else:
            self._data=data
        pool_size = mp.cpu_count()
        self._chunksize, extra = divmod(self._nchains, pool_size* 4)
        if extra:
            self._chunksize+=1


        self._paramarr=np.empty(nchains, object)
        for i in range(nchains):
            pdict={'stepsizes' : np.random.rand(self._spacedim),
                   'startpos'  : np.random.rand(self._spacedim)}
            self._paramarr[i]=pdict
        self._acceptanceratearr=None

    def getAcceptanceRate(self)->float:
        return self._acceptancerate

    def getStepSizes(self)->np.array:
        step_arr=np.array([self._paramarr[i]['stepsizes'] for i in range(self._nchains)])
        return step_arr

    def runmcmc(self, i:int)->int:
            stepsizes=self._paramarr[i]['stepsizes']
            startpos=self._paramarr[i]['startpos']
            mcmc_=mcmc(self._spacedim,data=self._data)
            _, acceptancerate=mcmc_(startpos, stepsizes, self._nsteps)
            return acceptancerate

    def runMCMC(self)->list:
        acceptarr=process_map(self.runmcmc, range(self._nchains), chunksize=self._chunksize)
        # train_arr=[]
        # for i in range(self.trainsize):
        #     train_arr.append(self.createTrain(i))
        self._acceptanceratearr=acceptarr
        return acceptarr

    def saveToFile(self, output: str)->None:
        #Let's make everything a nice dictionary first
        data={}
        data['Acceptance_Rate']=self._acceptanceratearr
        #This is grim, sorry!
        for j in range(self._spacedim):
            step_arr=[step['stepsizes'][j] for step in self._paramarr]
            data[f'Step_{j}']=step_arr

        dftosave=pd.DataFrame(data)
        dftosave.to_csv(output, index=False)
        print(f"Saved to {output}")

    def __call__(self, output: str)->None:
        print(f"Running {self._nchains} chains for {self._nsteps} steps with dimension {self._spacedim}")
        self.runMCMC()
        print(f"MCMC has been run, saving to {output}")
        self.saveToFile(output)


######################################

class classifier():
    def __init__(self, data_file: str, optimise_par_names: list, hyperparams: dict=None)->None:
        #Add way to generate good+bad dataframes here!
        '''
        data_file : points to file address for data stored in csv/other format
        optimise_par_names: list of names of parameter(s) you want to optimise, should be list of strings
        hyperparams: all the hyper params you could want!
        '''
        if not isinstance(optimise_par_names, list):
            raise TypeError(f"{optimise_par_names} is not a list, please provide as list")

        if not isinstance(data_file, str):
            raise TypeError(f"{data_file} is not a string, please give me a string")
        
        if not(isinstance(hyperparams, dict) or hyperparams is None):
            raise TypeError(f"{hyperparams} is not a dict, gimme a dict")

        self._hyperparams=hyperparams

        fulldata=pd.read_csv(data_file)
        print(fulldata)
        try:
            self._labeldf=fulldata[optimise_par_names]
        except ValueError:
            print(f"{optimise_par_names} is not in {data_file}")
        except:
            raise Exception(f"Sorry, something's gone wrong with your labels")

        data_df=fulldata.drop(optimise_par_names, axis=1)
        self._inputsize=len(data_df)
        if self._inputsize==0:
            raise ValueError("Test data must have length > 0")
        self._data_tensor=tf.convert_to_tensor(data_df)
        self._data_tensor=tf.reshape(self._data_tensor, (100,10))
        print(self._data_tensor)

        normaliser = Normalization(axis=-1)
        normaliser.adapt(self._data_tensor)
        self._model = Sequential([normaliser])
        self.setupNeuralNet()


    def getLabels(self)->list:
        return self._labels

    def getDataTensor(self):
        return self._data_tensor

    def getLabelDataFrame(self)->pd.DataFrame:
        return self._labeldf

    def getClassifier(self):
        return self._model

    def getModel(self):
        return self._model

    def setupNeuralNet(self)->None:
        #HARDCODED BAD DON'T DO THIS!
        #This is where the classifier lives, obviously this can be tuned. Need to make more customisable!
        self._model.add(Conv1D(32, 3, input_shape=(10, 100), activation='relu')) #Let's be spicy and add a conv alyer
        self._model.add(MaxPooling1D(3))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(8, activation='relu'))
        self._model.add(Dense(1, activation='sigmoid'))

        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __call__(self):
 
        self._model.fit(self._data_tensor, self._labeldf, epochs=150, batch_size=100)
        _, accuracy = self._model.evaluate(self._data_tensor, self._labeldf)
        print(f"accuracy is {accuracy}")



#################THIS IS TEMPORARY!####################
#This is where the classifier is going to live for now, obviously can be made more modular later I'm just lazy!

if __name__== "__main__":
    #Run a load of MCMC
    mc_runner=multi_mcmc(nchains=100, spacedim=100, nsteps=1000)
    mc_runner('test.csv')

    # cfier=classifier('test.csv', ['Acceptance_Rate'])
    # cfier()