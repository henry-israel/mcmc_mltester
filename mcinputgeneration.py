import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
import multiprocessing as mp
from keras.models import Sequential, load_model
from keras.layers import Dense, Normalization, Conv1D, MaxPooling1D, Dropout
from numba.experimental import jitclass
import numba as nb
import tensorflow as tf
from tqdm.contrib.concurrent import process_map

######################################

class mcmc:
    def __init__(self, spacedim: int=10, data=None):
  
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

    @property
    def mu(self)->float:
        return self._mu
    
    @mu.setter
    def mu(self,update_mu: float)->None:
        self._mu=update_mu
    
    @property
    def cov(self):
        return self._cov
    
    @cov.setter
    def cov(self,updated_cov)->None:
        self._cov=updated_cov

    @property
    def spacedim(self)->int:
        return self._spacedim

    @spacedim.setter
    def spacedim(self,updated_dim: int)->None:
        self._spacedim=updated_dim

    @property
    def accepted_steps(self)->list:
        return self._acceptedsteps

    @property
    def step_matrix(self):
        return self._stepmatrix

    @property
    def accepted_llhs(self)->list:
        return self._acceptedllhs

    @property
    def n_steps(self):
        return self._nsteps
    @property
    def acceptance_rate(self):
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
            pdict={'stepsizes' : np.random.rand(self._spacedim)/np.random.randint(1,10,size=self._spacedim),
                   'startpos'  : np.random.rand(self._spacedim)}
            self._paramarr[i]=pdict
        self._acceptanceratearr=None

    @property
    def acceptance_rate(self)->float:
        return self._acceptancerate

    @property
    def step_arr(self)->np.array:
        step_arr=np.array([self._paramarr[i]['stepsizes'] for i in range(self._nchains)])
        return step_arr

    def runMCMC(self, i:int)->int:
            stepsizes=self._paramarr[i]['stepsizes']
            startpos=self._paramarr[i]['startpos']
            mcmc_=mcmc(self._spacedim,data=self._data)
            _, acceptancerate=mcmc_(startpos, stepsizes, self._nsteps)
            return acceptancerate

    def runMCMC_MultiProc(self)->list:
        acceptarr=process_map(self.runMCMC, range(self._nchains), chunksize=self._chunksize)
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

    def saveModel(self, output: str ="model_data")->None:
        print(f"mean used is : {self._data[0]}, saving to {output}_mean.csv")
        np.savetxt(f"{output}_mean.csv",self._data[0],delimiter=',')
        print(f"saving covariance matrix to {output}_cov.csv")
        np.savetxt(f"{output}_cov.csv",self._data[1],delimiter=',')
        print('saved')
    
    def usePreviousModel(self, meanfile: str="model_data_mean.csv", covmatrixfile: str="model_data_cov.csv"):
        print("Using covariance matrix and mean from previous run")
        meanval=np.loadtxt(meanfile, delimiter=',')
        cov=np.loadtxt(covmatrixfile, delimiter=',')
        self._data=[meanval, cov]

    def __call__(self, output: str)->None:
        print(f"Running {self._nchains} chains for {self._nsteps} steps with dimension {self._spacedim}")
        self.runMCMC_MultiProc()
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
        try:
            self._labeldf=fulldata[optimise_par_names]
        except ValueError:
            print(f"{optimise_par_names} is not in {data_file}")
        except:
            raise Exception(f"Sorry, something's gone wrong with your labels")

        data_df=fulldata.drop(optimise_par_names, axis=1)
        self._nentries=len(data_df)
        self._ndim=len(data_df.columns)
        if self._ndim==0:
            raise ValueError("Test data must have at least one category > 0")
        self._data_tensor=tf.convert_to_tensor(data_df)
        self._data_tensor=tf.reshape(self._data_tensor, (self._nentries,self._ndim))

        normaliser = Normalization(axis=-1)
        normaliser.adapt(self._data_tensor)
        self._model = Sequential([normaliser])
        self.setupNeuralNet()

    @property
    def labels(self)->list:
        return self._labels

    @property
    def data_tensor(self):
        return self._data_tensor

    @property
    def label_data_frame(self)->pd.DataFrame:
        return self._labeldf

    @property
    def model(self):
        return self._model

    def setupNeuralNet(self)->None:
        #This is where the classifier lives, obviously this can be tuned. Need to make more customisable!
        self._model.add(Dense(30, input_dim=(self._nentries, self._ndim), activation='relu'))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(20, activation='relu'))
        self._model.add(Dense(10, activation='relu'))
        self._model.add(Dense(1, activation='sigmoid'))

        #Compile the boi
        self._model.compile(loss='mse',  optimizer='Adam')


    def __call__(self, output: str="modeloutput")->None:
 
        self._model.fit(self._data_tensor, self._labeldf, epochs=1500, batch_size=100, use_multiprocessing=True, steps_per_epoch=100)
        print(self._model.evaluate(self._data_tensor, self._labeldf))
        print(f"saving to {output}")
        self._model.save(output)

class model_analyser:
    #Analytics kit for NN
    def __init__(self, model_file: str, test_data_file: str, optimise_par_names: list=['Acceptance_Rate'])->None:
        print(f"Testing model from {model_file} on test_data_file")
        self._model=load_model(model_file)
        self._dataset=pd.read_csv(test_data_file)
        
        try:
            self._truevals=self._dataset[optimise_par_names]
        except ValueError:
            print(f"{optimise_par_names} is not in testing data")
        except:
            raise Exception(f"Sorry, something's gone wrong with your labels")
      
        data_df=self._dataset.drop(optimise_par_names, axis=1)
        nentries=len(data_df)
        ndim=len(data_df.columns)
        if ndim==0:
            raise ValueError("Test data must have at least one category > 0")
        self._data_tensor=tf.convert_to_tensor(data_df)
        self._data_tensor=tf.reshape(self._data_tensor, (nentries,ndim))

        print("Calculating predicted values")
        self._predictvals=[p[0] for p in self._model.predict(self._data_tensor)]
        print(self._predictvals)

    def truePredPlot(self, label: str=None)->plt.figure():
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        truearr=np.array(self._truevals.to_numpy())
        print(len(truearr)-len(self._predictvals))
        ax.plot(truearr, self._predictvals,'.')
        ax.set_xlabel(f"True Values {' : '+label if label is not None else ''}")
        ax.set_ylabel(f"Predicted Values {' : '+label if label is not None else ''}")
        return fig,ax

    def percentDiffPlot(self, label: str=None)->plt.figure():
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        truearr=np.array(self._truevals.to_numpy())
        predarr=np.array(self._predictvals)
        perdiff=200*np.abs(truearr-predarr)/(truearr+predarr)
        ax.hist(perdiff, bins=20)
        ax.set_xlabel(f"Binned percentage difference between true/predicted vals {label if label is not None else ''}")
        return fig,ax

    def __call__(self, outfile: str = "diagnostics.pdf", label: str=''):
        #Call here will run through all the diagnostic plots
        print("doing plots")
        pdf=backend_pdf.PdfPages(outfile)
        truepred_fig, truepred_ax=self.truePredPlot(label=label)
        pdf.savefig(truepred_fig)

        perdiff_fig, perdif_ax=self.percentDiffPlot(label=label)
        pdf.savefig(perdiff_fig)

        pdf.close()
        print(f"Save figures to {outfile}")





#################THIS IS TEMPORARY!####################
#This is where the classifier is going to live for now, obviously can be made more modular later I'm just lazy!

if __name__== "__main__":
    #Run a load of MCMC
    mc_runner=multi_mcmc(nchains=2000, spacedim=30, nsteps=10000)
    mc_runner('train_set.csv')
    mc_runner.saveModel()
    #Classifier
    cfier=classifier('train_set.csv', ['Acceptance_Rate'])
    cfier('model_2Kchains_dim39_10KStep')
   
    mc_test_runner=multi_mcmc(nchains=100, spacedim=30, nsteps=10000)
    mc_test_runner.usePreviousModel('model_data_mean.csv', 'model_data_cov.csv')
    mc_test_runner('test_set.csv')
    

    #Let's grab our diagnostics
    mod_ana=model_analyser ('model_2Kchains_dim39_10KStep', 'test_set.csv')
    mod_ana('diagnostics.pdf', label='Acceptance Rate')