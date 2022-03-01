import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
import multiprocessing as mp
from keras.models import Sequential, load_model
from keras.layers import Dense, Normalization, Conv1D, MaxPooling1D, Dropout
import tensorflow as tf
import tqdm as tqdm


######################################

class mcmc():
    def __init__(self, spacedim: int=10, data=None, debug=False):
  
        self._spacedim=spacedim
        if data is not None:
            self._mu=data[0]
            self._cov=data[1]
        
       
        else:
            self._mu=np.random.randn(spacedim)
            #Make cov matrix
            sqrt_cov=np.random.randn(spacedim,spacedim)
            self._cov=np.dot(sqrt_cov,sqrt_cov.T)
            print("HELLO")

        if self._spacedim!=len(self._mu):
            raise ValueError("spatial dimension and mean must be the same length!")

        self._acceptedsteps=[]
        self._numberaccepted=0
        self._acceptedllhs=[]
        self._stepmatrix=[]
        self._nsteps=0
        self._total_steps=0
        self._local_state=np.random.RandomState(None)
        self._debug=debug

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
        if np.all(x < 50) and np.all(x > -50):
            return scipy.stats.multivariate_normal(mean=self._mu, cov=self._cov).logpdf(x)
        else:
            return -1e6
    
    def acceptFunc(self, curr_step: list, prop_step: list)->bool:
        alpha=self._local_state.uniform(0,1)
        diff=min(prop_step-curr_step, 500) #Prevents pesky run time overflows 
        fact=min(1,np.exp(diff))
        if alpha<fact:
            return True
        else:
            return False

       
    def proposeStep(self,curr_step: list)->list:
        #This will slow everything down considerably :(
        prop_step = curr_step + [self._stepsizes*self._local_state.randn(len(curr_step))]
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
    

    def __call__(self, startpos: list=None, stepsize: int=None, nsteps: int=10000)->None:
        
        self._nsteps=nsteps
        
        if stepsize is None:
            stepsize=np.random.uniform(low=0, high=0.50, size=self._spacedim)
        self._stepsizes=stepsize
        self._stepmatrix=np.diag(stepsize)
        
        if startpos is None:
            startpos=np.random.normal(loc=0, scale=1.0, size=self._spacedim)

        #setup mcmc
        curr_step=startpos
        curr_llh=self.loglikelihood(startpos)
        
        
        self._acceptedsteps=np.zeros((nsteps,self._spacedim))
        self._acceptedllhs=np.zeros(nsteps)
        
        self._acceptedsteps[0]=curr_step
        self._acceptedllhs[0]=curr_llh
        

        for step in range(1,self._nsteps):
            prop_step=self.proposeStep(curr_step)
            prop_llh=self.loglikelihood(prop_step)
            if self.acceptFunc(curr_llh, prop_llh):
                curr_step=prop_step
                curr_llh=prop_llh
                self._numberaccepted+=1

            self._acceptedsteps[step]=curr_step
            self._acceptedllhs[step]=curr_llh
            self._total_steps=step

            
        return self._numberaccepted/self._total_steps

######################################
class multi_mcmc():
    def __init__(self, nchains: int=100, spacedim: int=10, nsteps: int=10000, data=None, debug=False)->None:
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
        self._pool_size = mp.cpu_count()
        print(f"Using {self._pool_size} CPUs")
        #self._chunk_size=np.floor(self._nchains/(self._pool_size* 4))
        self._acceptanceratearr=None
        self._stepsizes=np.random.normal(loc=0, scale=1.0, size=(self._nchains,self._spacedim))
        self._debug=debug

    @property
    def acceptance_rate(self)->float:
        return self._acceptancerate

    @property
    def step_arr(self)->np.array:
        return self._stepsizes

    def runMCMC(self, i:int)->int:
            mcmc_=mcmc(self._spacedim,data=self._data, debug=self._debug)
            acceptancerate=mcmc_(stepsize=self._stepsizes[i],nsteps=self._nsteps)
            return acceptancerate

    def runMCMC_MultiProc(self)->list:
        print("MCMC Progress : ")
        if not self._debug:
            acceptarr=[]
            with mp.get_context("spawn").Pool(self._pool_size) as pool:
                with tqdm.tqdm(total=self._nchains) as pbar:
                    for a in pool.imap_unordered(self.runMCMC, range(self._nchains)):
                        acceptarr.append(a)
                        pbar.update(1)
            pool.join()
            pool.close()
        #Non-Parallelized, for debugging (SLOW)
        else:
            acceptarr=[]
            for j in range(self._nchains):
                acceptarr.append(self.runMCMC(j))
                print(f"completed {j}/{self._nchains}")
        self._acceptanceratearr=acceptarr
        return acceptarr

    def saveToFile(self, output: str)->None:
        #Let's make everything a nice dictionary first
        data={}
        data['Acceptance_Rate']=self._acceptanceratearr
        #This is grim, sorry!
        for j in range(self._spacedim):
            step_arr=[step[j] for step in self._stepsizes]
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
 
        self._model.fit(self._data_tensor, self._labeldf, epochs=150, batch_size=100, use_multiprocessing=True, steps_per_epoch=100)
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
        self._truearr=[i[0] for i in np.array(self._truevals.to_numpy())]
        self._predarr=np.array(self._predictvals)

    def truePredPlot(self, label: str=None)->plt.figure():
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        truearr=self._truearr
        predarr=self._predarr
        print(len(truearr)-len(predarr))
        ax.plot(truearr, predarr,'.')
        ax.set_xlabel(f"True Values {' : '+label if label is not None else ''}")
        ax.set_ylabel(f"Predicted Values {' : '+label if label is not None else ''}")

        #LOBF
        grad, intercept = np.polyfit(truearr, predarr, 1)
        lobfvals=np.linspace(min(truearr), max(truearr),num=len(truearr))
        ax.plot(lobfvals, grad*lobfvals+intercept,'-',color='r')
        print(f"line of best fit is prediction={grad}*trueval+{intercept}")
        return fig,ax

    def percentDiffPlot(self, label: str=None)->plt.figure():
        truearr=self._truearr
        predarr=self._predarr
        fig = plt.figure()
        axes=fig.add_subplot(1,1,1)

        perdiff=200*np.abs(truearr-predarr)/(truearr+predarr)
        axes.hist(perdiff, bins=20)
        axes.set_xlabel(f"Binned percentage difference between true/predicted vals {label if label is not None else ''}")
        return fig,axes

    def __call__(self, outfile: str = "diagnostics.pdf", label: str=''):
        #Call here will run through all the diagnostic plots
        print("doing plots")
        pdf=backend_pdf.PdfPages(outfile)
        truepred_fig, _=self.truePredPlot(label=label)
        pdf.savefig(truepred_fig)

        perdiff_fig, _=self.percentDiffPlot(label=label)
        pdf.savefig(perdiff_fig)

        pdf.close()
        print(f"Save figures to {outfile}")


if __name__== "__main__":
    import time

    DIMENSION=45
    NSTEPS=10000

    mc_runner=multi_mcmc(nchains=10000, spacedim=DIMENSION, nsteps=NSTEPS, debug=False)
    mc_runner.usePreviousModel('model_data_mean.csv', 'model_data_cov.csv')
    start=time.time()
    mc_runner('train_set.csv')
    end=time.time()
    print(f"Chain generation took {end-start}s to run")
    mc_runner.saveModel()
   
    mc_test_runner=multi_mcmc(nchains=100, spacedim=DIMENSION, nsteps=NSTEPS)
    mc_test_runner.usePreviousModel('model_data_mean.csv', 'model_data_cov.csv')
    mc_test_runner('test_set.csv')

    #Classifier
    cfier=classifier('train_set.csv', ['Acceptance_Rate'])
    cfier('model_2Kchains_dim39_10KStep')

    #Let's grab our diagnostics
    mod_ana=model_analyser ('model_2Kchains_dim39_10KStep', 'test_set.csv')
    mod_ana('diagnostics.pdf', label='Acceptance Rate')