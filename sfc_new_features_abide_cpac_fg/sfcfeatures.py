# This file:sfcfeatures.py contains the classes to compute 
# the new features from static functional connectivity
# obtained from ABIDE preprocessed rs-fMRI multisite
# data, pipeline CPAC, strategy filt and global.
# Repository: https://github.com/pcdslab/ASD-DiagNet-Confounds

# Possibility to stop warnings
import warnings
warnings.filterwarnings('ignore') 

# Basic data manipulation and visualisation libraries
import numpy as np
import pandas as pd
import glob
import shutil 
import os
import numpy.ma as ma # for masked arrays
from neuroCombat import neuroCombat
import functools
import pyprind
import random
import pickle
import itertools
from deepdiff import DeepDiff  # For Deep Difference of 2 objects

#Statistical libraries
import scipy
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from patsy import dmatrices


#This class represents a rs-fMRI multisite database 
class MultiSiteData:
    # initializes a multisite data with an empty list
    def __init__(self,data_phenotypic_path,data_files_path):
        #super().__init__(**kwargs)
        self.data_phenotypic_path = data_phenotypic_path
        self.data_files_path = data_files_path
        self.subjects_id = []
        self.files = []
        self.labels = {}
        self.gender = {}
        self.age = {}
        self.eyes = {}
        self.medicated = {}
        self.FIQ = {}
        self.handedness = {}
        self.files_list = os.listdir(self.data_files_path)
        
    # helper function to get subject id from name of multisite file     
    def get_id(self,filename):
        f_split = filename.split('_')
        if f_split[3] == 'rois':
            id = '_'.join(f_split[0:3]) 
        else:
            id = '_'.join(f_split[0:2])
        return id
    
    # get subject id from name of multisite file     
    def get_subjects_id(self):
        #print ('get_subject_id: ')
        #print ('get_subject_id: ',files_list)
        for f in range(len(self.files_list)):
            self.subjects_id.append(self.get_id(self.files_list[f])) 
        
    #get multisite files list
    def get_multisite_file_list(self):
        for file in self.files_list:
            self.files.append(file)
            
    #get phenotypic multisite data 
    def get_phenotypic_data(self):
        df_labels = pd.read_csv(self.data_phenotypic_path)      
        df_labels.DX_GROUP = df_labels.DX_GROUP.map({1: 1, 2:0})
        for row in df_labels.iterrows():
            subj_id = row[1]['FILE_ID']
            y_label = row[1]['DX_GROUP']
            y_gender = row[1]['SEX']
            y_age = row[1]['AGE_AT_SCAN']
            y_eyes = row[1]['EYE_STATUS_AT_SCAN']
            y_medicated = row[1]['CURRENT_MED_STATUS']
            y_FIQ = row[1]['FIQ']
            y_handedness = row[1]['HANDEDNESS_CATEGORY']
            if subj_id == 'no_filename':
                continue
            assert(subj_id not in self.labels)
            self.labels[subj_id] = y_label
            assert(subj_id not in self.gender)
            self.gender[subj_id] = y_gender
            assert(subj_id not in self.age)
            self.age[subj_id] = y_age
            assert(subj_id not in self.eyes)
            self.eyes[subj_id] = y_eyes
            assert(subj_id not in self.medicated)
            self.medicated[subj_id] = y_medicated
            assert(subj_id not in self.FIQ)
            self.FIQ[subj_id] = y_FIQ
            assert(subj_id not in self.handedness)
            self.handedness[subj_id] = y_handedness
            

##############################################################
#This subclass represents the static functional connectivity 
# computed from rs-fMRI multisite database
# using the Pearson linear correlation function
class StaticFunctionalConnectivity(MultiSiteData):
    # initializes sfc (static fuctional connectivity) with an empty dictionary
    # This dictionary has a key equal to each subject id and two values for the 
    # subject: static functional connectivity vector and label (autism or control subject).
    # Initializes an instance of the parent class 
    
    def __init__(self,data_phenotypic_path,data_files_path,input_data_path,p_ROI):
        super().__init__(data_phenotypic_path,data_files_path)        
        self.input_data_path = input_data_path
        self.p_ROI = p_ROI
        self.sfc = {}
        super().get_subjects_id()
        super().get_phenotypic_data()
        super().get_multisite_file_list()
        
    # Compute static FC as the upper triangular part of the static FC matrix for 
    # a given subject
    def get_sfc_subject(self,subject):
        #print(subject)
        for file in self.files:
            if file.startswith(subject):
                df = pd.read_csv(os.path.join(self.data_files_path, file), sep='\t')
        with np.errstate(invalid="ignore"):
            sfc = np.nan_to_num(np.corrcoef(df.T))
        mask = np.invert(np.tri(sfc.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, sfc).compressed()
    
    # Compute static FC for all subjects
    def get_sfc_multisubject(self, save_file):
        print('computing sFC .... ') 
        pbar=pyprind.ProgBar(len(self.subjects_id))
        for subject in self.subjects_id: 
            self.sfc[subject] = (self.get_sfc_subject(subject), self.labels[subject])
            pbar.update()
            
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(self.sfc, open(self.input_data_path+'sfc_feature_file_'+self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_feature_file_', self.p_ROI)
        print(pbar)
        print('computation of static FC completed  ok with: ', self.p_ROI) 

##############################################################        
# This subclass represents the eigenvectors and eigenvalues 
# computed for the static functional connectivity matrix and 
# for the matrices corresponding to the new features derived 
# from the static functional connectivity values
class Eigenvectors(MultiSiteData):
    # Initializes an instance of the parent class
    def __init__(self,data_phenotypic_path,data_files_path,
                 input_data_path,p_ROI,feature):        
        super().__init__(data_phenotypic_path,data_files_path)
        self.eig_data = {}        
        self.p_ROI = p_ROI
        self.input_data_path = input_data_path        
        self.feature = feature
        print('please wait while the input files are opening...for feature',
             self.feature)
        f = open(self.input_data_path+self.feature+'_file_'+self.p_ROI+'.pkl', 'rb') 
        self.feat_values = pickle.load(f)        
        f.close
        print('input files opened ok...')
        super().get_subjects_id()
        
    # Compute the matrix for the static functional connectivity
    # or for the values obtained for the new features computed
    # with the multiple linear regression models, ComBat 
    # harmonization models, and the standard methods for 
    # a given subject   
    def get_feature_matrix(self,feat_values,n):
        mtx = np.ones((n,n))
        k = 0
        for i in range(n):
            for j in range(n):
                if j > i:
                    mtx[i][j] = feat_values[k]
                    mtx[j][i] = feat_values[k]
                    k +=1    
        return mtx 
                       
    # Compute eigenvectors and eigenvalue of the static FC matrix
    # for all subjects
    def get_eigenvectors(self, save_file): 
        print('computing eigenvectors and eigenvalues...')
        if self.p_ROI == 'cc200':
            n = 200        
        pbar=pyprind.ProgBar(len(self.subjects_id))
        for subject in self.subjects_id:    
            d = self.get_feature_matrix(self.feat_values[subject][0],n)
            eig_vals, eig_vecs = np.linalg.eig(d)

            for ev in eig_vecs.T:
                np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

            sum_eigvals = np.sum(np.abs(eig_vals))
            # Make a list of (eigenvalue, eigenvector, norm_eigval) tuples
            eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i], np.abs(eig_vals[i])/sum_eigvals)
                     for i in range(len(eig_vals))]

            # Sort the (eigenvalue, eigenvector) tuples from high to low
            eig_pairs.sort(key=lambda x: x[0], reverse=True)
            self.eig_data[subject] = {'eigvals':np.array([ep[0] for ep in eig_pairs]),
                       'norm-eigvals':np.array([ep[2] for ep in eig_pairs]),
                       'eigvecs':[ep[1] for ep in eig_pairs]}
            pbar.update()
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(self.eig_data, open(self.input_data_path+'eig_data_'+
                                            self.feature+'_file_'+self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file eig_data_',self.feature,'_file_', self.p_ROI)
        print(pbar)
        print('computation of eigenvectors and eigenvalues completed  ok with: ', self.p_ROI) 

##############################################################        
# This subclass represents the list of dictionaries with static  
# functional connectivity (sFC) and Fisher Z-transform of SFC 
# values and phenotypic and MRI scanner data.
# The list contains one dictionary for each feature
class Dictionaries(MultiSiteData):
    # Initializes an instance of the parent class and open files with sFC data
    def __init__(self,data_phenotypic_path,data_files_path,input_data_path,p_ROI,
                 number_features):        
        super().__init__(data_phenotypic_path,data_files_path)
        self.p_ROI = p_ROI
        self.number_features = number_features
        self.list_dict =[]
        self.input_data_path = input_data_path
        print('please wait while the input files are opening...')
        f = open(input_data_path+'sfc_feature_file_'+p_ROI+'.pkl', 'rb') 
        self.sfc = pickle.load(f)
        f.close
        f = open(input_data_path+'sfc_fz_feature_file_'+p_ROI+'.pkl', 'rb') 
        self.sfc_fz = pickle.load(f)
        f.close
        print('input files opened ok...')
        # creation of a dictionary with MRI vendor per site if Siemens then value  = 1, 
        # else value = 2 and 
        self.site_mri_ve = {'Caltech':1,'CMU':1,'KKI':2,'Leuven':2,'MaxMun':1,
                            'NYU':1,'OHSU':1,'Olin':1,'Pitt':1,'SBL':2,'SDSU':2,
                            'Stanford':2,'Trinity':2,'UCLA':1,'UM':2,'USM':1,
                            'Yale':1}
        # creation of a site scanner dictionary with an integer value for each site
        # as a key as ComBat scanner data.
        self.scanner_s = {'Caltech':1,'CMU':2,'KKI':3,'Leuven':4,'MaxMun':5,'NYU':6,
                          'OHSU':7,'Olin':8,'Pitt':9,'SBL':10,'SDSU':11,'Stanford':12,
                          'Trinity':13,'UCLA':14,'UM':15,'USM':16,'Yale':17}      
        super().get_subjects_id() 
        super().get_phenotypic_data()
        
    # creation of list of dictionaries and computation of 
    # lists of sfc and sfc_fz data for the combat harmonization
    # models. 
    def get_list_dict(self, save_file):
        print('computing dictionaries...')
        sfc_combat_l = []
        sfc_fz_combat_l = []
        pbar=pyprind.ProgBar(self.number_features)
        for k in range(0,self.number_features):
            # lists for the k_th feature with 1035 values, one value per subject
            sub_k = [] 
            sfc_k = []
            sfc_fz_k = []
            age_k = []
            eyes_k = []
            gender_k = []
            FIQ_k = []
            MRI_k = []               
            scanner_k = []
    
            for subject in self.subjects_id:
                sub_k.append(subject)
                sfc_k.append(self.sfc[subject][0][k])
                sfc_fz_k.append(self.sfc_fz[subject][0][k])
                age_k.append(self.age[subject])
                eyes_k.append(self.eyes[subject])
                gender_k.append(self.gender[subject])
                FIQ_k.append(self.FIQ[subject])
                site = subject.split('_')[0]
                #if k == 0:
                    #n_subj +=1
                    #print ('site for subject #', site,n_subj)
                MRI_k.append(self.site_mri_ve[site])
                scanner_k.append(self.scanner_s[site])
            
            sfc_combat_l.append(sfc_k)
            sfc_fz_combat_l.append(sfc_fz_k)
            # the dictionary for feature k with values for 1035 subjects
            dict_k= {'sub_k':sub_k,'sfc_k':sfc_k,'sfc_fz_k':sfc_fz_k,'age_k':age_k,
                     'eyes_k':eyes_k,'gender_k':gender_k,'FIQ_k':FIQ_k,
                     'MRI_k':MRI_k,'scanner_k':scanner_k}  
            # list of dictionaries,  one dict_k for each of  the 19900 features
            self.list_dict.append(dict_k) 
            pbar.update()
            
        sfc_combat = np.array(sfc_combat_l)
        sfc_fz_combat = np.array(sfc_fz_combat_l)        
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(self.list_dict, open(self.input_data_path+'sfc_fz_list_dict_k_file_'+
                                             self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_fz_list_dict_k_file_', self.p_ROI)
            pickle.dump(sfc_combat, open(self.input_data_path+'sfc_data_combat_file_'+
                                         self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_data_combat_file_', self.p_ROI)
            pickle.dump(sfc_fz_combat, open(self.input_data_path+'sfc_fz_data_combat_file_'+
                                            self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_fz_data_combat_file_', self.p_ROI)
        print(pbar)    
        print('computation of dictionaries completed  ok with: ', self.p_ROI) 

##############################################################
# This subclass represents the multiple linear regression residuals 
# computed from the static functional connectivity (sFC) and
# Fisher Z-transform of SFC values and phenotypic and MRI scanner data.
class MultipleLinearRegrResiduals:
    def __init__(self,input_data_path,p_ROI,number_features): 
        self.p_ROI = p_ROI
        self.number_features = number_features
        self.input_data_path = input_data_path        
        self.ind_var = ['age','FIQ','MRI','gender','AGM']
        self.dep_ind_var_sfc = {'age':'sfc_k ~ age_k','FIQ':'sfc_k ~ FIQ_k',
                                'MRI':'sfc_k ~ MRI_k','gender': 'sfc_k ~ gender_k',
                                'AGM':'sfc_k ~ age_k + gender_k + MRI_k'}
        self.dep_ind_var_sfc_fz = {'age':'sfc_fz_k ~ age_k','FIQ':'sfc_fz_k ~ FIQ_k',
                                   'MRI':'sfc_fz_k ~ MRI_k','gender': 'sfc_fz_k ~ gender_k',
                                   'AGM':'sfc_fz_k ~ age_k + gender_k + MRI_k'} 
        print('please wait while the input files are opening...')
        f = open(self.input_data_path+'sfc_fz_list_dict_k_file_'+p_ROI+'.pkl', 'rb')        
        self.list_dict  = pickle.load(f)
        f.close 
        print('input files opened ok...')
        
    def get_mlr_residuals(self,save_file):                   
        for var in self.ind_var:
            print('computing mlr residuals for variable:',var)
            mlr_resid = []
            mlr_resid_fz = []
            pbar=pyprind.ProgBar(self.number_features)
            for k in range(0,self.number_features):
                y, X = dmatrices(self.dep_ind_var_sfc[var], data=self.list_dict[k], 
                                 return_type='dataframe')
                if k == 0:
                    print ('mlr for ',var, ' and ',self.dep_ind_var_sfc[var])                                             
                y_fz, X_fz = dmatrices(self.dep_ind_var_sfc_fz[var] , 
                                       data=self.list_dict[k], return_type='dataframe')
                if k == 0:
                    print ('mlr for ',var, ' and ',self.dep_ind_var_sfc_fz[var]) 
        
                mod = sm.OLS(y, X)    # Describe model
                results = mod.fit()       # Fit model 
                mlr_resid.append(results.resid) # list of residuals
        
                mod_fz = sm.OLS(y_fz, X_fz)   
                results_fz = mod_fz.fit()        
                mlr_resid_fz.append(results_fz.resid) 
                pbar.update()
            if save_file: 
                print('please wait while the ouput data files are saved...')
                pickle.dump(mlr_resid, open(self.input_data_path+'sfc_mlr_res_' + 
                                                 var+'_file_'+self.p_ROI+'.pkl', 'wb'))
                print('Output data saved to file sfc_mlr_res_ ',var, '_file_ ', self.p_ROI)    
                pickle.dump(mlr_resid_fz, open(self.input_data_path+'sfc_fz_mlr_res_' + 
                                                    var+'_file_'+self.p_ROI+'.pkl', 'wb'))
                print('Output data saved to file sfc_fz_mlr_res_ ',var, '_file_ ', self.p_ROI)
        print(pbar)
        print('Computation of mlr residuals completed ok with: ', self.p_ROI)    
        
        
##############################################################
# This subclass represents the multiple linear regression  
# features computed from the multiple linear regression
# residuals
class MultipleLinearRegrFeatures(MultiSiteData):
    # Initializes an instance of the parent class and open files with sFC data
    def __init__(self,data_phenotypic_path,data_files_path,input_data_path,
                 p_ROI,variable):                         
        super().__init__(data_phenotypic_path,data_files_path)
        self.p_ROI = p_ROI
        self.input_data_path = input_data_path
        self.var = variable
        print('please wait while the input files are opening...')
        f = open(input_data_path+'sfc_mlr_res_'+variable+
                 '_file_'+p_ROI+'.pkl', 'rb') 
        mlr_res  = pickle.load(f)
        f.close
        f = open(input_data_path+'sfc_fz_mlr_res_'+variable+
                 '_file_'+p_ROI+'.pkl', 'rb') 
        mlr_res_fz  = pickle.load(f)
        f.close
        print('input files opened ok...')
        self.mlr_res_tr = np.array(mlr_res).transpose()
        self.mlr_res_fz_tr = np.array(mlr_res_fz).transpose()
        super().get_subjects_id()
        super().get_phenotypic_data()
        
    def get_mlr_features(self,save_file):
        print('computing mlr features for variable:',self.var)
        sfc_mlr_feat = {}
        sfc_fz_mlr_feat = {}
        subj_num = 0
        for subj in self.subjects_id:
            sfc_mlr_feat[subj] = (self.mlr_res_tr[subj_num],
                                  self.labels[subj]) 
            sfc_fz_mlr_feat[subj] = (self.mlr_res_fz_tr[subj_num],
                                     self.labels[subj]) 
            subj_num +=1
            
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(sfc_mlr_feat, open(self.input_data_path+'sfc_mlr_'+
                                           self.var+'_feature_file_'+self.p_ROI+
                                           '.pkl', 'wb'))
            print('Output data saved to file sfc_mlr_', self.var,'_feature_file')
            pickle.dump(sfc_fz_mlr_feat, open(self.input_data_path+'sfc_fz_mlr_'+
                                              self.var+'_feature_file_'+self.p_ROI+
                                              '.pkl', 'wb'))                                              
            print('Output data saved to file sfc_fz_mlr_', self.var,'_feature__file') 
        print('computation of mlr feature for independent variable ',
              self.var,' completed ok with: ', self.p_ROI)     
           
 ##############################################################
# This subclass represents the ComBat harmonization models 
# computed from the static functional connectivity (sFC) and
# Fisher Z-transform of SFC values and phenotypic and MRI scanner data.
class CombatHarmonization:
    def __init__(self,input_data_path,p_ROI): 
        self.p_ROI = p_ROI
        self.input_data_path = input_data_path        
        self.ind_var = ['age','FIQ','AFG']
        print('please wait while the input files are opening...')
        f = open(self.input_data_path+'sfc_fz_list_dict_k_file_'+p_ROI+'.pkl', 'rb') 
        self.list_dict  = pickle.load(f)
        f.close
        f = open(input_data_path+'sfc_data_combat_file_'+p_ROI+'.pkl', 'rb') 
        self.sfc_data = pickle.load(f)
        f.close
        f = open(input_data_path+'sfc_fz_data_combat_file_'+p_ROI+'.pkl', 'rb') 
        self.sfc_fz_data = pickle.load(f)
        f.close
        print('input files opened ok...')
                
    def get_combat_harm(self,save_file):
        age_d = self.list_dict[0]['age_k']
        FIQ_d = self.list_dict[0]['FIQ_k']
        gender_d = self.list_dict[0]['gender_k']
        scanner_d = self.list_dict[0]['scanner_k']
        for var in self.ind_var:
            if var == 'age':            
                covars = pd.DataFrame({'age':age_d,'scanner':scanner_d})
                continuous_cols = ['age']
                batch_col = 'scanner'
                combat_age = neuroCombat(dat=self.sfc_data,covars=covars,
                                     batch_col=batch_col,continuous_cols =continuous_cols)
                combat_fz_age = neuroCombat(dat=self.sfc_fz_data,covars=covars,
                                        batch_col=batch_col,continuous_cols =continuous_cols)
                combat_age_data = np.array(combat_age['data'])
                combat_fz_age_data = np.array(combat_fz_age['data'])
                print('Computations Combat finished ok for independent variable',var)
                print('###############################################')
            elif var == 'FIQ':
                covars = pd.DataFrame({'FIQ':FIQ_d,'scanner':scanner_d})
                continuous_cols = ['FIQ']
                batch_col = 'scanner'
                combat_FIQ = neuroCombat(dat=self.sfc_data,covars=covars,
                                     batch_col=batch_col,continuous_cols =continuous_cols)
                combat_fz_FIQ = neuroCombat(dat=self.sfc_fz_data,covars=covars,
                                     batch_col=batch_col,continuous_cols =continuous_cols)
                combat_FIQ_data = np.array(combat_FIQ['data'])
                combat_fz_FIQ_data = np.array(combat_fz_FIQ['data'])
                print('Computations Combat finished ok for independent variable',var)
                print('###############################################')
            elif var == 'AFG':
                covars = pd.DataFrame({'age':age_d,'FIQ':FIQ_d,'gender':gender_d,
                                   'scanner':scanner_d})
                continuous_cols = ['age','FIQ']
                batch_col = 'scanner'
                categorical_cols = ['gender'] 
                combat_AFG = neuroCombat(dat=self.sfc_data,covars=covars,
                                     batch_col=batch_col,continuous_cols =continuous_cols)
                combat_fz_AFG = neuroCombat(dat=self.sfc_fz_data,covars=covars,
                                        batch_col=batch_col,categorical_cols=categorical_cols,
                                        continuous_cols =continuous_cols)
                combat_AFG_data = np.array(combat_AFG['data'])
                combat_fz_AFG_data = np.array(combat_fz_AFG['data'])
                print('Computations Combat completed ok for independent variable',var)
                print('###############################################')

        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(combat_age_data, open(self.input_data_path+
                                         'sfc_combat_harm_data_age_file_'+self.p_ROI+'.pkl', 'wb'))
            pickle.dump(combat_fz_age_data, open(self.input_data_path+
                                            'sfc_fz_combat_harm_data_age_file_'+self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_fz_combat_harm_data_age_file ')
            pickle.dump(combat_FIQ_data, open(self.input_data_path+
                                         'sfc_combat_harm_data_FIQ_file_'+self.p_ROI+'.pkl', 'wb'))
            pickle.dump(combat_fz_FIQ_data, open(self.input_data_path+
                                            'sfc_fz_combat_harm_data_FIQ_file_'+self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_fz_combat_harm_data_FIQ_file')
            pickle.dump(combat_AFG_data, open(self.input_data_path+
                                         'sfc_combat_harm_data_AFG_file_'+self.p_ROI+'.pkl', 'wb'))
            pickle.dump(combat_fz_AFG_data, open(self.input_data_path+
                                            'sfc_fz_combat_harm_data_AFG_file_'+self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file sfc_fz_combat_harm_data_AFG_file')
        print('Computations Combat harmonization completed ok with: ', self.p_ROI) 
              
        
##############################################################
# This subclass represents the Combat harmonization  
# features computed from the results of tha ComBat harmonization 
# models
class CombatHarmonizationFeatures(MultiSiteData):
    # Initializes an instance of the parent class and open files with 
    # ComBat harmonization results          
    def __init__(self,data_phenotypic_path,data_files_path,input_data_path,
                 p_ROI,variable):                         
        super().__init__(data_phenotypic_path,data_files_path)
        self.p_ROI = p_ROI
        self.input_data_path = input_data_path
        self.var = variable
        print('please wait while the input files are opening...')
        f = open(input_data_path+'sfc_combat_harm_data_'+variable+
                 '_file_'+p_ROI+'.pkl', 'rb') 
        combat_harm  = pickle.load(f)
        f.close
        f = open(input_data_path+'sfc_fz_combat_harm_data_'+variable+
                 '_file_'+p_ROI+'.pkl', 'rb') 
        combat_harm_fz  = pickle.load(f)
        f.close
        self.combat_harm_tr = combat_harm.transpose()
        self.combat_harm_fz_tr = combat_harm_fz.transpose()
        super().get_subjects_id()
        super().get_phenotypic_data()
        
    def get_combat_features(self,save_file):
        print('computing combat features for variable:',self.var)
        sfc_combat_feat = {}
        sfc_fz_combat_feat = {}
        subj_num = 0
        for subj in self.subjects_id:
            sfc_combat_feat[subj] = (self.combat_harm_tr[subj_num],
                                  self.labels[subj]) 
            sfc_fz_combat_feat[subj] = (self.combat_harm_fz_tr[subj_num],
                                     self.labels[subj]) 
            subj_num +=1
            
        if save_file: 
            print('please wait while the ouput data files are saved...')
            pickle.dump(sfc_combat_feat, open(self.input_data_path+'sfc_combat_'+
                                           self.var+'_feature_file_'+self.p_ROI+
                                           '.pkl', 'wb'))
            print('Output data saved to file sfc_combat_', self.var,'_feature_file_')
            pickle.dump(sfc_fz_combat_feat, open(self.input_data_path+'sfc_fz_combat_'+
                                              self.var+'_feature_file_'+self.p_ROI+
                                              '.pkl', 'wb'))                                              
            print('Output data saved to file sfc_combat_fz_', self.var,'_feature__file') 
        print('Computations Combat harmonization features completed ok with: ', self.p_ROI) 
        
##############################################################
# This subclass represents the normalization methods  
# features: Fisher z-score of sFC and demeaning methods
class NormalizationMethodsFeatures(MultiSiteData):
    # Initializes an instance of the parent class and open files with sFC data
    def __init__(self,data_phenotypic_path,data_files_path,input_data_path,
                 p_ROI,number_features):                         
        super().__init__(data_phenotypic_path,data_files_path)
        self.p_ROI = p_ROI
        self.input_data_path = input_data_path
        self.number_features = number_features
        print('please wait while the input files are opening...')
        f = open(input_data_path+'sfc_feature_file_'+p_ROI+'.pkl', 'rb') 
        self.sfc = pickle.load(f)
        f.close
        print ('Files opened ok')
        super().get_subjects_id()
        super().get_phenotypic_data()
        super().get_multisite_file_list()
        
    # Compute the Fisher Z-transform of static FC for a given 
    # subject as the upper triangular  part of the Fisher
    # Z-transform of the static FC matrix 
    def get_sfc_fz_subject(self,subject):
        for file in self.files:
            if file.startswith(subject):
                df = pd.read_csv(os.path.join(self.data_files_path, file), sep='\t')            
        with np.errstate(invalid="ignore"):
            sfc_mtx = np.nan_to_num(np.corrcoef(df.T))
            sfc_fz_mtx = np.arctanh(sfc_mtx)
            mask = np.invert(np.tri(sfc_fz_mtx.shape[0], k=-1, dtype=bool))
            m = ma.masked_where(mask == 1, mask)
            return ma.masked_where(m, sfc_fz_mtx).compressed() 
    
    # Compute the Fishers Z-score transform of the static functional 
    # connectivity (sFC) for the 1035 ABIDE subjects
    def get_sfc_fz(self,save_file):
        print('computing Fishers Z-score transform of sFC')
        sfc_fz = {}
        pbar=pyprind.ProgBar(len(self.subjects_id))
        for subj in self.subjects_id:
            sfc_fz[subj] = (self.get_sfc_fz_subject(subj), self.labels[subj])
            pbar.update()
        
        if save_file:
            pickle.dump(sfc_fz, open(self.input_data_path+'sfc_fz_feature_file_'
                                     +self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file: sfc_fz_feature_file_',self.p_ROI,'.pkl')
        print(pbar)
        print('Computation of the Fishers Z-score transform of sFC completed ok with: ', 
              self.p_ROI)
    
    #compute static functional connectivity residual avg = delta_avg
    def get_residual_avg(self,save_file):
        print('computing residual avg')
        sfc_avg_feat = []
        sfc_res_avg = {}
        print('computation of delta_avg running..')
        for k in range(0,self.number_features):
            sfc_avg_l = []    
            for subj in self.subjects_id:
                sfc_avg_l.append(self.sfc[subj][0][k])        
            sfc_avg_feat.append(np.mean(np.array(sfc_avg_l))) 
            #pbar.update()        
        #compute delta_avg (sfc_res_avg)
        for subj in self.subjects_id:
            sfc_res_avg[subj] = (self.sfc[subj][0]-sfc_avg_feat, self.labels[subj])    
            #pbar.update()   

        
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(sfc_res_avg, open(self.input_data_path+'sfc_res_avg_feature_file_'
                                          +self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file: sfc_res_avg_feature_file_',self.p_ROI,'.pkl')
        print('computation of new feature: delta_avg completed ok with: ', self.p_ROI)
        
    #compute static functional connectivity residual average per site = delta_avg_site 
    def get_residual_avg_site(self,save_file):
        sites = ['Caltech','CMU','KKI','Leuven','MaxMun','NYU','OHSU',
                 'Olin','Pitt','SBL','SDSU','Stanford','Trinity',
                 'UCLA','UM','USM','Yale']
        sfc_res_avg_site = {}
        print('computation of delta_avg_site running..') 
        #compute overall average of sFC for a given site
        for SITE in sites: 
            sfc_site_l  = []        
            for subj in self.subjects_id:
                subj_split = subj.split('_')
                if subj_split[0] == SITE:
                    sfc_site_l.append(self.sfc[subj][0])  
                                    
            sfc_site_a = np.array(sfc_site_l)
            sfc_overall_avg_site = np.mean(sfc_site_a)
    
            #compute delta_avg_site for a given site
            for subj in self.subjects_id:        
                subj_split = subj.split('_')
                if subj_split[0] == SITE:
                    sfc_res_avg_site[subj] = (self.sfc[subj][0]- sfc_overall_avg_site, 
                                              self.labels[subj])
                            
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(sfc_res_avg_site, open(self.input_data_path+'sfc_res_avg_site_feature_file_'+
                                           self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file: sfc_res_avg_site_feature_file_',self.p_ROI,'.pkl')
        print('computation of new feature: delta_avg_site completed ok with: ', self.p_ROI)
        
    # compute static functional connectivity residual average per subject = delta_avg_subject
    def get_residual_avg_subj(self,save_file):
        sfc_res_avg_subj = {} 
        sfc_subj_l  = []
        print('computation of delta_avg_subj running..')
        #compute average of sFC for each subject over all features
        for subj in self.subjects_id:
            sfc_subj_l.append(self.sfc[subj][0])  
                                    
        sfc_subj_a = np.array(sfc_subj_l)
        sfc_avg_subj = np.mean(sfc_subj_a, axis=1)
        #compute delta_avg_subject
        k = 0
        for subj in self.subjects_id: 
            sfc_res_avg_subj[subj] = (self.sfc[subj][0]- sfc_avg_subj[k], self.labels[subj])
            k +=1 
            
        if save_file:
            print('please wait while the ouput data files are saved...')
            pickle.dump(sfc_res_avg_subj, open(self.input_data_path+'sfc_res_avg_subj_feature_file_'+
                                               self.p_ROI+'.pkl', 'wb'))
            print('Output data saved to file: sfc_res_avg_subj_feature_file_',self.p_ROI,'.pkl')    
        print('computation of new feature: delta_avg_subj completed ok with: ', self.p_ROI)           
            