# This file:asdiagnetconf.py contains the classes to  
# compute classificatio scores with the jupyter 
# notebook ASD-DiagNet-Confounds.
# Version: https://github.com/pcdslab/ASD-DiagNet-Confounds

# Possibility to stop warnings
import warnings
warnings.filterwarnings('ignore') 

# Basic data manipulation and visualisation libraries
import pandas as pd
import numpy as np
import os
from functools import reduce
import time
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pyprind
import sys
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
import functools
import numpy.ma as ma # for masked arrays
import pyprind
import random

#Statistical libraries
import scipy
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix

#This class represents a rs-fMRI multisite database 
class MultiSiteData:
    # initializes a multisite data with an empty list
    def __init__(self,data_phenotypic_path,data_files_path):
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
        files_list = os.listdir(self.data_files_path)
        for f in range(len(files_list)):
            self.subjects_id.append(self.get_id(files_list[f])) 
        
    #get multisite files list
    def get_multisite_file_list(self):
        #print (self.data_files_path)
        files_list = os.listdir(self.data_files_path)
        for file in files_list:
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
#This subclass represents the baseline and homogeneous
# subsamples
class SubSamples(MultiSiteData):    
    def __init__(self,data_phenotypic_path,data_files_path,
                p_center,max_add_control,subsample_length):
        super().__init__(data_phenotypic_path,data_files_path)        
        self.p_center = p_center
        self.max_add_control = max_add_control
        self.centers_dict = {}
        self.subsample = []
        super().get_subjects_id()
        super().get_phenotypic_data()
        super().get_multisite_file_list()
    
    # get homogeneous subsamples (hss) and centers_dict
    def get_hss(self):        
        if self.p_center == 'hss_age_10':
            self.subsample = ['KKI','MaxMun','NYU','OHSU','Olin',
                               'Pitt','SDSU','Stanford','UCLA','UM',
                               'USM','Yale'] 
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 0 and self.age[sub] <= 10: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
        elif self.p_center == 'hss_age_1015':
            self.subsample = ['KKI','Leuven','MaxMun','NYU','OHSU','Olin',
                               'Pitt','SDSU', 'Stanford','Trinity','UCLA',
                               'UM','USM','Yale'] 
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 10 and self.age[sub] <= 15: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_age_1520':
            self.subsample = ['Caltech','CMU','Leuven','MaxMun','NYU',
                                'OHSU','Olin','Pitt','SBL','SDSU',
                                'Trinity','UCLA','UM','USM','Yale']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 15 and self.age[sub] <= 20: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_age_1020':
            self.subsample = ['Caltech','CMU','KKI','Leuven','MaxMun',
                               'NYU','OHSU','Olin','Pitt','SBL',
                               'SDSU','Stanford','Trinity','UCLA','UM',
                               'USM','Yale']  
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 10 and self.age[sub] <= 20: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_age_20':
            self.subsample = ['Caltech','CMU','Leuven','MaxMun','NYU',
                               'Olin','Pitt','SBL','Trinity','UM','USM']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 20: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_FIQ_89':
            self.subsample = ['KKI','Leuven','MaxMun','NYU','OHSU',
                               'Olin','Pitt','SDSU','Stanford','Trinity',
                               'UCLA','UM','USM','Yale']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.FIQ[sub] > 0 and self.FIQ[sub] <= 89: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_FIQ_89_110':
            self.subsample = ['Caltech','CMU','KKI','Leuven','MaxMun',
                               'NYU','OHSU','Olin','Pitt','SDSU',
                               'Stanford','Trinity','UCLA','UM',
                               'USM','Yale']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.FIQ[sub] > 89 and self.FIQ[sub] <= 110: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_FIQ_110':
            self.subsample = ['Caltech','CMU','KKI','Leuven','MaxMun',
                               'NYU','OHSU','Olin','Pitt','SDSU',
                               'Stanford','Trinity','UCLA','UM',
                               'USM','Yale']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.FIQ[sub] > 110: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
        elif self.p_center == 'hss_age_1020_FIQ_89110':
            self.subsample = ['CMU','KKI','Leuven','MaxMun',
                               'NYU','OHSU','Olin','Pitt','SDSU',
                               'Stanford','Trinity','UCLA','UM',
                               'USM','Yale']
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 10 and self.age[sub] <= 20 and self.FIQ[sub] > 89 and self.FIQ[sub] <= 110: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
        elif self.p_center == 'hss_age_1020_FIQ_89':
            self.subsample = ['KKI','Leuven','MaxMun','NYU','OHSU',
                               'Olin','Pitt','Stanford','Trinity',
                               'UCLA','UM','USM','Yale']                             
            self.centers_dict = {}
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 10 and self.age[sub] <= 20 and self.FIQ[sub] > 0 and self.FIQ[sub] <= 89: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)    
        elif self.p_center == 'hss_FIQ_89_bal':
            self.subsample = ['KKI','Leuven','MaxMun','NYU','OHSU',
                               'Olin','Pitt','SDSU','Stanford','Trinity',
                               'UCLA','UM','USM','Yale']
            count_control = 0
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.FIQ[sub] > 0 and self.FIQ[sub] <= 89: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
                elif self.labels[sub] == 0 and  count_control < self.max_add_control:
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
                    count_control += 1     
        elif self.p_center == 'hss_age_1020_FIQ_89_bal':
            self.subsample = ['KKI','Leuven','MaxMun','NYU','OHSU',
                               'Olin','Pitt','SDSU','Stanford','Trinity',
                               'UCLA','UM','USM','Yale'] 
            count_control = 0
            for sub in self.subjects_id:
                key = sub.split('_')[0]
                if  self.age[sub] > 10 and self.age[sub] <= 20 and self.FIQ[sub] > 0 and self.FIQ[sub] <= 89: 
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
                elif self.labels[sub] == 0 and  count_control < self.max_add_control:
                    if key not in self.centers_dict:
                        self.centers_dict[key] = []
                    self.centers_dict[key].append(sub)
                    count_control += 1 
                    
    # get baseline subsamples (bss)                
    def get_bss(self):        
        if self.p_center  == 'bss_4':
            self.subsample = ['KKI','OHSU','Olin','USM']
        elif self.p_center == 'bss_5':
            self.subsample = ['KKI','NYU','OHSU','Olin','USM']
        elif self.p_center == 'bss_6':
            self.subsample = ['KKI','NYU','OHSU','Olin','UCLA','USM']
        elif self.p_center == 'bss_7':
            self.subsample = ['KKI','NYU','OHSU','Olin','UCLA','USM',
                              'Yale']
        elif self.p_center == 'bss_8':
            self.subsample = ['KKI','NYU','OHSU','Olin','Stanford',
                              'UCLA','USM','Yale']
        elif self.p_center == 'bss_9':
            self.subsample = ['CMU','KKI','NYU','OHSU','Olin',
                              'Stanford','UCLA','USM','Yale']
        elif self.p_center == 'bss_10':
            self.subsample = ['CMU','KKI','NYU','OHSU','Olin',
                              'Stanford','UCLA','UM','USM','Yale']
        elif self.p_center == 'bss_11':
            self.subsample = ['CMU','KKI','Leuven','NYU','OHSU',
                              'Olin','Stanford','UCLA','UM','USM',
                              'Yale']
        elif self.p_center == 'bss_12':
            self.subsample = ['CMU','KKI','Leuven','NYU','OHSU',
                              'Olin','Pitt', 'Stanford','UCLA','UM',
                              'USM','Yale']
        elif self.p_center == 'bss_13':
            self.subsample = ['CMU','KKI','Leuven','NYU','OHSU',
                              'Olin','Pitt','SDSU','Stanford','UCLA',
                              'UM','USM','Yale']
        elif self.p_center == 'bss_14':
            self.subsample = ['CMU','KKI','Leuven','NYU','OHSU',
                              'Olin','Pitt','SBL','SDSU','Stanford',
                              'UCLA','UM','USM','Yale'] 
        elif self.p_center == 'bss_15':    
            self.subsample = ['CMU','KKI','Leuven','MaxMun','NYU',
                              'OHSU','Olin','Pitt','SBL','SDSU',
                              'Stanford','UCLA','UM','USM','Yale']
        elif self.p_center == 'bss_16':    
            self.subsample = ['Caltech','CMU','KKI','Leuven','MaxMun',
                              'NYU','OHSU','Olin','Pitt','SBL',
                              'SDSU','Stanford','UCLA','UM','USM','Yale']                            
        elif self.p_center == 'whole':    
            self.subsample = ['Caltech','CMU','KKI','Leuven','MaxMun',
                              'NYU','OHSU','Olin','Pitt','SBL',
                              'SDSU','Stanford','Trinity','UCLA',
                              'UM','USM','Yale']
    
##############################################################
# This subclass represents the helper functions in 
# ASD-DiagNet
class HelperFunctions(MultiSiteData):    
    def __init__(self,data_phenotypic_path,data_files_path,feat_data,
                eig_data):
        super().__init__(data_phenotypic_path,data_files_path) 
        super().get_subjects_id()
        super().get_phenotypic_data()
        super().get_multisite_file_list()
        self.feat_data = feat_data
        self.eig_data = eig_data
        self.scores = [] 
        self.regions_inds = []
        self.centers_dict = {}
        self.subj_id_list = []
        self.control_autism = [0,0]
        self.norm_weights = []
        self.res = 0.0
                
    # compute regions indexes
    def get_regs_inds(self,samplesnames,reg_num):
        datas = []
        for sn in samplesnames:
            datas.append(self.feat_data[sn][0])
        datas = np.array(datas)
        avg=[]
        for ie in range(datas.shape[1]):
            avg.append(np.mean(datas[:,ie]))
        avg=np.array(avg)
        highs=avg.argsort()[-reg_num:][::-1]
        lows=avg.argsort()[:reg_num][::-1]
        reg_inds=np.concatenate((highs,lows),axis=0)
        self.regions_inds = reg_inds.tolist()

    # compute center dictionary    
    def get_centers_dict(self):        
        for sub in self.subjects_id:
            key = sub.split('_')[0]
            if key not in self.centers_dict:
                self.centers_dict[key] = []
            self.centers_dict[key].append(sub)
            
    # compute subject id list for a given subsample        
    def get_subj_id_list(self,length,subsample,centers_dict):        
        for sn in range(length):
            arr = np.array(centers_dict[subsample[sn]])
            self.subj_id_list = np.append(self.subj_id_list,arr)
            
    # compute number of control and autistic subjects        
    def get_number_subjects(self,subj_id_list):    
        for subj in subj_id_list:
            if self.labels[subj] == 0:
                self.control_autism[0] += 1
            else:
                self.control_autism[1] += 1
    
    # output of classification results
    def output_results(self,all_rp_res,repeat,output_name,
                       p_center,p_mode,p_ROI,p_feature):
        res = np.mean(np.array(all_rp_res),axis = 0)
        sd  = np.std(np.array(all_rp_res),axis = 0) 
        results = open(output_name, 'a')
        results.write('res-avg' + ',' + str(res[0]) + ',' + str(res[1]) + ',' 
                  + str(res[2])  + '\n')                          
        results.write('std' +  ',' + str(sd[0]) + ',' + str(sd[1]) + ',' 
                  + str(sd[2])  + '\n') 
        results.close() 
   
        print ('p_center: ',p_center)
        print ('p_mode: ',p_mode)
        print ('p_ROI ', p_ROI)
        print ('p_feature ',p_feature)    
        print('Average result of ',repeat,' repeats: ',res)
        print('Standard deviation of ',repeat,' repeats: ',sd )
        
    # print partial classification results
    def output_repeat_results(self,rp,r,start_time):   
        print('Result of repeat ', rp,':','averages:',r)
        finish_time = time.time()
        run_time = finish_time-start_time
        print('Running time:',run_time) 
    
    # Calculating Eros similarity
    def get_norm_weights(self,sub_list,num_dim):
        norm_weights_arr = np.zeros(shape=num_dim)
        for subj in sub_list:
            norm_weights_arr += self.eig_data[subj]['norm-eigvals'] 
        self.norm_weights = norm_weights_arr.tolist()

    def cal_similarity(self,d1, d2, weights, lim=None):    
        if lim is None:
            weights_arr = weights.copy()
        else:
            weights_arr = weights[:lim].copy()
            weights_arr /= np.sum(weights_arr)
        for i,w in enumerate(weights_arr):
            self.res += w*np.inner(d1[i], d2[i])
        return self.res 
    
    # compute classification scores    
    def confusion(self,g_turth,predictions):
        tn, fp, fn, tp = confusion_matrix(g_turth,predictions).ravel()
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        sensitivity = (tp)/(tp+fn)
        specificity = (tn)/(tn+fp)
        sc_tuple = (accuracy,sensitivity,specificity)
        self.scores = list(sc_tuple)
        
    # defining testing model
    def test(self,model, criterion, test_loader,
                 eval_classifier=False, num_batch=None,
                 device=None):
        #print('device:',device)
        test_loss, n_test, correct = 0.0, 0, 0
        all_predss=[]
        if eval_classifier:
            y_true, y_pred = [], []
        with torch.no_grad():
            model.eval()
            for i,(batch_x,batch_y) in enumerate(test_loader, 1):
                if num_batch is not None:
                    if i >= num_batch:
                        continue
                data = batch_x.to(device)
                rec, logits = model(data, eval_classifier)

                test_loss += criterion(rec, data).detach().cpu().numpy() 
                n_test += len(batch_x)
                if eval_classifier:
                    proba = torch.sigmoid(logits).detach().cpu().numpy()
                    preds = np.ones_like(proba, dtype=np.int32)
                    preds[proba < 0.5] = 0
                    all_predss.extend(preds)###????
                    y_arr = np.array(batch_y, dtype=np.int32)

                    correct += np.sum(preds == y_arr)
                    y_true.extend(y_arr.tolist())
                    y_pred.extend(proba.tolist())
                self.confusion(y_true,all_predss)
        return self.scores
    
###########################################################
# This subclass represents the Autoencoder of ASD-DiagNet
class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs=990, 
                 num_latent=200, tied=True,
                 num_classes=2, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent
        
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
    
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
         
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        
        if use_dropout:
            self.classifier = nn.Sequential (
                nn.Dropout(p=0.4),
                nn.Linear(self.num_latent, 1),
                
            )
        else:
            self.classifier = nn.Sequential (
                nn.Linear(self.num_latent, 1),
            )
            
         
    def forward(self, x, eval_classifier=False):
        x = self.fc_encoder(x)
        x = torch.tanh(x)
        if eval_classifier:
            x_logit = self.classifier(x)
        else:
            x_logit = None
        
        if self.tied:
            x = F.linear(x, self.fc_encoder.weight.t())
        else:
            x = self.fc_decoder(x)
            
        return x, x_logit
    
###########################################################        
# This subclass represents the data loader of ASD-DiagNet, 
# with the inner class DiagDataset.
class DiagDataLoader(HelperFunctions):
    def __init__(self,data_phenotypic_path,
                data_files_path,feat_data,
                 eig_data,num_dim):
        super().__init__(data_phenotypic_path,
                         data_files_path,feat_data,
                         eig_data)
        self.num_dim = num_dim
        
        
    def get_loader(self,data=None, samples_list=None,                
                   batch_size=64,num_workers=1, mode='train',
               *, augmentation=False, aug_factor=2, num_neighbs=5,
                   eig_data=None,similarity_fn=None, verbose=False,
                   regions=None):
        #Build and return data loader
        if mode == 'train':
            shuffle = True
        else:
            shuffle = False
            augmentation=False
        
        self.get_norm_weights(samples_list,self.num_dim)
        weights = np.array(self.norm_weights)
        dataset = self.DiagDataset(data=data,samples_list=samples_list,
                           augmentation=augmentation, aug_factor=aug_factor, 
                           num_neighbs=5,eig_data=eig_data,
                           similarity_fn=similarity_fn,verbose=verbose,
                           regs=regions,weights=weights)

        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,
                             num_workers=num_workers)
  
        return data_loader

   
    class DiagDataset(Dataset):
        def __init__(self,data=None,samples_list=None,
                     augmentation=False,aug_factor=2,
                     eig_data=None,num_neighbs=5,
                     similarity_fn=None, verbose=False,
                     regs=None,weights=None):
            
            self.regs=regs
            if data is not None:
                self.data = data.copy()            
            else:
                sys.stderr.write('Data is needed!')
                return                        
        
            self.subj_id = [subj for subj in samples_list]
            self.labels = np.array([self.data[subj][1] for subj in self.subj_id])
        
            current_subj_id = np.array(self.subj_id.copy())
            current_lab0_subj_id = current_subj_id[self.labels == 0]
            current_lab1_subj_id = current_subj_id[self.labels == 1]
                
            if augmentation:
                self.num_data = aug_factor * len(self.subj_id)
                #print ('self.num_data: ',self.num_data)
                self.neighbors = {}
                for subj in self.subj_id:
                    label = self.data[subj][1]
                    candidates = (set(current_lab0_subj_id) if label == 0 else set(current_lab1_subj_id))
                    candidates.remove(subj)
                    eig_subj = eig_data[subj]['eigvecs']
                    sim_list = []
                    for cand in candidates:
                        eig_cand = eig_data[cand]['eigvecs']
                        sim = similarity_fn(eig_subj, eig_cand,weights)
                        sim_list.append((sim, cand))
                    sim_list.sort(key=lambda x: x[0], reverse=True)
                    self.neighbors[subj] = [item[1] for item in sim_list[:num_neighbs]]
        
            else:
                self.num_data = len(self.subj_id)
        
        def __getitem__(self, index):
            if index < len(self.subj_id):
                subjname = self.subj_id[index]
                data = self.data[subjname][0].copy() #get_corr_data(fname, mode=cal_mode)    
                data = data[self.regs].copy()
                label = (self.labels[index],)
                return torch.FloatTensor(data), torch.FloatTensor(label)
            else:
                subj1 = self.subj_id[index % len(self.subj_id)]
                d1, y1 = self.data[subj1][0], self.data[subj1][1]
                d1=d1[self.regs]
                subj2 = np.random.choice(self.neighbors[subj1])
                d2, y2 = self.data[subj2][0], self.data[subj2][1]
                d2=d2[self.regs]
                assert y1 == y2
                r = np.random.uniform(low=0, high=1)
                label = (y1,)
                data = r*d1 + (1-r)*d2
                return torch.FloatTensor(data), torch.FloatTensor(label)

        def __len__(self):
            return self.num_data
        
        