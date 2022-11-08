#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import librosa
import librosa.display
import scipy as sp
import IPython.display as ipd
import matplotlib.pyplot as plt
from matplotlib import figure
import os
import math
import pandas as pd
import statistics as st
import csv
import soundfile as sf


# In[83]:


data_csv = pd.read_excel("C:/Users/MSI GF63/Desktop/P2M/metaData_Speech.xlsx")
dataset = "metaData_Speech.xlsx"
z=data_csv.iloc[:,0:2]

z.to_csv("data_new.csv", index=False)
data_split=pd.read_csv("data_new.csv")

cmap = plt.get_cmap('inferno')
tot_rows = data_csv.shape[0]
print(tot_rows)


# In[84]:


header = 'filename label mean_sc median_sc var_sc max_sc min_sc qunatile_25_sc qunatile_75_sc mean_ch_stft median_ch_stft var_ch_stft max_ch_stft min_ch_stft qunatile_25_ch_stft qunatile_75_ch_stft mean_rmse median_rmse var_rmse max_rmse min_rmse qunatile_rmse qunatile_rmse mean_spec_bw median_spec_bw var_spec_bw max_spec_bw min_sc qunatile_25_spec_bw qunatile_75_spec_bw mean_rolloff median_rolloff var_rolloff max_rolloff min_rolloff qunatile_25_rolloff qunatile_75_rolloff mean_zcr median_zcr var_zcr max_zcr min_zcr qunatile_25_zcr qunatile_75_zcr'
# for i in range(1, 21):
#     header += f' mfcc{i}'
print(header)
header = header.split()


# In[85]:


file = open('sig_feat_aa.csv', 'w')
with file:
    writer = csv.writer(file)
    writer.writerow(header)


# In[86]:


for i in range(tot_rows):
    source = data_split['filename'][i]
    label = data_split['label'][i]
    print(label)
    print("///////")
    print(source)
    print("///////")
    filename = 'C:/Users/MSI GF63/Desktop/P2M/wav/' + source

    x1,sr = librosa.load(filename, mono=True,)
    
    #   "removing silence"
    
    clips = librosa.effects.split(x1, top_db=25)
    wav_data = []
    for c in clips:
        data = x1[c[0]: c[1]]
        wav_data.extend(data)
    new_file_name='C:/Users/MSI GF63/Desktop/P2M/wav_new/' + source    
    sf.write(new_file_name, wav_data, sr)
    x1,sr = librosa.load(new_file_name, mono=True,)
   

    #   "Features extraction"
    
    
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    
    #Spectre Centroid
    sc_x1 = librosa.feature.spectral_centroid(y=x1, sr=sr, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]

    mean_sc=np.mean(sc_x1)
    median_sc=np.median(sc_x1)
    var_sc=np.var(sc_x1)
    max_sc=np.max(sc_x1)
    min_sc=np.min(sc_x1)
#     kurtosis_sc=sp.stats.kurtosis(sc_x1, bias=False)
#     skewness_sc=sp.stats.skew(sc_x1 ,bias=False)
    quantile_25_sc=np.quantile(sc_x1, .25)
    quantile_75_sc=np.quantile(sc_x1, .75)

#     print("the mean_sc = " ,mean_sc)
#     print("the median_sc = " ,median_sc)
#     print("the var_sc = " ,var_sc)
#     print("the max_sc = " ,max_sc)
#     print("the min_sc = " ,min_sc)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_sc = " ,quantile_25_sc)
#     print("the quantile_75_cs = " ,quantile_75_sc)
    
    
    # Chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=x1, sr=sr)
    
    mean_ch_stft=np.mean(chroma_stft)
    median_ch_stft=np.median(chroma_stft)
    var_ch_stft=np.var(chroma_stft)
    max_ch_stft=np.max(chroma_stft)
    min_ch_stft=np.min(chroma_stft)
#     kurtosis_ch_stft=sp.stats.kurtosis(chroma_stft, bias=False)
#     skewness_ch_stft=sp.stats.skew(chroma_stft,bias=False)
    quantile_25_ch_stft=np.quantile(chroma_stft, .25)
    quantile_75_ch_stft=np.quantile(chroma_stft, .75)
    
#     print("the mean_ch_stft = " ,mean_ch_stft)
#     print("the median_ch_stft = " ,median_ch_stft)
#     print("the var_ch_stft = " ,var_ch_stft)
#     print("the max_ch_stft = " ,max_ch_stft)
#     print("the min_ch_stft = " ,min_ch_stft)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_ch_stft = " ,quantile_25_ch_stft)
#     print("the quantile_75_ch_stft = " ,quantile_75_ch_stft)
    
    #rmse
    
    rmse = librosa.feature.rms(y=x1)
    
    mean_rmse=np.mean(rmse)
    median_rmse=np.median(rmse)
    var_rmse=np.var(rmse)
    max_rmse=np.max(rmse)
    min_rmse=np.min(rmse)
#     kurtosis_rmse=sp.stats.kurtosis(rmse, bias=False)
#     skewness_rmse=sp.stats.skew(rmse,bias=False)
    quantile_25_rmse=np.quantile(rmse, .25)
    quantile_75_rmse=np.quantile(rmse, .75)
    
#     print("the mean_rmse = " ,mean_rmse)
#     print("the median_rmse = " ,median_rmse)
#     print("the var_rmse = " ,var_rmse)
#     print("the max_rmse = " ,max_rmse)
#     print("the min_rmse = " ,min_rmse)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_rmse = " ,quantile_25_rmse)
#     print("the quantile_75_rmse = " ,quantile_75_rmse)
    
    
    #spec_bw
    spec_bw = librosa.feature.spectral_bandwidth(y=x1, sr=sr)
    
    mean_spec_bw=np.mean(spec_bw)
    median_spec_bw=np.median(spec_bw)
    var_spec_bw=np.var(spec_bw)
    max_spec_bw=np.max(spec_bw)
    min_spec_bw=np.min(spec_bw)
#     kurtosis_spec_bw=sp.stats.kurtosis(spec_bw, bias=False)
#     skewness_spec_bw=sp.stats.skew(spec_bw,bias=False)
    quantile_25_spec_bw=np.quantile(spec_bw, .25)
    quantile_75_spec_bw=np.quantile(spec_bw, .75)
    
#     print("the mean_spec_bw = " ,mean_spec_bw)
#     print("the median_spec_bw = " ,median_spec_bw)
#     print("the var_spec_bw = " ,var_spec_bw)
#     print("the max_spec_bw = " ,max_spec_bw)
#     print("the min_spec_bw = " ,min_spec_bw)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_spec_bw = " ,quantile_25_spec_bw)
#     print("the quantile_75_spec_bw = " ,quantile_75_spec_bw)
    
    
    #rolloff
    rolloff = librosa.feature.spectral_rolloff(y=x1, sr=sr)
    
    mean_rolloff=np.mean(rolloff)
    median_rolloff=np.median(rolloff)
    var_rolloff=np.var(rolloff)
    max_rolloff=np.max(rolloff)
    min_rolloff=np.min(rolloff)
#     kurtosis_rolloff=sp.stats.kurtosis(rolloff, bias=False)
#     skewness_rolloff=sp.stats.skew(rolloff,bias=False)
    quantile_25_rolloff=np.quantile(rolloff, .25)
    quantile_75_rolloff=np.quantile(rolloff, .75)
    
#     print("the mean_rolloff = " ,mean_rolloff)
#     print("the median_rolloff = " ,median_rolloff)
#     print("the var_rolloff = " ,var_rolloff)
#     print("the max_rolloff = " ,max_rolloff)
#     print("the min_rolloff = " ,min_rolloff)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_rolloff = " ,quantile_25_rolloff)
#     print("the quantile_75_rolloff = " ,quantile_75_rolloff)
      
 
    #zcr
    zcr = librosa.feature.zero_crossing_rate(x1)
    
    mean_zcr=np.mean(zcr)
    median_zcr=np.median(zcr)
    var_zcr=np.var(zcr)
    max_zcr=np.max(zcr)
    min_zcr=np.min(zcr)
#     kurtosis_zcr=sp.stats.kurtosis(zcr, fisher=False)
#     skewness_zcr=sp.stats.skew(zcr,bias=False)
    quantile_25_zcr=np.quantile(zcr, .25)
    quantile_75_zcr=np.quantile(zcr, .75)
#     mfcc = librosa.feature.mfcc(y=x1, sr=sr)
    
    
#     print("the mean_zcr = " ,mean_zcr)
#     print("the median_zcr = " ,median_zcr)
#     print("the var_zcr = " ,var_zcr)
#     print("the max_zcr = " ,max_zcr)
#     print("the min_zcr = " ,min_zcr)
# #     print("the kurtosis_zcr = " ,kurtosis_zcr)
# #     print("the skewness_zcr = " ,skewness_zcr)
#     print("the quantile_25_zcr = " ,quantile_25_zcr)
#     print("the quantile_75_zcr = " ,quantile_75_zcr)
    

    
    
    # "adding features to signal_features.csv"
    
    to_append = f'{source[:-3].replace(".", "")} {label} {mean_sc} {median_sc} {var_sc} {max_sc} {min_sc} {quantile_25_sc} {quantile_75_sc} {mean_ch_stft} {median_ch_stft} {var_ch_stft} {max_ch_stft} {min_ch_stft} {quantile_25_ch_stft} {quantile_75_ch_stft} {mean_rmse} {median_rmse} {var_rmse} {max_rmse} {min_rmse} {quantile_25_rmse} {quantile_75_rmse} {mean_spec_bw} {median_spec_bw} {var_spec_bw} {max_spec_bw} {min_spec_bw} {quantile_25_spec_bw} {quantile_75_spec_bw} {mean_rolloff} {median_rolloff} {var_rolloff} {max_rolloff} {min_rolloff} {quantile_25_rolloff} {quantile_75_rolloff} {mean_zcr} {median_zcr} {var_zcr} {max_zcr} {min_zcr} {quantile_25_zcr} {quantile_75_zcr} '
#     for e in mfcc:
#         to_append += f' {np.mean(e)}'
    file = open('sig_feat_aa.csv', 'a')
        
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
features_csv = pd.read_csv("sig_feat_aa.csv")
                                


# In[ ]:





# In[ ]:





# In[ ]:




