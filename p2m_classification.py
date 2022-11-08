#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import IPython.display as ipd
import matplotlib.pyplot as plt
from matplotlib import figure
import os
import math
import pandas as pd
import statistics as st
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #Seaborn is a library for making statistical graphics in Python
from scipy import stats
from scipy.stats import pearsonr #Pearson correlation coefficient and p-value for testing non-correlation.


# In[2]:


df= pd.read_excel("C:/Users/MSI GF63/Desktop/sig_feat_p2mm_modified.xlsx")
# df.head()


# In[3]:


l=df["Medhistory"].unique()
l0=df["Symptoms"].unique()
l3=df["Age"].unique()
l1=[]
l2=[]
for i in l0:
    l1+=i.split(',')
for i in l0:
    l2.append(i.split(','))

d={'None':0, 'pnts':0,'lung':10,'hbp':5,'diabetes':20,'hbp,otherHeart,diabetes':40,'pulmonary,heart,':25,
   'asthma,copd,':35,'asthma':30,'hiv':10,'longterm':40,'heart,valvular,otherHeart,':50,'copd,hbp':10,'hbp,otherHeart':20, 'hbp,diabetes':25,
       'asthma,lung,diabetes,long':100, 'asthma,diabetes,longterm,':90, 'cystic':20,
       'otherHeart':15, 'asthma,hbp,':35, 'valvular,diabetes,cancer,':90,
       'hbp,heart,otherHeart,':35, 'hbp,longterm,':45, 'asthma,copd,hiv,':45,
       'hbp,diabetes,':25, 'None,longterm,':40,
       'pulmonary,hbp,angina,otherHeart,':31, 'asthma,heart,':45,
       'lung,diabetes,longterm,':70, 'copd':5, 'asthma,copd,longterm,':75,
       'cancer':50, 'lung,hiv,':15, 'hbp,otherHeart,':20}

d0={'None':0, 'chills':10,'shortbreath':100,'fever':90,'dizziness':20,'pnts':0,'tightness':20,
  'wetcough':50, 'muscleache':70, 'drycough':70, 'headache':50, 'sorethroat':80, 'smelltasteloss':100}

d1={'50-59':5, '30-39':3, '40-49':4, '90-':9, '60-69':6, '20-29':2 ,'70-79':7, '0-19':1 ,'pnts':4,
 '16-19':1, '80-89':8,}


# In[4]:


for i in d :
    df=df.replace({'Medhistory': i}, {'Medhistory': d[i]}, regex=True)
for h in l0 :
    for i in h.split(',') :
        d0[h]=0
        d0[h]= d0[h]+ d0[i]
        df=df.replace({'Symptoms': h}, {'Symptoms': d0[h]}, regex=True)

for i in d1 :
    df=df.replace({'Age': i}, {'Age': d1[i]}, regex=True)


# In[5]:


# df.head(15)


# In[6]:


df=df.drop(columns=["Filename", "var_sc","qunatile_75_sc","mean_rolloff","var_rolloff","mean_zcr","qunatile_25_zcr","var_spec_bw"])


# In[7]:


# df.head()


# In[8]:


[l,c]=df.shape
# print(l)
# print(c)


# In[9]:


# df.describe()


# In[10]:


# df['label'].hist()


# In[11]:


# df.apply(lambda x: sum(x.isnull()),axis=0) #number of missing values in each column


# In[12]:


# df.dtypes


# In[13]:


# df.mean()


# In[ ]:





# In[ ]:





# In[108]:


matrice_corr = df.corr().round(1)
plt.figure(figsize=(15,15))
sns.heatmap(data=matrice_corr, annot=True)# ????


# In[15]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


# In[ ]:





# In[16]:


y=df.iloc[:,c-1]
x=df.iloc[:,:c-2]
# y_col=df.iloc[:,c-1]


# In[17]:


from sklearn.preprocessing import StandardScaler 
x = StandardScaler().fit_transform(x)
x


# In[18]:


#corr_mat_label=y_col.corr().round(2)


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:





# # import and apply PCA

# In[20]:


from sklearn.decomposition import PCA 
  
pca = PCA(0.98) 
  
pca.fit(x_train)

# explained_variance = pca.explained_variance_ratio_


# In[21]:


x_train = pca.transform(x_train)
x_test = pca.transform(x_test)


# In[22]:


type(x_test)


# In[23]:


x_test.shape


# In[ ]:





# In[24]:


x_train.shape


# In[25]:


from sklearn.model_selection import KFold ,StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                             random_state=1)
sss.get_n_splits(x, y)


# In[ ]:





# # model with logisticRegression
# 

# In[26]:


from sklearn.linear_model import LogisticRegression 
LR = LogisticRegression(random_state=0 ,solver='lbfgs', max_iter=1500)
LR.fit(x_train,y_train)
y_pred_LR=LR.predict(x_test)


# In[27]:


LR.score(x_train,y_train)


# In[28]:


LR.score(x_test,y_test)


# In[29]:


cm_LR=confusion_matrix(y_test,y_pred_LR)
cm_LR


# true positives:186 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives :33 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive:27 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative:72 -> prédiction est négative, la valeur réelle est positive

# In[30]:


mae_LR=mean_absolute_error(y_test,y_pred_LR)
print("Mean Absolute Error::",mae_LR)


# In[31]:


print("Accuracy of LR_classifier::",accuracy_score(y_test,y_pred_LR))


# In[32]:


print("Accuracy of LR_classifier with Kfold (StratifiedShuffle)::",np.mean(cross_val_score(LR,x,y,cv=sss)))


# the logisticRegression model have 68% accuracy 

# In[ ]:





# # Model  with SVM

# In[33]:


from sklearn.svm import SVC
svm = SVC(gamma='auto')
svm.fit(x_train, y_train)
y_pred_svm=svm.predict(x_test)


# In[34]:


svm.score(x_train,y_train)


# In[35]:


svm.score(x_test,y_test)


# In[36]:


cm_svm=confusion_matrix(y_test,y_pred_svm)
cm_svm


# true positives:186 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives :33 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive:27 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative:72 -> prédiction est négative, la valeur réelle est positive

# In[37]:


mae_svm=mean_absolute_error(y_test,y_pred_svm)
print("Mean Absolute Error::",mae_svm)


# In[38]:


print("Accuracy of LR_classifier::",accuracy_score(y_test,y_pred_svm))


# In[39]:


print("Accuracy of SVM with Kfold (StratifiedShuffle)::",np.mean(cross_val_score(svm,x,y,cv=sss)))


# In[ ]:





# In[ ]:





# #  model with DecisionTreeClassifier

# In[40]:


from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier(random_state=1)
DTC.fit(x_train,y_train)
y_pred_DTC= DTC.predict(x_test)


# In[41]:


DTC.score(x_train,y_train)


# In[42]:


DTC.score(x_test,y_test)


# In[43]:


cm_DTC=confusion_matrix(y_test,y_pred_DTC)
cm_DTC


# true positives:186 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives :33 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive:27 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative:72 -> prédiction est négative, la valeur réelle est positive

# In[44]:


mae_DTC=mean_absolute_error(y_test,y_pred_DTC)
print("Mean Absolute Error::",mae_DTC)


# In[45]:


print("Accuracy of DT_classifier::",accuracy_score(y_test,y_pred_DTC))


# In[46]:


print("Accuracy of DT_classifier with Kfol (StratifiedShuffle)::",np.mean(cross_val_score(DTC,x,y,cv=sss)))


# the DecisionTreeClassifier model have 68% accuracy

# In[ ]:





# # Changing the max_depth

# In[47]:


train_accuracy = []
test_accuracy =[]
for depth in range (1,20):
    dt_model = DecisionTreeClassifier(max_depth=depth,random_state=10)
    dt_model.fit(x_train,y_train)
    train_accuracy.append(dt_model.score(x_train,y_train))
    test_accuracy.append(dt_model.score(x_test,y_test))
    


# In[48]:


frame=pd.DataFrame({'max_depth':range(1,20), 'train_acc':train_accuracy, 'test_acc':test_accuracy})
frame


# In[49]:


plt.figure(figsize=(12,6))
plt.plot(frame['max_depth'],frame['train_acc'],marker='o')
plt.plot(frame['max_depth'],frame['test_acc'],marker='o')
plt.xlabel('Depth of tree')
plt.ylabel('performance')
plt.legend()


# In[50]:


DTC_m=DecisionTreeClassifier(max_depth=13,max_leaf_nodes=20,random_state=20)


# In[51]:


DTC_m.fit(x_train,y_train)
y_pred_DTCm= DTC_m.predict(x_test)


# In[52]:


DTC_m.score(x_train,y_train)


# In[53]:


DTC_m.score(x_test,y_test)


# In[54]:


mae_DTCm=mean_absolute_error(y_test,y_pred_DTCm)
print("Mean Absolute Error::",mae_DTCm)


# In[55]:


cm_DTCm=confusion_matrix(y_test,y_pred_DTCm)
cm_DTCm


# true positives:195 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives :49 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive:18 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative:56 -> prédiction est négative, la valeur réelle est positive

# In[56]:


print("Accuracy of DT_classifier_maxdepath=15::",accuracy_score(y_test,y_pred_DTCm))


# In[57]:


print("Accuracy of DT_classifier_maxdepath=15 with Kfol (StratifiedShuffle)::",np.mean(cross_val_score(DTC_m,x,y,cv=sss)))


# 
# # the DecisionTreeClassifier model with max_depth =3   have 76,7% accuracy

# # Random Forest Classifier
# 

# In[58]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=33)
RF.fit(x_train, y_train)
y_pred_RF= RF.predict(x_test)


# In[59]:


RF.score(x_test, y_test)


# In[60]:


RF.score(x_train, y_train)


# In[61]:


mae_RF=mean_absolute_error(y_test,y_pred_RF)
print("Mean Absolute Error::",mae_RF)


# In[62]:


cmRF=confusion_matrix(y_test,y_pred_RF)
cmRF


# true positives: 195 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives : 49 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive: 18 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative: 56 -> prédiction est négative, la valeur réelle est positive

# In[63]:


print("Accuracy of RF_classifier::",accuracy_score(y_test,y_pred_RF))


# In[64]:


print("Accuracy of RF_classifier with Kfol (StratifiedShuffle)::",np.mean(cross_val_score(RF,x,y,cv=sss)))


# In[ ]:





# 

# # the XGBoostClassifier 

# In[65]:


import xgboost as xgb
XGB= xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


# In[66]:


XGB.fit(x_train,y_train)


# In[67]:


y_pred_XGB = XGB.predict(x_test)


# In[68]:


XGB.score(x_test, y_test)


# In[69]:


XGB.score(x_train, y_train)


# In[70]:


mae_XGB=mean_absolute_error(y_test,y_pred_XGB)
print("Mean Absolute Error::",mae_XGB)


# In[71]:


cm_XGB=confusion_matrix(y_test,y_pred_XGB)
cm_XGB


# true positives:195 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives :49 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive:18 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative:56 -> prédiction est négative, la valeur réelle est positive

# In[72]:


print("Accuracy of XGBClassifier::",accuracy_score(y_test,y_pred_XGB))


# In[73]:


print("Accuracy of XGBClassifier with Kfol (StratifiedShuffle)::",np.mean(cross_val_score(XGB,x,y,cv=sss)))


# In[ ]:





# # Visualyse performance

# In[74]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_axes([0,0,1,1])
langs = ['LR', 'SVM', 'DTC', 'RF', 'XGB']
students = [np.mean(cross_val_score(LR,x,y,cv=sss)),np.mean(cross_val_score(svm,x,y,cv=sss)),np.mean(cross_val_score(DTC_m,x,y,cv=sss)),np.mean(cross_val_score(RF,x,y,cv=sss)),np.mean(cross_val_score(XGB,x,y,cv=sss))]
ax.bar(langs,students,color ='maroon',width = 0.4)
plt.title("Visualize model performance")
plt.show()


# In[75]:


print("la matrice de confusion de RF")
print(cmRF)
print("la matrice de confusion de XGB")
print(cm_XGB)
print("la matrice de confusion de svm")
print(cm_svm)


# true positives: 152 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives : 37 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive: 44 ->prédiction est positive,la valeur réelle est négative.
# 
# et la plus important c'est le False Negatvive
# 
# False Negative: 5 -> prédiction est négative, la valeur réelle est positive

# # 

# # Feature importance

# In[76]:


# Feature Importance
xgb.plot_importance(XGB)
plt.rcParams['figure.figsize'] = [7, 15]
plt.show()


# In[ ]:





# # Alternative using Dmatrix 

# In[77]:


data_dmatrix = xgb.DMatrix(data=x,label=y)


# In[78]:


# Parameters
params = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }


# In[79]:


# Create an XGBoost Classifier
xgb_clf = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)


# In[80]:


xgb.plot_importance(xgb_clf)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()


# # Model Interpretation

# In[81]:


import eli5
from eli5 import show_weights


# In[82]:


# Interpret Our Model
# show_weights(xgb_clf)


# using Shap

# In[83]:


import shap  
shap.initjs()


# In[84]:


# Create object that can calculate shap values
sh_explainer = shap.TreeExplainer(xgb_clf)


# In[85]:


# Calculate Shap values
# shap_values = sh_explainer.shap_values(x_train)


# In[86]:


# Summary Plot 
explainer = shap.Explainer(xgb_clf)
shap_values = explainer(x)
shap.plots.waterfall(shap_values[0],max_display=30)


# In[87]:


shap.plots.beeswarm(shap_values, max_display=25)


# To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impacts each feature has on the model output. The color represents the feature value (red high, blue low). This reveals for example that a high VAR_ZCR (% lower var_zcr of the signals) lowers the predicted covid illnesse.

# In[88]:


shap.plots.bar(shap_values,max_display=25)


# In[89]:


tlr=LR.predict(x_test[0:10])


# In[90]:


tsvm=svm.predict(x_test[0:10])
tsvm


# In[91]:


tdtcm=DTC_m.predict(x_test[0:10])


# In[92]:


trf=RF.predict(x_test[0:10])
trf


# In[93]:


txgb=XGB.predict(x_test[0:10])
txgb


# In[94]:


(tlr==tsvm)


# In[95]:


som=(svm.predict(x_test[0:1])==XGB.predict(x_test[0:1])).sum()
som


# In[ ]:





# In[96]:


(y_pred_svm==y_pred_RF).sum()


# In[97]:


(y_pred_svm==y_pred_XGB).sum()


# In[98]:


(y_pred_RF==y_pred_XGB).sum()


# In[ ]:





# In[ ]:





# # --------------------------------Visualyse performance---------------------------------------

# In[109]:


fig = plt.figure(figsize=(6,6))
ax = fig.add_axes([0,0,1,1])
langs = ['LR', 'SVM', 'DTC', 'RF', 'XGB']
students = [np.mean(cross_val_score(LR,x,y,cv=sss)),np.mean(cross_val_score(svm,x,y,cv=sss)),np.mean(cross_val_score(DTC_m,x,y,cv=sss)),np.mean(cross_val_score(RF,x,y,cv=sss)),np.mean(cross_val_score(XGB,x,y,cv=sss))]
ax.bar(langs,students,color ='maroon',width = 0.4)
plt.title("Visualize model performance")
plt.show()


# In[100]:


print("la matrice de confusion de RF")
print(cmRF)
print("----")
print("la matrice de confusion de XGB")
print(cm_XGB)
print("----")
print("la matrice de confusion de svm")
print(cm_svm)


# true positives: 152 ->prédiction est positive,valeur réelle est effectivement positive
# 
# True Negatives : 37 ->prédiction est négative ,valeur réelle est effectivement négative
# 
# False Positive: 44 ->prédiction est positive,la valeur réelle est négative.
# 
# False Negative: 5 -> prédiction est négative, la valeur réelle est positive

# # SVM est considérer comme le meilleur choix 

# #  --------------------------------Evaluation using a test audio------------------------------

# In[104]:


test_260_file = "C:/Users/MSI GF63/Desktop/P2M/wav/test_260.wav"
ipd.Audio(test_260_file)


# In[105]:


def vote_majoritaire(v,i):
    som=(svm.predict(v)+XGB.predict(v)+RF.predict(v))
    if (som[i]==3):
        print("you have a high probability of affection stay at home for the next 10 days ")
    elif (som[i]==2):
        print("you have a meduim probability of affection stay away from people at lest 1 m")
    elif (som[i]==1):
        print("it's recommanded to take it seriously")
    elif (som[i]==0):
        print("you are tested negative ")

        


# In[106]:


vote_majoritaire(x_test[0:l],5)


# In[ ]:





# # --------------------------------Thank You----------------------------------------

# In[ ]:





# In[ ]:





# In[ ]:





# In[102]:


# for i in range(1,5):
#     votegroup(x_test[i:i+1])


# In[ ]:





# In[ ]:




