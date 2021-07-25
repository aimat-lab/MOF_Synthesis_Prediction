

import os
import sys
import numpy as np
from scipy import stats
import math
import torch
from matplotlib import pyplot as plt
import matplotlib
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import sklearn.linear_model
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold
import seaborn as sns
import random
import copy
import joblib
import pickle
import yaml
#import MLP_model
#from models import getmodel,model_training
###METAL_PROPERTIES############
#import write_one_hot_encoding
#import write_one_hot_encoding_for_gas
#import encode_full_electronic_configuration
###############################

#####SOLVENT_PROPERTIES########

#import solvent_rdkit

############################

print("   ###   Libraries:")
print('   ---   Tensorflow:{}'.format(tf.__version__))
print('   ---   Pytorch:{}'.format(torch.__version__))
print('   ---   sklearn:{}'.format(sklearn.__version__))
print('   ---   rdkit:{}'.format(rdkit.__version__))


model_name = "MOF_NN"

#np.random.seed(1)
#random.seed(1)
s4="sed '601,672d' suffled_extended_big_additive_category.csv > new.csv"

os.system(s4)

def reg_stats(y_true,y_pred,scaler=None):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  if scaler:
    y_true_unscaled = scaler.inverse_transform(y_true)
    y_pred_unscaled = scaler.inverse_transform(y_pred)
  r2 = sklearn.metrics.r2_score(y_true,y_pred)
  mae = sklearn.metrics.mean_absolute_error(y_true_unscaled, y_pred_unscaled)
  return r2,mae


def train(df, target, hparam):
    fontname='Arial' 
    outdir=os.getcwd()

    print("start training")


    if not os.path.exists("%s/scatter_plots"%(outdir)):
        os.makedirs("%s/scatter_plots"%(outdir))

    if not os.path.exists("%s/models"%(outdir)):
        os.makedirs("%s/models"%(outdir))

    if not os.path.exists("%s/predictions_rawdata"%(outdir)):
        os.makedirs("%s/predictions_rawdata"%(outdir))

    use_rdkit = hparam["use_rdkit"]
    rdkit_l = hparam["rdkit_l"] # good number is 3-7
    rdkit_s = hparam["rdkit_s"] # good number is 2**10 - 2**13
    use_mfp = hparam["use_mfp"]
    mfp_l=hparam["mfp_l"] # good number is 2-3
    mfp_s=hparam["mfp_s"] # good number is 2**10 - 2**13
    fraction_training=hparam["fraction_training"] # 0.8
    num_epochs=hparam["num_epochs"] # 1000
    
    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(X)
    #sid=random.uniform(1,100000)
    for train_index, test_index in kf.split(X):
        #counter=counter+1
        #y_train, y_test = y[train_index], y[test_index]


    #train_index, test_index = sklearn.model_selection.train_test_split(df.index.tolist(), test_size=1.0-fraction_training)

        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(df[target].values.reshape(-1,1))
        
        y_train, y_test = y[train_index], y[test_index]
#        features_basic = [] # those could be the features computed with the other script using rdkit. but for now we only use fingerprints

        features_basic=['ASA [m^2/cm^3]','CellV [A^3]','Df','Di','Dif','NASA [m^2/cm^3]','POAV [cm^3/g]','POAVF','PONAV [cm^3/g]','PONAVF','density [g/cm^3]','total_SA_volumetric','total_SA_gravimetric','total_POV_volumetric','total_POV_gravimetric','mc_CRY-chi-0-all','mc_CRY-chi-1-all','mc_CRY-chi-2-all','mc_CRY-chi-3-all','mc_CRY-Z-0-all','mc_CRY-Z-1-all','mc_CRY-Z-2-all','mc_CRY-Z-3-all','mc_CRY-I-0-all','mc_CRY-I-1-all','mc_CRY-I-2-all','mc_CRY-I-3-all','mc_CRY-T-0-all','mc_CRY-T-1-all','mc_CRY-T-2-all','mc_CRY-T-3-all','mc_CRY-S-0-all','mc_CRY-S-1-all','mc_CRY-S-2-all','mc_CRY-S-3-all','D_mc_CRY-chi-0-all','D_mc_CRY-chi-1-all','D_mc_CRY-chi-2-all','D_mc_CRY-chi-3-all','D_mc_CRY-Z-0-all','D_mc_CRY-Z-1-all','D_mc_CRY-Z-2-all','D_mc_CRY-Z-3-all','D_mc_CRY-I-0-all','D_mc_CRY-I-1-all','D_mc_CRY-I-2-all','D_mc_CRY-I-3-all','D_mc_CRY-T-0-all','D_mc_CRY-T-1-all','D_mc_CRY-T-2-all','D_mc_CRY-T-3-all','D_mc_CRY-S-0-all','D_mc_CRY-S-1-all','D_mc_CRY-S-2-all','D_mc_CRY-S-3-all','sum-mc_CRY-chi-0-all','sum-mc_CRY-chi-1-all','sum-mc_CRY-chi-2-all','sum-mc_CRY-chi-3-all','sum-mc_CRY-Z-0-all','sum-mc_CRY-Z-1-all','sum-mc_CRY-Z-2-all','sum-mc_CRY-Z-3-all','sum-mc_CRY-I-0-all','sum-mc_CRY-I-1-all','sum-mc_CRY-I-2-all','sum-mc_CRY-I-3-all','sum-mc_CRY-T-0-all','sum-mc_CRY-T-1-all','sum-mc_CRY-T-2-all','sum-mc_CRY-T-3-all','sum-mc_CRY-S-0-all','sum-mc_CRY-S-1-all','sum-mc_CRY-S-2-all','sum-mc_CRY-S-3-all','sum-D_mc_CRY-chi-0-all','sum-D_mc_CRY-chi-1-all','sum-D_mc_CRY-chi-2-all','sum-D_mc_CRY-chi-3-all','sum-D_mc_CRY-Z-0-all','sum-D_mc_CRY-Z-1-all','sum-D_mc_CRY-Z-2-all','sum-D_mc_CRY-Z-3-all','sum-D_mc_CRY-I-0-all','sum-D_mc_CRY-I-1-all','sum-D_mc_CRY-I-2-all','sum-D_mc_CRY-I-3-all','sum-D_mc_CRY-T-0-all','sum-D_mc_CRY-T-1-all','sum-D_mc_CRY-T-2-all','sum-D_mc_CRY-T-3-all','sum-D_mc_CRY-S-0-all','sum-D_mc_CRY-S-1-all','sum-D_mc_CRY-S-2-all','sum-D_mc_CRY-S-3-all','lc-chi-0-all','lc-chi-1-all','lc-chi-2-all','lc-chi-3-all','lc-Z-0-all','lc-Z-1-all','lc-Z-2-all','lc-Z-3-all','lc-I-0-all','lc-I-1-all','lc-I-2-all','lc-I-3-all','lc-T-0-all','lc-T-1-all','lc-T-2-all','lc-T-3-all','lc-S-0-all','lc-S-1-all','lc-S-2-all','lc-S-3-all','lc-alpha-0-all','lc-alpha-1-all','lc-alpha-2-all','lc-alpha-3-all','D_lc-chi-0-all','D_lc-chi-1-all','D_lc-chi-2-all','D_lc-chi-3-all','D_lc-Z-0-all','D_lc-Z-1-all','D_lc-Z-2-all','D_lc-Z-3-all','D_lc-I-0-all','D_lc-I-1-all','D_lc-I-2-all','D_lc-I-3-all','D_lc-T-0-all','D_lc-T-1-all','D_lc-T-2-all','D_lc-T-3-all','D_lc-S-0-all','D_lc-S-1-all','D_lc-S-2-all','D_lc-S-3-all','D_lc-alpha-0-all','D_lc-alpha-1-all','D_lc-alpha-2-all','D_lc-alpha-3-all','func-chi-0-all','func-chi-1-all','func-chi-2-all','func-chi-3-all','func-Z-0-all','func-Z-1-all','func-Z-2-all','func-Z-3-all','func-I-0-all','func-I-1-all','func-I-2-all','func-I-3-all','func-T-0-all','func-T-1-all','func-T-2-all','func-T-3-all','func-S-0-all','func-S-1-all','func-S-2-all','func-S-3-all','func-alpha-0-all','func-alpha-1-all','func-alpha-2-all','func-alpha-3-all','D_func-chi-0-all','D_func-chi-1-all','D_func-chi-2-all','D_func-chi-3-all','D_func-Z-0-all','D_func-Z-1-all','D_func-Z-2-all','D_func-Z-3-all','D_func-I-0-all','D_func-I-1-all','D_func-I-2-all','D_func-I-3-all','D_func-T-0-all','D_func-T-1-all','D_func-T-2-all','D_func-T-3-all','D_func-S-0-all','D_func-S-1-all','D_func-S-2-all','D_func-S-3-all','D_func-alpha-0-all','D_func-alpha-1-all','D_func-alpha-2-all','D_func-alpha-3-all','f-lig-chi-0','f-lig-chi-1','f-lig-chi-2','f-lig-chi-3','f-lig-Z-0','f-lig-Z-1','f-lig-Z-2','f-lig-Z-3','f-lig-I-0','f-lig-I-1','f-lig-I-2','f-lig-I-3','f-lig-T-0','f-lig-T-1','f-lig-T-2','f-lig-T-3','f-lig-S-0','f-lig-S-1','f-lig-S-2','f-lig-S-3','sum-lc-chi-0-all','sum-lc-chi-1-all','sum-lc-chi-2-all','sum-lc-chi-3-all','sum-lc-Z-0-all','sum-lc-Z-1-all','sum-lc-Z-2-all','sum-lc-Z-3-all','sum-lc-I-0-all','sum-lc-I-1-all','sum-lc-I-2-all','sum-lc-I-3-all','sum-lc-T-0-all','sum-lc-T-1-all','sum-lc-T-2-all','sum-lc-T-3-all','sum-lc-S-0-all','sum-lc-S-1-all','sum-lc-S-2-all','sum-lc-S-3-all','sum-lc-alpha-0-all','sum-lc-alpha-1-all','sum-lc-alpha-2-all','sum-lc-alpha-3-all','sum-D_lc-chi-0-all','sum-D_lc-chi-1-all','sum-D_lc-chi-2-all','sum-D_lc-chi-3-all','sum-D_lc-Z-0-all','sum-D_lc-Z-1-all','sum-D_lc-Z-2-all','sum-D_lc-Z-3-all','sum-D_lc-I-0-all','sum-D_lc-I-1-all','sum-D_lc-I-2-all','sum-D_lc-I-3-all','sum-D_lc-T-0-all','sum-D_lc-T-1-all','sum-D_lc-T-2-all','sum-D_lc-T-3-all','sum-D_lc-S-0-all','sum-D_lc-S-1-all','sum-D_lc-S-2-all','sum-D_lc-S-3-all','sum-D_lc-alpha-0-all','sum-D_lc-alpha-1-all','sum-D_lc-alpha-2-all','sum-D_lc-alpha-3-all','sum-func-chi-0-all','sum-func-chi-1-all','sum-func-chi-2-all','sum-func-chi-3-all','sum-func-Z-0-all','sum-func-Z-1-all','sum-func-Z-2-all','sum-func-Z-3-all','sum-func-I-0-all','sum-func-I-1-all','sum-func-I-2-all','sum-func-I-3-all','sum-func-T-0-all','sum-func-T-1-all','sum-func-T-2-all','sum-func-T-3-all','sum-func-S-0-all','sum-func-S-1-all','sum-func-S-2-all','sum-func-S-3-all','sum-func-alpha-0-all','sum-func-alpha-1-all','sum-func-alpha-2-all','sum-func-alpha-3-all','sum-D_func-chi-0-all','sum-D_func-chi-1-all','sum-D_func-chi-2-all','sum-D_func-chi-3-all','sum-D_func-Z-0-all','sum-D_func-Z-1-all','sum-D_func-Z-2-all','sum-D_func-Z-3-all','sum-D_func-I-0-all','sum-D_func-I-1-all','sum-D_func-I-2-all','sum-D_func-I-3-all','sum-D_func-T-0-all','sum-D_func-T-1-all','sum-D_func-T-2-all','sum-D_func-T-3-all','sum-D_func-S-0-all','sum-D_func-S-1-all','sum-D_func-S-2-all','sum-D_func-S-3-all','sum-D_func-alpha-0-all','sum-D_func-alpha-1-all','sum-D_func-alpha-2-all','sum-D_func-alpha-3-all','sum-f-lig-chi-0','sum-f-lig-chi-1','sum-f-lig-chi-2','sum-f-lig-chi-3','sum-f-lig-Z-0','sum-f-lig-Z-1','sum-f-lig-Z-2','sum-f-lig-Z-3','sum-f-lig-I-0','sum-f-lig-I-1','sum-f-lig-I-2','sum-f-lig-I-3','sum-f-lig-T-0','sum-f-lig-T-1','sum-f-lig-T-2','sum-f-lig-T-3','sum-f-lig-S-0','sum-f-lig-S-1','sum-f-lig-S-2','sum-f-lig-S-3','MNC','MPC','pure_CO2_kH','pure_CO2_widomHOA','pure_methane_kH','pure_methane_widomHOA','pure_uptake_CO2_298.00_15000','pure_uptake_CO2_298.00_1600000','pure_uptake_methane_298.00_580000','pure_uptake_methane_298.00_6500000','logKH_CO2','logKH_CH4','CH4DC','CH4HPSTP']


        x_scaler_feat = StandardScaler()
        x_unscaled_feat=df[features_basic].values
        x_feat = x_scaler_feat.fit_transform(x_unscaled_feat)
        n_feat = len(features_basic)

            

##################COMBINE_THE_INPUTS####################################

#    x = np.hstack([x_fp,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14])
#    x_unscaled = np.hstack([x_unscaled_fp,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14])

#    x = np.hstack([x_fp,x_feat_1,x_feat_2])
#    x_unscaled = np.hstack([x_unscaled_fp,x_unscaled_feat_1,x_unscaled_feat_2])
#########################################################################


        x = x_feat
        x_unscaled=x_unscaled_feat
        x_train, x_test = x[train_index],x[test_index]
        x_unscaled_train, x_unscaled_test = x_unscaled[train_index], x_unscaled[test_index]

        print("   ---   Training and test data dimensions:")
        print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)






###############################################################


        print("   ---   Training and test data dimensions:")
        print ("I AM HEREEEEEEEEEEEE")
        print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)

     

        
        model =  RandomForestRegressor(max_depth=5)
        model.fit(x_train,y_train.ravel())
        #y_pred = model.predict(x_test)

        print("\n   ###   RandomForestRegressor:")
        y_pred_train = model.predict(x_train)
        r2_GBR_train,mae_GBR_train = reg_stats(y_train,y_pred_train,y_scaler)
        print("   ---   Training (r2, MAE): %.3f %.3f"%(r2_GBR_train,mae_GBR_train))
        y_pred_test = model.predict(x_test)
        r2_GBR_test,mae_GBR_test = reg_stats(y_test,y_pred_test,y_scaler)
        print("   ---   Testing (r2, MAE): %.3f %.3f"%(r2_GBR_test,mae_GBR_test))
        
 
target="temperature"
df = pd.read_csv("new.csv")




if os.path.exists("settings.yml"):
    user_settings = yaml.load(open("settings.yml","r"))
    hparam = yaml.load(open("settings.yml","r"))

train(df, target, hparam)







