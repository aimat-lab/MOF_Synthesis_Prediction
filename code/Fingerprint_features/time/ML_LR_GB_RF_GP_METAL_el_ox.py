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
import seaborn as sns
import random
import copy
import joblib
import pickle
import yaml
from sklearn.model_selection import KFold

###METAL_PROPERTIES############
#import write_one_hot_encoding
#import write_one_hot_encoding_for_gas
import encode_full_electronic_configuration
###############################

#####SOLVENT_PROPERTIES########

#import solvent_rdkit

############################



print("   ###   Libraries:")
print('   ---   Tensorflow:{}'.format(tf.__version__))
print('   ---   Pytorch:{}'.format(torch.__version__))
print('   ---   sklearn:{}'.format(sklearn.__version__))
print('   ---   rdkit:{}'.format(rdkit.__version__))

np.random.seed(1)
random.seed(1)


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
    
    #train_index, test_index = sklearn.model_selection.train_test_split(df.index.tolist(), test_size=1.0-fraction_training)
    X = np.array(df.index.tolist())
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(X)
    #sid=random.uniform(1,100000)
    counter=0
    for train_index, test_index in kf.split(X):
        counter=counter+1

        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(df[target].values.reshape(-1,1))
        y_train, y_test = y[train_index], y[test_index]
        features_basic = [] # those could be the features computed with the other script using rdkit. but for now we only use fingerprints

        if len(features_basic)>0:
            use_feat=True
        else:
            use_feat=False
   

#####METAL_DESCRIPTION_row_column_encoding_GOES_HERE@##########################

#    s1=write_one_hot_encoding.encode_row
#    
#    x_unscaled_feat_1=(s1)
#   
#    x_feat_1=x_unscaled_feat_1
#
#
#    
#    s2=write_one_hot_encoding.encode_column
#
#    x_unscaled_feat_2=(s2)
#
#    x_feat_2=x_unscaled_feat_2

#####METAL_DESCRIPTION_electronic_configuration_encoding_GOES_HERE@##########################
#    e1=write_one_hot_encoding_for_gas.encode_gas
    
#    x_unscaled_feat_1=(e1)
   
#    x_feat_1=x_unscaled_feat_1

#    e2=write_one_hot_encoding_for_gas.s1

#    x_unscaled_feat_2=(e2)

#    x_feat_2=x_unscaled_feat_2


#    e3=write_one_hot_encoding_for_gas.s2

#    x_unscaled_feat_3=(e3)

#    x_feat_3=x_unscaled_feat_3


#    e4=write_one_hot_encoding_for_gas.p1

#    x_unscaled_feat_4=(e4)

#    x_feat_4=x_unscaled_feat_4


#    e5=write_one_hot_encoding_for_gas.p2

#    x_unscaled_feat_5=(e5)

#    x_feat_5=x_unscaled_feat_5


#    e6=write_one_hot_encoding_for_gas.d1

#    x_unscaled_feat_6=(e6)

#    x_feat_6=x_unscaled_feat_6

#    e7=write_one_hot_encoding_for_gas.d2

#    x_unscaled_feat_7=(e7)

#    x_feat_7=x_unscaled_feat_7

#    e8=write_one_hot_encoding_for_gas.f1

#    x_unscaled_feat_8=(e8)

#    x_feat_8=x_unscaled_feat_8

#    e9=write_one_hot_encoding_for_gas.f2

#    x_unscaled_feat_9=(e9)

#    x_feat_9=x_unscaled_feat_9

##############################################################################
    
        e1=encode_full_electronic_configuration.s1
        x_unscaled_feat_1=e1
        x_feat_1=x_unscaled_feat_1
 
        e2=encode_full_electronic_configuration.s2
        x_unscaled_feat_2=e2
        x_feat_2=x_unscaled_feat_2

        e3=encode_full_electronic_configuration.s3
        x_unscaled_feat_3=e3
        x_feat_3=x_unscaled_feat_3

        e4=encode_full_electronic_configuration.s4
        x_unscaled_feat_4=e4
        x_feat_4=x_unscaled_feat_4

        e5=encode_full_electronic_configuration.s5
        x_unscaled_feat_5=e5
        x_feat_5=x_unscaled_feat_5

        e6=encode_full_electronic_configuration.s6
        x_unscaled_feat_6=e6
        x_feat_6=x_unscaled_feat_6

        e7=encode_full_electronic_configuration.p2
        x_unscaled_feat_7=e7
        x_feat_7=x_unscaled_feat_7

        e8=encode_full_electronic_configuration.p3
        x_unscaled_feat_8=e8
        x_feat_8=x_unscaled_feat_8

        e9=encode_full_electronic_configuration.p4
        x_unscaled_feat_9=e9
        x_feat_9=x_unscaled_feat_9

        e10=encode_full_electronic_configuration.p5
        x_unscaled_feat_10=e10
        x_feat_10=x_unscaled_feat_7

        e11=encode_full_electronic_configuration.d3
        x_unscaled_feat_11=e11
        x_feat_11=x_unscaled_feat_11

        e12=encode_full_electronic_configuration.d4
        x_unscaled_feat_12=e12
        x_feat_12=x_unscaled_feat_12


        e13=encode_full_electronic_configuration.d5
        x_unscaled_feat_13=e13
        x_feat_13=x_unscaled_feat_13

        e14=encode_full_electronic_configuration.f4
        x_unscaled_feat_14=e14
        x_feat_14=x_unscaled_feat_14

####################metal_oxidation_state#########################################
    
        e15=encode_full_electronic_configuration.o
        x_unscaled_feat_15=e15
        x_feat_15=x_unscaled_feat_15

#############SOLVENT_DESCRIPTION_GOES_HERE##################
#    e14=solvent_rdkit.x_solvent
#    x_unscaled_feat_14=(e14)
    #print (x_unscaled_feat_1)
#    x_feat_14=x_unscaled_feat_14

###############################################################




        if use_feat:
            x_scaler_feat = StandardScaler()
            x_unscaled_feat=df[features_basic].values
            x_feat = x_scaler_feat.fit_transform(x_unscaled_feat)
            n_feat = len(features_basic)

        if use_rdkit:
            x_unscaled_fp1 = np.array([Chem.RDKFingerprint(mol1, maxPath=rdkit_l, fpSize=rdkit_s) for mol1 in df['mol1'].tolist()]).astype(float)
            x_scaler_fp1 = StandardScaler()
            x_fp1 = x_scaler_fp1.fit_transform(x_unscaled_fp1)


            x_unscaled_fp2 = np.array([Chem.RDKFingerprint(mol2, maxPath=rdkit_l, fpSize=rdkit_s) for mol2 in df['mol2'].tolist()]).astype(float)
            x_scaler_fp2 = StandardScaler()
            x_fp2 = x_scaler_fp2.fit_transform(x_unscaled_fp2)
        if use_mfp:
            x_unscaled_mfp = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, mfp_l, nBits=mfp_s) for mol in df['mol'].tolist()]).astype(float)
            x_scaler_mfp = StandardScaler()
            x_mfp = x_scaler_mfp.fit_transform(x_unscaled_mfp)





##################COMBINE_THE_INPUTS####################################

        x = np.hstack([x_fp1,x_fp2,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        x_repeat = np.hstack([x_fp2,x_fp1,x_feat_1,x_feat_2,x_feat_3,x_feat_4,x_feat_5,x_feat_6,x_feat_7,x_feat_8,x_feat_9,x_feat_10,x_feat_11,x_feat_12,x_feat_13,x_feat_14,x_feat_15])
        x_unscaled = np.hstack([x_unscaled_fp1,x_unscaled_fp2,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])
        x_unscaled_repeat = np.hstack([x_unscaled_fp2,x_unscaled_fp1,x_unscaled_feat_1,x_unscaled_feat_2,x_unscaled_feat_3,x_unscaled_feat_4,x_unscaled_feat_5,x_unscaled_feat_6,x_unscaled_feat_7,x_unscaled_feat_8,x_unscaled_feat_9,x_unscaled_feat_10,x_unscaled_feat_11,x_unscaled_feat_12,x_unscaled_feat_13,x_unscaled_feat_14,x_unscaled_feat_15])
#    x = np.hstack([x_fp,x_feat_1,x_feat_2])
#    x_unscaled = np.hstack([x_unscaled_fp,x_unscaled_feat_1,x_unscaled_feat_2])
#########################################################################


    #x = x_feat
        x_train, x_test = x[train_index],x[test_index]
        x_unscaled_train, x_unscaled_test = x_unscaled[train_index], x_unscaled[test_index]
   
        r_train_index=[]
        for abcde in range(0,len(train_index)):
            if (df["nlinker"][train_index[abcde]])==2:
                print (df["nlinker"][train_index[abcde]])
                r_train_index.append(train_index[abcde])

        r_test_index=[]
        for abcde in range(0,len(test_index)):
            if (df["nlinker"][test_index[abcde]])==2:
                print (df["nlinker"][test_index[abcde]])
                r_test_index.append(test_index[abcde])

        x_r_train, x_r_test = x_repeat[r_train_index],x_repeat[r_test_index]
        x_r_unscaled_train, x_r_unscaled_test = x_unscaled_repeat[r_train_index], x_unscaled_repeat[r_test_index]


        y_r_train, y_r_test = y[r_train_index], y[r_test_index]

        x_train=np.vstack([x_train,x_r_train])
        y_train=np.vstack([y_train,y_r_train])
        x_test=np.vstack([x_test,x_r_test])
        y_test=np.vstack([y_test,y_r_test])
        print("   ---   Training and test data dimensions:")







        print("   ---   Training and test data dimensions:")
        print(x_train.shape,x_test.shape,y_train.shape, y_test.shape)


     

    ############################
    # RandomForestRegressor
    ############################
        model =  RandomForestRegressor(max_depth=7)
        #model =  RandomForestRegressor()
        model.fit(x_train,y_train.ravel())
    #y_pred = model.predict(x_test)

        print("\n   ###   RandomForestRegressor:")
        y_pred_train = model.predict(x_train)
        r2_GBR_train,mae_GBR_train = reg_stats(y_train,y_pred_train,y_scaler)
        print("   ---   Training (r2, MAE): %.3f %.3f"%(r2_GBR_train,mae_GBR_train))
        y_pred_test = model.predict(x_test)
        r2_GBR_test,mae_GBR_test = reg_stats(y_test,y_pred_test,y_scaler)
        print("   ---   Testing (r2, MAE): %.3f %.3f"%(r2_GBR_test,mae_GBR_test))
        joblib.dump(model, "%s/models/random_forest_regression.joblib"%(outdir))


        y_test_unscaled = y_scaler.inverse_transform(y_test)
        y_train_unscaled = y_scaler.inverse_transform(y_train)
        y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test)
        y_pred_train_unscaled = y_scaler.inverse_transform(y_pred_train)

        np.savetxt("%s/predictions_rawdata/y_real_test.txt"%(outdir), y_test_unscaled)
        np.savetxt("%s/predictions_rawdata/y_real_train.txt"%(outdir), y_train_unscaled)
        np.savetxt("%s/predictions_rawdata/y_RFR_test.txt"%(outdir), y_pred_test_unscaled)
        np.savetxt("%s/predictions_rawdata/y_RFR_train.txt"%(outdir), y_pred_train_unscaled)

        np.savetxt("./predictions_rawdata/y_real_"+str(counter)+"_test.txt", y_test_unscaled)
        np.savetxt("./predictions_rawdata/y_real_"+str(counter)+"_train.txt", y_train_unscaled)
        np.savetxt("./predictions_rawdata/y_RFR_"+str(counter)+"_test.txt", y_pred_test_unscaled)
        np.savetxt("./predictions_rawdata/y_RFR_"+str(counter)+"_train.txt", y_pred_train_unscaled)


        plt.figure()
        plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: r$^2$ = %.3f"%(r2_GBR_train))
        plt.scatter(y_pred_test_unscaled, y_test_unscaled, marker="o", c="C2", label="Testing: r$^2$ = %.3f"%(r2_GBR_test))
        plt.scatter(y_pred_train_unscaled, y_train_unscaled, marker="o", c="C1", label="Training: MAE = %.3f"%(mae_GBR_train))
        plt.scatter(y_pred_test_unscaled, y_test_unscaled, marker="o", c="C2", label="Testing: MAE = %.3f"%(mae_GBR_test))
        plt.plot(y_train_unscaled,y_train_unscaled)
        plt.title('RandomForestRegressor')
#    if plot_labels:
#        for counter,idx in enumerate(test_index):
#            ID=df['names'].tolist()[idx]
#            plt.text(y_test_unscaled[counter], y_pred_test_unscaled[counter], "%s"%(ID), fontsize=2)
#    if plot_all_labels:
#        for counter,idx in enumerate(train_index):
#            ID=df['names'].tolist()[idx]
#            plt.text(y_train_unscaled[counter], y_pred_train_unscaled[counter], "%s"%(ID), fontsize=2)
        plt.ylabel("Experimental Temperature [C]")
        plt.xlabel("Predicted Temperature [C]")
        plt.legend(loc="upper left")
        plt.savefig("%s/scatter_plots/full_data_RFR.png"%(outdir),dpi=300)
        plt.close()

target="time"
df = pd.read_csv("edited_full.csv")


df['mol1']=df['linker1smi'].apply(lambda smi: Chem.MolFromSmiles(smi))
atom_list1 = df['mol1'].apply(lambda m: [a.GetSymbol() for a in m.GetAtoms()]).tolist()
atom_set1 = list(set([i  for j in atom_list1 for i in j]))
print("   ---   Elements:")
print(atom_set1)

df['mol2']=df['linker2smi'].apply(lambda smi: Chem.MolFromSmiles(smi))
atom_list2 = df['mol2'].apply(lambda m: [a.GetSymbol() for a in m.GetAtoms()]).tolist()
atom_set2 = list(set([i  for j in atom_list2 for i in j]))
print("   ---   Elements:")
print(atom_set2)




if os.path.exists("settings.yml"):
    user_settings = yaml.load(open("settings.yml","r"))
    hparam = yaml.load(open("settings.yml","r"))

train(df, target, hparam)


