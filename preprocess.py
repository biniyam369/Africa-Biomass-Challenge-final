
import pandas as pd
import h5py
import numpy as np
import warnings
import gc,os
from time import time
import datetime,random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import argparse

def Parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir',
                      default='./data', help='input data path of dataset')
    args.add_argument('--output_dir',
                      default='./data', help='input data path of dataset')
    args = args.parse_args()
    return args

args = Parse_args()

warnings.simplefilter('ignore')

ncompo_genes = 80
ncompo_cells = 10

def Seed_everything(seed=369):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=369)

test_files = ['%s/images_test.h5'%args.input_dir,
              '%s/canopy_height_test.h5'%args.input_dir,
              '%s/shot_number_test.h5'%args.input_dir,
              '%s/cloud_test.h5'%args.input_dir,
              '%s/scl_test.h5'%args.input_dir,
              '%s/x_topleft_test.h5'%args.input_dir,
              '%s/y_topleft_test.h5'%args.input_dir,
              '%s/sample_submission.csv'%args.input_dir]

# test 
testset = h5py.File(test_files[0], "r")
test_images = np.array(testset['images'], dtype=np.float32)

train_files = ['%s/data/09072022_1154_train.h5'%args.output_dir,
               '%s/data/09072022_1154_val.h5'%args.output_dir,
               '%s/data/09072022_1154_test.h5'%args.output_dir]
    
# train & validate
trainset = h5py.File(train_files[0], "r")
validset = h5py.File(train_files[1], "r")
validset2 = h5py.File(train_files[2], "r")

train_images = np.array(trainset['images'],dtype=np.float32)
valid_images = np.array(validset['images'],dtype=np.float32)
valid_images2 = np.array(validset2['images'], dtype=np.float32)

train_biomasses = np.array(trainset['agbd'],dtype=np.float32)
valid_biomasses = np.array(validset['agbd'],dtype=np.float32)
valid_biomasses2 = np.array(validset2['agbd'],dtype=np.float32)


print(f"train dataset size {train_images.shape} train lab size {train_biomasses.shape}")
print(f"validate dataset size {valid_images.shape} validate lab size {valid_biomasses.shape}")
print(f"validate dataset2 size {valid_images2.shape} test lab size {valid_images2.shape}")
print(f"test dataset size {test_images.shape}")

def Feature(df):
    transformers = {}
    for i in range(12):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        temp = df[:, :, :, i].shape
        transformer.fit(df[:, :, :, i].reshape(-1,1))#transformer.fit(df[col].values.reshape(-1,1))
        df[:,:,:,i] = transformer.transform(df[:, :, :, i].reshape(-1,1)).reshape(temp)
        transformers[i] = transformer
        
    return df, transformers
