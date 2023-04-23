
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
import gc

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

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

train_path = data_dir / "processed"
train_path.mkdir(exist_ok=True)

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

train_data ={}
for col in ['agbd', 'cloud', 'images', 'lat', 'lon', 'scl']:
    trn_temp = np.array(trainset[col], dtype=np.float32)
    vld_temp = np.array(validset[col],dtype=np.float32)
    vld_temp2 = np.array(validset2[col],dtype=np.float32)
    train_data[col] = np.concatenate([trn_temp, vld_temp, vld_temp2])
    
    
def Feature(df):
    transformers = {}
    for i in range(12):
        transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
        temp = df[:, :, :, i].shape
        transformer.fit(df[:, :, :, i].reshape(-1,1))#transformer.fit(df[col].values.reshape(-1,1))
        df[:,:,:,i] = transformer.transform(df[:, :, :, i].reshape(-1,1)).reshape(temp)
        transformers[i] = transformer
        
    return df, transformers

train_images = train_data['images']

tt = np.concatenate([train_images, test_images])
tt, transformers = Feature(tt)
train_images = tt[:train_images.shape[0]]
test_images = tt[train_images.shape[0]:]

train_imagee = pd.DataFrame(train_data['images'].reshape(-1, 15*15*12))
train_imagee["ID"] = train_imagee.index.values

train_agbd = pd.DataFrame(train_data['agbd'].reshape(-1, 1)).rename(columns={0:'agbd'})
train_cloud = pd.DataFrame(train_data['cloud'].reshape(-1, 15*15))
train_cloud.rename(columns={i:f"cld-{i}" for i in train_cloud.columns}, inplace=True)

train_lat = pd.DataFrame(train_data['lat'].reshape(-1, 15*15))
train_lat.rename(columns={i:f"lat-{i}" for i in train_lat.columns}, inplace=True)

train_lon = pd.DataFrame(train_data['lon'].reshape(-1, 15*15))
train_lon.rename(columns={i:f"lon-{i}" for i in train_lon.columns}, inplace=True)

train_scl = pd.DataFrame(train_data['scl'].reshape(-1, 15*15))
train_scl.rename(columns={i:f"scl-{i}" for i in train_scl.columns}, inplace=True)

train_gt = pd.concat([train_agbd, train_cloud, train_lon, train_lat, train_scl], axis=1)
train_gt["ID"] = train_imagee.index.values

print(train_imagee.shape, train_cloud.shape,train_lat.shape,train_lon.shape,train_scl.shape,)


del train_cloud, train_lat, train_lon, train_scl, train_agbd
gc.collect()

train_imagee.loc[:,'fold'] = -1
train_gt.loc[:,'fold'] = -1

skf = MultilabelStratifiedKFold(n_splits = 5, shuffle=True, random_state=369)
for fold, (trn_index, val_index) in enumerate(skf.split(train_imagee,  train_gt)):
    train_imagee.loc[val_index,'fold'] = fold
    train_gt.loc[val_index,'fold'] = fold
    
    train_imagee.to_csv(f'./data/processed/train_images_{fold}.csv', index=False)
    train_gt.to_csv(f'./data/processed/train_gt_fld{fold}.csv', index=False)
    
    train_imagee.loc[:,'fold'] = -1
    train_gt.loc[:,'fold'] = -1
   
