
# pip install rioxarray
from skimage import morphology, measure

import rioxarray as rioxr
import xarray as xr
from xrspatial import multispectral
from tqdm import tqdm
import numpy.ma as ma
import rasterio as rio

# from scipy import stats
# from sklearn.model_selection import train_test_split
# from torchvision import transforms as T
# from torch.utils.data import Dataset
# import pytorch_lightning as pl
# import torch.nn.functional as F
# import torch

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
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
import uuid
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
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

# warnings.simplefilter('ignore')

data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

train_path = data_dir/"processed"
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

for col in ['images', 'cloud', 'lat', 'lon', 'scl']:
    print(len(input_data[col]))

def raster_featurization(idx, input_data, composite_type='plain'):
    
    feature_imgs = xr.DataArray(input_data['images'][idx].T, \
                              coords={'y':input_data['lat'][idx][:,0].squeeze(), \
                                      'x':input_data['lon'][idx][0,:].squeeze()}, \
                              dims=['band', 'y', 'x'])
    feature_cloud = xr.DataArray(input_data['cloud'][idx].T, \
                              coords={'y':input_data['lat'][idx][:,0].squeeze(), \
                                      'x':input_data['lon'][idx][0,:].squeeze()}, \
                              dims=['band', 'y', 'x'])
    feature_scl = xr.DataArray(input_data['scl'][idx].T, \
                              coords={'y':input_data['lat'][idx][:,0].squeeze(), \
                                      'x':input_data['lon'][idx][0,:].squeeze()}, \
                              dims=['band', 'y', 'x'])
    
    print(feature_imgs.shape, feature_cloud.shape, feature_scl.shape)

    if composite_type == 'plain':
        ## dst_composite_plain:        
        dst_composite = feature_imgs
        dst_numpy = dst_composite.values
    
    elif composite_type == "compo-0":
        ## dst_composite_compo-0:        
        dst_composite = xr.concat([feature_imgs, feature_cloud, feature_scl], dim="band")
        dst_numpy = dst_composite.values

    elif composite_type == "compo-1":
        ## dst_composite_compo-1:
        ##  0/B2/Blue, 1/B3/G, 2/B4/R, 3/B5/RE1, 4/B6/RE2, 5/B7/RE3, 6/B8/NIR1, 7/B8A/NIR2, 8/B11/SWIR1, 9/B12/SWIR2
        arvi = multispectral.arvi(nir_agg=feature_imgs.isel(band=6),
                              red_agg=feature_imgs.isel(band=2),
                              blue_agg=feature_imgs.isel(band=0))
        print(f'ARVI min/max: {arvi.min().values}/{arvi.max().values}')

        evi = multispectral.evi(nir_agg=feature_imgs.isel(band=7), # B8A is selected here over B08 due to its better fit to the MODIS band range (using which EVI was developed and tested)
                            red_agg=feature_imgs.isel(band=2),
                            blue_agg=feature_imgs.isel(band=0),
                            c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5)
        print(f'EVI min/max: {evi.min().values}/{evi.max().values}')

        gci = multispectral.gci(nir_agg=feature_imgs.isel(band=6),
                                green_agg=feature_imgs.isel(band=1))
        print(f'GCI min/max: {gci.min().values}/{gci.max().values}')

        nbr = multispectral.nbr(nir_agg=feature_imgs.isel(band=6),
                                swir2_agg=feature_imgs.isel(band=9))
        print(f'NBR min/max: {nbr.min().values}/{nbr.max().values}')

        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8),
                                  swir2_agg=feature_imgs.isel(band=9))
        print(f'NBR2 min/max: {nbr2.min().values}/{nbr2.max().values}')

        ndmi = multispectral.ndmi(nir_agg=feature_imgs.isel(band=6),
                                  swir1_agg=feature_imgs.isel(band=8))
        print(f'NDMI min/max: {ndmi.min().values}/{ndmi.max().values}')

        ndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2))
        print(f'NDVI min/max: {ndvi.min().values}/{ndvi.max().values}')

        savi = multispectral.savi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2))
        print(f'SAVI min/max: {savi.min().values}/{savi.max().values}')

        sipi = multispectral.sipi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2),
                                  blue_agg=feature_imgs.isel(band=0))
        print(f'SIPI min/max: {sipi.min().values}/{sipi.max().values}')

        # Combine:
        dst_composite = xr.concat([feature_imgs, arvi, evi, gci, nbr, nbr2, ndmi, ndvi, savi, sipi],
                                         dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values


    elif composite_type == "compo-2":
        ## dst_composite_compo-2:
        ##  0/B2/Blue, 1/B3/G, 2/B4/R, 3/B5/RE1, 4/B6/RE2, 5/B7/RE3, 6/B8/NIR1, 7/B8A/NIR2, 8/B11/SWIR1, 9/B12/SWIR2

        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8),
                                  swir2_agg=feature_imgs.isel(band=9))
        print(f'NBR2 min/max: {nbr2.min().values}/{nbr2.max().values}')

        ndmi = multispectral.ndmi(nir_agg=feature_imgs.isel(band=6),
                                  swir1_agg=feature_imgs.isel(band=8))
        print(f'NDMI min/max: {ndmi.min().values}/{ndmi.max().values}')

        ndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2))
        print(f'NDVI min/max: {ndvi.min().values}/{ndvi.max().values}')

        savi = multispectral.savi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2))
        print(f'SAVI min/max: {savi.min().values}/{savi.max().values}')

        sipi = multispectral.sipi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2),
                                  blue_agg=feature_imgs.isel(band=0))
        print(f'SIPI min/max: {sipi.min().values}/{sipi.max().values}')

        # Combine:
        dst_composite = xr.concat([feature_imgs, nbr2, ndmi, ndvi, savi, sipi],
                                         dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values


    elif composite_type == "compo-3":
        ## dst_composite_compo-3:
        ##  0/B2/Blue, 1/B3/G, 2/B4/R, 3/B5/RE1, 4/B6/RE2, 5/B7/RE3, 6/B8/NIR1, 7/B8A/NIR2, 8/B11/SWIR1, 9/B12/SWIR2
        arvi = multispectral.arvi(nir_agg=feature_imgs.isel(band=6),
                                  red_agg=feature_imgs.isel(band=2),
                                  blue_agg=feature_imgs.isel(band=0))
        print(f'ARVI min/max: {arvi.min().values}/{arvi.max().values}')

        evi = multispectral.evi(nir_agg=feature_imgs.isel(band=7), # B8A is selected here over B08 due to its better fit to the MODIS band range (using which EVI was developed and tested)
                                red_agg=feature_imgs.isel(band=2),
                                blue_agg=feature_imgs.isel(band=0),
                                c1=6.0, c2=7.5, soil_factor=1.0, gain=2.5)
        print(f'EVI min/max: {evi.min().values}/{evi.max().values}')

        gci = multispectral.gci(nir_agg=feature_imgs.isel(band=6),
                                green_agg=feature_imgs.isel(band=1))
        print(f'GCI min/max: {gci.min().values}/{gci.max().values}')

        nbr = multispectral.nbr(nir_agg=feature_imgs.isel(band=6),
                                swir2_agg=feature_imgs.isel(band=9))
        print(f'NBR min/max: {nbr.min().values}/{nbr.max().values}')


        # Combine:
        dst_composite = xr.concat([feature_imgs, arvi, evi, gci, nbr],
                                         dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values

    elif composite_type == "compo-4":
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        gndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6), \
                                     red_agg=feature_imgs.isel(band=1))
        # print(f'NDVI min/max: {gndvi_1.min().values}/{gndvi_1.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8), \
                                    swir2_agg=feature_imgs.isel(band=9))

        dst_composite = xr.concat([feature_imgs, gndvi, nbr2], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values

    elif composite_type == "compo-4-cloud":
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        gndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6), \
                                     red_agg=feature_imgs.isel(band=1))
        # print(f'NDVI min/max: {gndvi_1.min().values}/{gndvi_1.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8), \
                                    swir2_agg=feature_imgs.isel(band=9))

        dst_composite = xr.concat([feature_imgs, feature_cloud, gndvi, nbr2], dim="band")
        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values

    elif composite_type == "compo-4-scl":
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        gndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6), \
                                     red_agg=feature_imgs.isel(band=1))
        # print(f'NDVI min/max: {gndvi_1.min().values}/{gndvi_1.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8), \
                                    swir2_agg=feature_imgs.isel(band=9))

        dst_composite = xr.concat([feature_imgs, feature_scl, gndvi, nbr2], dim="band")

        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values

    elif composite_type == "compo-4-all":
        # 1: Sep, Oct, Nov, Dec; 2: Jan, Feb, Mar, Apr; 3: May, Jun, Jul, Aug
        gndvi = multispectral.ndvi(nir_agg=feature_imgs.isel(band=6), \
                                     red_agg=feature_imgs.isel(band=1))
        # print(f'NDVI min/max: {gndvi_1.min().values}/{gndvi_1.max().values}')
        nbr2 = multispectral.nbr2(swir1_agg=feature_imgs.isel(band=8), \
                                    swir2_agg=feature_imgs.isel(band=9))

        dst_composite = xr.concat([feature_imgs, feature_cloud, feature_scl, gndvi, nbr2], dim="band")

        # dst_numpy = dst_composite[[0, 10], :, :].values # TODO: Why 10 bands?
        dst_numpy = dst_composite.values

    else:
        raise Exception(f'Unknown composite_type')
    
    print(dst_numpy.shape, dst_composite.shape, feature_imgs.shape)
    return dst_numpy, dst_composite
        
def featrue_gene(input_data, composite_type, mode='train'):
    # TODO: Calculate NDSI: NIR1/NIR2, RE3/NIR1, SWIR1/SWIR2 (NDTI), G/NIR2, G/NIR1
    for idx in range(len(input_data['images'])):
        composite_type = "compo-4-all"
        dst_numpy, dst_composite = raster_featurization(3, train_data, composite_type)

        ## Fill NaNs with 0s (if less than 10%)
        dst_not_nans = np.count_nonzero(np.isnan(dst_numpy)) / np.prod(dst_numpy.shape) * 100

        # TODO: Interploation sometimes leads to
        if dst_not_nans > 0 and dst_not_nans <= 0.1:
            print(f"Interpolating {outdir}/{uID}_S2.tif as it contains {dst_not_nans:.3f}% NaNs")
            dst_composite = dst_composite.interpolate_na(dim='x', method="linear", fill_value="extrapolate")
            dst_composite = dst_composite.fillna(0) # TODO: Interploation sometimes leads to NaN: 45fb7068_S2 in 4SI
        elif dst_not_nans > 0.1 and dst_not_nans <= 100:
            print(f"Filling {outdir}/{uID}_S2.tif with 0s as it contains {dst_not_nans:.3f}% NaNs")
            # TODO: doesn't fill all NaNs ??
            dst_composite = dst_composite.fillna(0)
        break
        

    
#     ## TODO: Change ransform=self.transform(recalc=recalc_transform) to transform=None in ...\biomassters\Lib\site-packages\rioxarray\raster_array.py
#     dst_composite.rio.to_raster(f'{outdir}/{uID}_S2.tif', compress='DEFLATE', dtype="float32", nodata=None,
#                                 driver='GTiff', width=256, height=256, count=10, crs=None, transorm=None)

    
    
    
# def Feature(df):
#     transformers = {}
#     for i in range(12):
#         transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
#         temp = df[:, :, :, i].shape
#         transformer.fit(df[:, :, :, i].reshape(-1,1))#transformer.fit(df[col].values.reshape(-1,1))
#         df[:,:,:,i] = transformer.transform(df[:, :, :, i].reshape(-1,1)).reshape(temp)
#         transformers[i] = transformer
        
#     return df, transformers

# train_images = train_data['images']

# tt = np.concatenate([train_images, test_images])
# tt, transformers = Feature(tt)
# train_images = tt[:train_images.shape[0]]
# test_images = tt[train_images.shape[0]:]

# composite_type = "plain"
# featrue_gene(uID, train_data, composite_type, mode='train'):
# composite_type = "compo-4-all"
# dst_numpy, dst_composite = raster_featurization(3, train_data, composite_type)

# train_imagee = pd.DataFrame(train_data['images'].reshape(-1, 15*15*12))
# train_imagee["ID"] = train_imagee.index.values

# train_agbd = pd.DataFrame(train_data['agbd'].reshape(-1, 1)).rename(columns={0:'agbd'})
# train_cloud = pd.DataFrame(train_data['cloud'].reshape(-1, 15*15))
# train_cloud.rename(columns={i:f"cld-{i}" for i in train_cloud.columns}, inplace=True)

# train_lat = pd.DataFrame(train_data['lat'].reshape(-1, 15*15))
# train_lat.rename(columns={i:f"lat-{i}" for i in train_lat.columns}, inplace=True)

# train_lon = pd.DataFrame(train_data['lon'].reshape(-1, 15*15))
# train_lon.rename(columns={i:f"lon-{i}" for i in train_lon.columns}, inplace=True)

# train_scl = pd.DataFrame(train_data['scl'].reshape(-1, 15*15))
# train_scl.rename(columns={i:f"scl-{i}" for i in train_scl.columns}, inplace=True)

# train_gt = pd.concat([train_agbd, train_cloud, train_lon, train_lat, train_scl], axis=1)
# train_gt["ID"] = train_imagee.index.values

# print(train_imagee.shape, train_cloud.shape,train_lat.shape,train_lon.shape,train_scl.shape,)


# del train_cloud, train_lat, train_lon, train_scl, train_agbd
# gc.collect()

# train_imagee.loc[:,'fold'] = -1
# train_gt.loc[:,'fold'] = -1

# skf = MultilabelStratifiedKFold(n_splits = 5, shuffle=True, random_state=369)
# for fold, (trn_index, val_index) in enumerate(skf.split(train_imagee,  train_gt)):
#     train_imagee.loc[val_index,'fold'] = fold
#     train_gt.loc[val_index,'fold'] = fold
    
#     train_imagee.to_csv(f'./data/processed/train_images_fld{fold}.csv', index=False)
#     train_gt.to_csv(f'./data/processed/train_gt_fld{fold}.csv', index=False)
    
#     train_imagee.loc[:,'fold'] = -1
#     train_gt.loc[:,'fold'] = -1
   
