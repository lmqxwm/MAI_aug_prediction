# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 21:58:15 2022

@author: Mengqi Liu
"""
import os
import numpy as np
import pytorch_forecasting as pf
import pandas as pd
import pytorch_lightning as pl
import json
import argparse
from pathlib import Path
from shutil import copy
from models import *

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter("error", category=SettingWithCopyWarning)



def aug_data(data, mai_args, aug_type):
    if aug_type == "MAI":
        newdata = run_mae(mai_args, data)
    elif aug_type == "None":
        newdata = pseudo_aug(mai_args, data)    
    return newdata




def load_dataset(args, mai_args):
    if args.data_name == 'traffic':
        data_dir = os.path.join(os.path.join(args.base_dir, "datasets"), args.data_name) + '.csv'
        #data = pd.read_csv('E:/InterpoMAE/predict_task/datasets/traffic.csv', sep=';|,', header=0, index_col=False)
        
        data = pd.read_csv(data_dir, sep=';|,', header=0, index_col=False)
        columns_name = data.columns
        
        np_data = np.array(data)
        #print("Original data shape is", np_data.shape)
        #if len(list(np_data.shape))==2:
        #    np_data = np_data.reshape(list(np_data.shape)+[1])
        data = aug_data(np_data, mai_args, args.aug_type)
        data = pd.DataFrame(data, columns=columns_name)

        data['time'] = range(0,int(data.shape[0]))
        data['ID'] = 'traffic'
        data.rename(columns = {'Hour (Coded)':'Hour', 'Slowness in traffic (%)':'Slowness'}, inplace=True)
        training_cutoff = data.shape[0] - args.max_prediction_length
        
        training = pf.TimeSeriesDataSet(
            data[lambda x: x.time<training_cutoff],
            time_idx= 'time',  # column name of time of observation
            target= 'Slowness',  # column name of target to predict
            group_ids= ['ID'],  # column name(s) for timeseries IDs
            max_encoder_length=args.max_encoder_length,  # how much history to use
            max_prediction_length=args.max_prediction_length,  # how far to predict into future
            # covariates static for a timeseries ID
            #static_categoricals=[ ... ],
            #static_reals=[ ... ],
            # covariates known and unknown in the future to inform prediction
            #time_varying_unknown_reals=list(data.columns)[1:-3],
            time_varying_unknown_reals=["Slowness"],
            time_varying_known_reals=["time"]+list(data.columns)[1:-3],
            target_normalizer=pf.GroupNormalizer(groups=['ID']),
            add_relative_time_idx=False,
            add_target_scales=True,
            randomize_length=None,
        )
        validation = pf.TimeSeriesDataSet.from_dataset(
            training,
            data[lambda x: ~x.time<training_cutoff],
            predict=True,
            stop_randomization=True,
        )

    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0, 
                                            #batch_sampler=None, 
                                            #shuffle=True,
                                            )
    val_dataloader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0, 
                                            #batch_sampler=None, 
                                            #shuffle=True,
                                            )
    return training, validation, train_dataloader, val_dataloader



def load_arguments(base_dir):
    # Find the config for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default='config.json')
    args_dict = vars(parser.parse_args())

    # Load the config.json
    config_dir = args_dict['config_dir']
    #base_dir = Path(config_dir).parent

    #instance_name = os.path.basename(instance_dir)

    with open(config_dir, 'r') as f:
        config_dict = json.load(fp=f)

    config_dict['base_dir'] = base_dir
    #config_dict['instance_name'] = instance_name

    total_dict = {**config_dict, **args_dict}

    # Maintain dirs
    storage_dir = os.path.join(base_dir, 'storage')
    total_dict['storage_dir'] = storage_dir
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    experiment_dir = os.path.join(storage_dir, config_dict['experiment_name'])
    total_dict['experiment_dir'] = experiment_dir
    print(f'experiment_dir is {experiment_dir}')
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)

    args = argparse.Namespace(**total_dict)

    temp_config_dir = os.path.join(experiment_dir, 'config.json')
    copy(args.config_dir, temp_config_dir)

    f = open(temp_config_dir)
    config = json.load(f)
    f.close()

    config['config_dir'] = temp_config_dir

    # Maintain dirs for generated data
    gen_data_dir = os.path.join(experiment_dir, 'gen_data')
    config['gen_data_dir'] = gen_data_dir

    # Write the config: Dict into the config for the instance
    with open(temp_config_dir, 'w') as f:
        f.write(json.dumps(config))

    return args

def load_mai_arguments(base_dir, args_):
    # Find the config for experiments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mai_config_dir', default='mai_config.json')
    args_dict = vars(parser.parse_args())

    # Load the config.json
    config_dir = args_dict['mai_config_dir']
    with open(config_dir, 'r') as f:
        config_dict = json.load(fp=f)
    
    config_dict['model_dir'] = args_.experiment_dir

    args = argparse.Namespace(**config_dict)

    return args
