import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
import json
import numpy as np

from parse_args import parse_args_train_closeset
from utils.func_preparation import reproducibility, make_outputs_dir, create_dir
from utils.func_preparation import get_dataset_train, get_model, get_loss, get_callbacks
from utils.augmentations import get_training_augmentation, get_validation_augmentation

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from models.seg_module_base import SegmentationModuleBase

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold, train_test_split

def train_model_cross(args):
    
    # get dataset
    train_dataset, valid_dataset = get_dataset_train(args, get_training_augmentation, get_validation_augmentation)
    
    # info dataset
    all_index = train_dataset._get_images_index()
    num_classes = train_dataset._get_num_known_classes()
    args.num_classes = num_classes
    
    print('Known class.:', num_classes)
    print(train_dataset._class_to_idx())

    # tmp
    savedir = args.savedir
    
    # cross validation kfolds
    if args.train_type == 'cross':
        Kfold = KFold(n_splits=args.k_fold, shuffle=False)
        print('Num. K-folds.:', args.k_fold)
    
        for k, (train_index, test_index) in enumerate(Kfold.split(all_index)):
            
            print("Training K-fold.: ", k)
            
            # update savedir
            args.savedir = os.path.join(savedir, str(k))
            create_dir(args.savedir)
            
            # update parans to train 
            model = get_model(args)    
            loss = get_loss(args)
            callbacks = get_callbacks(args)
            
            train_sampler = SubsetRandomSampler(train_index)
            valid_sampler = SubsetRandomSampler(test_index)

            model = SegmentationModuleBase(
                args=args,
                model=model,
                loss=loss,
                batch_size=args.batch_size,
                trainset=train_dataset,
                train_sampler=train_sampler,
                valset=valid_dataset,
                valid_sampler=valid_sampler,
                num_classes=num_classes
            )
            model.cuda()
            
            trainer = pl.Trainer(
                        accelerator='cuda', 
                        max_epochs=args.max_epochs,
                        min_epochs=1,
                        precision=16, 
                        callbacks=callbacks,
                        num_sanity_val_steps=0,
                        log_every_n_steps=1
                    )
            
            # training
            trainer.fit(model)  
    else: # holdout
        print('Holdout / test size.:', args.test_size)
        train_index, test_index = train_test_split(all_index, test_size=args.test_size, shuffle=False)
        
        # update savedir
        args.savedir = os.path.join(savedir, 'holdout')
        create_dir(args.savedir)
        
        # update parans to train 
        model = get_model(args)    
        loss = get_loss(args)
        callbacks = get_callbacks(args)
        
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(test_index)

        model = SegmentationModuleBase(
            args=args,
            model=model,
            loss=loss,
            batch_size=args.batch_size,
            trainset=train_dataset,
            train_sampler=train_sampler,
            valset=valid_dataset,
            valid_sampler=valid_sampler,
            num_classes=num_classes
        )
        model.cuda()
        
        trainer = pl.Trainer(
                    accelerator='cuda', # 
                    max_epochs=args.max_epochs,
                    min_epochs=1,
                    precision=16, 
                    callbacks=callbacks,
                    num_sanity_val_steps=0,
                    log_every_n_steps=1
                )
        
        # training
        trainer.fit(model)  
            
    # root savedir
    args.savedir = savedir

if __name__ == '__main__':
    
    # get params
    args = parse_args_train_closeset()

    # reproducibility
    reproducibility(args)
    
    # create new directory to save outputs
    make_outputs_dir(args)
    
    # training the model
    train_model_cross(args)
    
    # save args
    with open(os.path.join(args.savedir, 'args.json'), "w") as write_file:
        json.dump(vars(args), write_file)
