import os
import json
import numpy as np
import pandas as pd 

from tqdm import tqdm
# from skimage import util

import torch
from torch.utils.data import DataLoader

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from parse_args import parse_args_train_report

from utils.func_preparation import reproducibility, load_model, get_dataset_test, save_openset_closeset_predictions, generete_report
from utils.augmentations import get_validation_augmentation, get_view

from skimage.morphology import opening

def compute_predictions_closetset(args):
    
    # get dataset and dataloader
    train_dataset, valid_dataset = get_dataset_test(args, get_view, get_validation_augmentation)
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)

    # info data
    categories = valid_dataset._get_classes_names()
    class_index = valid_dataset._get_classes_index()
    colors = valid_dataset._get_colors()
    
    # load trained model
    model = load_model(args)
    
    # predictions and labels
    embeddings, predictions, labels = [], [], []
    count_save_pred = 0
    
    for t, v in tqdm(zip(train_loader, valid_loader), total=len(valid_loader), desc='Compute Predictions Report'):
            
        # get batchs
        x_view, _, _ = t
        x, y, image_name = v
        x, y = x.float(), y.long()
        x, y = x.to(args.device), y.to(args.device)
        
        # multi lib models
        if args.model == 'seg_former':
            y_pred = model(x, y)
        else: 
            y_pred = model(x)
        
        # get embeddings to plot
        e = y_pred.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy()
        e = np.squeeze(e.reshape((e.shape[0] * e.shape[1], e.shape[2])))
        embeddings.append(e)
            
        # predictions to softmax and reshape
        # if args.loss == 'cross_entropy':
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        
        # reshape and torch tensor to numpy
        x_view = x_view.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy()
        y = y.detach().cpu().numpy().squeeze().astype(np.uint8)
        y_pred = y_pred.detach().cpu().numpy().squeeze().astype(np.uint8)
        
        # save images predictions
        if (args.save_images) and (args.num_save_images > count_save_pred):
            count_save_pred += 1
            
            image = x_view
            label = y
            prediction = y_pred

            # ISPRS dataset apply black pixel to ignored pixels
            image_name = str(image_name[0])
            
            save_openset_closeset_predictions(args, image, label, prediction, class_index, image_name, colors)
            
        # save predictions    
        predictions.append(y_pred)
        labels.append(y)

    # reshape
    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.reshape(np.stack(predictions), embeddings.shape[0])
    labels = np.reshape(np.stack(labels), embeddings.shape[0])
            
    return predictions, labels, categories, embeddings

if __name__ == '__main__':
    
    args, parser = parse_args_train_report()
    
    # args.folder_id = 'seg_former_prototycal_local_triplet_vaihingen_unknown_class_0_2078270235972'
    # args.k_idx = 0
    # args.name_checkpoint = 'prototycal_local_triplet_vaihingen_unknown_class_0_epoch=16-val_jac_epoch=0.805.ckpt'
    
    # load args folder for eval
    args_load = os.path.join(args.folder_path, args.folder_id, 'args.json')
    
    with open(args_load, 'rt') as f:
        args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=args)
        # only (True) use to train model
        args.mode_aug = True
        args.split = 'train'
        args.alpha = None # test because error
    
    # update params
    args.savedir = os.path.join(args.folder_path, args.folder_id)
    args.feature_extractor = False
    
    # reproducibility
    reproducibility(args)
    
    # start
    predictions, labels, categories, embeddings = compute_predictions_closetset(args)
    
    print("Predictions.:", np.unique(predictions))
    print("Labels......:", np.unique(labels))
    generete_report(args, predictions, labels, categories, 'openset_closeset')
    
    
    
    
    