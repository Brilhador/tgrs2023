import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from parse_args import parse_args_openset
from utils.func_preparation import reproducibility, save_openipcs_predictions, generete_report
from utils.func_preparation import load_model, get_dataset_test, get_dataset_train, save_best_scores_openipcs, compute_iou_value
from utils.augmentations import get_view, get_validation_augmentation
from sklearn import decomposition
from sklearn import metrics
from sklearn.metrics import f1_score
from skimage import util
from torch.nn.functional import interpolate
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split

def compute_openset_openipcs(args):

    model = load_model(args, feature_extraction=args.feature_extractor)
    model_full = train_model_pca(args, model)
    thresholds = compute_threshold_from_pca(args, model, model_full)
    predictions, labels, categories = compute_scores_from_pca(args, model, model_full, thresholds)
    generete_report(args, predictions, labels, categories, 'gopenipcs')

def train_model_pca(args, net):

    args.split = 'train'
    
    _, valid_dataset = get_dataset_train(args, get_view, get_validation_augmentation)
    all_index = valid_dataset._get_images_index()
    num_known_classes = valid_dataset._get_num_known_classes()

    if args.train_type == 'cross':
        Kfold = KFold(n_splits=args.k_fold, shuffle=False)
        for k, (train_index, test_index) in enumerate(Kfold.split(all_index)):
            if k == args.k_idx:
                valid_sampler = SubsetRandomSampler(test_index)
                val_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)
    else: 
        # print('Holdout / test size.:', args.test_size)
        _, val_index = train_test_split(all_index, test_size=args.test_size, shuffle=False)
        valid_sampler = SubsetRandomSampler(val_index)
        val_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)
    
    model_list = []
    net.eval()
    
    with torch.no_grad():
        
        ipca_training_time = [0.0 for c in range(num_known_classes)]
        
        for c in range(num_known_classes):
            
            # Computing Incremental PCA models from features.
            model = decomposition.IncrementalPCA(n_components=args.n_components)
            
            model_list.append(model)
        
        for i, data in enumerate(val_loader):
            
            print('Validation Batch %d/%d' % (i + 1, len(val_loader)))
            sys.stdout.flush()
            
            img, mask, _ = data
            img = img.cuda(args.device)
            mask = mask.cuda(args.device)
            
            if args.model == 'deeplabv3_resnet101' or args.model == 'deeplabv3': # gopenipcs
                features = net(img)
                outs = features['classifier.4']
                classif3 = features['classifier.3']
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                classif3 = interpolate(classif3, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs.squeeze(), classif3.squeeze()], 0)
                
            elif args.model == 'seg_former': # gopenipcs
                features = net(img, mask)
                hidden = features[1]
                decode_head = net.model.decode_head
                outs, last_classifier = decode_head(hidden)
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                last_classifier = interpolate(last_classifier, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs.squeeze(), last_classifier.squeeze()], 0)
                
            else: 
                raise NotImplementedError(f'This ({args.model}) model not supported')
                
            feat_flat = feat_flat.permute(1, 2, 0).contiguous().view(feat_flat.size(1) * feat_flat.size(2), feat_flat.size(0)).cpu().numpy()
            prds_flat = prds.cpu().numpy().ravel()
            true_flat = mask.cpu().numpy().ravel()
            
            for c in range(num_known_classes):
                
                tic = time.time()
                
                model_list[c] = partial_fit_ipca_model(model_list[c], feat_flat, true_flat, prds_flat, c)

                toc = time.time()
                ipca_training_time[c] += (toc - tic)
                
            # to debug
            if i > 25:
                break
    
    for c in range(num_known_classes):
        
        print('Time spent fitting model %d: %.2f' % (c, ipca_training_time[c]))
    
    model_full = {'generative': model_list}
    
    return model_full

def partial_fit_ipca_model(model, feat_np, true_np, prds_np, cl):
    
    if np.any((true_np == cl) & (prds_np == cl)):
        
        cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]
        
        try:
            model.partial_fit(cl_feat_flat)
        except ValueError:
            n_samples = len(cl_feat_flat)
            if args.n_components > n_samples:
                e = int((args.n_components / n_samples))
                cl_feat_flat = np.repeat(cl_feat_flat, e, axis=0)
                try:
                    model.partial_fit(cl_feat_flat)
                except ValueError:
                    pass
        
    return model

def compute_threshold_from_pca(args, net, model_full):

    # validation data
    args.split = 'train'
    test_view, test_dataset = get_dataset_train(args, get_view, get_validation_augmentation)
    
    all_index = test_dataset._get_images_index()
    num_known_classes = test_dataset._get_num_known_classes()

    if args.train_type == 'cross':
        Kfold = KFold(n_splits=args.k_fold, shuffle=False)
        for k, (train_index, test_index) in enumerate(Kfold.split(all_index)):
            if k == args.k_idx:
                valid_sampler = SubsetRandomSampler(test_index)
                view_loader = DataLoader(test_view, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)
                test_loader = DataLoader(test_dataset, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)
    else:
        _, test_index = train_test_split(all_index, test_size=args.test_size, shuffle=False)
        valid_sampler = SubsetRandomSampler(test_index)
        view_loader = DataLoader(test_view, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)
        test_loader = DataLoader(test_dataset, sampler=valid_sampler, batch_size=1, num_workers=args.num_workers, shuffle=False)

    img_list = []
    msk_list = []
    prd_list = []
    scr_list = []
    name_list = []

    # Setting network for evaluation mode.
    net.eval()

    with torch.no_grad():

        # Iterating over batches.
        for i, (view, data) in enumerate(zip(view_loader, test_loader)):
            
            print('Validation Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            x_view, _, _ = view
            imgs, true, image_name = data
            imgs = imgs.cuda(args.device)
            true = true.cuda(args.device)
                
            tic = time.time()

            # Forwarding
            if args.model == 'deeplabv3_resnet101' or args.model == 'deeplabv3':
                features = net(imgs)
                outs = features['classifier.4'] # ultima
                classif3 = features['classifier.3'] # penultima
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                classif3 = interpolate(classif3, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs, classif3], 1)
                
            elif args.model == 'seg_former':
                features = net(imgs, true)
                hidden = features[1]
                decode_head = net.model.decode_head
                outs, last_classifier = decode_head(hidden)
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                last_classifier = interpolate(last_classifier, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs, last_classifier], 1)
                
            else: 
                raise NotImplementedError(f'This ({args.model}) model not supported')

            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1)).cpu().numpy()
            prds_flat = prds.cpu().numpy().ravel()

            scores = score_pixelwise(model_full, feat_flat, prds_flat, num_known_classes)
            scores = scores.reshape(prds.size(0), prds.size(1), prds.size(2))

            # Transforming images to numpy.
            img_np = util.img_as_ubyte(x_view.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy())
            msk_np = util.img_as_ubyte(true.cpu().squeeze().numpy().astype(np.uint8))
            prd_np = util.img_as_ubyte(prds.cpu().squeeze().numpy().astype(np.uint8))
            scr_np = scores.squeeze()

            # Appending list with rows.
            img_list.append(img_np)
            msk_list.append(msk_np)
            prd_list.append(prd_np)
            scr_list.append(scr_np)
            name_list.append(str(image_name[0]))

            toc = time.time()
            print('        Elapsed Time: %.2f' % (toc - tic)) 
            
            # to debug
            if i > 200:
                break
                
    # ravel
    scr_list_ravel = np.array(scr_list).ravel()
    msk_list_ravel = np.array(msk_list).ravel()
    prd_list_ravel = np.array(prd_list).ravel()
    
    # remove 255 ignored class
    mask = np.ones(len(msk_list_ravel), dtype=bool)
    mask[msk_list_ravel == 255] = False
    scr_list_ravel = scr_list_ravel[mask,...]
    msk_list_ravel = msk_list_ravel[mask,...]
    prd_list_ravel = prd_list_ravel[mask,...]

    # get_thresholds
    print("Define threshold")
    thresholds = get_thresholds(scr_list_ravel, msk_list_ravel, num_known_classes)

    return thresholds

def compute_scores_from_pca(args, net, model_full, thresholds):

    # data
    args.split = 'test'
    test_view, test_dataset = get_dataset_test(args, get_view, get_validation_augmentation)
    view_loader = DataLoader(test_view, batch_size=1, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    num_known_classes = test_dataset._get_num_known_classes()
    categories = test_dataset._get_classes_names()
    class_index = test_dataset._get_classes_index()
    colors = test_dataset._get_colors()

    img_list = []
    msk_list = []
    prd_list = []
    scr_list = []
    name_list = []

    # Setting network for evaluation mode.
    net.eval()

    with torch.no_grad():

        # Iterating over batches.
        for i, (view, data) in enumerate(zip(view_loader, test_loader)):
            
            print('Test Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            x_view, _, _ = view
            imgs, true, image_name = data
            imgs = imgs.cuda(args.device)
            true = true.cuda(args.device)
                
            tic = time.time()

            # Forwarding
            if args.model == 'deeplabv3_resnet101' or args.model == 'deeplabv3':
                features = net(imgs)
                outs = features['classifier.4'] # ultima
                classif3 = features['classifier.3'] # penultima
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                classif3 = interpolate(classif3, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs, classif3], 1)
                
            elif args.model == 'seg_former':
                features = net(imgs, true)
                hidden = features[1]
                decode_head = net.model.decode_head
                outs, last_classifier = decode_head(hidden)
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
                last_classifier = interpolate(last_classifier, size=(224,224), mode="bilinear", align_corners=False)
                soft_outs = F.softmax(outs, dim=1)
                prds = soft_outs.data.max(1)[1]
                feat_flat = torch.cat([outs, last_classifier], 1)
                
            else: 
                raise NotImplementedError(f'This ({args.model}) model not supported')

            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1)).cpu().numpy()
            prds_flat = prds.cpu().numpy().ravel()

            scores = score_pixelwise(model_full, feat_flat, prds_flat, num_known_classes)
            scores = scores.reshape(prds.size(0), prds.size(1), prds.size(2))

            # Transforming images to numpy.
            img_np = util.img_as_ubyte(x_view.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy())
            msk_np = util.img_as_ubyte(true.cpu().squeeze().numpy().astype(np.uint8))
            prd_np = util.img_as_ubyte(prds.cpu().squeeze().numpy().astype(np.uint8))
            scr_np = scores.squeeze()

            # Appending list with rows.
            img_list.append(img_np)
            msk_list.append(msk_np)
            prd_list.append(prd_np)
            scr_list.append(scr_np)
            name_list.append(str(image_name[0]))

            toc = time.time()
            print('        Elapsed Time: %.2f' % (toc - tic)) 
                
    # ravel
    scr_list_ravel = np.array(scr_list).ravel()
    msk_list_ravel = np.array(msk_list).ravel()
    prd_list_ravel = np.array(prd_list).ravel()
    
    # remove 255 ignored class
    mask = np.ones(len(msk_list_ravel), dtype=bool)
    mask[msk_list_ravel == 255] = False
    scr_list_ravel = scr_list_ravel[mask,...]
    msk_list_ravel = msk_list_ravel[mask,...]
    prd_list_ravel = prd_list_ravel[mask,...]
    
    print("Search best thresholding")
    
    # save best param 
    th_score_best = 0.0
    best_score = 0.0
    best_predictions = []
    
    # run over indices
    for i, th in enumerate(thresholds):
        
        full_prd = np.copy(prd_list_ravel)
        full_prd[scr_list_ravel > th] = num_known_classes
        
        # get f1 score
        if args.metric_eval == 'f1score':
            score = f1_score(msk_list_ravel, full_prd, average='macro')
        else:
            score = compute_iou_value(full_prd, msk_list_ravel, num_known_classes+1, ignore_index=None)
        
        if score > best_score:
            best_score = score
            th_score_best = th
            best_predictions = np.copy(full_prd)

    # Show best score
    if args.metric_eval == 'f1score':
        print("Best Score F1-Score.:", best_score)
    else:
        print("Best Score mIoU.:", best_score)

    # save best params       
    save_best_scores_openipcs(args, best_score, th_score_best, args.metric_eval)

    if args.save_images:
        count_save_pred = 0
        for img, msk, prd, scr, name in zip(img_list, msk_list, prd_list, scr_list, name_list):
            
            scr_ravel = np.array(scr).ravel()
            full_prd = np.copy(prd.ravel())
            full_prd[scr_ravel > th_score_best] = num_known_classes
            prd_openipcs = full_prd.reshape(prd.shape[0], prd.shape[1])
            
            if (args.save_images) and (args.num_save_images > count_save_pred):
                count_save_pred += 1
                print("Plot predictions image")
                save_openipcs_predictions(args, img, msk, prd_openipcs, class_index, name, colors)
        
    # return predictions to compute generate report
    return best_predictions, msk_list_ravel, categories

def score_pixelwise(model_full, feat_np, prds_np, num_known_classes):
    
    scores = np.zeros_like(prds_np, dtype=np.float)
    for c in range(num_known_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            # when the pixel is unknown there will be a big difference between the feature vectors
            # transform --> inverse_transform
            if hasattr(model_full['generative'][c], 'n_samples_seen_'):
                feat_recovered = model_full['generative'][c].inverse_transform(model_full['generative'][c].transform(feat_np[feat_msk, :]))
                scores[feat_msk] = np.abs((feat_np[feat_msk, :] - feat_recovered)).sum(axis=-1)
            else: 
                scores[feat_msk] = 0
                
    return scores

def get_thresholds(scr, msk, n_known):
    
    thresholds = []
    for c in range(n_known):
    
        bin_msk = (msk == c)

        fpr, tpr, ths = metrics.roc_curve(bin_msk, scr)

        ths_ = [0.02, 0.05, 0.10, 0.15]
        
        for t in ths_:
            for i in range(len(ths)):
                if tpr[i] >= t:
                    thresholds.append(ths[i])
                    break
                    
    return [np.mean(thresholds)]

if __name__ == '__main__':
    
    args, parser = parse_args_openset()

    # args.folder_id = 'seg_former_prototycal_local_triplet_vaihingen_unknown_class_0_2078270235972'
    # args.k_idx = 0
    # args.name_checkpoint = 'prototycal_local_triplet_vaihingen_unknown_class_0_epoch=16-val_jac_epoch=0.805.ckpt'
    
    # args.folder_id = 'seg_former_prototycal_local_triplet_vaihingen_unknown_class_1_4_2078911047034'
    # args.k_idx = 0
    # args.name_checkpoint = 'prototycal_local_triplet_vaihingen_unknown_class_1_4_epoch=19-val_jac_epoch=0.791.ckpt'
    
    # load args folder for eval
    args_load = os.path.join(args.folder_path, args.folder_id, 'args.json')
    
    with open(args_load, 'rt') as f:
        args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=args)
        args.mode_aug = True
        # print(args)
    
    # update params
    args.savedir = os.path.join(args.folder_path, args.folder_id)
    args.feature_extractor = True
    
    # reproducibility
    reproducibility(args)
    
    # start
    args.batch_size = 1
    args.plot_tsne = False
    compute_openset_openipcs(args)