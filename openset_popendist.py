# from skimage.morphology import opening
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.functional import interpolate

from skimage import util
from sklearn.metrics import f1_score
from sklearn import metrics
from utils.augmentations import get_view, get_validation_augmentation
from utils.func_preparation import load_model, get_dataset_test, get_dataset_train, save_best_scores_popendist, compute_iou_value
from utils.func_preparation import reproducibility, save_popendist_predictions, generete_report, generete_report_unknown
from parse_args import parse_args_openset
import torch.multiprocessing
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.nn import PairwiseDistance
from statistics import mean
import json
import numpy as np
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.multiprocessing.set_sharing_strategy('file_system')

def compute_popendist(args):

    # load model closeset
    model = load_model(args, feature_extraction=args.feature_extractor)

    # save folder
    args.save_folder = 'popendist'

    threshold = compute_threshold(args, model)
    predictions, labels, categories = compute_scores(args, model, threshold)

    generete_report(args, predictions, labels, categories, args.save_folder)
    generete_report_unknown(args, predictions, labels, args.save_folder)

def compute_threshold(args, net):

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

    # info data
    num_known_classes = test_dataset._get_num_known_classes()
    categories = test_dataset._get_classes_names()
    class_index = test_dataset._get_classes_index()
    colors = test_dataset._get_colors()
    
    # Prototype Anchors
    magnitude = 3
    anchor_prototypes = torch.zeros((num_known_classes, num_known_classes)).cuda()
    for i in range(num_known_classes): # num_classes
        anchor_prototypes[i][i] = magnitude
    
    # distance function
    distance_function = PairwiseDistance(p=2)
        
    img_list = []
    msk_list = []
    prd_list = []
    scr_list = []
    name_list = []
    prob_list = {}
    
    # Setting network for evaluation mode.
    net.eval()
    
    with torch.no_grad():

        # Iterating over batches.
        for i, (view, data) in enumerate(zip(view_loader, test_loader)):
            
            print('Validation Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            x_view, _, _ = view
            img, mask, image_name = data

            # Casting tensors to cuda.
            img = img.cuda(args.device)
            mask = mask.cuda(args.device)

            # deeplabv3
            if args.model == 'deeplabv3_resnet101' or args.model == 'deeplabv3':
                features = net(img)
                outs = features['out']
            elif args.model == 'seg_former':
                features = net(img, mask)
                hidden = features[1]
                decode_head = net.model.decode_head
                outs, _ = decode_head(hidden)
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
            else: 
                raise NotImplementedError(f'This ({args.model}) model not supported')
            
            # 
            feat_flat = outs
            
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)
            # soft_outs = torch.nn.functional.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.data.max(1)[1]

            # Obtaining posterior predictions.
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1))
            
            # append all predictions by class
            distances = [[] for c in range(num_known_classes)]
            
            for c in range(num_known_classes):
                    
                # outliers_prds[c][outliers_prds[c] <= 0] = 0
                distances[c] = distance_function(anchor_prototypes[c], feat_flat).cpu().numpy()
            
            # 
            feat_flat = feat_flat.cpu().numpy()            
            prds_flat = prds.cpu().numpy().ravel()
            true_flat = mask.cpu().numpy().ravel()
            
            # tranpose
            distances = np.transpose(distances)
            
            # compute score by pixel
            scrs_flat = []
            
            # min distance
            for idx, (f, p, t, d) in enumerate(zip(feat_flat, prds_flat, true_flat, distances)):

                # get min distance
                score = np.min(d)
                
                # anomaly scores by pixel
                scrs_flat.append(score) 
                
            # Transforming images to numpy.
            img_np = util.img_as_ubyte(x_view.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy())
            
            # Appending list with rows.
            img_list.append(img_np)
            msk_list.append(true_flat)
            prd_list.append(prds_flat)
            scr_list.append(np.array(scrs_flat))
            # prob_list = np.concatenate((prob_list, prob_iforest), axis=0)
            name_list.append(str(image_name[0]))
            
            # to debug
            # if i > 100:
            #     break
    
    # ravel
    msk_list_ravel = np.array(msk_list).ravel()
    prd_list_ravel = np.array(prd_list).ravel()
    scr_list_ravel = np.array(scr_list).ravel()
    
    # remove 255 ignored class
    mask = np.ones(len(msk_list_ravel), dtype=bool)
    mask[msk_list_ravel == 255] = False
    scr_list_ravel = scr_list_ravel[mask,...]
    msk_list_ravel = msk_list_ravel[mask,...]
    prd_list_ravel = prd_list_ravel[mask,...]
    
    # get thresholds (mean preset values TPR)
    print("Define threshold")
    thresholds = get_threshold(scr_list_ravel, msk_list_ravel, num_known_classes)
    
    return thresholds

def compute_scores(args, net, threshold):

    # use only val data to extract mavs
    args.split = 'test'

    # get dataset and dataloader
    test_view, test_dataset = get_dataset_test(args, get_view, get_validation_augmentation)
    view_loader = DataLoader(test_view, batch_size=1, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

    # info data
    num_known_classes = test_dataset._get_num_known_classes()
    categories = test_dataset._get_classes_names()
    class_index = test_dataset._get_classes_index()
    colors = test_dataset._get_colors()
    
    # Prototype Anchors
    magnitude = 3
    anchor_prototypes = torch.zeros((num_known_classes, num_known_classes)).cuda()
    for i in range(num_known_classes): # num_classes
        anchor_prototypes[i][i] = magnitude
    
    # distance function
    distance_function = PairwiseDistance(p=2)
        
    img_list = []
    msk_list = []
    prd_list = []
    scr_list = []
    name_list = []
    prob_list = {}
    
    # Setting network for evaluation mode.
    net.eval()
    
    with torch.no_grad():

        # Iterating over batches.
        for i, (view, data) in enumerate(zip(view_loader, test_loader)):
            
            print('Test Batch %d/%d' % (i + 1, len(test_loader)))
            sys.stdout.flush()
            
            # Obtaining images, labels and paths for batch.
            x_view, _, _ = view
            img, mask, image_name = data

            # Casting tensors to cuda.
            img = img.cuda(args.device)
            mask = mask.cuda(args.device)

            # deeplabv3
            if args.model == 'deeplabv3_resnet101' or args.model == 'deeplabv3':
                features = net(img)
                outs = features['out']
            elif args.model == 'seg_former':
                features = net(img, mask)
                hidden = features[1]
                decode_head = net.model.decode_head
                outs, _ = decode_head(hidden)
                outs = interpolate(outs, size=(224,224), mode="bilinear", align_corners=False)
            else: 
                raise NotImplementedError(f'This ({args.model}) model not supported')
            
            # 
            feat_flat = outs    
            
            # Computing probabilities.
            soft_outs = F.softmax(outs, dim=1)
            # soft_outs = torch.nn.functional.softmax(outs, dim=1)

            # Obtaining prior predictions.
            prds = soft_outs.data.max(1)[1]

            # Obtaining posterior predictions.
            feat_flat = feat_flat.permute(0, 2, 3, 1).contiguous().view(feat_flat.size(0) * feat_flat.size(2) * feat_flat.size(3), feat_flat.size(1))
            
            # append all predictions by class
            distances = [[] for c in range(num_known_classes)]
            
            for c in range(num_known_classes):
                    
                # outliers_prds[c][outliers_prds[c] <= 0] = 0
                distances[c] = distance_function(anchor_prototypes[c], feat_flat).cpu().numpy()
            
            # 
            feat_flat = feat_flat.cpu().numpy()            
            prds_flat = prds.cpu().numpy().ravel()
            true_flat = mask.cpu().numpy().ravel()
            
            # tranpose
            distances = np.transpose(distances)
            
            # compute score by pixel
            scrs_flat = []
            
            # min distance
            for idx, (f, p, t, d) in enumerate(zip(feat_flat, prds_flat, true_flat, distances)):

                # get min distance
                score = np.min(d)
                
                # anomaly scores by pixel
                scrs_flat.append(score) 
                
            # Transforming images to numpy.
            img_np = util.img_as_ubyte(x_view.permute(0, 2, 3, 1).detach().cpu().squeeze().numpy())
            
            # Appending list with rows.
            img_list.append(img_np)
            msk_list.append(true_flat)
            prd_list.append(prds_flat)
            scr_list.append(np.array(scrs_flat))
            # prob_list = np.concatenate((prob_list, prob_iforest), axis=0)
            name_list.append(str(image_name[0]))
            
            # to debug
            # if i > 200:
            #     break
    
    # ravel
    msk_list_ravel = np.array(msk_list).ravel()
    prd_list_ravel = np.array(prd_list).ravel()
    scr_list_ravel = np.array(scr_list).ravel()
    
    # remove 255 ignored class
    mask = np.ones(len(msk_list_ravel), dtype=bool)
    mask[msk_list_ravel == 255] = False
    scr_list_ravel = scr_list_ravel[mask,...]
    msk_list_ravel = msk_list_ravel[mask,...]
    prd_list_ravel = prd_list_ravel[mask,...]
    
    # save best param 
    th_best = 0.0
    th_score_best = 0.0
    best_score = 0.0
    best_predictions = []
    
    print("Open Set Prediction")
    for i, th in enumerate(threshold):
            
        full_prd = np.copy(prd_list_ravel)
        full_prd[scr_list_ravel > th] = num_known_classes

        # get f1 score
        if args.metric_eval == 'f1score':
            score = f1_score(msk_list_ravel, full_prd, average='macro')
        else:
            score = compute_iou_value(full_prd, msk_list_ravel, num_known_classes+1, ignore_index=None)
        
        if score > best_score:
            best_score = score
            th_best = th
            best_predictions = np.copy(full_prd)
    
    # Show best score
    if args.metric_eval == 'f1score':
        print("Best Score F1-Score.:", best_score)
    else:
        print("Best Score mIoU.:", best_score)
    
    save_best_scores_popendist(args, best_score, th_best, args.metric_eval)
    
    # compute enhancer
    predictions = []
    count_save_pred = 0
    for img, msk, prd, scr, name in zip(img_list, msk_list, prd_list, scr_list, name_list):
        
        scr_ravel = np.array(scr).ravel()
        full_prd = np.copy(prd.ravel())
        full_prd[scr_ravel > th_score_best] = num_known_classes
        prd_openipcs = full_prd.reshape(img.shape[0], img.shape[1])
        msk = msk.reshape(img.shape[0], img.shape[1])
        
        if (args.save_images) and (args.num_save_images > count_save_pred):
            count_save_pred += 1
            print("Plot predictions image")
            save_popendist_predictions(args, img, msk, prd_openipcs, class_index, name, colors)
        
    # return predictions to compute generate report
    return best_predictions, msk_list_ravel, categories

def get_threshold(scr, msk, n_known):
    
    thresholds = []
    for c in range(n_known):
    
        bin_msk = (msk == c)

        fpr, tpr, ths = metrics.roc_curve(bin_msk, scr)

        ths_ = [0.02, 0.05, 0.1, 0.15]
        
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
        # t_args = argparse.Namespace()
        args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=args)
        args.mode_aug = True
        args.th = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # print(args)

    # update params
    args.savedir = os.path.join(args.folder_path, args.folder_id)
    args.feature_extractor = True

    # reproducibility
    reproducibility(args)

    # start
    args.batch_size = 1
    compute_popendist(args)
