import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def open_preprocessing_isprs(args, image_path, segm_path):
    img = Image.open(image_path).convert('RGB')
    segm = Image.open(segm_path)
    segm = np.array(segm)
    segm = encode_segmap_isprs(segm)
    segm = remove_open_class(args.split, segm, args.original_num_classes, args.openset_idx)
    # segm = Image.fromarray(segm)
    return np.array(img), np.array(segm)

def remove_open_class(split, segm, num_class, open_idx):
    
    closeset_idx = [i for i in range(0, num_class)]
    for i in open_idx:
        closeset_idx.remove(i)
        segm[segm == i] = 128 # tmp ids
        
    class_map = dict(zip(closeset_idx, range(len(closeset_idx))))
    
    if split == 'train':
        ignore_index = 255 # closeset_idx
    else:
        ignore_index = num_class - (len(open_idx))
        
    class_map[128] = ignore_index
    closeset_idx.append(128)
        
    # somente este precisa ser excutado em toda imagem
    for i in closeset_idx:
        segm[segm == i] = class_map[i]
    
    return segm

def encode_segmap_isprs(mask):
     
    # 
    palette = [    
                        
                        (255, 255, 255),    # Impevious surfaces or Roads               (white)
                        (  0,   0, 255),    # Buildings                                 (blue)
                        (  0, 255, 255),    # Low vegetation                            (cyan)
                        (  0, 255,   0),    # Trees                                     (green)
                        (255, 255,   0),    # Cars                                      (yellow)
                        (255,   0,   0),    # Clutter or unknown                        (red)
                        (  0,   0,   0)     # Boundaries/Make Boundaries                (black) 
                        
                        ]
        
    class_names =  ['roads', 'buildings', 'low veg.', 'trees', 'cars'] # clutter and boundaries are ignoreds
       
    # class_map
    class_map = {}
    class_map[0] = 0
    class_map[1] = 1
    class_map[2] = 2
    class_map[3] = 3
    class_map[4] = 4
    class_map[5] = 255
    class_map[6] = 255
    
    # original index
    class_index = range(0, len(palette))
    
    # print(palette)
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
    palette_idx = dict(zip(class_index, palette))
    
    # for c in class_index:
    for c in class_index:
        aux = mask == palette_idx[c]
        aux = np.logical_and(np.logical_and(aux[:,:,0], aux[:,:,1]), aux[:,:,2])     
        new_mask[aux] = class_map[c]
    
    # map to unknown class
    new_mask = np.squeeze(new_mask)
            
    return new_mask

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, odgt, augmentation, **kwargs):
        
        self.args = args
        self.augmentation = augmentation
        self.segm_downsampling_rate = 8
        
        self.target = []
        self.images = []
        self.indexes = []
        
        # update indexes and class names
        self.tmp_names = self._get_categories()
        self.args.original_num_classes = len(self.tmp_names)
        self.class_names = []
        for i in range(0, len(self.tmp_names)):
            if i not in args.openset_idx:
                self.class_names.append(self.tmp_names[i])
        
        # update num_classes
        self.num_classes = len(self.class_names)
        self.class_index = [i for i in range(self.num_classes)]
        
        # include unknown class in test
        if self.args.split == "test": 
            #open-set segmentation
            self.class_names.append("unknown")
        
        # get path from odgt
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')][0]

        for i, s in enumerate(self.list_sample):
            self.indexes.append(i)
            self.images.append(s['fpath_img'])
            self.target.append(s['fpath_segm'])
            
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
    
    def __getitem__(self, index):
        
        image_name = self.images[index].split('/')[-1]
        
        if self.args.dataset == 'vaihingen' or self.args.dataset == 'potsdam':
            image, target = open_preprocessing_isprs(self.args, self.images[index], self.target[index])
        else:
            print('x')
        
        # apply augmentation
        sample = self.augmentation(image=image, mask=target)
        image, target = sample['image'], sample['mask']

        return image, target, image_name
    
    def __len__(self):
        return self.num_sample
    
    def _class_to_idx(self, dictionary=False):
        if dictionary:
            return dict(zip(self.class_names, range(self.num_classes)))
        return list(zip(self.class_names, range(self.num_classes)))
    
    def _get_images_index(self):
        return self.indexes
    
    def _get_images_labels(self):
        return self.images, self.targets
    
    def _get_classes_index(self):
        return [x for x in range(len(self.class_names))]
    
    def _get_classes_names(self):
        return self.class_names
    
    def _get_num_known_classes(self):
        if self.args.split == 'train':
            return len(self.class_names) 
        else:
            # return len(self.class_names) - len(self.args.openset_idx)
            return len(self.class_names) - 1
    
    def _get_num_unknown_classes(self):
        return len(self.args.openset_idx)
    
    def _get_categories(self):
        
        if self.args.dataset == 'vaihingen' or self.args.dataset == 'potsdam':
            class_names = [     'roads',        # 0
                                'buildings',    # 1
                                'low veg.',     # 2
                                'trees',        # 3
                                'cars']         # 4
        else: 
            raise NotImplementedError(f'This ({self.args.dataset}) dataset not supported')
        
        return class_names
    
    def _get_colors(self):
           
        if self.args.dataset == 'vaihingen' or self.args.dataset == 'potsdam':    
            palette =   [ 
                            (192, 192, 192),    # Impevious surfaces or Roads               (gray)
                            (  0,   0, 255),    # Buildings                                 (blue)
                            (  0, 255, 255),    # Low vegetation                            (cyan)
                            (  0, 255,   0),    # Trees                                     (green)
                            (255, 255,   0),    # Cars                                      (yellow)
                            (200,   0,   0)     # Unknown                                   (red)
                        ]
        else: 
            raise NotImplementedError(f'This ({self.args.dataset}) dataset not supported')
            
        colors = palette.copy()

        #remove openset class
        for i in self.args.openset_idx:
            del colors[i]
            
        return np.array(colors)