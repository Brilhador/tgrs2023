import torch
import numpy as np 
import random 
import cv2 
import albumentations as albu
import time

def reproducibility(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cv2.setRNGSeed(args.seed)
    torch.backends.cudnn.deterministic = True
    
def outputs_dir(args):
    idx = '_'.join(map(str, args.openset_idx))
    args.savedir = args.encoder + '_' + args.decoder + '_' + args.dataset + '_' + 'unknown_class_' + idx + '_' + str(int(round(time.time() * args.seed)))
    
class Denormalize(object):
    
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):
        
        self.denormalize = albu.Compose(
            [
                albu.Normalize(
                    mean=tuple(-m / s for m, s in zip(mean, std)),
                    std=tuple(1.0 / s for s in std),
                    max_pixel_value=1.0,
                ),
                albu.FromFloat(max_value=255, dtype="uint8"),
            ]
        )
        
    def __call__(self, img):
        if  isinstance(img, torch.Tensor):
            img = np.transpose(img.cpu().detach().numpy().squeeze(), (1, 2, 0))
        return self.denormalize(image=img)['image']
    
def segmap_to_rgb(input, num_class, class_index, colors=None):

    if colors is None: 
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor(class_index)[:, None] * palette
        colors = torch.as_tensor([i for i in range(num_class)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

    r = np.zeros_like(input).astype(np.uint8)
    g = np.zeros_like(input).astype(np.uint8)
    b = np.zeros_like(input).astype(np.uint8)

    for i, c in enumerate(class_index):
    # for c in class_index:
        idx = input == c
        r[idx] = colors[i, 0]
        g[idx] = colors[i, 1]
        b[idx] = colors[i, 2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize_cv2(img, mean, denominator):
    if mean.shape and len(mean) != 4 and mean.shape != img.shape:
        mean = np.array(mean.tolist() + [0] * (4 - len(mean)), dtype=np.float64)
    if not denominator.shape:
        denominator = np.array([denominator.tolist()] * 4, dtype=np.float64)
    elif len(denominator) != 4 and denominator.shape != img.shape:
        denominator = np.array(denominator.tolist() + [1] * (4 - len(denominator)), dtype=np.float64)

    img = np.ascontiguousarray(img.astype("float32"))
    cv2.subtract(img, mean.astype(np.float64), img)
    cv2.multiply(img, denominator.astype(np.float64), img)
    return img

def normalize_numpy(img, mean, denominator):
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img

def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[-1] == 3:
        return normalize_cv2(img, mean, denominator)
    return normalize_numpy(img, mean, denominator)
        
    
