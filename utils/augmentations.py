import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import cv2

def get_training_augmentation(args):
    
    width = args.input_width
    height = args.input_height
    p = args.prob_augmentation
    
    # https://dev.to/itminds/improve-your-deep-learning-models-with-image-augmentation-4j7f
    train_transform = [
        
        # zoom in or out transformation
        albu.OneOf([
            albu.Sequential([
                albu.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0, mask_value=255),
                albu.RandomResizedCrop(height=height, width=width, scale=(0.1, 0.9), ratio=(0.75, 1.25)),
            ]),
            albu.Resize(height=height, width=width, always_apply=True)
        ], p=1),
        
        # Flip / Geometry transformations
        albu.Sequential([
            albu.OneOf([
                albu.VerticalFlip(p=p),
                albu.HorizontalFlip(p=p),
                # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=90, shift_limit=0.5, p=p, border_mode=0, mask_value=255),
            ], p=1),
            albu.RandomRotate90(p=p)
        ]),
            
        # Noise transformations
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=p),
            albu.GaussianBlur(blur_limit=(5), p=p),
            # albu.GaussNoise(p=p),
        ], p=1),
        
        # Dropout transformations
        # albu.GridDropout(mask_fill_value=255, p=p),
        albu.OneOf([
            albu.GridDropout(mask_fill_value=255, p=p),
            albu.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=p, mask_fill_value=255),
        ], p=1),
        
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return albu.Compose(val_transform)

def get_view(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
        ToTensorV2()
    ]
    return albu.Compose(val_transform) 

def get_view_numpy(args):
    
    width = args.input_width
    height = args.input_height
    
    """Add paddings to make image shape divisible by 32"""
    val_transform = [
        albu.Resize(height=height, width=width, always_apply=True),
    ]
    return albu.Compose(val_transform) 