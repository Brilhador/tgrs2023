from argparse import ArgumentParser

output_dir = "/home/anderson/Documents/prototypical-triplet-learning/outputs/"

def parse_args_train_closeset():

    parser = ArgumentParser(description="Close Set Semantic Segmentation")

    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    # model settings
    parser.add_argument('--model', type=str, default='seg_former', choices=['seg_former', 'deeplabv3_resnet101'], help='model name')
    parser.add_argument('--pretrained', type=bool, default=True, help='pre trained on IMAGENET1k_V1')

    # training setting
    parser.add_argument('--dataset', type=str.lower, default='vaihingen', choices=['vaihingen', 'potsdam'], help='dataset name')
    parser.add_argument('--split', type=str.lower, default='train', choices=['train', 'val', 'train+val','test'], help='dataset split step')
    parser.add_argument('--batch_size', type=int, default=32, help='set the batch size')
    parser.add_argument('--max_epochs', type=int, default=2, help='the number of epochs')
    parser.add_argument('--input_width', type=int, default=224, help='input width of model') 
    parser.add_argument('--input_height', type=int, default=224, help='input height of model') 
    parser.add_argument('--num_workers', type=int, default=8, help='the number of parallel threads')
    parser.add_argument('--prob_augmentation', type=float, default=0.5, help='probability of apply the data augmentation')
    parser.add_argument('--openset_idx', nargs='*', type=int, default=[2,3], help='class indexes defined as unknown')
    
    # cross-validation settings
    parser.add_argument('--train_type', type=str, default='holdout', choices=['holdout', 'cross'], help='train type: cross-validation or holdout') # only cross implemented
    parser.add_argument('--k_fold', type=int, default='3', help='number of k-folds')
    parser.add_argument('--test_size', type=float, default=0.3, help='test size in float (0.2 = 20%)')
    
    # optimization settings 
    parser.add_argument('--optim', type=str.lower,default='nadam',choices=['nadam','sgd','adam','rmsprop'], help="select optimizer")
    parser.add_argument('--lr', type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--w_decay', type=float, default=0.00001, help="weight decay")
    parser.add_argument('--momentum', type=float,default=0.09, help="define momentum value")
    parser.add_argument('--loss', type=str.lower,default='cross_entropy', choices=['cross_entropy', 'prototycal_triplet'], help="select loss functons")

    # triplet loss params
    parser.add_argument('--triplet_margin_global', type=float, default=1.5, help="define the margin to loss functons")
    parser.add_argument('--triplet_margin_local', type=float, default=0.5, help="define the margin to loss functons")
    parser.add_argument('--magnitude', type=float, default=3, help="")
    
    # extra setting
    parser.add_argument('--metrics', type=str.lower, nargs='*', default=['miou'], choices=['fscore', 'miou'], help="select metrics functons")
    parser.add_argument('--early_stopping', type=bool, default=False, help='early stopping based in val_loss')   
    parser.add_argument('--patience', type=int, default=5, help="define patience value")
    parser.add_argument('--feature_extractor', type=bool, default=False, help='')   
    
    # cuda settings
    parser.add_argument('--device', type=str, default='cuda', help="running on CPU or CUDA (GPU)")
    
    # local save
    parser.add_argument('--savedir', default=output_dir, help="directory to save the model snapshot")

    args = parser.parse_args()
    
    return args

def parse_args_train_report():

    parser = ArgumentParser(description="Close Set Semantic Segmentation - Report")

    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    parser.add_argument('--folder_id', type=str, default='', help="directory to load args and models")
    parser.add_argument('--folder_path', type=str, default=output_dir, help="directory to load args and models")     
    parser.add_argument('--save_images', type=bool, default=True, help="save images from predictions") 
    parser.add_argument('--num_save_images', type=int, default=100, help="save images from predictions") 
    parser.add_argument('--posprocessing', type=str, default='None', choices=['slic', 'opening', 'None'], help="")
    
    parser.add_argument('--k_idx', type=str, help='') 
    parser.add_argument('--name_checkpoint', type=str, help='') 
     
    args = parser.parse_args()
    
    return args, parser

def parse_args_openset():
    
    parser = ArgumentParser(description="Open Set Semantic Segmentation - Report")
    
    # global seed
    parser.add_argument('--seed', type=int, default='1234', help='global seed')

    # IPCS settings
    parser.add_argument('--n_components', type=int, default=64, help="")
    
    parser.add_argument('--folder_id', type=str, default='', help="directory to load args and models")
    parser.add_argument('--folder_path', type=str, default=output_dir, help="directory to load args and models")    
    parser.add_argument('--save_images', type=bool, default=True, help="save images from predictions")  
    parser.add_argument('--num_save_images', type=int, default=100, help="save images from predictions") 

    parser.add_argument('--k_idx', type=str, help='') 
    parser.add_argument('--name_checkpoint', type=str, help='') 
    
    # metric eval
    parser.add_argument('--metric_eval', type=str, default='miou', choices=['f1score', 'miou'], help="metric to eval best search")
    
    args = parser.parse_args()
    
    return args, parser