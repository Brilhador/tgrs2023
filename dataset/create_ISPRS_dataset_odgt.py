import os
import numpy as np 
from PIL import Image
import json

def create_odgt(root, out_dir, dataset, split):
    
    # get dir images and annotations
    images_dir = os.path.join(root, dataset, split, 'JPEGImages', 'data')
    targets_dir = os.path.join(root, dataset, split, 'Masks', 'data')
    
    # list of dict to dump in file
    out_dir_files = []
    
    for image_name in os.listdir(images_dir):
        
        # get target
        if split == 'Test':
            target_name = image_name.replace("IRRG", "label")
        else:
            target_name = image_name.replace("IRRG", "label_noBoundary")
        
        # files paths
        image_path = os.path.join(images_dir, image_name)
        target_path = os.path.join(targets_dir, target_name)
        
        # Open target to verify index in list 
        target = np.array(Image.open(target_path))
        
        image = Image.open(image_path)
        width, height = image.size
        
        dict_entry = {
            "dbName": "ISPRS_" + dataset,
            "width": width,
            "height": height,
            "fpath_img": image_path,
            "fpath_segm": target_path,
        }

        out_dir_files.append(dict_entry)
            
    print("Total images: " + str(len(out_dir_files)))
            
    file = os.path.join(root, out_dir)
            
    with open(file, "w") as outfile:
        json.dump(out_dir_files, outfile)
        
    print("odgt saved: " + file)
      
def main():
    
    root_dir = 'ISPRS_BENCHMARK_DATASETS'
    # dataset = 'Potsdam'
    dataset = 'Vaihingen'
    # split = 'Train'
    split = 'Test'
    # train images
    out_dir = 'ISPRS_'+dataset+'_'+split+'.odgt'
    create_odgt(root_dir, out_dir, dataset, split)
    
if __name__ == '__main__':
    main()