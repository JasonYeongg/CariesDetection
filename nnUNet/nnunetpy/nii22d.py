import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
from nnunet.utilities.file_conversions import convert_2d_segmentation_nifti_to_img
from tqdm import tqdm
import glob

if __name__ == '__main__':

    task = 'Task201_Caries'
    location = 'inferTs'
    targetname = 'Predict1'
    base = r'/home/dentall/Desktop/jason_test/nnUNetFrame/DATASET/'
    # this folder should have the training and testing subfolders

    target_images = join(base, task)
	
    oExist = True
    while os.path.exists(target_images):
        os.rename(target_images, target_images + r"_delete")
        if oExist:
            shutil.rmtree(target_images + r"_delete")
            oExist = False
    os.mkdir(target_images)
    target_images = join(target_images, targetname)
    os.mkdir(target_images)


    base1 = r'/home/dentall/Desktop/jason_test/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/'
    image_cases = join(base1, task, location)
    image_paths = glob.glob(image_cases + "/*")
    for t in tqdm(image_paths):
        filepath, filename = os.path.split(t)
        if filename[-7:] != ".nii.gz":
            continue
        unique_name = filename[:-7] + ".jpg" # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
		
        input_image_file = join(image_cases, t)
        output_image_file = join(target_images, unique_name)

        convert_2d_segmentation_nifti_to_img(input_image_file, output_image_file) #jason add something in def

