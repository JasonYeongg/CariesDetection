import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
from tqdm import tqdm

if __name__ == '__main__':

    '''
	
    nnUnet最初适用于3D数据，当处理2D数据时需要将数据做成一个伪3D的图像，
    形状为（X，Y）的图像需要转换为形状为（1，X，Y）的图像，结果图像必须以nifti格式保存，
    将第一轴（具有形状1的轴）的间距设置为大于其他轴的值。

    '''

    # now start the conversion to nnU-Net:
    task_name = 'Task201_Caries'
    path = "/home/dentall/Desktop/jason_test/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data"
    target_base = join(path, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")
	
    oExist = True
    while os.path.exists(target_base):
        os.rename(target_base, target_base + r"_delete")
        if oExist:
            shutil.rmtree(target_base + r"_delete")
            oExist = False
    os.mkdir(target_base)

    for dist in [r"/images", r"/labels"]:
        for ddist in [r"Tr",r"Ts"]:
            os.mkdir(target_base+dist+ddist)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir_tr = join(task_name, 'training', 'output')
    images_dir_tr = join(task_name, 'training', 'input')
    training_cases = subfiles(labels_dir_tr, suffix='.jpg', join=False)
    for t in tqdm(training_cases):
        unique_name = t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
		
        '''
        分别依次读取每个数据（这里是3通道的彩色2d图像），
		获取数据名称unique_name（去掉后缀），输入图像文件为input_image_file，
		标注图像文件为input_segmentation_file，输出图像文件为output_image_file
		（不带后缀，后面会分成3个模态3个输出图像文件），输出标注文件output_seg_file。
        '''

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))
								  

    # now do the same for the test set
    labels_dir_ts = join(task_name, 'testing', 'output')
    images_dir_ts = join(task_name, 'testing', 'input')
    testing_cases = subfiles(labels_dir_ts, suffix='.jpg', join=False)
    for ts in tqdm(testing_cases):
        unique_name = ts[:-4]
        input_segmentation_file = join(labels_dir_ts, ts)
        input_image_file = join(images_dir_ts, ts)

        output_image_file = join(target_imagesTs, unique_name)
        output_seg_file = join(target_labelsTs, unique_name)

        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == 255).astype(int))

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Red', 'Green', 'Blue'),
                          labels={0: 'background', 1: 'caries'}, dataset_name=task_name, license='hands off!')

