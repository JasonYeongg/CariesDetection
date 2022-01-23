# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import multiprocessing as mp
import tempfile
import time
import warnings
import numpy as np
import os, json, cv2, random, glob
import PIL.Image as Image
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from shapely.geometry.polygon import Polygon
import urllib.request
import shutil

import sys
sys.path.append("demo/")#添加其他文件夹
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

# constants
WINDOW_NAME = "SOLOv2 detections"

def split_mask_n_duplicate_array(masks,sscores):
			
    dmasks = []
    dscores = []
	
    for mask,sscore in zip(masks,sscores):
        mask8 = convert_bool_uint8(mask)
        cnts = []
        cnts = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) >= 2:
            for cc in cnts:
                if (len(cc) < 3):
                    continue
                ccc = np.zeros((mask.shape[0], mask.shape[1]),np.uint8)
                cv2.fillPoly(ccc, [cc], 255)
                ccc = ccc>200
                dmasks.append(ccc)
                dscores.append(sscore)
        else:
            dmasks.append(mask)
            dscores.append(sscore)
    dmasks = np.array(dmasks)
		
    return dmasks, dscores

def checkans(dataname, data, score):

    with open( "data.json") as f:
        tm_text = json.load(f)

    filename = str("https://storage.googleapis.com/taipei_medical/" + str(dataname) + ".jpg")
		
    #記錄當下圖片的結果
    TP=0
    FP=0
    checkedp = 0
    emptyp = 0
	
    skip = 0

    for key,value in tm_text.items():
        if skip:
            skip = 0
            continue
			
        raw_data = tm_text[key]
		
        url,filename2 = os.path.split(str(raw_data['filename']))
        filename2, suffix = os.path.splitext(filename2)
		
        if int(filename2) == 267:
            skip = 1#268 bug
			
        if(filename == raw_data['filename']):
            oimg = io.imread(filename)
            py, px = oimg.shape[:2]
		
            t_mask = []
            maxx = 0
            minx = px
            caries = np.zeros((py,px),np.uint8)
	
            for anno in raw_data['regions']:
                if(anno == None):
                    continue
                region = anno['region_attributes']
			
                temp = Image.new("L", [px,py], 0)
                if 'detail' in region and region['detail'] == 'caries':
                    xs = anno['shape_attributes']['all_points_x']
                    ys = anno['shape_attributes']['all_points_y']
                    combined = list(map(tuple, np.vstack((xs, ys)).T))

                    ImageDraw.Draw(temp).polygon(combined, outline=1, fill=1)
                    temp = np.array(temp)
                    temp = convert_bool_uint8(temp)
                    caries = caries + temp
		
    #讀取圖片
    overlay = oimg.copy()
    overlayo = oimg.copy()
		
    #caries = datahandlet.convert_bool_uint8(caries) #將caries值map到0和255並轉成uint8
    caries = caries > 0 #mix all type caries to boolean
    caries = caries * 1 #convert boolean to 0/255
    caries = caries.astype(np.uint8)
    cariessize = np.count_nonzero(caries)

    data,score = split_mask_n_duplicate_array(data,score)
    sortdata =  sorted(zip(score,data), reverse=True  , key = lambda x:  x[0])
	
    selectdata = []
    selectedist = []
    sortlist = sortdata.copy()
    while (sortlist): #NMS
        clearlist = []
        for num in range(1,len(sortlist)):
            #iou = intersection_over_union((sortlist[0][1].to("cpu").numpy())*1, (sortlist[num][1].to("cpu").numpy())*1)
            iou = intersection_over_union(sortlist[0][1], sortlist[num][1])
            if iou > 0.05:
                clearlist.append(num)
        while (clearlist):
            del sortlist[clearlist[len(clearlist)-1]]
            del clearlist[len(clearlist)-1]
        selectedist.append(sortlist[0])
        del sortlist[0]
    selectdata = selectedist.copy();
	
    print(selectdata)
    if(len(selectdata)) <=  2:
        threshold = 0
    else:
        sum=0
        '''
        for num in range(0,(len(selectdata)-1)):
            sum += selectdata[num][0]
        threshold = sum/(len(selectdata)*0.7)
        '''
        threshold = (selectdata[0][0] + selectdata[(len(selectdata)-1)][0])/2
		
    masks = []
		
    for dlist in selectdata:
        if dlist[0] > threshold:
            masks.append(dlist[1])
	
    pcaries = np.zeros((py,px),np.uint8)
    for pdata in masks:
        pdatasize = convert_bool2one_uint8(pdata)
        #pdatasize = (pdata.to("cpu").numpy())*1
        #pdatasize = (pdatasize.astype(np.uint8))
        ptruesize = np.count_nonzero((caries + pdatasize) >=2)

        pcaries = pdatasize + pcaries

        if  int(ptruesize) > 0:
            TP += 1
        else:
            FP += 1

    pcaries = convert_bool2one_uint8(pcaries)

    cnts = []
    cnts = cv2.findContours(caries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for cc in cnts:
        if (len(cc) < 3):
            continue
        ccc = np.zeros((py,px),np.uint8)
        cv2.fillPoly(ccc, [cc], 1)

        pchecksize = np.count_nonzero((pcaries + ccc) >=2)
		
        if  int(pchecksize) > 0:
            checkedp += 1
        else:
            emptyp += 1
				
    print(">>>>>>>>>>>>> TP:",TP,"  FP:" ,FP," gotP:",checkedp," emptyP:",emptyp)
    return TP, FP, checkedp, emptyp
	
def intersection_over_union(frame1, frame2):
    # compute the area of intersection rectangle
    interArea = np.count_nonzero(np.logical_and(frame1 ,frame2))                 #predict of img.shape * 0.0005 maybe too small
    # compute the area of both the prediction and ground-truth
    # rectangles
    frame1Area = np.count_nonzero(frame1)
    frame2Area = np.count_nonzero(frame2)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(frame1Area + frame2Area - interArea)
    #iou = interArea / float(frame1Area)
    iou = interArea / float(frame1Area + frame2Area - interArea)
    if iou <= 0.05:
        if ((interArea >=  (frame1Area*0.8)) or (interArea >=  (frame2Area*0.8))):
            iou = 0.9
        elif ((frame1Area < (0.00005*frame1.shape[0]*frame1.shape[1])) or (frame2Area < (0.00005*frame1.shape[0]*frame1.shape[1]))):
            iou = 0.9
    #if interArea > 0:
        #plt.title((interArea,frame1Area,int(frame1Area*0.8),frame2Area,int(frame2Area*0.8),iou))
        #plt.imshow((frame1*10)+(frame2*20))
        #plt.show()
		
    return iou
	
def confusion_matrix(TP,TN,FP,FN, save_dir='', names=["normal","caries"], normalize=True):
	
    matrix = np.zeros((2,2))
    matrix[0, 0] = TN
    matrix[0, 1] = FN
    matrix[1, 0] = FP
    matrix[1, 1] = TP
    array = matrix / ((matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    fig = plt.figure(figsize=(12, 9), tight_layout=True)
    sn.set(font_scale=1.0)  # for label size
    labels = (0 < len(names) < 99) and len(names) == 2  # apply names to ticklabels
    sn.heatmap(array, annot = True, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
        xticklabels=names if labels else "auto",
        yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel('True')
    fig.axes[0].set_ylabel('Predicted')
    fig.savefig(save_dir+"confusion_matrix.png", dpi=250)

def extract_bboxes(masks):
    bboxes = []
    for mask in masks:
        # print(mask.shape)
        m = GenericMask(mask.to("cpu").numpy(), mask.shape[0], mask.shape[1])
        bboxes.append(m.bbox())
    return bboxes

def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image
	
def convert_bool2one_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *1 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":

     #ori in demo/demopy
	 #detectron2/detectron2/data/datasets/builtin_meta.py   中COCO_CATEGORIES修改为COCO_CATEGORIES_caries
	 #修改detectron2/utils/visualizer.py #400 edited !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!up!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
     #python predict.py  --config-file AdelaiDet/configs/SOLOv2/R50_3x.yaml  --input caries/val --opts MODEL.WEIGHTS caries/model_final.pth 
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
	
    outdir_result = r"clu/result"
    Exist = True
    while os.path.exists(outdir_result):
        os.rename(outdir_result, outdir_result + r"_delete")
        if Exist:
            shutil.rmtree(outdir_result + r"_delete")
            Exist = False
    os.mkdir(outdir_result)

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    TTP = 0
    FFP = 0
    includep = 0
    emptyp = 0
	
    if args.input:
        each_paths = glob.glob(str(args.input[0]) + "/*")
        if (len(args.input) <= 0): log.warning("eachdata_paths is empty\n")

        t = tqdm(each_paths) # Create progress bar
        t.set_description("loaded val data")
        for path in t:

            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
			

            filepath, full_filename = os.path.split(path)
            filename, suffix = os.path.splitext(full_filename)

            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
			
            scores = predictions["instances"].to("cpu").scores.tolist() 
            masks = np.asarray(predictions["instances"].to("cpu").pred_masks) 

            TP, FP, ip, ep = checkans(filename, masks, scores)
            #predictions, visualized_output = demo.run_on_image(img,dataname=filename)
            TTP += TP
            #TTN += TN
            FFP += FP
            #FFN += FN
            includep += ip
            emptyp += ep

            #noted that precision is TP/TP+FP ,recall  ip/ip+ep

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            filepath, filename = os.path.split(path)
			
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                #cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                cv2.imwrite(outdir_result+"/"+str(filename ), visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
					
        if (TTP == 0) and (FFP ==0):
            FFPnz = 1
        else:
            FFPnz = FFP

        #confusion_matrix(TP,TN,FP,FN,save_dir="caries/result/")
        print('----------------------------------------------------------------------------')
        #print(f'	> Num {TP+TN+FP+FN} -- TP: {TP} - TN: {TN} - FP: {FP} - FN: {FN}	')
        print(f'	> Num {TTP+FFP} -- TP: {TTP} - FP: {FFP} - include: {includep} - empty: {emptyp}')
        print('----------------------------------------------------------------------------')
        print(" > Accuracy: %.2f%%"% ((TTP/(TTP+FFPnz))*100))
        print('----------------------------------------------------------------------------')
		
        #save result as  txt
        r = open(outdir_result + r"/0result.txt", "w")
        r.write("---------------------------------------------------------------------------- \n")
        r.write("> Num "+ str(TTP+FFP) +" -- TP: "+ str(TTP) +" - FP: "+ str(FFP) +" - include: "+ str(includep) +" - empty: "+ str(emptyp) +" \n")
        r.write("---------------------------------------------------------------------------- \n")
        r.write("> Accuracy: "+ str(((TTP/(TTP+FFPnz))*100)) +" \n")
        r.write("---------------------------------------------------------------------------- \n")
        r.close()

    
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
