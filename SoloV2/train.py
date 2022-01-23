from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

import numpy as np
import os, json, cv2, random, cv2, glob
import urllib.request
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
import shutil
from collections import OrderedDict
import torch
from tensorflow.keras.preprocessing.image import img_to_array#图片转为array
from torch.nn.parallel import DistributedDataParallel

import pywt
import pywt.data

import logging

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
	
class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

	
def json_handle(outdir_ori,outdir_train,outdir_val,img_dir,numdata):

    #url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"
    #response = urllib.request.urlopen(url)
    #tm_text = json.loads(response.read())
    #'''

    #json_file = os.path.join(img_dir, "data.json")
    with open( "data.json") as f:
        tm_text = json.load(f)
		
    #'''
    trainp = glob.glob(outdir_train+"/*")
    valp = glob.glob(outdir_val+"/*")
    traindata = []
    valdata = []
    
    t = tqdm(trainp) # Create progress bar
    t.set_description("loaded train")
    for c in t:
        filepath, full_filename = os.path.split(c)
        filename, suffix = os.path.splitext(full_filename)
        traindata.append(filename)
		
    t = tqdm(valp) # Create progress bar
    t.set_description("loaded val")
    for n in t:
        filepath, full_filename = os.path.split(n)
        filename, suffix = os.path.splitext(full_filename)
        valdata.append(filename)	
    #'''
		
    dataset_dictst = []
    dataset_dictsv = []
	
    #save dataset as  txt
    #datatxtt = open(img_dir + r"/datasett.txt", "w")
    #datatxtv = open(img_dir + r"/datasetv.txt", "w")
	
    skip = 0

    for key,value in tqdm(tm_text.items()):
        if skip:
            skip = 0
            continue
			
        record = {}
        raw_data = tm_text[key]
		
        if(raw_data['filename'] == None):
            continue
        filename = raw_data['filename']
		
        oimg = io.imread(filename)
        overlay = oimg.copy()
        height, width = oimg.shape[:2]
		
        url,filename = os.path.split(str(filename))
        filename, suffix = os.path.splitext(filename)

        if int(filename) >= numdata: 
            break
        elif int(filename) == 267:
            skip = 1#268 bug
			
        if filename not in np.hstack([traindata,valdata]):
            continue
		
        record["image_id"] = key
        record["height"] = height
        record["width"] = width
		
        cariescheck = 0
        objs = []
        #boxtxt = []
        caries = np.zeros((height, width),np.uint8)
		
        for anno in raw_data['regions']:
            if(anno == None):
                continue
            region = anno['region_attributes']

            temp = Image.new("L", [width, height,], 0)

            if 'detail' in region and region['detail'] == 'caries':
                cariescheck = 1
                px = anno['shape_attributes']['all_points_x']
                py = anno['shape_attributes']['all_points_y']
                if(len(px) < 3):
                    continue
					
                combined = list(map(tuple, np.vstack((px, py)).T))
                ImageDraw.Draw(temp).polygon(combined, outline=1, fill=1)
                temp = np.array(temp)
                temp = convert_bool_uint8(temp)
				
                if (np.count_nonzero(temp)/(width*height)) >= 0.055:
                    continue
                if (int(np.min(px))/int(np.max(px))) <= 0.04:
                    continue

                caries = caries + temp
					
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
				
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
                #boxtxt.append((0, np.min(px), np.min(py), np.max(px), np.max(py)))
            else:
                continue
		
        record["annotations"] = objs 
		
        if  random.random() > 0.09: 
            record["file_name"] = outdir_train +"/" + filename+".jpg"
            dataset_dictst.append(record)
            cv2.imwrite( outdir_train + "/" +str(filename)+".jpg", oimg) 
			
        else: 
            record["file_name"] = outdir_val + "/" + filename+".jpg"
            dataset_dictsv.append(record)
            cv2.imwrite( outdir_val + "/" +str(filename)+".jpg", oimg) 

    return dataset_dictst,dataset_dictsv

if __name__ == '__main__': #防止調用的時候這一行被運行

    outdir = r"caries"
    outdir_ori = outdir + r"/ori"
    outdir_train = outdir + r"/train"
    outdir_val = outdir + r"/val"
	
    '''
    for dist in [outdir_ori,outdir_train,outdir_val]:
        Exist = True
        while os.path.exists(dist):
            os.rename(dist, dist + r"_delete")
            if Exist:
                shutil.rmtree(dist + r"_delete")
                Exist = False
        os.mkdir(dist)
    '''
    dataset_dictst, dataset_dictsv = json_handle(outdir_ori,outdir_train,outdir_val,"caries",9000)#2000

    for d in ['train']:
        DatasetCatalog.register("caries_" + d, lambda d=d: dataset_dictst)
        MetadataCatalog.get("caries_" + d).set(thing_classes=["caries"])
	
    for d in ['val']:
        DatasetCatalog.register("caries_" + d, lambda d=d: dataset_dictsv)
        MetadataCatalog.get("caries_" + d).set(thing_classes=["caries"])
		
    caries_metadata = MetadataCatalog.get("caries_train")
	
    itr = 3000
    lr = 0.0001

    cfg = get_cfg()
    cfg.OUTPUT_DIR = "caries/" 
    cfg.merge_from_file("AdelaiDet/configs/SOLOv2/R50_3x.yaml") 
    cfg.DATASETS.TRAIN = ("caries_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = itr    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
	
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
	
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
	
    eachdata_paths = glob.glob("caries/val/*")
    if (len(eachdata_paths) <= 0): log.warning("eachdata_paths is empty\n")
    t = tqdm(eachdata_paths) # Create progress bar
    t.set_description("loaded val data")
    for each_path in t:
        im = cv2.imread(each_path)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=caries_metadata, 
                       scale=0.5, 
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        filepath, filename = os.path.split(each_path)
        outputs["instances"].pred_boxes = extract_bboxes(outputs["instances"].pred_masks)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow("val",out.get_image()[:, :, ::-1])
        cv2.imwrite("caries/result/"+str(filename ), out.get_image()[:, :, ::-1])
        #cv2.waitKey (0)  

    #evaluator = trainer.build_evaluator(["caries","normal"], cfg, ["caries_val"])
    #evaluator = COCOEvaluator("caries_val")
    #val_loader = build_detection_test_loader(cfg, "caries_val")
    #print(inference_on_dataset(trainer.model, val_loader, evaluator))
    #print(evaluator)