#!/usr/bin/env python
# encoding: utf-8
from skimage import feature
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil

import logging
# log的部分
logging.basicConfig(level='INFO',
                    format='[%(asctime)s %(levelname)-8s %(module)-12s:Line %(lineno)-3d] %(message)s ',
                    datefmt='[%H:%M:%S]')

log = logging.getLogger('datahandle.py')

def increase_brightness(image, saturation, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    
    lim = 255 - saturation
    s[s > lim] = 255
    s[s <= lim] += saturation
    
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def convert_bool_uint8(image, n=0):
    #將image值map到0和255並轉成uint8
    image = image > n #mix all type caries to boolean
    image = image *255 #convert boolean to 0/255
    image = image.astype(np.uint8)
    return image

def get_teeth_width(t_mask, px, py):
    log.info("running function \"get_teeth_width\" \n")
    t_w = [] #teeth width
    for loop in range(t_mask.shape[2]):
        row = 0
        start = 0
        end = 0
        w_cdd = [] #width candidate
        while row < int(py*0.75): #減少搜到牙根部分，雙牙根會影響大小
            for i in range(px - 1):
                if t_mask[row,i+1,loop] == 255 and t_mask[row,i,loop] == 0:
                    start = i
                if t_mask[row,i+1,loop] == 0 and t_mask[row,i,loop] == 255:
                    end = i
                    break
            if start < end:
                w_cdd.append(end-start)
            row = row + int(py/15)
        if len(w_cdd) >= 5:
            w_cdd.pop()
            w_cdd.pop()
            w_cdd.pop(0)
            w_cdd.pop(0)
        elif len(w_cdd) >= 3:
            #去頭去尾解決outlier
            w_cdd.pop()
            w_cdd.pop(0)
        #乘以1.1彌補牙齒上寬下窄
        t_w.append(int(np.mean(w_cdd)*1.1))
    return t_w

def get_teeth_avg(t_mask, t_w):
    #取框平均值，其中若牙齒多餘2則去掉第一和最後一顆再決定框的平均
    #因爲多顆牙齒的時候，牙齒很可能在邊緣，也就是牙齒只有部分
    twidth = 0
    if (t_mask.shape[2]) == 1:
        twidth = t_w[0]
    elif (t_mask.shape[2]) == 2:
        twidth = sum(i for i in t_w)/2
    else:
        twidth = sum(t_w[1:(t_mask.shape[2])-1])/((t_mask.shape[2])-2)
    return int(twidth)

def only_teeth(img, t_mask, px, py, twidth, mcolor=0):
    blackmask = 255 * np.ones((py,px,3), np.uint8)
    whitemask = 255 * np.ones((py,px,3), np.uint8)
    kernel = np.ones((int(twidth*0.00),int(twidth*0.00)),dtype=np.uint8)
    for loop in range(t_mask.shape[2]):
        blackmask[(cv2.dilate(t_mask[:,:,loop],kernel))>0] = 0#create blackmask
	
    img[blackmask>0] = mcolor #add blackmask on ori
    return img

def get_edges(t_mask, twidth, reducenum=0.12):
    #利用縮小後的edge取框使得框在裏面一點
    edges = []
    reduce = np.ones((int(twidth*reducenum),int(twidth*reducenum)),dtype=np.uint8)
    for loop in range(t_mask.shape[2]):
        edge = cv2.erode(t_mask[:,:,loop],reduce)
        edge = feature.canny(edge, sigma=3)
        edge[edge>0] = 1
        edges.append(edge)
    return edges

def intersection_over_union(frame1, frame2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(frame1[1], frame2[1]) #求兩個窗口左上角x座標最大值 
    yA = max(frame1[0], frame2[0]) #求兩個窗口左上角y座標最大值 
    xB = min(frame1[3], frame2[3]) #求兩個窗口右下角x座標最小值 
    yB = min(frame1[2], frame2[2]) #兩個窗口右下角y座標最小值 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    frame1Area = (frame1[3] - frame1[1] +1 ) * (frame1[2] - frame1[0] +1)
    frame2Area = (frame2[3] - frame2[1] +1 ) * (frame2[2] - frame2[0] +1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(frame1Area + frame2Area - interArea)
    #iou = interArea / float(frame1Area)
    # return the intersection over union value
    return iou
    
def get_croplists(gums, twidth, edges, px, py, overlapsize=0.2, \
                    boost=15, cut_y=0.8, cut_x_s=0.1, cut_x_e=0.9):
    log.info("running function \"get_croplists\" \n")
    #cut_x_s 控制x軸從多少比例開始，預設值0.1
    #cut_x_e 控制x軸到多少比例結束，預設值0.9
    #cut_y 控制y軸到多少比例結束，預設值1
    croplists = []#儲存圖片crop的位置
    #for size in range(3,6): 我傾向與只用一個大小
    #計算牙齒的框的大小
    size = 4
    bordersize = int(twidth*size*0.05) #0.15 0.2 0.25
    #控制不讓bordersize太大/小
    if bordersize < (py/50):#40
        bordersize = int(py/50)
    elif bordersize > (py/5):#40
        bordersize = int(py/5)
    boostj = int(bordersize/boost)
    boosti = int(bordersize/boost)
    overlap = int(bordersize*overlapsize) #決定框之間可以overlap多少(越小越多)
    fault = int(bordersize/4)#accept how many gums in frame
    #進行框選
    for loop in range(len(edges)):#edges中包含每顆牙齒的edge
        cnt = 0
        i = 0
        croplist = [] #儲存某顆牙齒的crop的位置
        #gums 的threshold,包含的gums太多就會捨棄
        croplist.append((-100,-100,-50,-50))
        while i < (int(py*cut_y)):#太下面的都是牙根可以不要，加速loop
            j = int(px * cut_x_s) #前面cut_x_s白色去掉
            while j < ( int(px * cut_x_e)): #後面cut_x_s白色去掉
                if (edges[loop][i, j] == True) :
                    if ((i-bordersize) >= 0) and ((j-bordersize) >= 0) and ((i+bordersize) < int(py*cut_y)) and ((j+bordersize) < int(px)): #check boundaries
                        for c in croplist:
                            if (i+overlap <= c[0] or i-overlap >= c[2]) or (j+overlap <= c[1] or j-overlap >= c[3]): #check overlap
                                cnt = cnt + 1
                        if cnt == len(croplist):
                            gumscheck = gums[(i-bordersize):(i+bordersize-fault*2), (j-bordersize):(j+bordersize)]#框的正下方25%以上有一點gums就不要
                            if np.count_nonzero(gumscheck) == 0:
                                croplist.append((i-bordersize,j-bordersize,i+bordersize,j+bordersize))
                                if (j + boostj) < (int(px*cut_x_e)):
                                    j = j + 1 + boostj #boost the loop
                            cnt = 0
                        else:
                            cnt = 0
                j = j + 1
            if (i+ boosti) < (int(py*cut_y)):
                i = i + 1+ boosti #boost the loop
            else:
                i = i + 1
        del croplist[0]
        croplists.append(croplist)
    return bordersize,croplists

def crop_by_caries(caries,bordersize,px,py): #專門對caries進行crop
    log.info("running function \"crop_by_caries\" \n")
    
    caries_crop = []
    boostj = int(bordersize/15)
    boosti = int(bordersize/15)
    overlap = int(bordersize*0.2) #決定框之間可以overlap多少(越小越多)
    cnt = 0
    i = 0
    caries_crop.append((-100,-100,-50,-50))
    while i < py:
        j = 0
        while j < px:
            if (caries[i, j] != 0) :
                if ((i-bordersize) >= 0) and ((j-bordersize) >= 0) and ((i+bordersize) < py) and ((j+bordersize) < px): #check boundaries
                    for c in caries_crop:
                        if (i+overlap <= c[0] or i-overlap >= c[2]) or (j+overlap <= c[1] or j-overlap >= c[3]): #check overlap
                            cnt = cnt + 1
                    if cnt == len(caries_crop):
                        caries_crop.append((i-bordersize,j-bordersize,i+bordersize,j+bordersize))
                        if (j + boostj) < px:
                            j = j + 1 + boostj #boost the loop
                        cnt = 0
                    else:
                        cnt = 0
            j = j + 1
        if (i+ boosti) < py:
            i = i + 1+ boosti #boost the loop
        else:
            i = i + 1
    del caries_crop[0]
    return caries_crop
	
def add_edges(edges,oimg):
    log.info("running function \"add_edges\" \n")
   
    #產生所有牙齒的edge的圖
    t_edge = np.zeros((py,px),np.uint8)
    for loop in edges:
        t_edge = np.logical_or(t_edge,loop)
    t_edge = np.logical_not(t_edge)
    #將edge的部分的值變成0
    oimg[:, :, 0] = oimg[:, :, 0] * t_edge
    oimg[:, :, 1] = oimg[:, :, 1] * t_edge
    oimg[:, :, 2] = oimg[:, :, 2] * t_edge

    return oimg
	
def  find_side(edges,oimg):
    log.info("running function \"find_side\" \n")
	
    allhline = []
    side = np.zeros((oimg.shape[0],oimg.shape[1]),dtype=np.uint8)
    i = 0
    for edge in edges:
        i = i+1
        hline = np.zeros((oimg.shape[0],oimg.shape[1],3),dtype=np.uint8)
        lines = cv2.HoughLines(convert_bool_uint8(edge), 1, np.pi/180, 8)
        for sub_lines in lines:
            for line in sub_lines:
                rho = line[0]
                theta= line[1]
                #rho是從(0,0)到直綫的距離，theta從y負數軸開始逆時針
                #https://blog.csdn.net/yl_best/article/details/88744997
                if  (theta > (np.pi*0.3)) and (theta < (np.pi*0.7)):  #54度-126度
                    cos_theta = np.cos(theta); sin_theta = np.sin(theta)
                    x = rho * cos_theta; y = rho * sin_theta
                    x1 = int(x + 1500 * (-sin_theta))
                    y1 = int(y + 1500 * (cos_theta))
                    x2 = int(x - 1500 * (-sin_theta))
                    y2 = int(y - 1500 * (cos_theta))
                    cv2.line(hline, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
        hline = cv2.cvtColor(hline,cv2.COLOR_BGR2GRAY)
        hline = hline.astype(np.bool)
                
        hline = np.logical_and(hline,edge)
        hline = np.logical_xor(hline,edge)
        allhline.append(hline)
		
        side = np.logical_or(hline,side)
        img = convert_bool_uint8(side)
        cv2.imwrite(outdir_swoverview + filename + "_side.jpg", img)

    return side, allhline

def crop_image(img,oimg,side,caries,samplew,croplists,bordersize,outdir,predict=0):
    #log.info("running function \"crop_image\" \n")
    #fault會使得我們注意框的中間的caries數量，如果太少就認爲這個框是normal
    fault = int(bordersize/4)#accept how many caries in frame
    fcheck = int(bordersize*0.8)#accept how many houghline in frame
    predict_list = []
	
    cnts = []
    cnts = cv2.findContours(caries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	
    #將圖crop出來
    for loop in range(len(croplists)):
        croplist = croplists[loop]
        for c in croplist:
            sidecheck = side[(c[0]+fcheck):(c[2]-fcheck), (c[1]+fcheck):(c[3]-fcheck)]
            if np.count_nonzero(sidecheck) > 0:
                isside = 1
            else:
                isside = 0.5

            cariescheck = caries[(c[0]+fault):(c[2]-fault), (c[1]+fault):(c[3]-fault)]
            crop = img[c[0]:c[2], c[1]:c[3],:]
            '''
            cropemptycheck = crop > 50 #check crop empty or not
            如果大於50...那剛好框到全部caries的就會被去掉了....
            cropemptycheck = cropemptycheck *255 # 0 means black
            '''
            cropemptycheck = convert_bool_uint8(crop,0)
            deciderbool,samplew = caries_decider(c,caries,samplew,cnts,fault,isside)
            #if 80%black then byebye
            if np.count_nonzero(cropemptycheck) > ((c[3]-c[1])*(c[2]-c[0]))*0.2:
                if deciderbool > 0 :
                    if predict == 0:
                        cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(255,0,0),1)
                        #cv2.rectangle(oimg,(c[1]+fault,c[0]+fault),(c[3]-fault,c[2]-fault),(255,255,0),1) #感覺會讓圖片很亂，先去掉
                        cv2.imwrite(outdir + r"/00001/" + str(c) + ".jpg", crop)
                    else:
                        #append (label,location,image)
                        predict_list.append((1,c,crop))
                else:
                    if predict == 0:
                        cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(0,255,0),1)
                        cv2.imwrite(outdir + r"/00000/" + str(c) + ".jpg", crop)
                    else:
                        #append (label,location,image)
                        predict_list.append((0,c,crop))
    return oimg,samplew,predict_list

def crop_image_conly(img,oimg,side,caries,samplew,croplists,bordersize,outdir,predict=0):
    log.info("running function \"crop_image\" \n")
    #fault會使得我們注意框的中間的caries數量，如果太少就認爲這個框是normal
    fault = int(bordersize/4)#accept how many caries in frame
    fcheck = int(bordersize*0.8)#accept how many houghline in frame
    predict_list = []
	
    cnts = []
    cnts = cv2.findContours(caries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#caries輪廓
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	
    #將圖crop出來
    for loop in range(len(croplists)):
        croplist = croplists[loop]
        for c in croplist:
            sidecheck = side[(c[0]+fcheck):(c[2]-fcheck), (c[1]+fcheck):(c[3]-fcheck)]
            if np.count_nonzero(sidecheck) > 0:
                isside = 1
            else:
                isside = 0.5

            cariescheck = caries[(c[0]+fault):(c[2]-fault), (c[1]+fault):(c[3]-fault)]
            crop = img[c[0]:c[2], c[1]:c[3],:]
            '''
            cropemptycheck = crop > 50 #check crop empty or not
            如果大於50...那剛好框到全部caries的就會被去掉了....
            cropemptycheck = cropemptycheck *255 # 0 means black
            '''
            cropemptycheck = convert_bool_uint8(crop,0)
            deciderbool,samplew = caries_decider(c,caries,samplew,cnts,fault,isside)
            #if 80%black then byebye
            if np.count_nonzero(cropemptycheck) > ((c[3]-c[1])*(c[2]-c[0]))*0.2:
                if deciderbool > 0 :
                    if predict == 0:
                        cv2.rectangle(oimg,(c[1],c[0]),(c[3],c[2]),(255,0,0),1)
                        cv2.imwrite(outdir + r"/00001/" + str(c) + ".jpg", crop)
                else:
                    if predict == 0:
                        del samplew[-1]
    return oimg,samplew,predict_list
	
def caries_decider(c,caries,samplew,cnts,fault,isside):
    log.info("running function \"caries_decider\" \n")
    
    if(np.count_nonzero(caries[(c[0]):(c[2]), (c[1]):(c[3])])==0):
        decider = 0
        samplew.append((c, 1*isside))#normal  sw 
        return decider,samplew
    else:
        selectedcariesArea = np.count_nonzero(caries[(c[0]):(c[2]), (c[1]):(c[3])]) #框到的蛀牙面積
        selectedArea = ((c[0])-(c[2]))*((c[1])-(c[3])) #框的面積
        
        size = []
        temp = []
        for cc in cnts:
            mask = np.zeros(caries.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [cc], [255,255,255])
            check = mask[(c[0]):(c[2]), (c[1]):(c[3])]
            
            ccArea = np.count_nonzero(check) #框到某塊蛀牙區域的蛀牙面積
            cariesArea = np.count_nonzero(mask) #該蛀牙區域的面積
            
            # (selectedcariesArea/selectedArea) = 框內蛀牙面積，(ccArea/cariesArea) = 蛀牙在蛀牙區域的面積
            if (((selectedcariesArea/selectedArea)>=0.4) and ((ccArea/cariesArea)>=0.4) and (cariesArea!=0)): 
                temp.append((c, 1.5*isside))#caries  sw
                size.append(5)
            elif (((selectedcariesArea/selectedArea)>=0.05) and ((ccArea/cariesArea)>=0.40) and (cariesArea!=0)):
                temp.append((c, 1*isside))#caries  sw
                size.append(4)
            elif (((selectedcariesArea/selectedArea)>=0.40) and ((ccArea/cariesArea)>=0.05) and (cariesArea!=0)):
                temp.append((c, 0.6*isside))#caries  sw
                size.append(3)
            elif (((selectedcariesArea/selectedArea)>=0.05) and ((ccArea/cariesArea)>=0.05) and (cariesArea!=0)):
                temp.append((c, 0.1*isside))#caries  sw
                size.append(2)
            elif (((selectedcariesArea/selectedArea)>=0.01) and ((ccArea/cariesArea)>=0.01) and (cariesArea!=0)):
                temp.append((c, 0.05*isside))#caries  sw
                size.append(1)
            else:
                temp.append((c, 0.95*isside))#normal  sw 
                size.append(0)
				
        maxtemp = max(size)
        maxlocat = size.index(maxtemp)
        if maxtemp == 0: 
            decider = 0 
        else: 
            decider = 1 
        samplew.append(temp[maxlocat])
    return decider,samplew

if __name__ == '__main__':

    outdir_data = r"../data"
    outdir_cv = r"../data/cv"
    for directory in [outdir_data, outdir_cv]:
        maked = False
        while not os.path.exists(directory):
            if not maked:
                log.warning("%s does not exist, creating...\n" % directory)
                os.mkdir(directory); maked = True
    outdir_overview = r"../data/overview"
    outdir_swoverview = r"../data/swoverview"
    Exist = True
    while os.path.exists(outdir_overview):
        os.rename(outdir_overview, outdir_overview + r"_delete")
        os.rename(outdir_swoverview, outdir_swoverview + r"_delete")
        if Exist:
            log.warning("%s exist, will remove then create it...\n" % outdir_overview)
            shutil.rmtree(outdir_overview + r"_delete"); Exist = False
            shutil.rmtree(outdir_swoverview + r"_delete"); Exist = False
    log.info("creating %s \n" % outdir_overview)
    os.mkdir(outdir_overview)
    os.mkdir(outdir_swoverview)
    outdir_overview = outdir_overview + "/"
    outdir_swoverview = outdir_swoverview + "/"
	
    with open( "data.json") as f:
        tm_text = json.load(f)
		
    skip = 0

    for key,value in tqdm(tm_text.items()):
        if skip :
            skip = 0
            continue
			
        record = {}
        raw_data = tm_text[key]
		
        if(raw_data['filename'] == None):
            continue
        filename = raw_data['filename']
		
        oimg = io.imread(filename)
        height, width = oimg.shape[:2]
		
        url,filename = os.path.split(str(filename))
        filename, suffix = os.path.splitext(filename)
		
        outdir = r"../data/cv/" + filename
        Exist = True
        while os.path.exists(outdir):
            os.rename(outdir, outdir + r"_delete")
            if Exist:
                log.warning("%s exist, removing...\n" % outdir)
                shutil.rmtree(outdir + r"_delete"); Exist = False
        log.info("creating %s \n" % outdir)
        os.mkdir(outdir)
        os.mkdir(outdir + r"/00000"); os.mkdir(outdir + r"/00001"); os.mkdir(outdir + r"/sw")

        if int(filename) >= numdata: 
            break
        elif int(filename) == 267 or int(filename) == 38:
            skip = 1#268 bug
		
        record["image_id"] = key
        record["height"] = height
        record["width"] = width
		
        maxx = 0
        minx = width
        teethmask = []
        caries = Image.new("L", [width,height], 0)
        gums = np.zeros((height,width),np.uint8)
			
        for anno in raw_data['regions']:
            if(anno == None):
                continue
            region = anno['region_attributes']
			
            teeth = Image.new("L", [width,height], 0)
            if region['type'] == 'tooth':
                xs = anno['shape_attributes']['all_points_x']
                ys = anno['shape_attributes']['all_points_y']
                if(len(xs) < 3):
                    continue
                maxx = max(max(xs),maxx)
                minx = min(min(xs),minx)
                combined = list(map(tuple, np.vstack((xs, ys)).T))
                ImageDraw.Draw(teeth).polygon(combined, outline=1, fill=1)
                teeth = np.array(teeth)
                teethmask.append(teeth)
                showmask = showmask + teeth

            temp = Image.new("L", [width, height,], 0)
            if 'detail' in region and region['detail'] == 'caries':
                xs = anno['shape_attributes']['all_points_x']
                ys = anno['shape_attributes']['all_points_y']
                combined = list(map(tuple, np.vstack((xs, ys)).T))

                ImageDraw.Draw(temp).polygon(combined, outline=1, fill=1)
                temp = np.array(temp)
                temp = convert_bool_uint8(temp)
                caries = caries + temp

				
        teethmask = np.transpose((convert_bool_uint8(np.array(teethmask))), (1, 2, 0))
	
        teethwidth = int((maxx-minx)/(teethmask.shape[2]))#計算每顆牙齒的寬度
        #t_w = get_teeth_width(t_mask,px,py) #計算每顆牙齒的寬度
        #twidth = get_teeth_avg(t_mask,t_w) #取得平均牙齒的大小
			
        oimg = only_teeth(oimg,teethmask,width,height,teethwidth)#除了teeth以外的部分都變成黑色
        img = oimg.copy()
		
        edges = get_edges(teethmask,teethwidth) #利用縮小後的edge取框使得框在裏面一點
        bordersize, croplists = get_croplists(gums,teethwidth,edges,width,height) #獲取crop的位置(y/row,x/col,y/row+h,x/col+w)
		
        ########can add more augmentation here -------------------------------------#######
        twidth1 = teethwidth*1.3
        bordersize1, croplists1 = get_croplists(gums,twidth1,edges,width,height,0.2,15) #獲取crop的位置
        twidth2 = teethwidth*0.7
        bordersize2, croplists2 = get_croplists(gums,twidth2,edges,width,height,0.1,30) #獲取crop的位置
	
        side,hline = find_side(edges,oimg)
			
        caries = np.array(caries)
        caries = convert_bool_uint8(caries) #將caries值map到0和255並轉成uint8
		
        #將caries部分上紅色,這部分先完成避免框的顔色改變
        bimg = overlay[:,:,0]
        gimg = overlay[:,:,1]
        rimg = overlay[:,:,2]
        bimg[caries == 255] = 0
        gimg[caries == 255] = 0
        rimg[caries == 255] = 255
        overlay[:,:,0] = bimg
        overlay[:,:,1] = gimg
        overlay[:,:,2] = rimg
        oimg = oimg - (overlayo*0.4).astype(np.uint8) + (overlay *0.4).astype(np.uint8)
        oimg = add_edges(edges,oimg)
        cv2.imwrite(outdir_overview + filename + "_O.jpg", oimg)
        swmap = oimg.copy()
		
        samplew = []    
        oimg,samplew,predict_list = crop_image(img,oimg,side,caries,samplew,croplists,bordersize,outdir)
        oimg,samplew,predict_list = crop_image(img,oimg,side,caries,samplew,croplists1,bordersize1,outdir)
        oimg,samplew,predict_list = crop_image(img,oimg,side,caries,samplew,croplists2,bordersize2,outdir)
			
        oimg,samplew,predict_list = crop_image_conly(img,oimg,side,caries,samplew,croplists3,bordersize3,outdir)
			
        #remake without repeat element
        sampleweight=[]
        for i in samplew:
            if i not in sampleweight:
                sampleweight.append(i)

        #for show sw
        mask = np.zeros((oimg.shape[0],oimg.shape[1],3),dtype=np.uint8)
        maskbw = np.zeros((oimg.shape[0],oimg.shape[1],3),dtype=np.uint8)
        for each in sampleweight: 
            c,sw = each[0],each[1]
            cv2.rectangle(mask,(c[1],c[0]),(c[3],c[2]),((170*sw),(170*sw),(170*sw)),2)#Cause got sw 1.5 so 170*1.5 = 255
            cv2.rectangle(maskbw,(c[1],c[0]),(c[3],c[2]),(255,255,255),2)
        mask = cv2.applyColorMap((cv2.bitwise_not(cv2.cvtColor(mask , cv2.COLOR_BGR2GRAY))), cv2.COLORMAP_RAINBOW)
        np.putmask(swmap, maskbw > 0, mask)
        cv2.imwrite(outdir_swoverview + filename + "_SW.jpg", swmap)
			
			
        #save samplew as  txt
        c = open(outdir + r"/sw/c.txt", "w")
        sw = open(outdir + r"/sw/sw.txt", "w")
        c.write("c \n")#pd.read_csv read without header(first line) so need to add something 
        sw.write("sw \n")
        for row in sampleweight:
            c.write(str(row[0]) + "\n")
            sw.write(str(row[1]) + "\n")
        c.close()
        sw.close()