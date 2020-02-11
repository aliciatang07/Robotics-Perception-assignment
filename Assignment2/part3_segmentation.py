import os
import sys

import cv2
import numpy as np
import numpy.ma as ma
import kitti_dataHandler
from matplotlib import pyplot as plt
import math


# FP:  indicates a predicted object mask had no associated ground truth object mask.
# TP/TP+FP
def cal_preicion(et_seg, gt_seg):
    # TP: number of pixel with the same value 
    # FP: number of pixel et_seg = 0 gt_seg = 255

    tp = np.sum(et_seg == gt_seg)
    fp = np.sum([ et == 0 and gt == 255 for (et,gt) in zip(et_seg,gt_seg)])
    return tp/(tp+fp)

# FN: indicates a ground truth object mask had no associated predicted object mask.
# TP/TP+FN
def cal_recall(et_seg, gt_seg):
    # TP: number of pixel with the same value 
    # FN: number of pixel et_seg = 255 gt_seg = 0

    tp = np.sum(et_seg == gt_seg)
    fn = np.sum([ et == 255 and gt == 0 for (et,gt) in zip(et_seg,gt_seg)])
    return tp/(tp+fn)

def cal_fscore(recall, precision):
    return 2*(precision*recall)/(precision+recall)


def main_train():

    ################
    # Options
    ################

    # Input dir and output dir:train
    depth_dir = 'data/train/est_depth'
    image_dir = 'data/train/left'
    output_dir = 'data/train/est_segmentation'
    bbox_coord_dir = 'data/train/bbox_coord'
    gt_seg_dir = 'data/train/gt_segmentation'
    sample_list = ['000001', '000002', '000003', '000004', '000005','000006',
    '000007','000008','000009', '000010']

    ################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for sample_name in sample_list:
        image = cv2.imread(os.path.join(image_dir,sample_name+".png"))
        image_mask = np.ones(image.shape[:2])*255

        distance_range = 10

        image_path = os.path.join(image_dir, sample_name + ".png")

        # Read depth map
        depth_map = cv2.imread(os.path.join(depth_dir, sample_name + ".png"),cv2.IMREAD_ANYDEPTH)

        gt_seg = cv2.imread(os.path.join(gt_seg_dir, sample_name + ".png"),cv2.IMREAD_ANYDEPTH)

        # Discard depths less than 10cm from the camera
        depth_map[depth_map <0.1] = 0

        file_object = open(os.path.join(bbox_coord_dir,sample_name+".txt"),"r")
        obj = file_object.readlines()
        precision_scores = 0
        recall_scores = 0
        fscores = 0
        count = 0
        for i in range(len(obj)):
            # print("######{} \n".format(i))
            data = obj[i].split(" ")
            # print(data)
            if(len(data)>6):
                # print("goes here")
                continue   

            type_name,x,y,w,h,_ = data
            if(type_name != "car"):
                # print("all depth zero")
                continue

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

         
            # Read 2d bbox
            # print("x {} y {} w {} h {}".format(x,y,w,h))

            # deal with the case that bounding box outside image 
            if(x<0):
                w += x
                x = 0 
            if(y<0):
                h += y
                y = 0 

            bbox_depth = depth_map[int(y):int(y+h),int(x):int(x+w)]
            # print(bbox_depth)

            # Estimate the average depth of the objects
            # avg_depth = np.average(bbox_depth)
            avg_depth = np.nanmean(np.nanmean(np.where(bbox_depth!=0,bbox_depth,np.nan),1))
            if(math.isnan(avg_depth)):
                continue
            # print(avg_depth)
            
            # Find the pixels within a certain distance from the centroid
            mask1 = ma.masked_inside(bbox_depth, avg_depth-distance_range, avg_depth+distance_range)

            # if(mask1.mask == False):
            #     print("no points inside range ")
            #     continue
            int_mask = mask1.mask.astype(int)
            # print(int_mask)
            int_mask[int_mask == 0] = 255
            int_mask[int_mask == 1] = 0
            # print(int_mask)
            # if it's already 0 doesn't override it with 255
            instance_shape = int_mask.shape
            temp = image_mask[int(y):int(y+h),int(x):int(x+w)].flatten()
            int_mask = int_mask.flatten()
            image_mask[int(y):int(y+h),int(x):int(x+w)] = np.array([elem if origin!=0 else origin for origin,elem in zip(temp,int_mask)]).reshape(instance_shape)
            # image_mask[int(y):int(y+h),int(x):int(x+w)] = int_mask
            est =  image_mask[int(y):int(y+h),int(x):int(x+w)].flatten()
            gt = gt_seg[int(y):int(y+h),int(x):int(x+w)].flatten()
            precision = cal_preicion(est, gt)
            recall = cal_recall(est, gt)
            fscore = cal_fscore(recall,precision)
            precision_scores += precision
            recall_scores += recall
            fscores += fscore
            count+= 1

        precision_scores = precision_scores/count
        recall_scores = recall_scores/count
        fscores = fscores/count
        print("image {} segmentation precision:{} and recall:{} and F-score:{}".format(sample_name, precision_scores, recall_scores,fscores))
        # plt.imshow(image_mask)
        cv2.imwrite(os.path.join(output_dir,sample_name + ".png"),image_mask)


def main_test():

    ################
    # Options
    ################
    depth_dir = 'data/test/est_depth'
    image_dir = 'data/test/left'
    output_dir = 'data/test/est_segmentation'
    bbox_coord_dir = 'data/test/bbox_coord'
    sample_list = ['000011', '000012', '000013', '000014', '000015']


    ################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for sample_name in sample_list:
        image = cv2.imread(os.path.join(image_dir,sample_name+".png"))
        image_mask = np.ones(image.shape[:2])*255

        distance_range = 10

        image_path = os.path.join(image_dir, sample_name + ".png")

        # Read depth map
        depth_map = cv2.imread(os.path.join(depth_dir, sample_name + ".png"),cv2.IMREAD_ANYDEPTH)


        # Discard depths less than 10cm from the camera
        depth_map[depth_map <0.1] = 0

        file_object = open(os.path.join(bbox_coord_dir,sample_name+".txt"),"r")
        obj = file_object.readlines()
        for i in range(len(obj)):
            # print("######{} \n".format(i))
            data = obj[i].split(" ")
            # print(data)
            if(len(data)>6):
                # print("goes here")
                continue   

            type_name,x,y,w,h,_ = data
            if(type_name != "car"):
                # print("all depth zero")
                continue

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

         
            # Read 2d bbox
            # print("x {} y {} w {} h {}".format(x,y,w,h))

            # deal with the case that bounding box outside image 
            if(x<0):
                w += x
                x = 0 
            if(y<0):
                h += y
                y = 0 

            bbox_depth = depth_map[int(y):int(y+h),int(x):int(x+w)]
            # print(bbox_depth)

            # Estimate the average depth of the objects
            # avg_depth = np.average(bbox_depth)
            avg_depth = np.nanmean(np.nanmean(np.where(bbox_depth!=0,bbox_depth,np.nan),1))
            if(math.isnan(avg_depth)):
                continue
            # print(avg_depth)

            # Find the pixels within a certain distance from the centroid
            mask1 = ma.masked_inside(bbox_depth, avg_depth-distance_range, avg_depth+distance_range)

            int_mask = mask1.mask.astype(int)
            # print(int_mask)
            int_mask[int_mask == 0] = 255
            int_mask[int_mask == 1] = 0
            # print(int_mask)
            # if it's already 0 doesn't override it with 255
            instance_shape = int_mask.shape
            temp = image_mask[int(y):int(y+h),int(x):int(x+w)].flatten()
            int_mask = int_mask.flatten()
            image_mask[int(y):int(y+h),int(x):int(x+w)] = np.array([elem if origin!=0 else origin for origin,elem in zip(temp,int_mask)]).reshape(instance_shape)
            est =  image_mask[int(y):int(y+h),int(x):int(x+w)].flatten()
           
        cv2.imwrite(os.path.join(output_dir,sample_name + ".png"),image_mask)
     


if __name__ == '__main__':
    if(sys.argv[1] == "train"):
        print("#####TRAINING MODE")
        main_train()
    if(sys.argv[1] == "test"):
        print("#####TESTING MODE")
        main_test()

