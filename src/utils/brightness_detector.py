import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import configparser
import numpy as np
from utils.brightness_iou_calculation import bb_intersection_over_union
import sys
sys.path.insert(1, '../model_files/darknet')

import darknet


class Darknetv4:
    #self.network = None
    #self.class_name = None
    #self.class_colors = None

    def __init__(self, config,data,weights):
        self.network, self.class_names,self.class_colors = darknet.load_network(config,data,weights,1)
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)

    def image_detection(self,image_path, thresh):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=thresh)
        #darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, self.class_colors)

        #darknet.print_detections(detections)
        return detections
    
    def bbox2points(self,bbox, img_h, img_w):
        """
        From bounding box yolo format to corner points cv2 rectangle
        """
        x, y, w, h = bbox 
        # Resized the coordinates based on image shape #(self.width, self.height)
        x,y,w,h = (x*img_w)/self.width, (y*img_h)/self.height, (w*img_w)/self.width, (h*img_h)/self.height
        xmin = int(round(x - (w / 2)) )
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))    
        return xmin, ymin, xmax, ymax

    def convert_bboxesFormat(self,detections,image,conf_thresh):
        pred_bboxes = []
        img_h, img_w = image.shape[:2]
        for label, confidence, bbox in detections:
            left, top, right, bottom = self.bbox2points(bbox, img_h, img_w)
            height = bottom-top
            width = right-left
            if(abs(left)>0 and abs(top)>0 and abs(width) < img_w and abs(height) < img_h\
                and (abs(height) > 10 and abs(width) > 10)):
                pred_bboxes.append([left, top, right, bottom, float(confidence), label])                       
        return pred_bboxes

    def cross_class_nms(self, bboxes,nms_thresh):    
        sup_nms_index =[]
        bboxes.sort(key=lambda x: x[4],reverse=True)    
        for i,bbi in enumerate(bboxes):    
            for _,bbj in enumerate(bboxes[i+1:]):
                #print(bbi,bbj,bb_intersection_over_union(bbi,bbj))
                if(bb_intersection_over_union(bbi,bbj) > nms_thresh):
                    sup_nms_index.append(bboxes.index(bbj))
        filtered_pred_output = [bboxes[i] for i in range(len(bboxes)) if i not in sup_nms_index]
        return filtered_pred_output
        
    def image_detection_img(self,img, conf_thresh=0.7, nms_thresh=0.45,img_name=None):

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),  interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=conf_thresh, hier_thresh=.5, nms=nms_thresh )
        if img_name:
            image_pred = image_resized#darknet.draw_boxes(detections, image_resized, self.class_colors)
            cv2.imwrite(img_name,image_pred)
        detections = self.convert_bboxesFormat(detections,image_rgb,conf_thresh)
        detections = self.cross_class_nms(detections,nms_thresh)
        annot_image = img.copy()
        for box in detections:
            left, top, right, bottom, conf, label = box
            cv2.rectangle(annot_image,(left,top),(right,bottom),(95,17,224),8)
        if img_name != None:
            cv2.imwrite('/comads_work/data/images/Output/'+img_name,annot_image)
        # darknet.free_image(self.darknet_image)
        # darknet.print_detections(detections)
        return detections

# yolo_version = 'yoloV4'
    
# config1 = configparser.ConfigParser()
# config1.read('config.ini')

# config = config1[yolo_version]['yolo_cfg']
# weights = config1[yolo_version]['yolo_weights']
# data = config1[yolo_version]['yolo_data']

# with open(r'img_list.txt','r') as f:
#     img_list = f.read().splitlines()

# # config = './cfg/yolov4.cfg'
# # data='./cfg/coco.data'
# # weights = './yolov4.weights'
# obj = Darknetv4(config,data,weights)

# t1 = time.time()
# for i in img_list:
#     t2 = time.time()
#     # detections = obj.image_detection('./dog.jpg',0.7)
#     detections = obj.image_detection(i,0.7)
#     print("Time Taken for {}: {} secs".format(os.path.basename(i),time.time()-t2))

# print(time.time()-t1,' seconds for 10 images')
# print(detections)
