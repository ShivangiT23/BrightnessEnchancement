import os
import cv2
import numpy as np
from modules.YoloClassifier import Yolov4Classifier
import random
import glob
import shutil
import configparser
import xml.etree.ElementTree as ET
from utils.utils import *

if __name__ == "__main__":
        config = configparser.ConfigParser()
        config_filepath="./config/data_generation_config.ini"
        config.read(config_filepath)
        data_gen = "data_gen"
        img_width=int(config[data_gen]["img_width"])
        img_height=int(config[data_gen]["img_height"])
        batch_size=int(config[data_gen]["batch_size"])    
        a = float(config[data_gen]["lower_dim"])
        b = float(config[data_gen]["upper_dim"])
        c = float(config[data_gen]["lower_bright"])
        d = float(config[data_gen]["upper_bright"])
        gap = float(config[data_gen]["gap"])
        n_steps = (b-a)/gap
        n_steps_bright =(d-c)/gap
        list_iter_dim= [int(100*(b-gap*i))/100 for i in range(int(n_steps+1))]
        list_iter_bright= [int(100*(c+gap*i))/100 for i in range(int(n_steps_bright+1))]
        list_final=list_iter_bright+list_iter_dim
        yolo = Yolov4Classifier()
        data_dump=[["fname","gamma","class","gt_len","pred_len"]]

        folder_dump = 'temp'
        if os.path.exists(folder_dump):
            shutil.rmtree(folder_dump)
        os.makedirs(folder_dump)

        folder = config[data_gen]["folder"]
        data_dump=[]
        
        for images in glob.glob(folder+"/**/*.png"):
            row=None
            head,image_name=os.path.split(images)
            file_basename = images[images.find('flip')+5:] if 'flip' in os.path.basename(images) else os.path.basename(images)
            print(image_name)
            img=cv2.imread(images)
            head1,qual=os.path.split(head)
            annotation=os.path.join(head,image_name[:-3]+"xml")
            annot=read_annotations(annotation)
            print(qual)
            y_pred=yolo.transform_predict([[img,annot,None,"good",1]])
            pred = y_pred[0][1]
            if pred == "correct":
                row=[images, 1,"correct",y_pred[0][2],y_pred[0][3],file_basename]
            else:
                row = [images,"none", "wrong",y_pred[0][2],y_pred[0][3],file_basename]
            dump_path = os.path.join('temp',os.path.basename(images))
            cv2.imwrite(dump_path,img)      
            img_copy=cv2.imread(dump_path)
            a = np.sum((img!=img_copy).astype(np.int))
            print("-------------------- a:",a)
            data_dump.append(row)
            print(row)


        import csv
        csv_file=folder = config[data_gen]["csv_file"]
        with open(csv_file,'w', newline='') as file:
            writer = csv.writer(file)
            for line in data_dump:
                writer.writerow(line)
