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


        ## Added for too dim and too bright
        e=0.01
        f= 0.3
        g=2.5
        h=4
        gap_ = 0.05
        n_steps = (f-e)/gap_
        n_steps_bright =(h-g)/gap_
        list_iter_too_dim = [int(100*(f-gap_*i))/100 for i in range(int(n_steps+1))]
        list_iter_too_bright = [int(100*(g+gap_*i))/100 for i in range(int(n_steps_bright+1))]
        # print(list_iter_too_dim)
        # print(list_iter_too_bright)
        list_iter_bright.extend(list_iter_too_bright)
        list_iter_dim.extend(list_iter_too_bright)
        
        def apply_gamma():
            prob = np.random.uniform(0,1)
            if (prob<=0.6):
                return None,"good"
            elif (prob>0.6 and prob<=0.78):    
                return list_iter_dim, "gamma_dim"
            elif(prob>0.78 and prob<=0.92):
                return list_iter_bright, "gamma_bright"
            elif(prob>0.92 and prob<=0.96):
                return list_iter_too_dim, "alert_dim"
            else:
                return list_iter_too_bright, "alert_bright"

        # folder_dump = '/comads_work/data/new_images'
        folder_dump = '/comads_work/data/synthetic_augmented_2'
        # if os.path.exists(folder_dump):
        #     shutil.rmtree(folder_dump)
        # os.makedirs(folder_dump)
        yolo = Yolov4Classifier()
        data_dump=[]
        
        folder = config[data_gen]["folder"]
        data_dump=[["fname","gamma","class","gt_len","pred_len"]]
        all_images = glob.glob(folder+"/**/*.jpg",recursive=True)
        # all_images = glob.glob(folder+"/*.png")
        for images in all_images:
            print(images)
            row=None
            head,image_name=os.path.split(images)
            
            img=cv2.imread(images)
            head1,qual=os.path.split(head)
            annotation=os.path.join(head,image_name[:-3]+"xml")
            annot=read_annotations(annotation)
            # print(qual)
            
            
            y_pred=yolo.transform_predict([[img,annot,None,"good",1]])
            pred = y_pred[0][1]
            if pred == "correct":
                dir= os.path.join(folder_dump,'good')
                # cv2.imwrite(os.path.join(dir,os.path.basename(images)[:-3]+"png"),img)
                with open('/comads_work/data/add_im.txt','a') as f:
                    f.write(os.path.join(dir,os.path.basename(images)[:-3]+"png"))
                    # shutil.copy(annotation,dir)
                    f.write("\n")

        #         list_final,label = apply_gamma()
        #         print("applying - label - ",label)
                
        #         if(list_final):
        #             #print("in qual=dim")
        #             for gamma in list_final:
        #                 # gamma_corrected = np.array(255*(img / 255) **(gamma), dtype = 'uint8')
        #                 # if('alert_bright' in label):
        #                 print("trying...",gamma)
        #                 gamma_corrected = gamma_correction(img,gamma)
        #                 ##yolo code
        #                 y_pred=yolo.transform_predict([[gamma_corrected,annot,None,"good",1.0]])
        #                 pred = y_pred[0][1]
                        
        #                 if pred == "wrong":
        #                     print(pred)
        #                     print("gamma: ",gamma)
        #                     dir= os.path.join(folder_dump,label)
        #                     if not os.path.exists(dir):
        #                         os.makedirs(dir)
        #                     cv2.imwrite(os.path.join(dir,os.path.basename(images)[:-3]+"png"),gamma_corrected)
        #                     shutil.copy(annotation,dir)
        #                     row=[images, gamma,label,y_pred[0][2],y_pred[0][3]]                       
        #                     break   
        #         else:
        #             gamma = 1
        #             dir= os.path.join(folder_dump,label)
        #             if not os.path.exists(dir):
        #                 os.makedirs(dir)
        #             cv2.imwrite(os.path.join(dir,os.path.basename(images)[:-3]+"png"),img)
        #             shutil.copy(annotation,dir)
        #             row = [images, gamma,label,y_pred[0][2],y_pred[0][3]] 
        #     if row:
        #         data_dump.append(row)
        #     # print(row)

        # import csv
        # csv_file=folder = config[data_gen]["csv_file"]
        # with open(csv_file,'w', newline='') as file:
        #     writer = csv.writer(file)
        #     for line in data_dump:
        #         writer.writerow(line)