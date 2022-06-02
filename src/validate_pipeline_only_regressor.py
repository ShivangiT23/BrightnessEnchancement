import cv2
import os
import numpy as np
import glob
import sklearn 
from sklearn.pipeline import Pipeline, make_pipeline
from utils.brightness_detector import Darknetv4
import csv
import joblib
import json
from modules.SVM_Classifier import SvmClassifier
from RF_Classifier import RFClassifier
from modules.GammaCorrector import FixedGammaCorrection,VariableGammaCorrection
from modules.YoloClassifier import Yolov4Classifier
from utils.utils import *

def main():
    with open("./config/validation_config.json", "r") as jsonfile:
        config = json.load(jsonfile)

    obj = joblib.load(config['classifier_model_path'])
    pipe=Pipeline(steps=[('regressor',obj),('pillcount',Yolov4Classifier())]) 
    yolov4only = Yolov4Classifier()
    # file1=open(config['validation_txt'],'r')
    # lines=file1.read().splitlines() 
    lines = glob.glob('/comads_work/data/new_images/**/*.png')
    content = [x.strip() for x in lines]
    data_dump=[]
    header = ["image_path","yolo without brightness","brightness prediction","yolo after brightness","gamma_pred"]
    data_dump.append(header)
    for line in content:
        if 'fname' in line:
            continue
        image=line
        print(line)
        img = cv2.imread(image)
        head, image_name=os.path.split(image)
        head1, quality = os.path.split(head)
        annot_name = image[:-3]+"xml"
        annot = read_annotations(annot_name)
        pooled_image = get_maxpooled(img)
        xtr=[img,annot,pooled_image]
        if "good" in quality or 'okay' in quality:
                type_qual="correct"
                ytr=[quality,"correct"]
        else:
                type_qual="wrong"
                ytr=[quality,"wrong"]
        y_pred=pipe.predict([xtr])
        reg_pred=obj.transform([xtr])
        # print(len(reg_pred[0]))
        # # print(os.path.basename(line))
        # # y_pred_class=yolov4only.transform_predict([[reg_pred[0][0],annot,image_name,reg_pred[0][3],'1']],None,image_name)
        y_pred_class=yolov4only.transform_predict([[img,annot,image_name,'good','1']],None,image_name)
        
        y_pred_wo_class=yolov4only.transform_predict([[img,annot,image_name,'good','1']],None,None)
        # print(y_pred[0][-1])
        row=[image_name,y_pred_wo_class[0][1], y_pred[0][0],y_pred[0][1],y_pred[0][-1]]
        data_dump.append(row)

    # with open(config['output_csv_path'],'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for line in data_dump:
    #         writer.writerow(line)


if __name__ == "__main__":
    main()
