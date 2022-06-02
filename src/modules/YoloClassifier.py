from utils.brightness_detector import Darknetv4
from utils.brightness_iou_calculation import one_image_IOU
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
import cv2
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('./config/yolo_config.ini')

yolo_version = 'yoloV4'

data = config[yolo_version]['yolo_data']
cfg = config[yolo_version]['yolo_cfg']
weights = config[yolo_version]['yolo_weights']

dartnet_obj = Darknetv4 (cfg, data, weights)


class Yolov4Classifier(BaseEstimator,TransformerMixin):
    def __init__(self,):
        self.yoloobj = dartnet_obj

    def fit(self, X, y):
        return self

    def transform_predict(self,X,y=None,img_name=None):
        '''
        Does Yolo prediction for each image
        Does FN,FP comparison with GT of iimages
        #return alert/wrong/correct
        '''
        img_index = 0
        annot_index = 1
        pred_class_index = 3
        regressed_gamma_index = 4
        #y_labels = [a[1] for a in y] #annotations
        prediction_classes = []
        for index in range(len(X)):
            if("alert" not in X[index][pred_class_index]):
                # print(img_name)
                detections = self.yoloobj.image_detection_img(X[index][img_index],img_name=img_name)
                class_ = one_image_IOU(detections,X[index][annot_index])
                prediction_classes.append([X[index][pred_class_index],
                                        class_,
                                        len(X[index][annot_index]),
                                        len(detections),
                                        X[index][regressed_gamma_index]
                                        ])
            else:
                class_="alert"
                prediction_classes.append([X[index][pred_class_index],class_,0,0,X[index][regressed_gamma_index]])
        return prediction_classes
        
    def transform_predict_and_save(self,X,y=None,image_name = None):
        return self.transform_predict(X,y,image_name)

    def transform(self, X, y=None):
        return self.transform_predict(X,y)

    def predict(self, X, y=None):
        return self.transform_predict(X,y)
