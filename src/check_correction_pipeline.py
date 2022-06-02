
import cv2
import os
import numpy as np
import glob
#from sklearn.ensemble import RandomForestClassifier
import shutil
from sklearn.metrics import make_scorer,confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import  Pipeline, make_pipeline
from sklearn.utils import shuffle
from modules.GammaCorrector import FixedGammaCorrection
from modules.YoloClassifier import Yolov4Classifier
from RF_Classifier import RFClassifier
from modules.SVM_Classifier import SvmClassifier
import csv
from utils.utils import *
import json

with open("./config/training_config.json", "r") as jsonfile:
    config = json.load(jsonfile)
data_csv = config['data_csv_path']
images = []
y = []
qual = []
# with open(data_csv, 'r') as file:
#     reader = csv.reader(file, delimiter = '\t')
#     index = 0
#     for row in reader:
#         # if(index==0):
#         #     index+=1
#         #     continue
#         row = row[0].split(',')
#         if('alert' not in row[2]):         
#             images.append(row[0])
#             #y.append(float(row[1]))
#             #qual.append(row[2])
#         else:
#             images.append(row[0])
#             #y.append(None)
#             #qual.append(row[2])
# X_train=[]
# y_train=[]
# index = 0
# correct = 0
# folder_dump = 'temp_check_correctness'
# if os.path.exists(folder_dump):
#     shutil.rmtree(folder_dump)
# os.makedirs(folder_dump)
# for image in images:
X_train=[]
y_train=[]
index = 0
correct = 0
gamma_corrector = FixedGammaCorrection(gammaDimValue=1.0,gammaBrightValue=1.0)
yolo_classifier = Yolov4Classifier()
data_folder = '/comads_work/data/synthetic_augmented'
for image in glob.glob(data_folder+'/**/*.png'):
    img=cv2.imread(image)
    #img = cv2.resize(img,(604,604))
    head, image_name=os.path.split(image)
    head1, quality = os.path.split(head)
    annot_name = image[:-3]+"xml"
    annot = read_annotations(annot_name)
    corrected_img = gamma_corrector.transform_image(img,quality)
    y_pred=yolo_classifier.transform_predict([[corrected_img,annot,None,quality,1.0]])
    pred = y_pred[0][1]
    if pred == "correct":
        if 'good' not in image:
            print("CORRECTED",image)
        correct+=1
    else:
        if 'good' in image:
            print("INCORRECTED",image)
        # cv2.imwrite(os.path.join(folder_dump,os.path.basename(image)[:-3]+"png"),corrected_img)

print("Finally total correct:",correct)
print("total:",len(images))