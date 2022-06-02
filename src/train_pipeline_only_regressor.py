#!/usr/bin/env python
# coding: utf-8


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
from modules.YoloClassifier import Yolov4Classifier
from modules.Regressor import GammaRegressor
import csv
from utils.utils import *
import json

with open("./config/training_config.json", "r") as jsonfile:
    config = json.load(jsonfile)
data_csv = config['data_csv_path']
images = []
y = []
qual = []
gamma_val = []
with open(data_csv, 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    index = 0
    for row in reader:
        if(index==0):
            index+=1
            continue
        row = row[0].split(',')
        images.append(row[0])
        gamma_val.append(float(row[1]))
        
X_train=[]
y_train=[]
index = 0
for image in images:
    img=cv2.imread(image)
    #img = cv2.resize(img,(604,604))
    head, image_name=os.path.split(image)
    head1, quality = os.path.split(head)
    annot_name = image[:-3]+"xml"
    annot = read_annotations(annot_name)
    X_train.append([img,annot,get_maxpooled(img)])
    if "okay" in quality or "good" in quality:
        type_qual="correct"
    else:
        type_qual="wrong"
    y_train.append([quality,type_qual,gamma_val[index]])
    index+=1


X_train, y_train = shuffle(X_train, y_train)

model_name = "gamma_regressor"
obj = GammaRegressor()

pipe=Pipeline(steps=[('classifier',obj),('pillcount',Yolov4Classifier())]) 

# gamma_params = config['gamma_params']
## REMOVED GAMMA PARAMS
classifier_params={
        "classifier__max_depth": [5,7,9,12,15],
        "classifier__min_samples_split": [ 4, 5],
        "classifier__n_estimators": [50,70,80]    
    }
scorer = make_scorer(score_func_only_regressor,greater_is_better=True) 
grid_search = GridSearchCV(pipe, classifier_params, n_jobs=1,cv=config["cross_validation"],verbose =2,scoring=scorer)
grid_search.fit(X_train,y_train)
best_grid = grid_search.best_estimator_

f = open('pooled_'+model_name+'.txt','w')
print("Best Grid:",file=f)
print(best_grid,file=f)

import joblib
saved_path_classifier = os.path.join(config['saved_model_dir'],'gbt_best_'+model_name+'_with_pooling.pkl')
joblib.dump(best_grid[0],saved_path_classifier)