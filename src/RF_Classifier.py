from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np

class RFClassifier(BaseEstimator,TransformerMixin):
    def __init__(self,criterion='entropy',n_estimators=40,max_depth=25,min_samples_split=5):
        self.criterion=criterion
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split = min_samples_split
        self.rf = RandomForestClassifier(criterion=self.criterion,n_estimators=self.n_estimators,
                                         max_depth=self.max_depth,min_samples_split=self.min_samples_split)
        self.rf_model = None


    def fit(self, X, y):
        '''
        take v channel -> find histogram -> convert to batch 
        take out class labels for RF Classifier
        RF -> FIT
        '''
        #TODO: X shape length should be 4
        #input("Hii")

        rfdata = [a[2] for a in X]
        labels = [a[0] for a in y]
        self.rf = self.rf.fit(rfdata, labels)
        return self

    def transform(self, X, y=None):
        '''
        Given image, it is appending the label of classifier along with the image.
        '''
        rfdata = [a[2] for a in X]
        pred = self.rf.predict(rfdata)
        X_post = []
        for index in range(len(pred)):
            X_post.append([X[index][0],X[index][1],X[index][2],pred[index]])
        #print(X_post)
        return X_post
    
    def predict_classifier(self,X):
        return self.rf.predict(X)
