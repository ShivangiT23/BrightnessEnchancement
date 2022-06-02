
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
import cv2
import numpy as np

class SvmClassifier(BaseEstimator,TransformerMixin):
    def __init__(self,C=1,gamma=1,kernel='linear',degree=2):
        self.C=C
        self.gamma=gamma
        self.kernel=kernel
        self.degree=degree
        self.svm = SVC(C=self.C,gamma=self.gamma,kernel=self.kernel,degree=self.degree)
        self.svm_model = None

    def fit(self, X, y):
        '''
        take v channel -> find histogram -> convert to batch 
        take out class labels for SVM Classifier
        SVM -> FIT
        '''
        pooledData = [a[2] for a in X]
        labels = [a[0] for a in y]
        self.svm = self.svm.fit(pooledData, labels)
        return self

    def transform(self, X, y=None):
        '''
        Given image, it is appending the label of classifier along with the image.
        '''
        print("Transform of SVM:")
        pooledData = [a[2] for a in X]
        pred = self.svm.predict(pooledData)
        X_post = []
        for index in range(len(pred)):
            X_post.append([X[index][0],X[index][1],X[index][2],pred[index]])
        #print(X_post)
        return 
    
    def predict_classifier(self,X):
        return self.svm.predict(X)