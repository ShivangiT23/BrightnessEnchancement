from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
import cv2
import numpy as np
import skimage.measure
from sklearn.linear_model import LinearRegression, Lasso, GammaRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

class GammaRegressor(BaseEstimator,TransformerMixin):
    def __init__(self,n_estimators=10,max_depth=8,min_samples_split=3,criterion='mse'):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        # self.reg = RandomForestRegressor(
        #                                 criterion='mse',
        #                                 n_estimators=self.n_estimators,
        #                                 max_depth=self.max_depth,
        #                                 min_samples_split=self.min_samples_split, 
        #                                 random_state=0)
        self.reg =  GradientBoostingRegressor(
                            criterion='mse',
                            n_estimators=self.n_estimators,
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split, 
                            random_state=0
                        )

    def fit(self, X, y):
        '''
        filter gamma dim and bright images from the batch and then apply fit to both. 
        '''
        img_index = 0
        feature_index = 2
        label_index = 3
        gamma_index = 2
        batchX = [itr[feature_index] for itr in X]
        batchy = [itr[gamma_index] for itr in y]
        self.reg.fit(batchX,batchy)
        return self
    
    def apply_gamma(self,X,gamma):
        return np.array(255*(X/255) **(1/gamma), dtype = 'uint8')

    def transform_image(self,X,gamma):
        return np.array(255*(X/255) **(1./gamma), dtype = 'uint8')
        # return np.array(255*(X/255) **(gamma), dtype = 'uint8')
        

    def transform(self,X, y=None):
        '''
        input: [[img,annot,,label],[img,label]]
        '''
        img_index = 0
        label_index = 3
        feature_index = 2
        pred = 1
        batchX = [itr[feature_index] for itr in X]
        pred = self.reg.predict(batchX)
        index = 0        
        for p in pred:
            p = round(p, 2)
            
            label = "good"
            if p>2.0 or p<0.3:
                label = "alert"
            X[index].append(label)
            
            X[index].append(p)
            X[index][img_index] = self.transform_image(X[index][img_index],1.0/p)
            # X[index][img_index] = self.transform_image(X[index][img_index],p)
            index+=1
        return X
