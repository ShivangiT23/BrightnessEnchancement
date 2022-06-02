from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
import cv2
import numpy as np
import skimage.measure
from sklearn.linear_model import LinearRegression, Lasso, GammaRegressor


class FixedGammaCorrection(BaseEstimator,TransformerMixin):
    def __init__(self,gammaDimValue=1.0/0.5,gammaBrightValue=1.0/1.2,image=None):
        self.gammaDimValue=gammaDimValue
        self.gammaBrightValue=gammaBrightValue

    def fit(self, X, y):
        return self
    
    def apply_gamma(self,X,gamma):
        return np.array(255*(X/255) **(1/gamma), dtype = 'uint8')

    def transform_image(self,X,class_pred):
        if class_pred=="gamma_bright":
            return np.array(255*(X/255) **(1/self.gammaBrightValue), dtype = 'uint8')
        if class_pred=="gamma_dim":
            return np.array(255*(X / 255) **(1/self.gammaDimValue), dtype = 'uint8')
        else:
            return X

    def transform(self,X, y=None):
        '''
        input: [[img,annot,maxpool,label],[img,label]]
        '''
        img_index = 0
        label_index = 3
        #print("transform of gamma")
        #print("X_Labels received by gamma:",X_labels)
        #print("y_Labels received by gamma:",y)
        for index in range(len(X)):
            if X[index][label_index]=="gamma_bright":
                #print('gamma_bright')
                X[index][img_index] = np.array(255*(X[index][img_index]/255) **(1/self.gammaBrightValue), dtype = 'uint8')
            elif X[index][label_index]=="gamma_dim":
                #print('gamma_dim')
                X[index][img_index] = np.array(255*(X[index][img_index] / 255) **(1/self.gammaDimValue), dtype = 'uint8')
        return X


class VariableGammaCorrection(BaseEstimator,TransformerMixin):
    def __init__(self,image=None):
        # self.dim_reg = LinearRegression()
        # self.bright_reg = LinearRegression()
        self.dim_reg = Lasso()
        self.bright_reg = Lasso()

    def fit(self, X, y):
        '''
        filter gamma dim and bright images from the batch and then apply fit to both. 
        '''
        batch_dim = []
        batch_dim_y = []
        batch_bright = []
        batch_bright_y = []
        img_index = 0
        feature_index = 2
        label_index = 3
        gamma_index = 2
        for index in range(len(X)):
            if X[index][label_index]=="gamma_bright":
                batch_bright.append(X[index][feature_index])
                batch_bright_y.append(y[index][gamma_index])
            if X[index][label_index]=="gamma_dim":
                batch_dim.append(X[index][feature_index])
                batch_dim_y.append(y[index][gamma_index])
        if len(batch_dim)>0:
            self.dim_reg.fit(batch_dim,batch_dim_y)
        if len(batch_bright)>0:
            self.bright_reg.fit(batch_bright,batch_bright_y)
        return self
    
    def apply_gamma(self,X,gamma):
        return np.array(255*(X/255) **(1/gamma), dtype = 'uint8')

    def transform_image(self,X,gamma):
        return np.array(255*(X/255) **(1./gamma), dtype = 'uint8')
        

    def transform(self,X, y=None):
        '''
        input: [[img,annot,,label],[img,label]]
        '''
        img_index = 0
        label_index = 3
        feature_index = 2
        pred = 1
        for index in range(len(X)):
            if X[index][label_index]=="gamma_bright":
                pred = self.bright_reg.predict([X[index][feature_index]])
                pred = pred[0]
                #print("predicted:",pred)
                X[index][img_index] = self.transform_image(X[index][img_index],pred)
            elif X[index][label_index]=="gamma_dim":
                pred = self.dim_reg.predict([X[index][feature_index]])
                pred=pred[0]
                #print("predicted:",pred)
                X[index][img_index] = self.transform_image(X[index][img_index],pred)
            X[index].append(pred)
            print(pred,len(X[index]))
        return X
