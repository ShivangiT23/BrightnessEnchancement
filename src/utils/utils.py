import xml.etree.ElementTree as ET
import numpy as np
import cv2
import skimage.measure
from keras.utils.np_utils import to_categorical
from sklearn.metrics import make_scorer,confusion_matrix, classification_report,accuracy_score


def __gamma_correction(img,gamma=1.0):        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l = cv2.split(lab)[0]
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255.0
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    l_adjusted = cv2.LUT(l, table)
    lab[:,:,0] = l_adjusted
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def _gamma_correction(img,gamma=1.0):    
    print("in gamma...")    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255.0
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    l_adjusted = cv2.LUT(img, table)
    return l_adjusted

def gamma_correction(img,gamma=1.0):    
    return np.array(255*(img / 255) **(1.0/gamma), dtype = 'uint8')

def read_annotations(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()
	bboxes = []
	for boxes in root.iter('object'):
		ymin , xmin, ymax, xmax = None, None, None, None
		for box in boxes.findall("bndbox"):
			ymin = int(box.find("ymin").text)
			xmin = int(box.find("xmin").text)
			ymax = int(box.find("ymax").text)
			xmax = int(box.find("xmax").text)
		list_with_single_boxes = [xmin, ymin, xmax, ymax]
		bboxes.append(list_with_single_boxes)
	return bboxes

def confusion_matrix_3_class(y_test,y_pred,index=1):
    conf = np.zeros((2,3))
    map_index={
        'wrong':0,
        'correct':1,
        'alert':2
    }
    for i in range(len(y_test)):
        #print(y_test[i],y_pred[i])
        conf[map_index[y_test[i][1]],map_index[y_pred[i][1]]]+=1
    return conf

def get_hist(img):
    hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    channel_values = np.array(hsv_img[:,:,2])
    channel_array = channel_values.flatten()
    hist,_ = np.histogram(channel_array,150,[0,255])
    return hist

def get_maxpooled(img):
    im = cv2.resize(img,(128,128))
    hsv_img=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    channel_values = np.array(hsv_img[:,:,2])
    im_data = skimage.measure.block_reduce(channel_values, (4,4), np.max)
    im_data = im_data.reshape(-1)
    return im_data

def score_func(y,y_pred):
    print("Confusoin Matrix:\n",confusion_matrix_3_class(y,y_pred))
    ######## FOR CORRECT WRONG SCORER ##################
    map_labels_to_indexes = {
        'alert':2,
        'wrong':0,
        'correct':1
    }
    y_wout_class = [map_labels_to_indexes[a[1]] for a in y]
    y_pred_3class = [map_labels_to_indexes[a[1]] for a in y_pred]
    

    one_hot_score_gt =  to_categorical(y_wout_class,num_classes=2)
    one_hot_score_pred =  to_categorical(y_pred_3class,num_classes=3)
    #print(one_hot_score_gt)
    #print(one_hot_score_pred)
    score_matrix = 0
    cost_array=np.array([[-0.5,2,0.1],[-2,0.1,-2]]) #rows: w/0,c/1 cols: w,c,a
    for i in range(len(one_hot_score_gt)):
        #print(i, one_hot_score_gt[i,:].T,one_hot_score_pred[i,:],one_hot_score_gt[i,:].T.dot(cost_array).dot(one_hot_score_pred[i,:].reshape((3,1))))
        score_matrix +=  one_hot_score_gt[i,:].T.dot(cost_array).dot(one_hot_score_pred[i,:].reshape((3,1)))
    print("Score:",score_matrix) 
    #############################################################
    map_labels_to_indexes_5_classes = {
        'alert_bright':0,
        'alert_dim':1,
        'gamma_bright':2,
        'gamma_dim':3,
        'good':4
    }
    y_wout_class = [map_labels_to_indexes_5_classes[a[0]] for a in y]
    y_pred_5_class = [map_labels_to_indexes_5_classes[a[0]] for a in y_pred]
    one_hot_score_gt =  to_categorical(y_wout_class,num_classes=5)
    one_hot_score_pred =  to_categorical(y_pred_5_class,num_classes=5)
    acc = accuracy_score(y_wout_class,y_pred_5_class)  
    print("Acc:",acc)
    ######################################################################
    pred_gamma = np.array([x[4] for x in y_pred])
    gt_gamma = np.array([x[2] for x in y])
    esp = 0.00000001
    diff = (pred_gamma-gt_gamma).reshape(1,-1)
    dist = np.sqrt(esp+(diff).dot(diff.T))
    gamma_score = 1.0/dist
    print("Gamma:",gamma_score)
    #print("Conf Matrix\n",confusion_matrix(one_hot_score_gt,one_hot_score_pred))
    print("Total Score:",(acc + score_matrix[0]+gamma_score)/3.)
    return (acc + score_matrix[0]+gamma_score)/3.


def score_func_only_regressor(y,y_pred):
    print("Confusoin Matrix:\n",confusion_matrix_3_class(y,y_pred))
    ######## FOR CORRECT WRONG SCORER ##################
    map_labels_to_indexes = {
        'alert':2,
        'wrong':0,
        'correct':1
    }
    y_wout_class = [map_labels_to_indexes[a[1]] for a in y]
    y_pred_3class = [map_labels_to_indexes[a[1]] for a in y_pred]
    

    one_hot_score_gt =  to_categorical(y_wout_class,num_classes=2)
    one_hot_score_pred =  to_categorical(y_pred_3class,num_classes=3)
    #print(one_hot_score_gt)
    #print(one_hot_score_pred)
    score_matrix = 0
    cost_array=np.array([[-0.5,2,0.1],[-2,0.1,-2]]) #rows: w/0,c/1 cols: w,c,a
    
    # cost_array=np.array([[-0.5,3,0.1],[-3,3,-3]]) #rows: w/0,c/1 cols: w,c,a
    # cost_array=np.array([[-0.5,3,0.5],[-3,1,-3]]) #rows: w/0,c/1 cols: w,c,a
    for i in range(len(one_hot_score_gt)):
        #print(i, one_hot_score_gt[i,:].T,one_hot_score_pred[i,:],one_hot_score_gt[i,:].T.dot(cost_array).dot(one_hot_score_pred[i,:].reshape((3,1))))
        score_matrix +=  one_hot_score_gt[i,:].T.dot(cost_array).dot(one_hot_score_pred[i,:].reshape((3,1)))
    print("Score:",score_matrix) 
    ######################################################################
    pred_gamma = np.array([x[4] for x in y_pred])
    gt_gamma = np.array([x[2] for x in y])
    esp = 0.00000001
    diff = (pred_gamma-gt_gamma).reshape(1,-1)
    dist = np.sqrt(esp+(diff).dot(diff.T))
    gamma_score = 1.0/dist
    print("Gamma:",gamma_score)
    #print("Conf Matrix\n",confusion_matrix(one_hot_score_gt,one_hot_score_pred))
    print("Total Score:",( score_matrix[0]+gamma_score)/2.)
    return ( score_matrix[0]+gamma_score)/2.
