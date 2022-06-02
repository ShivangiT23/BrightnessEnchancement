import os
import cv2
import numpy as np
import random
import glob
import shutil
import configparser
import xml.etree.ElementTree as ET
from utils.utils import *
import csv
import pandas
from sklearn.model_selection import StratifiedShuffleSplit


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_filepath="./config/data_generation_config.ini"
    config.read(config_filepath)
    csv_file= config["data_gen"]["csv_file"]
    df = pandas.read_csv(csv_file,header=None)
    df.columns = ['fname','gamma','yolopred','gt_len','pred_len','file_basename']
    indices = list(range(len(df)))
    df['quality'] = df['fname'].apply(lambda x: os.path.basename(os.path.dirname(x)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    x = np.array(indices)
    y = np.array(df['quality'].tolist())
    sss.get_n_splits(x, y)
    print(x[:5])
    print(y[:5])
    print(sss.split(x, y))
    for train_index, test_index in sss.split(x, y):
        print(x[train_index])
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    df_train = df.loc[x_train]
    df_test = df.loc[x_test]
    print(df_train.shape)
    df_train.to_csv('./intermediate_results/train_syn.csv',index=False)
    df_test.to_csv('./intermediate_results/test_syn.csv',index=False)
    df_test['fname'].to_csv('./intermediate_results/valset.txt', sep=' ', index=False)
