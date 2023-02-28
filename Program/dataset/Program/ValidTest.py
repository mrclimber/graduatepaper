from re import X
from tifffile import TiffFile
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from tifffile import imwrite
import os
from PIL import Image
import openpyxl
import shutil

book_ehime = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_愛媛_20210830.xlsx')
book_shizuoka = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_静岡_20210710.xlsx')

ws_ehime = book_ehime.worksheets[0]
ws_shizuoka = book_shizuoka.worksheets[0]

train_dir = '/content/dataset/train_valid/all'
train_good_dir = '/content/dataset/train_valid/good'
train_bad_dir = '/content/dataset/train_valid/bad'
test_good_dir = '/content/dataset/test/good'
test_bad_dir = '/content/dataset/test/bad'
shutil.rmtree(train_dir)
os.mkdir(train_dir)
shutil.rmtree(train_good_dir)
os.mkdir(train_good_dir)
shutil.rmtree(train_bad_dir)
os.mkdir(train_bad_dir)
shutil.rmtree(test_good_dir)
os.mkdir(test_good_dir)
shutil.rmtree(test_bad_dir)
os.mkdir(test_bad_dir)

for i in range(132):
    a = 'A' + str(i+2)
    a_ehime = str(ws_ehime[a].value)
    a_shizuoka = str(ws_shizuoka[a].value)
    t = 'S' + str(i+2)
    type_ehime = str(ws_ehime[t].value)
    type_shizuoka = str(ws_shizuoka[t].value)
    x1 = np.random.choice(2,p=[0.8,0.2])
    x2 = np.random.choice(2,p=[0.8,0.2])
    
    for j in range(5):
        path_ehime = "/media/masaki/627D-A8B0/graduatepaper/data/OriginalData/AllData/" # パスを格納
        i_new_ehime = a_ehime.zfill(3)
        j_new = str(j+1)
        name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
        name_ehime = path_ehime+name
        is_file_ehime = os.path.isfile(name_ehime)
        path_shizuoka = "/media/masaki/627D-A8B0/graduatepaper/data/OriginalData/AllData/" # パスを格納
        i_new_shizuoka = a_shizuoka.zfill(3)
        name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
        name_shizuoka = path_shizuoka+name
        is_file_shizuoka = os.path.isfile(name_shizuoka)
        # IDで分ける
        if is_file_ehime:
            img = io.imread(name_ehime) # これでOK
            if x1 == 0:
                path = "/content/dataset/train_valid/all/" # パスを格納
                name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
                name = path+name
                imwrite(name, img)
                if type_ehime == "良性":
                    path = "/content/dataset/train_valid/good/" # パスを格納
                    name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                elif type_ehime == "悪性":
                    path = "/content/dataset/train_valid/bad/" # パスを格納
                    name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                    
            else:
                if type_ehime == "良性":
                    path = "/content/dataset/test/good/" # パスを格納
                    name = 'MRIT2画像_愛媛_test_'+i_new_ehime+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                elif type_ehime == "悪性":
                    path = "/content/dataset/test/bad/" # パスを格納
                    name = 'MRIT2画像_愛媛_test_'+i_new_ehime+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
    
        if is_file_shizuoka:
            img = io.imread(name_shizuoka) # これでOK
            if x2 == 0:
                path = "/content/dataset/train_valid/all/" # パスを格納
                name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
                name = path+name
                imwrite(name, img)
                if type_shizuoka == "良性":
                    path = "/content/dataset/train_valid/good/" # パスを格納
                    name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                elif type_shizuoka == "悪性":
                    path = "/content/dataset/train_valid/bad/" # パスを格納
                    name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                    
            else:
                if type_shizuoka == "良性":
                    path = "/content/dataset/test/good/" # パスを格納
                    name = 'MRIT2画像_静岡_test_'+i_new_shizuoka+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)
                elif type_shizuoka == "悪性":
                    path = "/content/dataset/test/bad/" # パスを格納
                    name = 'MRIT2画像_静岡_test_'+i_new_shizuoka+'_'+j_new+'.tif'
                    name = path+name
                    imwrite(name, img)