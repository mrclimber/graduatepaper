import openpyxl
import numpy as np
import cv2
from tifffile import imwrite
import os
import random
import shutil
from skimage import io

# # ブックを取得
book_ehime = openpyxl.load_workbook('/content/dataset/耳下腺情報シート_愛媛_20210830.xlsx')
book_shizuoka = openpyxl.load_workbook('/content/dataset/耳下腺情報シート_静岡_20210710.xlsx')

ws_ehime = book_ehime.worksheets[0]
ws_shizuoka = book_shizuoka.worksheets[0]


ehimelist = []
shizuokalist = []
lenehime = []
lenshizuoka = []

for i in range(132):
    k = str(i+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    type = str(ws_ehime[t].value)
    a_ehime = str(ws_ehime[a].value)
    b_ehime = str(ws_ehime[b].value)
    i_new_ehime = a_ehime.zfill(3)
    ins = []
    if b_ehime != "None":
        for j in range(5):
            j_new = str(j+1)
            filename_ehime = '/content/dataset/NewData/Ehime/original/all/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
            is_file_ehime = os.path.isfile(filename_ehime)
            if is_file_ehime:
                ins.append(i_new_ehime+'_'+j_new)
        if ins != []:
            ehimelist.append(ins)
            lenehime.append(len(ins))

for i in range(132):
    k = str(i+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    type = str(ws_shizuoka[t].value)
    a_shizuoka = str(ws_shizuoka[a].value)
    b_shizuoka = str(ws_shizuoka[b].value)
    i_new_shizuoka = a_shizuoka.zfill(3)
    ins = []
    if b_shizuoka != "None":
        for j in range(5):
            j_new = str(j+1)
            filename_shizuoka = '/content/dataset/NewData/Shizuoka/original/all/MRIT2画像_静岡_' + i_new_shizuoka + '_' + j_new + '.tif'
            is_file_shizuoka = os.path.isfile(filename_shizuoka)
            if is_file_shizuoka:
                i_new_shizuoka = str(int(i_new_shizuoka)+132)
                ins.append(i_new_shizuoka+'_'+j_new)
                i_new_shizuoka = str(int(i_new_shizuoka) - 132).zfill(3)
        if ins != []:
            shizuokalist.append(ins)
            lenshizuoka.append(len(ins))
listehime = []
for i in lenehime:
    listins = []
    if i == 1:
        listins = [5]
    elif i == 2:
        listins = [3,2]
    elif i == 3:
        listins = [2,2,1]
    elif i == 4:
        listins = [2,1,1,1]
    else:
        listins = [1]
    listehime.append(listins)

listshizuoka = []
for i in lenshizuoka:
    listins = []
    if i == 1:
        listins = [5]
    elif i == 2:
        listins = [3,2]
    elif i == 3:
        listins = [2,2,1]
    elif i == 4:
        listins = [2,1,1,1]
    else:
        listins = [1]
    listshizuoka.append(listins)

for i in range(132):
    a = 'A' + str(i+2)
    a = str(ws_ehime[a].value)
    t = 'S' + str(i+2)
    type = str(ws_ehime[t].value)
    # print(type)
    for j in range(5):
        path = "/content/dataset/OriginalData/Ehime/" # パスを格納
        i_new_ehime = a.zfill(3)
        j_new = str(j+1)
        name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
        name = path+name
        is_file = os.path.isfile(name)
        if is_file:
            img = io.imread(name) # これでOK
            name_new = '/content/dataset/OriginalData/Data/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
            imwrite(name_new, img)            

for i in range(132):
    a = 'A' + str(i+2)
    a = str(ws_shizuoka[a].value)
    t = 'S' + str(i+2)
    type = str(ws_shizuoka[t].value)
    # print(type)
    for j in range(5):
        path = "/content/dataset/OriginalData/Shizuoka/" # パスを格納
        i_new_shizuoka = a.zfill(3)
        j_new = str(j+1)
        name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
        name = path+name
        is_file = os.path.isfile(name)
        if is_file:
            img = io.imread(name) # これでOK
            name_new = '/content/dataset/OriginalData/Data/MRIT2画像_静岡_' + i_new_shizuoka + '_' + j_new + '.tif'
            imwrite(name_new, img)       


l = 0
for ins in listehime:
    m = lenehime[l]
    k = str(l+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    type = str(ws_ehime[t].value)
    a_ehime = str(ws_ehime[a].value)
    b_ehime = str(ws_ehime[b].value)
    i_new_ehime = a_ehime.zfill(3)
    r = 1
    for j in ins:
        for n in range(j-1):
            r_new = str(r)
            name = '/content/dataset/OriginalData/Ehime/MRIT2画像_愛媛_' + i_new_ehime + '_' + r_new + '.tif'
            img = io.imread(name)
            listab = []
            a = random.uniform(-5.0,-15.0)
            b = random.uniform(5.0,15.0)
            listab.append(a)
            listab.append(b)
            h, w, c = img.shape
            
            sig=random.uniform(20,40)
            noise_gaussian=np.random.normal(0,sig,np.shape(img))
            img_noise=img+np.floor(noise_gaussian) #画像にノイズを付加
            img_noise[img_noise>255]=255 # 255超える場合は255
            img_noise[img_noise<0]=0 # 255超える場合は255
            img_new=img_noise.astype(np.uint8)
            m_new = str(m+1)
            name_new = '/content/dataset/OriginalData/Data/MRIT2画像_愛媛_' + i_new_ehime + '_' + m_new + '.tif'
            imwrite(name_new, img_new)
            m = m + 1
        r += 1
    l += 1

l = 0
for ins in listshizuoka:
    m = lenshizuoka[l]
    k = str(l+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    type = str(ws_shizuoka[t].value)
    a_shizuoka = str(ws_shizuoka[a].value)
    b_shizuoka = str(ws_shizuoka[b].value)
    i_new_shizuoka = a_shizuoka.zfill(3)
    r = 1
    for j in ins:
        for n in range(j-1):
            r_new = str(r)
            name = '/content/dataset/OriginalData/Shizuoka/MRIT2画像_静岡_' + i_new_shizuoka + '_' + r_new + '.tif'
            img = io.imread(name)
            h, w, c = img.shape
            
            sig=random.uniform(20,40)
            noise_gaussian=np.random.normal(0,sig,np.shape(img))
            img_noise=img+np.floor(noise_gaussian) #画像にノイズを付加
            img_noise[img_noise>255]=255 # 255超える場合は255
            img_noise[img_noise<0]=0 # 255超える場合は255
            img_new=img_noise.astype(np.uint8)
            m_new = str(m+1)
            name_new = '/content/dataset/OriginalData/Data/MRIT2画像_静岡_' + i_new_shizuoka + '_' + m_new + '.tif'
            imwrite(name_new, img_new)
            m = m + 1
        r += 1
    l += 1