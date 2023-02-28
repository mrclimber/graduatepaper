from genericpath import isfile
import openpyxl
import numpy as np
import cv2
from tifffile import imwrite
import os
import random
import shutil
from skimage import io

# # ブックを取得
book_ehime = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_愛媛_20210830.xlsx')
book_shizuoka = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_静岡_20210710.xlsx')

ws_ehime = book_ehime.worksheets[0]
ws_shizuoka = book_shizuoka.worksheets[0]

dir_num = ["1","2","3","4","5"]
cond = ["all","bad","good"]


for i in dir_num:
    for j in cond:
        RemDir="/content/dataset/train_valid/" + i + "/validation/" + j
        shutil.rmtree(RemDir)
        os.mkdir(RemDir)


ehimelist = []
shizuokalist = []
lenehime = []
lenshizuoka = []
alllist = []

for i in range(132):
    k = str(i+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    fac = ws_ehime[a].value
    type = str(ws_ehime[t].value)
    a_ehime = str(ws_ehime[a].value)
    b_ehime = str(ws_ehime[b].value)
    i_new_ehime = a_ehime.zfill(3)
    ins = []
    if b_ehime != "None":
        for j in range(5):
            j_new = str(j+1)
            filename_ehime = '/media/masaki/627D-A8B0/graduatepaper/data/NewData/Ehime/original/all/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
            is_file_ehime = os.path.isfile(filename_ehime)
            if is_file_ehime:
                ins.append(i_new_ehime+'_'+j_new)
                if j == 0:
                    alllist.append(fac)
        if ins != []:
            ehimelist.append(ins)
            lenehime.append(len(ins))

for i in range(132):
    k = str(i+2)
    a = 'A' + k
    b = 'B' + k
    t = 'S' + k
    fac = ws_shizuoka[a].value
    type = str(ws_shizuoka[t].value)
    a_shizuoka = str(ws_shizuoka[a].value)
    b_shizuoka = str(ws_shizuoka[b].value)
    i_new_shizuoka = a_shizuoka.zfill(3)
    ins = []
    if b_shizuoka != "None":
        for j in range(5):
            j_new = str(j+1)
            filename_shizuoka = '/media/masaki/627D-A8B0/graduatepaper/data/NewData/Shizuoka/original/all/MRIT2画像_静岡_' + i_new_shizuoka + '_' + j_new + '.tif'
            is_file_shizuoka = os.path.isfile(filename_shizuoka)
            if is_file_shizuoka:
                i_new_shizuoka = str(int(i_new_shizuoka)+132)
                ins.append(i_new_shizuoka+'_'+j_new)
                i_new_shizuoka = str(int(i_new_shizuoka) - 132).zfill(3)
                if j == 0:
                    alllist.append(fac+132)
        if ins != []:
            shizuokalist.append(ins)
            lenshizuoka.append(len(ins))

print(alllist)

random.shuffle(alllist)

print(alllist)
print(len(alllist))

div = int(len(alllist) / 5)
mod = len(alllist) % 5
divlist = []
if mod == 0:
    divlist = [div,div,div,div,div]
if mod == 1:
    divlist = [div+1,div,div,div,div]
if mod == 2:
    divlist = [div+1,div+1,div,div,div]
if mod == 3:
    divlist = [div+1,div+1,div+1,div,div]
if mod == 4:
    divlist = [div+1,div+1,div+1,div+1,div]

s = 0
t = None
type=""
for i in range(5):
    dir = str(i+1)
    x = divlist[i]
    for j in range(x):
        fac = alllist[s + j]
        # print(fac)
        if fac >= 133:
            name = "静岡"
            fac -= 132
        else:
            name = "愛媛"
        i_new = str(fac).zfill(3)

        # print(name)

        if name == "静岡":
            t = 'S' + str(fac+1)
            type = str(ws_shizuoka[t].value)
            # print(type)
        else:
            for k in range(132):
                r = str(k+2)
                a = 'A' + r
                a_new = ws_ehime[a].value
                # print(a_new)
                if a_new == fac:
                    t = 'S' + r
                    # print(t)
                    type = str(ws_ehime[t].value)
                    # print(type)
                    break

        for k in range(5):
            j_new = str(k+1)
            filename = '/content/dataset/train_valid/all/MRIT2画像_' + name + '_' + i_new + '_' + j_new + '.tif'
            is_file = isfile(filename)
            if is_file:
                img = io.imread(filename)
                path = '/content/dataset/train_valid/' + dir + '/validation/all/MRIT2画像_' + name + '_' + i_new + '_' + j_new + '.tif'
                imwrite(path,img)
                # print(type)
                if type == "良性":
                    path = '/content/dataset/train_valid/' + dir + '/validation/good/MRIT2画像_' + name + '_' + i_new + '_' + j_new + '.tif'
                    imwrite(path,img)
                if type == "悪性":
                    path = '/content/dataset/train_valid/' + dir + '/validation/bad/MRIT2画像_' + name + '_' + i_new + '_' + j_new + '.tif'
                    imwrite(path,img)

    s += x