import openpyxl
import numpy as np
import cv2
from tifffile import imwrite
import os
import random
import shutil
from skimage import io
import glob

book_ehime = openpyxl.load_workbook('/content/耳下腺情報シート_愛媛_20210830.xlsx')
book_shizuoka = openpyxl.load_workbook('/content/耳下腺情報シート_静岡_20210710.xlsx')
                                                                                                                                                        
ws_ehime = book_ehime.worksheets[0]
ws_shizuoka = book_shizuoka.worksheets[0]

dir_list = ["/2/","/3/","/4/","/5/"]
cond_list = ["bad", "good"]
amppha_waylist = ["AmpPha", "AmpPha1", "AmpPha2", "AmpPha3"]
highlow_waylist = ["HighLow", "HighLow1", "HighLow2", "HighLow3"]
norm_lenlist = [250, 1250]
mix_lenlist = [500, 2500]

# 
def MakeListNorm(cond, x):
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
        a_ehime = str(ws_ehime[a].value)
        b_ehime = str(ws_ehime[b].value)
        i_new_ehime = a_ehime.zfill(3)
        ins = []
        if b_ehime != "None":
            for j in range(5):
                j_new = str(j+1)
                filename_ehime = '/content/dataset/new/train_valid' + x + 'clip/' + cond + '/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
                is_file_ehime = os.path.isfile(filename_ehime)
                if is_file_ehime:
                    ins.append(i_new_ehime+'_'+j_new)
            if ins != []:
                ehimelist.append(ins)
                alllist.append(ins)
                lenehime.append(len(ins))


    for i in range(132):
        k = str(i+2)
        a = 'A' + k
        b = 'B' + k
        t = 'S' + k
        a_shizuoka = str(ws_shizuoka[a].value)
        b_shizuoka = str(ws_shizuoka[b].value)
        i_new_shizuoka = a_shizuoka.zfill(3)
        ins = []
        if b_shizuoka != "None":
            for j in range(5):
                j_new = str(j+1)
                filename_shizuoka = '/content/dataset/new/train_valid' + x + 'clip/' + cond + '/MRIT2画像_静岡_' + i_new_shizuoka + '_' + j_new + '.tif'
                is_file_shizuoka = os.path.isfile(filename_shizuoka)
                if is_file_shizuoka:
                    i_new_shizuoka = str(int(i_new_shizuoka)+132)
                    ins.append(i_new_shizuoka+'_'+j_new)
                    i_new_shizuoka = str(int(i_new_shizuoka) - 132).zfill(3)
            if ins != []:
                shizuokalist.append(ins)
                alllist.append(ins)
                lenshizuoka.append(len(ins))

    return alllist


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def ReadImgNorm(cond, i, alllist, x):
    t1 = random.choice(random.choice(alllist)).split('_')
    t2 = random.choice(random.choice(alllist)).split('_')
    if t1[0] == "132":
        div1 = 0
    else:
        div1 = int(int(t1[0]) / 132)
    if t2[0] == "132":
        div2 = 0
    else:
        div2 = int(int(t2[0]) / 132)
    if t1[0] == "132":
        num1 = 132
    else:
        num1 = int(t1[0]) % 132
    if t2[0] == "132":
        num2 = 132
    else:
        num2 = int(t2[0]) % 132
    num1 = str(num1)
    num2 = str(num2)
    num1_new = num1.zfill(3)
    num2_new = num2.zfill(3)   
    t = 'S' + str(i+2)
    if div1:
        name1 = "静岡"
    else:
        name1 = '愛媛'
    if div2:
        name2 = '静岡'
    else:
        name2 = '愛媛'
    
    j1_new = str(t1[1])
    j2_new = str(t2[1])

    filename1 = '/content/dataset/new/train_valid' + x + 'clip/' + cond + '/MRIT2画像_' + name1 + '_' + num1_new + '_' + j1_new + '.tif'
    filename2 = '/content/dataset/new/train_valid' + x + 'clip/' + cond + '/MRIT2画像_' + name2 + '_' + num2_new + '_' + j2_new + '.tif'
    img1 = imread(filename1,0)
    img1 = cv2.resize(img1, (128, 128))

    img2 = imread(filename2,0)
    img2 = cv2.resize(img2, (128, 128))

    return img1, img2

def WriteImgNorm(dst1, dst2, cond, i, way, c):
    name3 = '/content/dataset/new/train/' + way + '/all/' + cond + '/MRIT2画像_' + way + '_' + str(i+c*2000) + '.tif'
    imwrite(name3, dst1)
    name4 = '/content/dataset/new/train/' + way + '/all/' + cond + '/MRIT2画像_' + way + '_' + str(i+c*2000+1) + '.tif'
    imwrite(name4, dst2)


def GetFiles(x):
    good_files = glob.glob("/content/dataset/new/train_valid" + x + "clip/good/*")
    bad_files = glob.glob("/content/dataset/new/train_valid" + x + "clip/bad/*")
    goodlen = len(good_files)
    badlen = len(bad_files)
    return good_files, bad_files, goodlen, badlen

def ReadImgMix(good_files, bad_files, goodlen, badlen):
    arr = np.random.rand(1,2)
    arr /= arr.sum(axis=1)[:,np.newaxis]

    x1 = np.random.choice(2,p=[0.5,0.5])
    if x1 == 0:
        cond1 = "good"
        point1 = random.randint(0,goodlen-1)
        print("good:",good_files[point1])
        img1 = imread(good_files[point1],0)
    else:
        cond1 = "bad"
        point1 = random.randint(0,badlen-1)
        print("bad:",bad_files[point1])
        img1 = imread(bad_files[point1],0)

    img1 = cv2.resize(img1, (128, 128))

    x2 = np.random.choice(2,p=[0.5,0.5])
    if x2 == 0:
        cond2 = "good"
        point2 = random.randint(0,goodlen-1)
        print("good:",good_files[point2])
        img2 = imread(good_files[point2],0)
    else:
        cond2 = "bad"
        point2 = random.randint(0,badlen-1)
        print("bad:",bad_files[point2])
        img2 = imread(bad_files[point2],0)
    img2 = cv2.resize(img2, (128, 128))

    return img1, img2, cond1, cond2, arr

def WriteImgMix(arr, dst1, dst2, cond1, cond2, i, way, c):
    if arr[0][0] > arr[0][1]:
        name1_new = "/content/dataset/new/train/" + way + "/all/" + cond1 + "/MRIT2画像_" + way + "_" + str(i+c*2000) + '.tif'
        imwrite(name1_new,dst1)
        name2_new = "/content/dataset/new/train/" + way + "/all/" + cond2 + "/MRIT2画像_" + way + "_" + str(i+c*2000+1) + '.tif'
        imwrite(name2_new,dst2)
    else:
        name1_new = "/content/dataset/new/train/" + way + "/all/" + cond1 + "/MRIT2画像_" + way + "_" + str(i+c*2000) + '.tif'
        imwrite(name1_new,dst1)
        name2_new = "/content/dataset/new/train/" + way + "/all/" + cond2 + "/MRIT2画像_" + way + "_" + str(i+c*2000+1) + '.tif'
        imwrite(name2_new,dst2)

def AmpPhaChange(img1, img2):
    dft1 = cv2.dft(np.float32(img1),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)  # type: ignore
    dft_shift1 = np.fft.fftshift(dft1)
    magnitude1,angle1 = cv2.cartToPolar(dft_shift1[:,:,0],dft_shift1[:,:,1])

    dft2 = cv2.dft(np.float32(img2),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)  # type: ignore
    dft_shift2 = np.fft.fftshift(dft2)

    magnitude2,angle2 = cv2.cartToPolar(dft_shift2[:,:,0],dft_shift2[:,:,1])

    dft_shift3_real, dft_shift3_imag = cv2.polarToCart(magnitude1, angle2)
    dft_shift3 = np.stack([dft_shift3_real, dft_shift3_imag], axis=-1)
    f_ishift3 = np.fft.ifftshift(dft_shift3)
    img_back3 = cv2.idft(f_ishift3)
    dst1 = np.clip(img_back3[:,:,0],0,255).astype(np.uint8)

    dft_shift4_real, dft_shift4_imag = cv2.polarToCart(magnitude2, angle1)
    dft_shift4 = np.stack([dft_shift4_real, dft_shift4_imag], axis=-1)
    f_ishift4 = np.fft.ifftshift(dft_shift4)
    img_back4 = cv2.idft(f_ishift4)
    dst2 = np.clip(img_back4[:,:,0],0,255).astype(np.uint8)

    return dst1, dst2

def HighLowChange(img1, img2):
    dft1 = cv2.dft(np.float32(img1),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE) * arr[0][0] # type: ignore
    dft_shift1 = np.fft.fftshift(dft1)

    dft2 = cv2.dft(np.float32(img2),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE) * arr[0][1] # type: ignore
    dft_shift2 = np.fft.fftshift(dft2)

    rows1, cols1 = img1.shape
    crow1,ccol1 = rows1//2 , cols1//2

    mask_low = np.zeros((rows1,cols1,2),np.uint8)

    mask_low[crow1-10:crow1+10, ccol1-10:ccol1+10] = 1

    mask_high = np.ones((rows1,cols1,2),np.uint8)

    mask_high[crow1-10:crow1+10, ccol1-10:ccol1+10] = 0

    fshift1_low = dft_shift1*mask_low

    fshift1_high = dft_shift1*mask_high

    fshift2_low = dft_shift2*mask_low

    fshift2_high = dft_shift2*mask_high

    fshift1 = fshift1_low + fshift2_high

    fshift2 = fshift1_high + fshift2_low

    f_ishift1 = np.fft.ifftshift(fshift1)
    img_back1 = cv2.idft(f_ishift1)
    dst1 = np.clip(img_back1[:,:,0],0,255).astype(np.uint8)

    f_ishift2 = np.fft.ifftshift(fshift2)
    img_back2 = cv2.idft(f_ishift2)
    dst2 = np.clip(img_back2[:,:,0],0,255).astype(np.uint8)
    return dst1, dst2

def AmpPhaNorm(x, c):
    for cond in cond_list:
        alllist = MakeListNorm(cond,x)
        for num in norm_lenlist:
            for i in range(1, num, 2):
                for k in range(2):        
                    img1, img2 = ReadImgMix(cond, i, alllist, x)
                    dst1, dst2 = AmpPhaChange(img1, img2)
                    WriteImgNorm(dst1, dst2, cond, i, amppha_waylist[k], c)

def AmpPhaMix(x, c):
    good_files, bad_files, goodlen, badlen = GetFiles(x)
    for num in norm_lenlist:
        for i in range(1, num, 2):
            for k in range(2): 
                img1, img2, cond1, cond2, arr = ReadImgMix(x, good_files, bad_files, goodlen, badlen)
                dst1, dst2 = AmpPhaChange(img1, img2)
                WriteImgMix(arr, dst1, dst2, cond1, cond2, i, amppha_waylist[k+2], c)


def HighLowNorm(x, c):
    for cond in cond_list:
        alllist = MakeListNorm(cond,x)
        for num in norm_lenlist:
            for i in range(1, num, 2):
                for k in range(2):
                    img1, img2 = ReadImgMix(cond, i, alllist, x)
                    dst1, dst2 = AmpPhaChange(img1, img2)
                    WriteImgNorm(dst1, dst2, cond, i, highlow_waylist[k], c)

def HighLowMix(x, c):
    good_files, bad_files, goodlen, badlen = GetFiles(x)
    for num in norm_lenlist:
        for i in range(1, num, 2):
            for k in range(2): 
                img1, img2, cond1, cond2, arr = ReadImgMix(x, good_files, bad_files, goodlen, badlen)
                dst1, dst2 = AmpPhaChange(img1, img2)
                WriteImgMix(arr, dst1, dst2, cond1, cond2, i, highlow_waylist[k+2], c)

def main():
    c = 0
    for x in dir_list:
        AmpPhaNorm(x, c)
        AmpPhaMix(x, c)
        HighLowNorm(x, c)
        HighLowMix(x, c)
        c += 1

if __name__ == "__main__":
    main()