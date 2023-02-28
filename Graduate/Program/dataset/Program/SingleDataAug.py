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

dir_list = ["2/","3/","4/","5/"]
evenway = ["AmpPha", "AmpPha2", "HighLow", "HighLow2", "HighPass", "LowPass", "base"]
oddway = ["AmpPha1", "AmpPha3", "HighLow1", "HighLow3", "HighPass1", "LowPass1", "base1"]


def MakeList(h):
    ehimelist = []
    shizuokalist = []
    lenehime = []
    lenshizuoka = []
    for i in range(132):
        k = str(i+2)
        a = 'A' + k
        b = 'B' + k
        a_ehime = str(ws_ehime[a].value)
        b_ehime = str(ws_ehime[b].value)
        i_new_ehime = a_ehime.zfill(3)
        ins = []
        if b_ehime != "None":
            for j in range(5):
                j_new = str(j+1)
                filename_ehime = '/content/dataset/train_valid/'+ h +'/validation/all/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
        a_ehime = str(ws_ehime[a].value)
        b_ehime = str(ws_ehime[b].value)
        i_new_ehime = a_ehime.zfill(3)
        ins = []
        if b_ehime != "None":
            for j in range(5):
                j_new = str(j+1)
                filename_ehime = '/content/dataset/train_valid/'+ h +'/validation/all/MRIT2画像_愛媛_' + i_new_ehime + '_' + j_new + '.tif'
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
                filename_shizuoka = '/content/dataset/train_valid/'+ h +'/validation/all/MRIT2画像_静岡_' + i_new_shizuoka + '_' + j_new + '.tif'
                is_file_shizuoka = os.path.isfile(filename_shizuoka)
                if is_file_shizuoka:
                    i_new_shizuoka = str(int(i_new_shizuoka)+132)
                    ins.append(i_new_shizuoka+'_'+j_new)
                    i_new_shizuoka = str(int(i_new_shizuoka) - 132).zfill(3)
            if ins != []:
                shizuokalist.append(ins)
                lenshizuoka.append(len(ins))

    return ehimelist, lenehime, shizuokalist, lenshizuoka

def Cut(img, w, x, y, z):
    img_cut = img[x:x+z,w:w+y]

def Bright(img):
    gamma = random.uniform(1.1,1.4)
    LUT_Table=np.zeros((256,1),dtype='uint8')
    for lut in range(len(LUT_Table)):
        LUT_Table[lut][0]=255*(float(lut)/255)**(1.0/gamma)
        img_bright = cv2.LUT(img,LUT_Table)
        return img_bright

def Dark(img):
    gamma = random.uniform(0.75,0.9)
    LUT_Table=np.zeros((256,1),dtype='uint8')
    for lut in range(len(LUT_Table)):
        LUT_Table[lut][0]=255*(float(lut)/255)**(1.0/gamma)
        img_dark = cv2.LUT(img,LUT_Table)
        return img_dark

def Noise(img):
    sig=random.uniform(20,40)
    noise_gaussian=np.random.normal(0,sig,np.shape(img))
    img_noise=img+np.floor(noise_gaussian) #画像にノイズを付加
    img_noise[img_noise>255]=255 # 255超える場合は255
    img_noise[img_noise<0]=0 # 255超える場合は255
    img_noise=img_noise.astype(np.uint8)
    return img_noise

def Ave(img_noise):
    img_ave = cv2.blur(img_noise,(5,5))
    return img_ave

def Bilat(img_noise):
    img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2LAB)
    img_bilat = cv2.bilateralFilter(img_noise, 15, sigmaColor=50, sigmaSpace=20)
    img_bilat = cv2.cvtColor(img_bilat, cv2.COLOR_LAB2BGR)
    return img_bilat

def Gaussian(img_noise):
    img_gaussian = cv2.GaussianBlur(img_noise,(11,11),0)
    return img_gaussian

def Median(img_noise):
    img_median = cv2.medianBlur(img_noise,11)
    return img_median

def ShearRange():
    listab = []
    listcd = []
    c = random.uniform(-3.5,-7.5)
    d = random.uniform(3.5,7.5)
    listab.append(c)
    listab.append(d)
    e = random.uniform(-3.5,-7.5)
    f = random.uniform(3.5,7.5)
    listcd.append(e)
    listcd.append(f)
    theta1 = np.deg2rad(random.choice(listab))
    theta2 = np.deg2rad(random.choice(listcd))

    return theta1, theta2

def ShiftRange():
    listab = []
    c = random.randint(-20,-10)
    d = random.randint(10,20)
    listab.append(c)
    listab.append(d)
    x_diff = random.choice(listab)
    listcd = []
    e = random.randint(-20,-10)
    f = random.randint(10,20)
    listcd.append(e)
    listcd.append(f)
    y_diff = random.choice(listcd)

    return x_diff, y_diff

def RoundRange():
    listab = []
    c = random.uniform(-10.0,-30.0)
    d = random.uniform(10.0,30.0)
    listab.append(c)
    listab.append(d)
    #回転角を指定
    angle = random.choice(listab)

    return angle

def Xshear(img):
    height, width, _ = img.shape
    theta_x1, theta_x2 = ShearRange()
    mat = np.float32([[1, np.tan(theta_x1), 0], [np.tan(theta_x2), 1, 0]])
    img_xshear = cv2.warpAffine(img, mat, (width, height))
    return img_xshear

def Yshear(img):
    height, width, _ = img.shape
    theta_y1, theta_y2 = ShearRange()
    mat = np.float32([[1, np.tan(theta_y1), 0], [np.tan(theta_y2), 1, 0]])
    img_yshear = cv2.warpAffine(img, mat, (width, height))
    return img_yshear

def Shift(img):
    height, width, _ = img.shape

    x_diff, y_diff = ShiftRange()

    # 画像をシフト
    img_shift = np.roll(img, shift=(y_diff, x_diff), axis=(0, 1))
    
    if x_diff >= 0:
        img_shift[:, :x_diff] = 0
    else:
        img_shift[:, width + x_diff:] = 0

    if y_diff >= 0:
        img_shift[:y_diff] = 0
    else:
        img_shift[height + y_diff:] = 0
    
    return img_shift


def Round(img, w, x, y, z):
    height, width, _ = img.shape
    center = ((int(w)+int(y/2)),(int(x)+int(z)))
    angle = RoundRange()
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, angle , scale)
    #アフィン変換
    img_round = cv2.warpAffine(img, trans, (width,height))
    return img_round

def HighPass(img, range):

    dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)  # type: ignore
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2

    mask = np.ones((rows,cols,2),np.uint8)

    mask[crow-range:crow+range, ccol-range:ccol+range] = 0

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    dst = np.clip(img_back[:,:,0],0,255).astype(np.uint8)

    return dst

def LowPass(img, range):

    dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE)  # type: ignore
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow,ccol = rows//2 , cols//2

    mask = np.ones((rows,cols,2),np.uint8)

    mask[crow-range:crow+range, ccol-range:ccol+range] = 1

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    dst = np.clip(img_back[:,:,0],0,255).astype(np.uint8)

    return dst

def Write(name, img):
    imwrite(name, img)

def bright(img, w, x, y, z, type, i_new, j_new, place):
    img_bright = Bright(img)
    img_bright = Cut(img_bright, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_bright_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_bright)
    for r in range(5):
        r_new = str(r+1)
        img_bright = Bright(img)
        img_bright = Cut(img_bright, w, x, y, z)
        for way in evenway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_bright_" + i_new + "_" + j_new+ "_" + r_new+ ".tif"
            Write(name, img_bright)    

def dark(img, w, x, y, z, type, i_new, j_new, place):
    img_dark = Dark(img)
    img_dark = Cut(img_dark, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_dark_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_dark)
    for r in range(5):
        r_new = str(r+1)
        img_dark = Dark(img)
        img_dark = Cut(img_dark, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_dark_" + i_new + "_" + j_new+ "_" + r_new+ ".tif"
            Write(name, img_dark)

def ave(img, w, x, y, z, type, i_new, j_new, place):
    img_ave = Ave(img)
    img_ave = Cut(img_ave, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_ave_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_ave)
    for r in range(5):
        r_new = str(r+1)
        img_ave = Ave(img)
        img_ave = Cut(img_ave, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_ave_" + i_new + "_" + j_new+  "_" + r_new+ ".tif"
            Write(name, img_ave)

def bilat(img, w, x, y, z, type, i_new, j_new, place):
    img_bilat = Bilat(img)
    img_bilat = Cut(img_bilat, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_bilat_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_bilat)
    for r in range(5):
        r_new = str(r+1)
        img_bilat = Bilat(img)
        img_bilat = Cut(img_bilat, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_bilat_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_bilat)           

def gaus(img, w, x, y, z, type, i_new, j_new, place):
    img_gaus = Gaussian(img)
    img_gaus = Cut(img_gaus, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_gaus_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_gaus)
    for r in range(5):
        r_new = str(r+1)
        img_gaus = Gaussian(img)
        img_gaus = Cut(img_gaus, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_gaus_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_gaus) 
        
def median(img, w, x, y, z, type, i_new, j_new, place):
    img_median = Median(img)
    img_median = Cut(img_median, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_median_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_median)
    for r in range(5):
        r_new = str(r+1)
        img_median = Median(img)
        img_median = Cut(img_median, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_median_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_median) 

def noise(img, w, x, y, z, type, i_new, j_new, place):
    img_noise = Noise(img)
    img_noise_cut = Cut(img_noise, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_noise_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_noise)
    for r in range(5):
        r_new = str(r+1)
        img_noise = Noise(img)
        img_noise_cut = Cut(img_noise, w, x, y, z)         
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_noise_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_noise)                
    return img_noise


def xshear(img, w, x, y, z, type, i_new, j_new, place):
    img_xshear = Xshear(img, type)
    img_xshear = Cut(img_xshear, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_xshear_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_xshear)
    for r in range(5):
        r_new = str(r+1)
        img_xshear = Xshear(img, type)
        img_xshear = Cut(img_xshear, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_xshear_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_xshear)                

def yshear(img, w, x, y, z, type, i_new, j_new, place):
    img_yshear = Yshear(img)
    img_yshear = Cut(img_yshear, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_yshear_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_yshear)
    for r in range(5):
        r_new = str(r+1)
        img_yshear = Yshear(img)
        img_yshear = Cut(img_yshear, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_yshear_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_yshear)        

def shift(img, w, x, y, z, type, i_new, j_new, place):
    img_shift = Shift(img)
    img_shift = Cut(img_shift, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_shift_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_shift)
    for r in range(5):
        r_new = str(r+1)
        img_shift = Shift(img)
        img_shift = Cut(img_shift, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_shift_" + i_new + "_" + j_new+   "_" + r_new+ ".tif" 
            Write(name, img_shift)  

def round(img, w, x, y, z, type, i_new, j_new, place):
    img_round = Round(img, w, x, y, z)
    img_round = Cut(img_round, w, x, y, z)
    for way in evenway:
        name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_round_" + i_new + "_" + j_new+ ".tif"
        Write(name, img_round)
    for r in range(5):
        r_new = str(r+1)
        img_round = Round(img, w, x, y, z)
        img_round = Cut(img_round, w, x, y, z)
        for way in oddway:
            name = "/content/dataset/new/train/" + way + "/all/" + type + "/MRIT2画像_" + place + "_round_" + i_new + "_" + j_new +  "_" + r_new + ".tif" 
            Write(name, img_round)  



def highpass(img, w, x, y, z, type, i_new, j_new, place):
    range = 10
    img_high = HighPass(img, range)
    img_high = Cut(img_high, w, x, y, z)
    name = "/content/dataset/new/train/HighPass/all/" + type + "/MRIT2画像_" + place + "_high_" + i_new + "_" + j_new+ ".tif"
    Write(name, img_high)
    for r in range(5):
        r_new = str(r+1)
        name = "/content/dataset/new/train/HighPass/all/" + type + "/MRIT2画像_" + place + "_high_" + i_new + "_" + j_new+ "_" + r_new +  ".tif"
        Write(name, img_high)

def lowpass(img, w, x, y, z, type, i_new, j_new, place):
    range = 10
    img_low = LowPass(img, range)
    img_low = Cut(img_low, w, x, y, z)
    name = "/content/dataset/new/train/LowPass/all/" + type + "/MRIT2画像_" + place + "_low_" + i_new + "_" + j_new+ ".tif"
    Write(name, img_low)
    for r in range(5):
        r_new = str(r+1)
        name = "/content/dataset/new/train/LowPass/all/" + type + "/MRIT2画像_" + place + "_low_" + i_new + "_" + j_new+ "_" + r_new +  ".tif"
        Write(name, img_low)


def function(img, w, x, y, z, type, i_new, j_new, place):
    bright(img, w, x, y, z, type, i_new, j_new, place)
    dark(img, w, x, y, z, type, i_new, j_new, place)
    img_noise = noise(img, w, x, y, z, type, i_new, j_new, place)
    ave(img_noise, w, x, y, z, type, i_new, j_new)
    bilat(img_noise, w, x, y, z, type, i_new, j_new)
    gaus(img_noise, w, x, y, z, type, i_new, j_new)
    median(img_noise, w, x, y, z, type, i_new, j_new)
    xshear(img, w, x, y, z, type, i_new, j_new, place)
    yshear(img, w, x, y, z, type, i_new, j_new, place)
    shift(img, w, x, y, z, type, i_new, j_new, place)
    round(img, w, x, y, z, type, i_new, j_new, place)
    highpass()
    lowpass()

def Flag(n, j):
    if n == 1:
        if j == 0 or j == 1 or j == 2 or j == 3 or j == 4:
            j_flag = 1
    if n == 2:
        if j == 0 or j == 2 or j == 3:
            j_flag = 1
        if j == 1 or j == 4:
            j_flag = 2
    if n == 3:
        if j == 0 or j == 3:
            j_flag = 1
        if j == 1 or j == 4:
            j_flag = 2
        if j == 2:
            j_flag = j + 1
    if n == 4:
        if j == 0 or j == 4:
            j_flag = 1
        else:
            j_flag = j + 1
    if n == 5:
        j_flag = j + 1

    return j_flag


def EhimeInfo(ehimelist, i):
    x = ehimelist[i]
    x1 = x[0].split('_')
    for count in range(132):
        im = str(count+2)
        fd = 'A' + im
        fd_new = ws_ehime[fd].value
        if fd_new == int(x1[0]):
            t = 'S' + im
            type = str(ws_ehime[t].value)
            break
    a = 'A' + im
    a_ehime = str(ws_ehime[a].value)
    i_new_ehime = a_ehime.zfill(3)
    return i_new_ehime, im

def CheckFlagEhime(j_flag, im):
    if j_flag == 1:
        w = 'W'+im
        x = 'X'+im
        y = 'Y'+im
        z = 'Z'+im
    if j_flag == 2:
        w = 'AA'+im
        x = 'AB'+im
        y = 'AC'+im
        z = 'AD'+im
    if j_flag == 3:
        w = 'AE'+im
        x = 'AF'+im
        y = 'AG'+im
        z = 'AH'+im
    if j_flag == 4:
        w = 'AI'+im
        x = 'AJ'+im
        y = 'AK'+im
        z = 'AL'+im
    if j_flag == 5:
        w = 'AM'+im
        x = 'AN'+im
        y = 'AO'+im
        z = 'AP'+im

    w = ws_ehime[w].value
    x = ws_ehime[x].value
    y = ws_ehime[y].value
    z = ws_ehime[z].value
    return w, x, y, z

def ShizuokaInfo(shizuokalist, i):
    x = shizuokalist[i]
    x1 = x[0].split('_')
    ye = int(x1[0])-132
    for count in range(132):
        im = str(count+2)
        fd = 'A' + im
        fd_new = ws_shizuoka[fd].value
        if fd_new == ye:
            t = 'S' + im
            type = str(ws_shizuoka[t].value)
            break
    a = 'A' + im
    a_shizuoka = str(ws_shizuoka[a].value)
    i_new_shizuoka = a_shizuoka.zfill(3)

    return i_new_shizuoka, im

def CheckFlagShizuoka(j_flag, im):
    if j_flag == 1:
        w = 'U'+im
        x = 'V'+im
        y = 'W'+im
        z = 'X'+im
    if j_flag == 2:
        w = 'Y'+im
        x = 'Z'+im
        y = 'AA'+im
        z = 'AB'+im
    if j_flag == 3:
        w = 'AC'+im
        x = 'AD'+im
        y = 'AE'+im
        z = 'AF'+im
    if j_flag == 4:
        w = 'AG'+im
        x = 'AH'+im
        y = 'AI'+im
        z = 'AJ'+im
    if j_flag == 5:
        w = 'AK'+im
        x = 'AL'+im
        y = 'AM'+im
        z = 'AN'+im
    
    w = ws_shizuoka[w].value
    x = ws_shizuoka[x].value
    y = ws_shizuoka[y].value
    z = ws_shizuoka[z].value

def Ehime(h, im, j_flag, w, x, y, z, ehimelist, lenehime):
    for i in range(lenehime):
        i_new_ehime, im, type = EhimeInfo(ehimelist, i)

        for j in range(5):
            path = "/content/dataset/train_valid/" + h + "/datalist/" # パスを格納
            j_new = str(j+1)
            name = 'MRIT2画像_愛媛_'+i_new_ehime+'_'+j_new+'.tif'
            name = path+name
            is_file = os.path.isfile(name)
            if is_file:
                img = io.imread(name) # これでOK
                n = lenehime[i]

                j_flag = Flag(n, j)

                w, x, y, z = CheckFlagEhime(j_flag, im)


                function(img, w, x, y, z, type, i_new_ehime, j_new, "愛媛")


def Shizuoka(h, im, j_flag, w, x, y, z, shizuokalist, lenshizuoka):
    for i in range(lenshizuoka):
        i_new_shizuoka, im, type = ShizuokaInfo(shizuokalist, i)

        for j in range(5):
            path = "/content/dataset/train_valid/" + h + "/datalist/" # パスを格納
            j_new = str(j+1)
            name = 'MRIT2画像_静岡_'+i_new_shizuoka+'_'+j_new+'.tif'
            name = path+name
            is_file = os.path.isfile(name)
            if is_file:
                img = io.imread(name) # これでOK
                n = lenshizuoka[i]

                j_flag = Flag(n, j)

                w, x, y, z = CheckFlagShizuoka(j_flag, im)

                function(img, w, x, y, z, type, i_new_shizuoka, j_new, "静岡")


def main():
    for h in dir_list:
        ehimelist, lenehime, shizuokalist, lenshizuoka = MakeList(h)
        Ehime(h, "", 0, 0, 0, 0, 0, ehimelist, lenehime)
        Shizuoka(h, "", 0, 0, 0, 0, 0, shizuokalist, lenshizuoka)

if __name__ == "__main__":
    main()