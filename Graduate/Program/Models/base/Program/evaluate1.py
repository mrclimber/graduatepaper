from tensorflow.python.keras.models import load_model
import os
from unicodedata import name

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.applications.vgg16 import VGG16

from keras_preprocessing.image import load_img,img_to_array


import numpy as np
from keras.applications.vgg16 import preprocess_input

import openpyxl

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# matplotlib inline
from sklearn.metrics import roc_curve,roc_auc_score
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
from skimage import io

classes = ['good','bad']
nb_classes = len(classes)
batch_size_for_data_generator = 20

img_rows, img_cols = 128, 128

base_dir = "/media/masaki/627D-A8B0/graduatepaper/data/model"

train_dir = os.path.join(base_dir, 'new/train/base/2345/all') # ここ
validation_dir = os.path.join(base_dir, 'new/validation/1') # ここ
test_dir = os.path.join(base_dir, 'new/test_new') # ここ

train_good_dir = os.path.join(train_dir, 'good')
train_bad_dir = os.path.join(train_dir, 'bad')

validation_good_dir = os.path.join(validation_dir, 'good')
validation_bad_dir = os.path.join(validation_dir, 'bad')

test_good_dir = os.path.join(test_dir, 'good')
test_bad_dir = os.path.join(test_dir, 'bad')

img_rows, img_cols = 128, 128

print('total training good images:', len(os.listdir(train_good_dir)),train_good_dir)
print('total training bad images:', len(os.listdir(train_bad_dir)),train_bad_dir)

print('total validation good images:', len(os.listdir(validation_good_dir)),validation_good_dir)
print('total validation bad images:', len(os.listdir(validation_bad_dir)),validation_bad_dir)

print('total test good images:', len(os.listdir(test_good_dir)),test_good_dir)
print('total test bad images:', len(os.listdir(test_bad_dir)),test_bad_dir)


test_dir = os.path.join(base_dir, 'new/test_new')

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

model = load_model('/media/masaki/627D-A8B0/graduatepaper/data/Models/base/modellist/2345/acc/weights.45-0.22-0.90-0.24-0.90.hdf5')
# model.summary()

f = open('/media/masaki/627D-A8B0/graduatepaper/data/Models/base/result/2345/acc/resultSample.txt', 'w', encoding='UTF-8')

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='binary',
    batch_size=batch_size_for_data_generator)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)

# 識別問題のaccuracy
print('test acc:', test_acc)

f.write('test acc:')
f.write(str(test_acc))
f.write('\n')


filename = os.path.join(test_dir, 'all')

book_ehime = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_愛媛_20210830.xlsx')
ws_ehime = book_ehime.worksheets[0]

book_shizuoka = openpyxl.load_workbook('/media/masaki/627D-A8B0/graduatepaper/data/耳下腺情報シート_静岡_20210710.xlsx')
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
            filename_ehime = '/media/masaki/627D-A8B0/graduatepaper/data/model/new/alltest_new/MRIT2画像_愛媛_test_' + i_new_ehime + '_' + j_new + '.tif'
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
            filename_shizuoka = '/media/masaki/627D-A8B0/graduatepaper/data/model/new/alltest_new/MRIT2画像_静岡_test_' + i_new_shizuoka + '_' + j_new + '.tif'
            is_file_shizuoka = os.path.isfile(filename_shizuoka)
            if is_file_shizuoka:
                i_new_shizuoka = str(int(i_new_shizuoka)+132)
                ins.append(i_new_shizuoka+'_'+j_new)
                i_new_shizuoka = str(int(i_new_shizuoka) - 132).zfill(3)
        if ins != []:
            shizuokalist.append(ins)
            lenshizuoka.append(len(ins))

y_true = []
y_pred = []
y_true_ehime = []
y_pred_ehime = []
y_true_shizuoka = []
y_pred_shizuoka = []
y_new_pred_ehime = []
y_new_pred_shizuoka = []

i = 0
im=""
for length in lenehime:
    x = ehimelist[i]
    x1 = x[0].split('_')
    # print(x1)
    for count in range(132):
        im = str(count+2)
        fd = 'A' + im
        fd_new = ws_ehime[fd].value
        if fd_new == int(x1[0]):
            t = 'S' + im
            type = str(ws_ehime[t].value)
            break
    a = 'A' + im
    b = 'B' + im
    a_ehime = str(ws_ehime[a].value)
    b_ehime = str(ws_ehime[b].value)
    i_new_ehime = a_ehime.zfill(3)

    if type == "悪性":
        t_ehime = 1
    else:
        t_ehime = 0

    # print(type)
    y_ins_ehime = []
    flag_ehime = 0

    for j in range(5):
        path = "/media/masaki/627D-A8B0/graduatepaper/data/model/new/alltest_new/" # パスを格納
        j_new = str(j+1)
        name = 'MRIT2画像_愛媛_test_'+i_new_ehime+'_'+j_new+'.tif'
        name = path+name
        is_file = os.path.isfile(name)
        if is_file:
            flag_ehime = 1
            img = load_img(name, target_size=(img_rows, img_cols))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            predict = model.predict(preprocess_input(x))
            print(predict)
            if predict < 0.5:
                predictions = 0
            else:
                predictions = 1
            y_ins_ehime.append(predictions)
    if flag_ehime:
        y_pred_ehime.append(y_ins_ehime)
        y_true_ehime.append(t_ehime)
        # y_pred.append(y_ins_ehime)
        y_true.append(t_ehime)
        # print("a")
    i += 1
i = 0
for length in lenshizuoka:
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
    b = 'B' + im
    a_shizuoka = str(ws_shizuoka[a].value)
    b_shizuoka = str(ws_shizuoka[b].value)
    i_new_shizuoka = a_shizuoka.zfill(3)

    if type == "悪性":
        t_shizuoka = 1
    else:
        t_shizuoka = 0

    y_ins_shizuoka = []
    flag_shizuoka = 0

    for j in range(5):
        path = "/media/masaki/627D-A8B0/graduatepaper/data/model/new/alltest_new/" # パスを格納
        j_new = str(j+1)
        name = 'MRIT2画像_静岡_test_'+i_new_shizuoka+'_'+j_new+'.tif'
        name = path+name
        is_file = os.path.isfile(name)
        if is_file:
            flag_shizuoka = 1
            img = load_img(name, target_size=(img_rows, img_cols))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            predict = model.predict(preprocess_input(x))
            if predict < 0.5:
                predictions = 0
            else:
                predictions = 1
            y_ins_shizuoka.append(predictions)
    if flag_shizuoka:
        y_pred_shizuoka.append(y_ins_shizuoka)
        y_true_shizuoka.append(t_shizuoka)
        # y_pred.append(y_ins_shizuoka)
        y_true.append(t_shizuoka)
        # print("b")
    i += 1

# print(y_pred_ehime)
# print(y_true_ehime)

for i in range(len(y_pred_ehime)):
    length = len(y_pred_ehime[i])
    if length % 2:
        length_1 = length + 1
    else:
        length_1 = length
    halflength = round(length_1/2)
    bad = 0
    for j in range(length):
        if y_pred_ehime[i][j] == 1:
            bad += 1
    if halflength <= bad:
        y_pred.append(1)
        y_new_pred_ehime.append(1)
    else:
        y_pred.append(0)
        y_new_pred_ehime.append(0)
for i in range(len(y_pred_shizuoka)):
    length = len(y_pred_shizuoka[i])
    if length % 2:
        length_1 = length + 1
    else:
        length_1 = length
    halflength = round(length_1/2)
    bad = 0
    for j in range(length):
        if y_pred_shizuoka[i][j] == 1:
            bad += 1
    if halflength <= bad:
        y_pred.append(1)
        y_new_pred_shizuoka.append(1)
    else:
        y_pred.append(0)
        y_new_pred_shizuoka.append(0)

print(y_pred_shizuoka)
print(y_true_shizuoka)

print("y_new_pred_ehime:",y_new_pred_ehime)
print("y_true_ehime:",y_true_ehime)
print("y_new_pred_shizuoka:",y_new_pred_shizuoka)
print("y_true_shizuoka:",y_true_shizuoka)
print("y_pred:",y_pred)
print("y_true:",y_true)


cm = confusion_matrix(y_true,y_pred)
print(cm)


tn, fp, fn, tp = cm.flatten()

fpr, tpr, thresholds = roc_curve(y_true, y_pred)

f.write('tn:')
f.write(str(tn))
f.write('\n')

f.write('fp:')
f.write(str(fp))
f.write('\n')

f.write('fn:')
f.write(str(fn))
f.write('\n')

f.write('tp:')
f.write(str(tp))
f.write('\n')

plt.clf()

sns.heatmap(cm)
plt.savefig('/media/masaki/627D-A8B0/graduatepaper/data/Models/base/result/2345/acc/sklearn_confusion_matrix_sample.png')

plt.clf()

plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig('/media/masaki/627D-A8B0/graduatepaper/data/Models/base/result/2345/acc/sklearn_roc_curve_sample.png')

# accuracy=(TP+TN)/(TP+TN+FP+FN) すべてのサンプルのうちの正解したサンプルの割合
print("正解率",accuracy_score(y_true, y_pred))
f.write('正解率:')
f.write(str(accuracy_score(y_true, y_pred)))
f.write('\n')

# precision=(TP)/(TP+FP) 陽性と予測されたサンプルのうちの正解したサンプルの割合
print("適合率",precision_score(y_true, y_pred,pos_label=0))
f.write('適合率:')
f.write(str(precision_score(y_true, y_pred,pos_label=0)))
f.write('\n')

# recall=(TP)/(TP+FN) 実際に要請のサンプルのうちの正解したサンプルの割合
print("再現率",recall_score(y_true, y_pred,pos_label=0))
f.write('再現率:')
f.write(str(recall_score(y_true, y_pred,pos_label=0)))
f.write('\n')

# F1-measure=(2*precision*recall)/(precision+recall)=(2*TP)/(2*TP+FP+FN) 適合率と再現率の調和平均
print("F1値",f1_score(y_true, y_pred,pos_label=0))
f.write('F1率:')
f.write(str(f1_score(y_true, y_pred,pos_label=0)))
f.write('\n')

print("fpr:",fpr)
f.write('fpr:')
f.write(str(fpr))
f.write('\n')

print("tpr:",tpr)
f.write('tpr:')
f.write(str(tpr))
f.write('\n')

print("thresholds:",thresholds)
f.write('thresholds:')
f.write(str(thresholds))
f.write('\n')

print("roc_auc_score:",roc_auc_score(y_true,y_pred))

f.write('roc_auc_acore:')
f.write(str(roc_auc_score(y_true,y_pred)))
f.write('\n')

f.close()

y_true = str(y_true)
y_pred = str(y_pred)

with open('/media/masaki/627D-A8B0/graduatepaper/data/Models/base/result/2345/acc/testSample.txt', 'w') as f:
    f.write('true:')
    f.write(y_true)
    f.write('\n')
    f.write('pred:')
    f.write(y_pred)


f.close()