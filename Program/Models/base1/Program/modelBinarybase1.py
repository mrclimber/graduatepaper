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
from tensorflow.keras import optimizers
import time

f = open('/home/masaki/graduatepaper/data/Models/base1/result/2345/result.txt', 'w', encoding='UTF-8') # ここ

classes = ['good','bad']
nb_classes = len(classes)
batch_size_for_data_generator = 20

base_dir = "/home/masaki/graduatepaper/data/model"

train_dir = os.path.join(base_dir, 'new/train/base1/2345/select') # ここ
validation_dir = os.path.join(base_dir, 'new/validation/1') # ここ
test_dir = os.path.join(base_dir, 'new/test') # ここ

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

# rescale (正規化)必要
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='binary',
    batch_size=batch_size_for_data_generator,
    shuffle=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    classes=classes,
    class_mode='binary',
    batch_size=batch_size_for_data_generator,
    shuffle=True)

input_tensor = Input(shape=(img_rows, img_cols, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
vgg16.summary()

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
model.summary()

# 要調整
vgg16.trainable = True
set_trainable = False
for layer in vgg16.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        
for layer in vgg16.layers:
    print(layer, layer.trainable )
    
print("-------------------------------------------------")

for layer in model.layers:
    print(layer, layer.trainable )

start = time.perf_counter()

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

EPOCHS = 100
STEPS = 25
fpath_acc = '/home/masaki/graduatepaper/data/Models/base1/modellist/2345/acc/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
cp_cb = ModelCheckpoint(
    filepath=fpath_acc,
    monitor='val_acc', 
    verbose=1, 
    save_best_only=True, 
    mode='auto')
fpath_loss = '/home/masaki/graduatepaper/data/Models/base1/modellist/2345/loss/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
cs_cb = ModelCheckpoint(
    filepath=fpath_loss,
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='auto')


history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=5, # ここ
    verbose=1,
    callbacks=[cp_cb,cs_cb]
    )

end = time.perf_counter()

elapse = end - start

print("elapse:",elapse)

f.write('elapse:')
f.write(str(elapse))
f.write('\n')

hdf5_file = os.path.join(base_dir, 'model.h5')
model.save_weights(hdf5_file)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(history.epoch) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('/home/masaki/graduatepaper/data/Models/base1/result/2345/accuracy.png') #ここ

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('/home/masaki/graduatepaper/data/Models/base1/result/2345/loss.png') # ここ


f.close()