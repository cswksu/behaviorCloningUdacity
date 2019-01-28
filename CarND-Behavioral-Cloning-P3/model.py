# -*- coding: utf-8 -*-
"""
Udacity SDCND Behavioral Cloning Project
Christian Wiles
"""

import numpy as np #needed for np array creation
import csv #needed to read in CSV from simulator
from keras.models import Sequential #to create sequential model
import keras.layers as kl #import all keras layers
from PIL import Image #image processing to mirror drive.py


lines=[] #store lines from csv file

#open csv file, read in each line
with open('../data/realData1/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    
    
images=[] #store all images in this array
angle=[] #store all steering angles in array
rZoom=224.0/70.0 #needed to stretch image vertically to 244 px
hZoom=224.0/320.0 #shrink image horizontally to 244 px
ang_offset=2.0 #angle correction for left and right cameras

for line in lines: #for each line of data from CSV
    source_path=line[0] #path of center image
    left_source_path=line[1] #path of left image
    right_source_path=line[2] #path of right image
    image = Image.open(source_path) #open center image
    im = np.asarray(image) #turn into np array
    image_l=Image.open(left_source_path) #open left image
    im_l=np.asarray(image_l) #turn into array
    image_r=Image.open(right_source_path) #open right image
    im_r=np.asarray(image_r) #turn into array
    angleNorm=float(line[3])/25.0 #normalize steering angle
    
    #add offset to left and right pictures
    angleNormL=angleNorm+ang_offset/25.0
    angleNormR=angleNorm-ang_offset/25.0
    
    angle.append(angleNorm) #append center angle to list
    angle.append(-angleNorm) #append inverse of center angle to list
    angle.append(angleNormL) #append left angle to list
    angle.append(angleNormR) #append right angle to list
    
    images.append(im) #apend center image to list
    images.append(np.flip(im,axis=1)) #flip center image, add to list
    images.append(im_l) #add left image
    images.append(im_r) #add right image
    
x_train=np.array(images) #create array of training data images
y_train=np.array(angle) #create array of training output

model=Sequential() #create new model (similar to NVIDIA model presented)

#add processing layer to center and scale pixel data
model.add(kl.Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(kl.Cropping2D(cropping=((70,20),(0,0)))) #crop image

#begin NVIDIA network architecture, add dropout to reduce overfitting
model.add(kl.Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(kl.Dropout(0.25))
model.add(kl.Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(kl.Dropout(0.25))
model.add(kl.Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(kl.Dropout(0.25))
model.add(kl.Convolution2D(64,3,5,activation="relu"))
model.add(kl.Dropout(0.25))
model.add(kl.Convolution2D(64,3,5,activation="relu"))
model.add(kl.Dropout(0.25))
model.add(kl.Flatten())
model.add(kl.Dense(100))
model.add(kl.Dense(50))
model.add(kl.Dense(1)) #1 layer output for steering

model.compile('adam','mean_squared_error',['accuracy']) #use MSE as loss
model.fit(x_train, y_train, epochs=15, validation_split=0.3)
#train, using 30% of shuffled data as validation data for 15 epochs

model.save('model.h5') #save model for drive.py use