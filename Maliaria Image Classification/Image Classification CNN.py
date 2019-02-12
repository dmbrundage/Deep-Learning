# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:46:39 2019

@author: dmbru
"""

from keras.layers import Dense, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy as np
from keras.layers.advanced_activations import LeakyReLU

def get_size_statistics(DIR):
  heights = []
  widths = []
  for img in os.listdir(DIR): 
    path = os.path.join(DIR, img)
    data = np.array(Image.open(path)) #PIL Image library
    heights.append(data.shape[0])
    widths.append(data.shape[1])
  avg_height = sum(heights) / len(heights)
  avg_width = sum(widths) / len(widths)
  print("Average Height: " + str(avg_height))
  print("Max Height: " + str(max(heights)))
  print("Min Height: " + str(min(heights)))
  print('\n')
  print("Average Width: " + str(avg_width))
  print("Max Width: " + str(max(widths)))
  print("Min Width: " + str(min(widths)))

#fit the data to our mode
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))#2 represent output layer neurons 
model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('desktop\\cell_images\\Train', target_size = (50, 50), batch_size = 100, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('desktop\\cell_images\\Test', target_size = (50, 50), batch_size = 100, class_mode = 'binary')

#fit the data to our mode
model.fit_generator(training_set, steps_per_epoch = 193, epochs = 25, validation_data = test_set, validation_steps = 41,
                     workers = 4, verbose =1)
#use_multiprocessing = True, workers = 4

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Desktop\\CNN New\\Parasite\\C37BP2_thinF_IMG_20150620_132847a_cell_76.png', target_size = (50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
    prediction = 'Parasitized'
else:
    prediction = 'Uninfected'


print(prediction)