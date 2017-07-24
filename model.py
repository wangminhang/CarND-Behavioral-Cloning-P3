# import necessary library
import csv
import cv2
import sklearn
import random
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D,MaxPooling2D
from sklearn.model_selection import train_test_split

# 
file_dic = "../P3_data/data/"
samples = []
with open(file_dic+"driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def generator_center_camera(samples, batch_size = 32):
    file_path = "../P3_data/data/IMG/"
    num_samples = len(samples)
    while 1: # always run
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = file_path+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def generator_multi_camera(samples, batch_size=32):
    file_path = "../P3_data/data/IMG/"
    num_samples = len(samples)
    correction = 0.2
    while 1: # always run
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                flipped_center_angle = (center_angle) * -1.0
                for i in range(3):
                    name = file_path+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    images.append(image)
                    flipped_image = cv2.flip(image, 1)
                    images.append(flipped_image)
                    if i == 0:
                        angle = center_angle
                        flipped_angle = flipped_center_angle
                    elif i == 1:
                        angle = center_angle + correction
                        flipped_angle = flipped_center_angle + correction
                    else:
                        angle = center_angle - correction
                        flipped_angle = flipped_center_angle - correction
                    angles.append(angle)
                    angles.append(flipped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
if 1: #0:only one camera; 1:three cameras
    train_generator = generator_multi_camera(train_samples, batch_size = 32)
    validation_generator = generator_multi_camera(validation_samples, batch_size = 32)
else:
    train_generator = generator_center_camera(train_samples, batch_size = 32)
    validation_generator = generator_center_camera(validation_samples, batch_size = 32)

ch, row, col = 3, 160, 320  #image format

model = Sequential()
# preprocess input images 
model.add(Lambda(lambda x: x/255 - 0.5,
        input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,20),(10,10))))

# NVIDIA Dave2
model.add(Convolution2D(16, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*4, validation_data=validation_generator, nb_val_samples=len(validation_samples)*4, nb_epoch=6)

# save model 
model.save('model.h5')
