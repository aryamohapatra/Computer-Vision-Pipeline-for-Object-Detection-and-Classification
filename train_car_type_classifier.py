import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
#import numpy as np
#import pandas as pd
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#import os
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt

def create_model():
    image_size = 160
    img_shape = (image_size, image_size, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                  include_top=False,
                                                  weights='imagenet')
    base_model.trainable = False #Freezing the base model
    model = tf.keras.Sequential([
                              base_model,
                              Flatten(),
                              Dense(1024,activation='relu'),
                              Dropout(0.2),
                              Dense(1024,activation='relu'),
                              Dropout(0.2), 
                              Dense(512,activation='relu'),
                              Dropout(0.2), 
                              keras.layers.Dense(1, activation='sigmoid')])
    model.summary()
    
    model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.01),,
              metrics=['acc'])
    return model
    
def classfication():

    #Compiling our model
    model = create_model()
    model.summary()
    train_data = ImageDataGenerator(rescale=1/255,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.4,
                                    fill_mode='nearest')
    
    train_generator = train_data.flow_from_directory(
            'training_data/Train',
            target_size=(160, 160),  #Shaping the train image to requisite size
            batch_size=128,
            class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1/255)
    validation_generator = validation_datagen.flow_from_directory(
            'training_data/Validation',
            target_size=(160,160),
            batch_size=128,
            class_mode='binary')
    
    history = model.fit_generator(
                      train_generator,
                      steps_per_epoch=8,  
                      epochs=17,
                      validation_data=validation_generator,
                      validation_steps=8,
                      verbose=1)

    return model, history

if __name__=='__main__':
    model, history = classfication()
    model.save('mobilenet_cars.h5')
    
    # Plot Learning Curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([0.4,1.1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,2.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()