
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES = 4
IMAGE_RESIZE = 224
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 10 #10
BATCH_SIZE = 32
lr = 0.0001


MODEL_POOLING = 'avg' # None
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy' #categorical_crossentropy
LOSS_METRICS = ['accuracy']

trImgNum = 34848
valImgNum = 5288

STEPS_PER_EPOCH_TRAINING = trImgNum//BATCH_SIZE
STEPS_PER_EPOCH_VALIDATION = valImgNum//BATCH_SIZE

from keras.applications.resnet50 import preprocess_input

# from tensorflow.keras.applications.efficientnet import preprocess_input
# from keras.applications.inception_v3 import preprocess_input
# from keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
image_size = IMAGE_RESIZE
train_data_generator = ImageDataGenerator(rescale= 1./255,
                                          rotation_range=90,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2)

validation_data_generator = ImageDataGenerator(rescale= 1./255)

tr_dir = "D:/json_train"
train_generator = train_data_generator.flow_from_directory(
        tr_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical') #categorical

val_dir = "D:/json_val"
validation_generator = validation_data_generator.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

#보라색 없애는 코드
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os

# filenames = os.listdir('D:/json_train/[원천]탈모_3.중증')
# for f in filenames:
#     img = image.load_img('D:/json_train/[원천]탈모_3.중증/'+f)
#     img_array = image.img_to_array(img)
#     img_array = img_array.reshape((1,) + img_array.shape)
    
#     datagen = ImageDataGenerator(rescale= 1./255,
#                                  rotation_range=90,
#                                  horizontal_flip=True,
#                                  vertical_flip=True,
#                                  width_shift_range=0.3,
#                                  height_shift_range=0.3) #no arguments, no augmentations
#     save_to_dir = 'D:/json_train/[원천]탈모_3.중증/'
    
#     i = 0
#     for batch in datagen.flow(img_array, batch_size=1, save_format='jpg'):
#         img_save = image.array_to_img(batch[0], scale=True) #scale=False did the trick
#         img_save.save(save_to_dir+f+fr'augment_{i}.jpg') #save image manually
#         if i == 18:
#             break
#         i += 1
        
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import os
# import numpy as np
# filenames = os.listdir('D:/json_train/[원천]탈모_2.중등도')
# for f in filenames:
#     img = load_img('D:/json_train/[원천]탈모_2.중등도/'+f)  
#     x = img_to_array(img) 
#     # Reshape the input image 
#     x = x.reshape((1, ) + x.shape)  
#     i = 0

#     # generate 5 new augmented images 
#     for batch in train_data_generator.flow(x, batch_size = 1, 
#                       save_to_dir ='D:/json_train/[원천]탈모_2.중등도/',  
#                       save_prefix =f, save_format ='jpg'):
#         i += 1
#         if i > 2: 
#             break

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.efficientnet import EfficientNetB0
# from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, GlobalMaxPool2D, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout,Flatten
# from keras import regularizers

model = Sequential()
base_model = ResNet50(include_top = False, weights = "imagenet", pooling=MODEL_POOLING)#include_top = False
# model.add(Dropout(0.5)) # 0.5
# base_model.summary()
model.add(base_model)
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))
model.layers[0].trainable = True

# model.summary()


opt = optimizers.SGD(learning_rate=lr, momentum=0.5)
model.compile(optimizer = opt, loss = OBJECTIVE_FUNCTION, metrics = LOSS_METRICS)


cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = EARLY_STOP_PATIENCE, 
                                  mode = 'min')
cb_checkpointer = ModelCheckpoint(filepath = 'D:/best_weight_json_4.hdf5', 
                                  monitor = 'val_loss', 
                                  save_best_only = True, mode='min')

weight_for_0 = (1 / 16869) * (trImgNum / 4.0)
weight_for_1 = (1 / 13346) * (trImgNum / 4.0)
weight_for_2 = (1 / 3797) * (trImgNum / 4.0)
weight_for_3 = (1 / 836) * (trImgNum / 4.0)
class_weights = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}

fit_history = model.fit_generator(
        train_generator,
        epochs = NUM_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
        validation_steps=STEPS_PER_EPOCH_VALIDATION,
        validation_data=validation_generator,
        callbacks=[cb_checkpointer, cb_early_stopper],
        class_weight=class_weights
)


print(fit_history.history.keys())

plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(fit_history.history['accuracy'])  
plt.plot(fit_history.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222)  
plt.plot(fit_history.history['loss'])  
plt.plot(fit_history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()

# predict - confusion matrix

import pandas as pd
label = validation_generator.classes
predictions = model.predict(validation_generator)

guess = [np.argmax(i) for i in predictions]
df_re = pd.DataFrame([label,guess])
df_re = df_re.T
from sklearn.metrics import confusion_matrix
print(confusion_matrix(df_re.iloc[:,0], df_re.iloc[:,1]))

# convert tflite
import tensorflow as tf
saved_model_dir = "D:/"
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)