from keras.callbacks import EarlyStopping, LambdaCallback
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Conv2D, \
    MaxPooling2D, \
    AveragePooling2D, \
    Flatten, \
    Dense, \
    GlobalAveragePooling2D, \
    Activation, \
    BatchNormalization, \
    Dropout, \
    Input, \
    add, \
    Concatenate
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras import optimizers
#from keras.callbacks import callbacks
import matplotlib.pyplot as plt
#from keras.optimizers import Adadelta
from datetime import datetime
import os
now = datetime.now()
date_time = now.strftime("pol_%m-%d-%Y_%H:%M:%S")

train_data_dir='/home/mcv/datasets/MIT_split/train'
val_data_dir='/home/mcv/datasets/MIT_split/test'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 256
img_height = 256
batch_size  = 32
number_of_epoch = 150
validation_samples = 807

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 114.717
        x[ 1, :, :] -= 115.455
        x[ 2, :, :] -= 109.076
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 114.717
        x[:, :, 1] -= 115.455
        x[:, :, 2] -= 109.076
    return x

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print('Building the model...\n')

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(img_width,img_height,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (1,1), padding='same', activation='relu'))
model.add(BatchNormalization())
# aqui
model.add(GlobalAveragePooling2D())
#model.add(Dropout(0.2))
#model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units = 8, activation='softmax'))

'''model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(img_width,img_height,3), padding='same', activation='relu'))
#model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
#model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dense(units = 8, activation='softmax'))'''

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print('Done!\n')

os.mkdir(date_time)
print(model.summary())
plot_model(model, to_file=(date_time + '/model_'+date_time+'.png'), show_shapes=True, show_layer_names=True)


train_datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             preprocessing_function=preprocess_input,
                             rotation_range=5,
                             width_shift_range=0.,
                             height_shift_range=0.,
                             shear_range=0.,
                             zoom_range=0.,
                             channel_shift_range=0.,
                             fill_mode='nearest',
                             cval=0.,
                             horizontal_flip=True,
                             vertical_flip=False,
                             rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(val_data_dir,
                                                   target_size=(img_width, img_height),
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_data_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')


history = model.fit_generator(train_generator,
                              steps_per_epoch=(int(1881 // batch_size) + 1),
                              nb_epoch=number_of_epoch,
                              validation_data=validation_generator,
                              validation_steps=(int(validation_samples // batch_size) + 1))

result = model.evaluate_generator(test_generator, val_samples=validation_samples)
print(result)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(date_time + '/accuracy_' + date_time + '.jpg')
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(date_time + '/loss_' + date_time + '.jpg')