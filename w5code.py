from keras.models import Sequential, Model, load_model
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.utils import plot_model
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop, Adadelta, Adam

import os
from os.path import join
import argparse
import itertools as it

parser = argparse.ArgumentParser(description='Define the model')
parser.add_argument('--data', default='/home/mcv/datasets/MIT_split', 
                    choices=['/home/mcv/datasets/MIT_split','data/MIT'],
                    help='path to data (default: /home/mcv/datasets/MIT_split)')
parser.add_argument('--epochs', default=30, type=int, help='define epochs size')
parser.add_argument('--batch_size', default=32, type=int, help='define batch size')
parser.add_argument('--workers', default=16, type=int, help='define workers number')
parser.add_argument('--optimizer', default='adam', choices=['adadelta','rmsprop','adam'], help='define optimizer')
parser.add_argument('--last_layer', default=False, type=bool, help='if True train just last layer')
parser.add_argument('--lr', default=1e-3, type=float, help='define learning rate')

args = parser.parse_args()

train_data_dir=args.data+'/train'
test_data_dir='/home/mcv/datasets/MIT_split/test'
img_width = 224
img_height=224
batch_size=args.batch_size
number_of_epoch=args.epochs
validation_samples=807

model_dir=join('w5','grid_search')
os.makedirs(model_dir, exist_ok=True)

layer2 = 0
layer3 = 0

#build model
def build_model(train_generator, validation_generator, params=None):
    global layer2
    global layer3

    model = Sequential()
    model.add(Convolution2D(params['num_filters1'], params['kernel_size1'], strides=4, padding='valid', input_shape=(224, 224, 3)))#, kernel_initializer='glorot_normal'
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    model.add(Convolution2D(params['num_filters2'], 3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    if params['num_layers']>2:
        model.add(Convolution2D(params['num_filters3'], 3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
        model.add(Dropout(params['dropout']))

        model_dir2 = join(model_dir, str(params['num_layers']), 'model'+str(layer3), str(params['dropout']))
        layer3 += 1
    else:

        model_dir2 = join(model_dir, str(params['num_layers']), 'model'+str(layer2), params['optimizer'])
        layer2 += 1        
    
    model.add(Convolution2D(8,1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    os.makedirs(model_dir2, exist_ok=True)
    plot_model(model, to_file=join(model_dir2,'model.png'), show_shapes=True, show_layer_names=True)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer=params['optimizer'],
                metrics=['accuracy'])

    tensorboard_cb = TensorBoard(log_dir=model_dir2 + '/logs', batch_size=batch_size, histogram_freq=0,
                                    update_freq='epoch')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    checkpointcallback = ModelCheckpoint(filepath=model_dir2+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',period=1)

    history=model.fit_generator(train_generator,
            steps_per_epoch=(int((train_generator.samples)//batch_size)+1),
            nb_epoch=params['num_layers']*number_of_epoch,
            validation_data=validation_generator,
            validation_steps= (int((validation_generator.samples)//batch_size)+1),
            workers=args.workers,
            callbacks=[checkpointcallback, reduce_lr, tensorboard_cb])
    
    model.save(join(model_dir2,'model.h5'))

    return model, history


datagen = ImageDataGenerator(
    rotation_range=5,
    horizontal_flip=True,
    rescale=1./255,
    validation_split=.2)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

validation_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


best_models = [
    {
    'optimizer': ['adadelta'],
    'num_layers': [3],
    'kernel_size1': [7],
    'num_filters1': [32],
    'num_filters2': [16],
    'num_filters3': [64],
    'dropout': [0.5]
    },
    {
    'optimizer': ['adadelta'],
    'num_layers': [3],
    'kernel_size1': [5],
    'num_filters1': [16],
    'num_filters2': [16],
    'num_filters3': [64],
    'dropout': [0.5]
    }

]

combinations=[]
for all_params in best_models:
    keys = all_params.keys()
    values = (all_params[key] for key in keys)
    combinations.extend([dict(zip(keys, combination)) for combination in it.product(*values)])

results=[]

for params in combinations:
    model, _ = build_model(train_generator, validation_generator, params)

    result = model.evaluate_generator(test_generator, steps=(int((validation_samples)//batch_size)+1), workers=8, verbose=1)

    results.append(result[1])

    print("For the optimizer "+ params['optimizer']+" we got and accuracy of "+ str(result[1]))

print(results)
