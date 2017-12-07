import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import pickle
from keras.utils import to_categorical
import os
# dimensions of our images.
img_width, img_height, channels = 160, 160, 3

# from collections import Counter
# train_count = Counter(train_y)
# print(train_count)
# val_count = Counter(validation_y)
# print(val_count)

# train_y_int = [ord(x)-97 for x in train_y]
# train_y = to_categorical(train_y_int)
# #train_y = [list(x) for x in zip(*train_y)]

# val_y_int = [ord(x)-97 for x in validation_y]
# validation_y = to_categorical(val_y_int)
# #validation_y = [list(x) for x in zip(*validation_y)]

# print(test_y)
top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = './data/train'
validation_data_dir = './data/validation'
#test_data_dir = './data/test'

# nb_train_samples = len(train_y)
# nb_validation_samples = len(validation_y)
# nb_test_samples = len(test_y)
epochs = 50
batch_size = 128


def create_labels(parent_dir):
    labels = []
    for class_ in os.listdir(parent_dir):
        for file in os.listdir(os.path.join(parent_dir, class_)):
            if(file[0]=='d'):
                os.remove(os.path.join(parent_dir, class_, file))
            else:
                labels.append(ord(class_)-97)
    return labels

train_y_ = create_labels(train_data_dir)
train_y_.sort()
train_y = to_categorical(train_y_)

validation_y_ = create_labels(validation_data_dir)
validation_y_.sort()
validation_y = to_categorical(validation_y_)

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    model = applications.mobilenet.MobileNet(
        input_shape=(img_width, img_height, channels), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', classes=122)

    generator_train = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator_train, len(train_y) // batch_size)
    np.save(open('bottleneck_features_train1.npy', 'wb'),
            bottleneck_features_train)
    #pickle.dump(generator_train, open('gen.pkl','wb'))
    # generator_validation = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_validation = model.predict_generator(
    #     generator_validation, len(validation_y) // batch_size)
    # np.save(open('bottleneck_features_validation1.npy', 'wb'),
    #         bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train1.npy', 'rb'))
    #validation_data = np.load(open('bottleneck_features_validation1.npy', 'rb'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax'))

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_y,
              epochs=epochs,
              batch_size=batch_size)
    #model.fit_generator(generator_train,
    #                     24// batch_size,
    #                     epochs=epochs)
    model.save_weights(top_model_weights_path)
    #test_data = np.load(open('bottleneck_features_test.npy', 'rb'))
    #model.evaluate_generator(test_data, 32)


def test_model():

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

#save_bottlebeck_features()
train_top_model()
# test_model()
