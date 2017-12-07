import os
import pickle
import shutil
import numpy as np
dir_data = './temp'


def create_labels(parent_dir):
    labels = np.zeros((25042, 24))
    for class_ in os.listdir(parent_dir):
        for file in os.listdir(os.path.join(parent_dir, class_)):
            labels[ord(class_)-97]=1
    return labels


def class_dir(parent_path):
    for i in range(97, 123):
        if(i != 122 and i != 106):
            if not os.path.exists(os.path.join(parent_path, chr(i))):
                os.makedirs(os.path.join(parent_path, chr(i)))


def shift(main_data_dir, train_dir):
    sub_dir = os.listdir(main_data_dir)
    for data in sub_dir:
        data1 = os.path.join(main_data_dir, data)
        print(data1)
        print(os.listdir(data1))
        for alphabet in os.listdir(data1):
            letters = os.path.join(main_data_dir, data, alphabet)
        print('leters', letters)
        print('alphabet', os.listdir(letters))
        if(len(os.listdir(letters)) > 0):
            print('True')
            for image in os.listdir(letters):
                image_file_path = os.path.join(
                    main_data_dir, data, alphabet, image)
                print(image_file_path)
                # print(image_file_path, os.path.join(
                #     train_dir, alphabet, image))
                os.rename(image_file_path, os.path.join(
                    train_dir, alphabet, image))


def shift1(inital, train):
    for data in os.listdir(inital):
        for alphabet in os.listdir(os.path.join(inital, data)):
            for image in os.listdir(os.path.join(inital, data, alphabet)):
                if not os.path.exists(os.path.join('..' + train, alphabet, image)):
	                train__ = train
	                print((os.path.join(train__, alphabet, image)))
	                shutil.move((os.path.join(inital, data, alphabet, image)),
	                            (os.path.join(train__, alphabet, image)))

class_dir('./a')
#shift1(dir_data, './train')
# labels_train = create_labels('./train')
# labels_validation = create_labels('./validation')
# labels_test = create_labels('./test')

pickle.dump([labels_train, labels_validation], open('labels.pkl','wb'))
