import os
main_data_dir = './dataset5'
data = './data'
train_dir = './data/train'
path_file = './'
if not os.path.exists(data):
    os.makedirs(data)


def class_dir(parent_path):
    for i in range(97, 122):
        if(i != 122 and i != 106):
            if not os.path.exists(os.path.join(parent_path, chr(i))):
                os.makedirs(os.path.join(parent_path, chr(i)))

counter = 0
label_train = []
for class_ in os.listdir(data):
	for file in os.listdir(os.path.join(data, class_)):
			label_train.append(class_)

print(len(label_train))
class_dir(train_dir)
sub_dir = os.listdir(main_data_dir)
for data in sub_dir:
	data1 = os.path.join(main_data_dir, data)
	for alphabet in os.listdir(data1):
		letters = os.path.join(main_data_dir, data, alphabet)
        #print(letters)
        for image in os.listdir(letters):
            image_file_path = os.path.join(
                main_data_dir, data, alphabet, image)
            # print(image_file_path)
            # print(image_file_path, os.path.join(
            #     train_dir, alphabet, image))
            os.rename(image_file_path, os.path.join(
                train_dir, alphabet, image))
