import os
import math
import csv
import numpy as np
from PIL import Image


label_str_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def convert_images_and_labels(rows, dataset_type, dataset_path, wf_label):
	dataset = [row[:-1] for row in rows if row[-1] == dataset_type]
	num_dataset = len(dataset)
	#csv.writer(open('train.csv', 'w+')).writerows([header[:-1]] + dataset)
	
	for i in range(num_dataset):
		label = dataset[i][0]
		img_size = int(math.sqrt(len(dataset[i][1].split()))) # 48
		img_arr = np.zeros((img_size, img_size), dtype=np.uint8)
		for idx, val in enumerate(dataset[i][1].split()):
			img_arr[idx/img_size, idx%img_size] = val
		img_name = os.path.join(dataset_path, label_str_list[int(label)], dataset_type + '_' + str(i) + '.png')
		img = Image.fromarray(img_arr)
		img.save(img_name)
		wf_label.write(img_name.replace('\\', '/') + ' ' + label + '\n')
	print(dataset_type, num_dataset)


def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)



data_path = os.path.join('data')
make_dir(data_path)
fer2013_path = os.path.join(data_path, 'fer2013')
make_dir(fer2013_path)
train_path = os.path.join(fer2013_path, 'train')
make_dir(train_path)
test_path = os.path.join(fer2013_path, 'test')
make_dir(test_path)
for dataset_path in [train_path, test_path]:
	for label in label_str_list:
		label_path = os.path.join(dataset_path, label)
		make_dir(label_path)

csvr = csv.reader(open('fer2013.csv'))
header = csvr.next() #next(csvr)
rows = [row for row in csvr]

wf_label = open(os.path.join(data_path, 'fer2013_labels.txt'), 'w')
convert_images_and_labels(rows, 'Training',    train_path, wf_label)
convert_images_and_labels(rows, 'PublicTest',   test_path, wf_label)
convert_images_and_labels(rows, 'PrivateTest',  test_path, wf_label)
wf_label.close()
