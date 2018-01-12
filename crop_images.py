import os
import argparse
import numpy as np
from PIL import Image


def crop_img(img_path, img_row_num, img_size):
	img = Image.open(img_path)
	img = np.array(img, dtype='uint8')
	return img[img_row_num*img_size:(img_row_num+1)*img_size, :, :]


def to_crop_img_process(cropped_imgs, gen_dirs, dataset, training_type, img_num, img_row_num, img_size):
	for i, dir in enumerate(gen_dirs):
		result_path = os.path.join(dir, 'results')
		if training_type == 'multi':
			img_path = os.path.join(result_path, dataset+'_'+str(img_num)+'_fake.png')
		else:
			img_path = os.path.join(result_path, str(img_num)+'_fake.png')
		cropped_img = crop_img(img_path, img_row_num, img_size)
		if i > 0:
			cropped_imgs[i*img_size:(i+1)*img_size, img_size:, :] = cropped_img[:, img_size:, :]
		else:
			cropped_imgs[i*img_size:(i+1)*img_size, :, :] = cropped_img


def save_crop_imgs(cropped_imgs, dataset, training_type, img_num, img_row_num):
	crop_result_path = 'crop_result'
	if not os.path.exists(crop_result_path):
		os.mkdir(crop_result_path)

	if training_type == 'multi':
		cropped_img_name = os.path.join(crop_result_path, dataset+'_'+str(img_num)+'_'+str(img_row_num)+'_fake.png')
	else:
		cropped_img_name = os.path.join(crop_result_path, str(img_num)+'_'+str(img_row_num)+'_fake.png')
	cropped_imgs = Image.fromarray(cropped_imgs)
	cropped_imgs.save(cropped_img_name)



parser = argparse.ArgumentParser()
parser.add_argument('--img_num', help='img_num', type=int, default=1)
parser.add_argument('--img_row_num', help='img_row_num', type=int, default=0)
parser.add_argument('--dataset', help='dataset', type=str, default='fer2013')
parser.add_argument('--type', help='training_type', type=str, default='single')
args = parser.parse_args()

img_num = args.img_num
img_row_num = args.img_row_num
dataset = args.dataset # ['fer2013', 'ferg_db']
training_type = args.type

num_classes = 7
img_size = 64

if training_type == 'multi':
	gen_dirs = ['stargan_both_orig', ] # 'stargan_both_res_pixel_dropout'
	num_dirs = len(gen_dirs_multi)
	cropped_imgs = np.ones((img_size*num_dirs, img_size*(num_classes*2+1), 3), dtype='uint8') * 255
else:
	gen_dirs = ['stargan_fer2013_orig', 'stargan_fer2013_rec', 'stargan_fer2013_res_rec', 'stargan_fer2013_dropout', 'stargan_fer2013_res_pixel', 'stargan_fer2013_res_pixel_dropout']
	num_dirs = len(gen_dirs)
	cropped_imgs = np.ones((img_size*num_dirs, img_size*(num_classes+1), 3), dtype='uint8') * 255

to_crop_img_process(cropped_imgs, gen_dirs, dataset, training_type, img_num, img_row_num, img_size)
save_crop_imgs(cropped_imgs, dataset, training_type, img_num, img_row_num)
